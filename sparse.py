import os
import time
import pickle
import asyncio
import tiktoken
import numpy as np
import numpy.typing as npt
from dotenv import load_dotenv
from cohere import AsyncClientV2
from typing import AsyncGenerator
from chonkie import SentenceChunker
from anthropic import AsyncAnthropic
from prompts import ALPHA_TEMPLATE, FINAL_RESPONSE_TEMPLATE
from sklearn.feature_extraction.text import TfidfVectorizer
from numba import (
    njit,
    prange,
    get_thread_id,
    get_num_threads
)
from openai import (
    Timeout,
    APIError,
    AsyncOpenAI,
    RateLimitError,
    APIConnectionError
)
from tenacity import (
    retry,
    wait_exponential,
    stop_after_attempt,
    retry_if_exception_type
)

load_dotenv()

N_THREADS = get_num_threads()
oai_client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)
ant_client = AsyncAnthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)
cohere_client = AsyncClientV2(
    api_key=os.getenv("COHERE_API_KEY"),
)
TOKENIZER = tiktoken.get_encoding("o200k_base")
# use a faster chunker in prod
CHUNKER = SentenceChunker(
    tokenizer_or_token_counter=TOKENIZER,
    chunk_size=2048,
    chunk_overlap=200,
    min_sentences_per_chunk=1,
)

def load(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise e
    
def save(file_path: str, data: dict) -> None:
    with open(file_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

@njit(
    fastmath=True,
    parallel=True,
    cache=True,
    nogil=True,
    boundscheck=False,
    inline="always",
    error_model="numpy",
)
def normalize(data: npt.NDArray, indices: npt.NDArray, indptr: npt.NDArray, shape: tuple) -> tuple:
    local_sums = np.zeros((N_THREADS, shape[1]), dtype=np.float32)
    for i in prange(shape[0]):
        tid = get_thread_id()
        for j in range(indptr[i], indptr[i+1]):
            local_sums[tid, indices[j]] += data[j]
    sums = local_sums.sum(axis=0)
    sums[sums == 0] = 1
    for i in prange(shape[0]):
        for j in range(indptr[i], indptr[i+1]):
            data[j] /= sums[indices[j]]
    return data, indices, indptr

@njit(
    fastmath=True,
    parallel=True,
    cache=True,
    nogil=True,
    boundscheck=False,
    inline="always",
    error_model="numpy",
)
def pagerank(
    pi: npt.NDArray,
    data: npt.NDArray,
    indices: npt.NDArray,
    indptr: npt.NDArray,
    alpha: float,
    pers: npt.NDArray
) -> npt.NDArray:
    pi_next = np.zeros_like(pi)
    for j in prange(len(pi)):
        for i in range(indptr[j], indptr[j+1]):
            pi_next[indices[i]] += data[i] * pi[j]
    return (1 - alpha) * pi_next + alpha * pers

class Retriever:
    def __init__(self):
        self._vectorizer = TfidfVectorizer(norm='l2', sublinear_tf=True, dtype=np.float32)
        self._sim_thresh = 0.27
        self._max_iter = 18
        self._top_k = 50
        self._top_n = 10

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type((RateLimitError, APIConnectionError, APIError, Timeout)),
    )
    async def _get_alpha(self, query: str) -> float:
        return float((await oai_client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.0,
            top_p=0.0,
            messages=[{"role": "user", "content": ALPHA_TEMPLATE.format(query=query)}]
        )).choices[0].message.content)
    
    def ingest(self, text: str) -> tuple[dict, float]:
        start = time.time()
        chunks = [chunk.text for chunk in CHUNKER(text)]
        print(f"Chunking: {time.time() - start:.2f} seconds")
        start = time.time()
        E = self._vectorizer.fit_transform(chunks)
        A = E @ E.T
        A.data[A.data < self._sim_thresh] = 0
        A.eliminate_zeros()
        A.data, A.indices, A.indptr = normalize(A.data, A.indices, A.indptr, A.shape)
        return {
            "full_text": text,
            "chunks": chunks,
            "vectorizer": self._vectorizer,
            "embeddings": E,
            "adj_matrix": {
                "data": A.data,
                "indices": A.indices,
                "indptr": A.indptr,
            }
        }, start
    
    async def query(self, query: str, data: dict) -> AsyncGenerator[str, None]:
        vectorizer, chunks, E, data, indices, indptr = (
            data["vectorizer"], data["chunks"], data["embeddings"], data["adj_matrix"]["data"],
            data["adj_matrix"]["indices"], data["adj_matrix"]["indptr"]
        )
        alpha = await self._get_alpha(query)
        pers = (E @ vectorizer.transform([query]).T).toarray().flatten()
        pers /= (pers.sum() + 1e-9)
        pi = np.ones(n_chunks := len(chunks), dtype=np.float32) / n_chunks
        for _ in range(self._max_iter):
            if np.allclose(
                pi,
                pi_next := pagerank(pi, data, indices, indptr, alpha, pers)
            ):
                break
            pi = pi_next
        k = min(self._top_k, len(pi))
        top_k = np.argpartition(pi, -k)[-k:]
        top_k.sort()
        final_chunks = [chunks[i] for i in top_k]
        reranked = await cohere_client.rerank(
            model="rerank-v3.5",
            query=query,
            documents=final_chunks,
            top_n=self._top_n,
        )
        context = "\n\n\n\n".join(
            f"{i}. {final_chunks[result.index]}" for i, result in enumerate(reranked.results, 1)
        )
        print(f"Estimated token count of final context: {len(TOKENIZER.encode(context))}")
        try:
            async with ant_client.messages.stream(
                model="claude-3-7-sonnet-20250219",
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text", "text": FINAL_RESPONSE_TEMPLATE.format(context=context, query=query)
                        }
                    ]
                }],
                temperature=0.0,
                top_p=0.0,
                max_tokens=8192,
            ) as stream:
                async for chunk in stream.text_stream:
                    yield chunk
        except Exception as e:
            yield str(e)

async def test(user: str, files: list[str], query: str) -> None:
    os.makedirs(f"users/{user}", exist_ok=True)
    assert user, "User is required"
    assert files, "At least one file is required"
    assert query, "Query is required"
    retriever = Retriever()

    text = ""
    for file in files:
        with open(file, "r") as f:
            text += "\n\n" + f.read()
    print(f"Estimated token count of all file contents: {len(TOKENIZER.encode(text))}")
    
    data, start = retriever.ingest(text)
    save(f"users/{user}/data.pkl", data)
    print(f"Ingestion: {time.time() - start:.2f} seconds")
    
    first = True
    start = time.time()
    async for chunk in retriever.query(query, load(f"users/{user}/data.pkl")):
        if first:
            print(f"TTFT: {time.time() - start:.2f} seconds")
            first = False
        print(chunk, end="", flush=True)

if __name__ == "__main__":
    user = "test"
    files = ["book1.txt", "book2.txt"] # ~1M tokens
    query = ""
    asyncio.run(test(user=user, files=files, query=query))
