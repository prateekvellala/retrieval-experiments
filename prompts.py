ALPHA_TEMPLATE = """
Is this query asking about broad themes, summaries, or patterns (rather than specific facts or details)? Answer '0.0' for broad/global queries, '0.6' for specific/local queries.

---Query---
{query}
"""

FINAL_RESPONSE_TEMPLATE = """
Answer the query using the given context.

---Instructions---
1. Each statement must be supported by an EXACT word-for-word quote from the context.
2. Citations must use the format: <citation></citation>.
3. Citations must be placed INLINE (on the same line) at the end of the statement they support - never on the next line.
4. When multiple quotes support a single statement, cite them as separate citations one after another, like: <citation></citation><citation></citation>.

---Context---
{context}

---Query---
{query}
"""
