"""Prompt templates for grounded RAG generation."""

SYSTEM_INSTRUCTION = (
    "You are a helpful search assistant. Answer the user's question using ONLY "
    "the provided passage excerpts. If the passages do not contain enough "
    "information, say you cannot answer confidently from the sources. "
    "Cite supporting passages as [DOC <doc_id>] after each claim. "
    "Do not invent facts not present in the passages."
)


def build_user_prompt(query: str, context_blocks: str) -> str:
    return (
        f"Question: {query.strip()}\n\n"
        f"Retrieved passages:\n{context_blocks}\n\n"
        "Answer in clear natural language with citations."
    )
