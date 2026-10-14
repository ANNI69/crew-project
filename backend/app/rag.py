UNAVAILABLE = "The requested information is unavailable in the provided documents."

SYSTEM_PROMPT = (
    "You are a question-answering assistant. Answer the question using ONLY the "
    "context provided below. Be concise. If the context does not contain the "
    f'answer, reply exactly: "{UNAVAILABLE}"'
)


def get_llm(settings):
    from openai import OpenAI

    return OpenAI(api_key=settings.groq_api_key, base_url="https://api.groq.com/openai/v1")


def build_context(hits: list[dict]) -> str:
    return "\n\n".join(
        f"[{i + 1}] {hit['doc_name']}:\n{hit['text']}" for i, hit in enumerate(hits)
    )


def answer_question(
    question: str,
    hits: list[dict],
    settings,
    llm=None,
) -> dict:
    """Generate an answer grounded in retrieved chunks.

    Returns {"answer": str, "grounded": bool, "sources": [...]}.
    """
    sources = [
        {
            "document": hit["doc_name"],
            "excerpt": hit["text"],
            "score": round(hit["score"], 3),
        }
        for hit in hits
    ]
    if not hits:
        return {"answer": UNAVAILABLE, "grounded": False, "sources": []}

    llm = llm or get_llm(settings)
    response = llm.chat.completions.create(
        model=settings.groq_model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{build_context(hits)}\n\nQuestion: {question}",
            },
        ],
    )
    answer = (response.choices[0].message.content or "").strip()
    if "information is unavailable" in answer.lower():
        # the LLM declined because the retrieved context lacks the answer
        return {"answer": UNAVAILABLE, "grounded": False, "sources": []}
    return {"answer": answer, "grounded": True, "sources": sources}
