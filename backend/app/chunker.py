import re


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into chunks, preferring paragraph boundaries.

    Paragraphs are packed into chunks up to `chunk_size` characters. Paragraphs
    longer than `chunk_size` are hard-split on word boundaries with `overlap`
    characters carried over between consecutive pieces.
    """
    text = re.sub(r"\r\n?", "\n", text or "").strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = ""
        if len(para) <= chunk_size:
            current = para
        else:
            words = para.split()
            buf = ""
            for word in words:
                if buf and len(buf) + len(word) + 1 > chunk_size:
                    chunks.append(buf)
                    tail = buf[-overlap:] if overlap else ""
                    buf = f"{tail} {word}".strip() if tail else word
                else:
                    buf = f"{buf} {word}".strip()
            current = buf

    if current:
        chunks.append(current)
    return [c for c in chunks if c]
