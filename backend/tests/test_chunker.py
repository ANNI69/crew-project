from app.chunker import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_short_text_is_single_chunk():
    text = "The Eiffel Tower is in Paris."
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert chunks == [text]


def test_chunks_respect_max_size():
    text = " ".join(f"word{i}" for i in range(2000))
    chunks = chunk_text(text, chunk_size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 260 for c in chunks)  # size + one word slack


def test_content_is_preserved():
    text = "alpha beta gamma delta epsilon zeta eta theta"
    chunks = chunk_text(text, chunk_size=25, overlap=5)
    joined = " ".join(chunks)
    for word in text.split():
        assert word in joined


def test_overlap_between_hard_split_pieces():
    text = " ".join(f"w{i}" for i in range(100))
    chunks = chunk_text(text, chunk_size=60, overlap=20)
    assert len(chunks) > 2
    # tail of one chunk reappears at the start of the next
    assert chunks[1].startswith(chunks[0][-20:].strip())


def test_paragraphs_preferred_over_mid_sentence_splits():
    text = "First paragraph here.\n\nSecond paragraph here."
    chunks = chunk_text(text, chunk_size=500, overlap=10)
    assert chunks == [text]
