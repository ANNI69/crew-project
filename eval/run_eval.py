"""Evaluation runner for the RAG API.

Usage:
  1. Start the backend:  uvicorn app.main:app --port 8000  (with GROQ_API_KEY set)
  2. Run:                python eval/run_eval.py [--base-url http://localhost:8000]

Ingests the sample documents, asks every question in eval/questions.json, and
scores each answer:
  - answerable:  grounded answer containing at least one expected keyword
  - unanswerable: model correctly reports the information is unavailable
  - multi:       grounded answer drawing on >= min_sources chunks, with all
                 expected keywords present

Exits non-zero if any question fails.
"""

import argparse
import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def normalize(text: str) -> str:
    # gpt-oss sometimes emits narrow/no-break spaces (U+202F, U+00A0, U+2009)
    return (
        text.lower()
        .replace("\u202f", " ")
        .replace("\u00a0", " ")
        .replace("\u2009", " ")
    )


def contains_any(text: str, keywords: list[str]) -> bool:
    lower = normalize(text)
    return any(normalize(k) in lower for k in keywords)


def contains_all(text: str, keywords: list[str]) -> bool:
    lower = normalize(text)
    return all(normalize(k) in lower for k in keywords)


def score(qtype: str, result: dict, spec: dict) -> tuple[bool, str]:
    grounded = result["grounded"]
    answer = result["answer"]

    if qtype == "unanswerable":
        # passes if the retrieval threshold abstained OR the LLM declined,
        # since both produce the same explicit "unavailable" statement
        ok = "unavailable" in normalize(answer) or "don't know" in normalize(answer)
        return ok, "correctly reported unavailable" if ok else f"expected unavailable, got: {answer[:80]}"

    if not grounded or not answer.strip():
        return False, "not grounded / empty answer"

    if qtype == "answerable":
        ok = contains_any(answer, spec.get("expect_any", []))
        return ok, "keyword found" if ok else f"expected one of {spec.get('expect_any')} in: {answer[:80]}"

    if qtype == "multi":
        sources = len(result.get("sources", []))
        ok = sources >= spec.get("min_sources", 2) and contains_all(answer, spec.get("expect_all", []))
        detail = f"{sources} sources"
        if not ok:
            detail += f", expected {spec.get('expect_all')} in: {answer[:80]}"
        return ok, detail

    return False, f"unknown question type {qtype}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()

    suite = json.loads((ROOT / "eval" / "questions.json").read_text())
    client = httpx.Client(base_url=args.base_url, timeout=60)

    print(f"Ingesting {len(suite['documents'])} documents...")
    for rel_path in suite["documents"]:
        path = ROOT / rel_path
        with path.open("rb") as f:
            response = client.post(
                "/documents", files={"file": (path.name, f, "text/plain")}
            )
        response.raise_for_status()
        body = response.json()
        print(f"  {body['name']}: {body['chunks']} chunks")

    passed = failed = 0
    print(f"\nRunning {len(suite['questions'])} questions...\n")
    for spec in suite["questions"]:
        try:
            response = client.post("/ask", json={"question": spec["question"]})
            response.raise_for_status()
            ok, detail = score(spec["type"], response.json(), spec)
        except httpx.HTTPError as exc:
            ok, detail = False, f"request failed: {exc}"
        passed += ok
        failed += not ok
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {spec['id']:<8} ({spec['type']:<11}) {detail}")

    total = passed + failed
    print(f"\n{passed}/{total} passed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
