from __future__ import annotations

from pathlib import Path

from retriever import HybridRetriever


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    retriever = HybridRetriever(repo_root=repo_root)
    print(f"Indexed {len(retriever.records)} chunks")
    print(retriever.describe_backend())


if __name__ == "__main__":
    main()
