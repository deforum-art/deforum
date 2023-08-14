from pathlib import Path

filler_words = set(
    [s.strip() for s in (Path(__file__).resolve().parent / "stopwords_combined.txt").read_text().splitlines()]
)
