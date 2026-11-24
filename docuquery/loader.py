"""Load unstructured (text/markdown) and structured (CSV) data into a common record format."""
from pathlib import Path

import pandas as pd


def load_text_documents(folder):
    """Read every .txt / .md file in a folder (unstructured data)."""
    records = []
    folder = Path(folder)
    for path in sorted(list(folder.glob("*.txt")) + list(folder.glob("*.md"))):
        records.append({"source": path.name, "text": path.read_text(encoding="utf-8")})
    return records


def load_csv_records(csv_path):
    """Turn each CSV row into a searchable text record (structured data)."""
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path).fillna("")
    records = []
    for i, row in df.iterrows():
        text = " | ".join(f"{col}: {row[col]}" for col in df.columns)
        records.append({"source": f"{csv_path.name}#row{i + 1}", "text": text})
    return records
