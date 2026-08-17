import csv
from pathlib import Path

from langchain_core.documents import Document


def load_csv_file(path: Path) -> list[Document]:
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        return [
            Document(
                page_content="\n".join(
                    f"{column}: {value}"
                    for column, value in row.items()
                ),
                metadata={
                    "source": str(path),
                    "file_type": "csv",
                    "row": row_number,
                },
            )
            for row_number, row in enumerate(reader, start=1)
        ]
