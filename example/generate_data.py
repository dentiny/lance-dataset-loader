"""Create the bundled synthetic datasets, or write copies to a fresh directory."""

import argparse
import json
from pathlib import Path

import lance
import pyarrow as pa

FIXTURES = {
    "coding": {"rows": 8, "tokens_per_sequence": 8, "first_id": 0, "first_token": 100},
    "math": {"rows": 12, "tokens_per_sequence": 4, "first_id": 0, "first_token": 500},
    "general": {"rows": 8, "tokens_per_sequence": 4, "first_id": 0, "first_token": 800},
    "coding_v2": {
        "rows": 8,
        "tokens_per_sequence": 8,
        "first_id": 100,
        "first_token": 1200,
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "data")
    output = parser.parse_args().output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field("source", pa.string()),
            pa.field("text", pa.string()),
            pa.field("input_ids", pa.list_(pa.int32())),
        ]
    )
    for name, spec in FIXTURES.items():
        path = output / f"{name}.lance"
        if path.exists():
            print(f"Keeping existing dataset: {path}")
            continue
        rows = []
        for index in range(spec["rows"]):
            row_id = spec["first_id"] + index
            length = spec["tokens_per_sequence"]
            start = spec["first_token"] + index * length
            rows.append(
                {
                    "id": row_id,
                    "source": name,
                    "text": f"Synthetic {name} sample {row_id}; token IDs are illustrative.",
                    "input_ids": list(range(start, start + length)),
                }
            )
        lance.write_dataset(
            pa.Table.from_pylist(rows, schema=schema),
            str(path),
            data_storage_version="2.0",
            enable_v2_manifest_paths=False,
        )
        print(f"Created {name}: {len(rows)} rows, {spec['tokens_per_sequence']} tokens/sequence")
    catalog = {
        "description": "Synthetic examples; token IDs are not produced by a real tokenizer.",
        "storage_version": "2.0",
        "datasets": {
            name: {"path": f"{name}.lance", "version": 1, **spec} for name, spec in FIXTURES.items()
        },
    }
    (output / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")


if __name__ == "__main__":
    main()
