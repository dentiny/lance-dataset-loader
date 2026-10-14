"""Physically concatenate compatible Lance tables into a new dataset."""

import tempfile
from collections import Counter
from pathlib import Path

import lance
import pyarrow as pa
from _common import DATA_DIR, show_batch

from lance_dataset_loader import LanceSource, MixtureDataset


def main():
    # This small-data example materializes the input tables; large jobs should
    # stream RecordBatches instead. Unlike mixing, each input row is copied once.
    tables = [
        lance.dataset(str(DATA_DIR / f"{name}.lance"), version=1).to_table()
        for name in ("coding", "math", "general")
    ]
    combined = pa.concat_tables(tables)
    with tempfile.TemporaryDirectory(prefix="lance-merge-") as directory:
        destination = str(Path(directory) / "merged.lance")
        lance.write_dataset(combined, destination, data_storage_version="2.0")
        source = LanceSource(destination, version=1)
        print("Merged rows:", source.snapshot.row_count)
        print("Original sources:", dict(Counter(combined.column("source").to_pylist())))
        dataset = MixtureDataset({"merged": source}, {"merged": 1}, global_batch_size=4)
        with dataset.iter_batches() as batches:
            batch = next(batches)
            show_batch(batch)
            dataset.commit(batch)
    print("Temporary merged output removed; bundled input datasets were untouched.")


if __name__ == "__main__":
    main()
