"""Run with `uv run python examples/train_and_resume.py`. Uses real local Lance files."""

import tempfile
from pathlib import Path

import lance
import pyarrow as pa

from lance_dataset_loader import LanceSource, MixtureDataset


def collect(dataset, steps, **kwargs):
    result = []
    with dataset.iter_batches(**kwargs) as batches:
        for _ in range(steps):
            batch = next(batches)
            # Replace this with tensor conversion and the actual training step.
            result.append(
                [(s.source_id, row["id"]) for s, row in zip(batch.samples, batch.rows, strict=True)]
            )
            dataset.commit(batch)
    return result


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        sources = {}
        for name, count in (("coding", 7), ("math", 11)):
            uri = str(root / f"{name}.lance")
            lance.write_dataset(
                pa.table({"id": range(count), "input_ids": [[1, 2, 3, 4]] * count}), uri
            )
            sources[name] = LanceSource(uri, token_column="input_ids", tokens_per_sequence=4)

        def create():
            return MixtureDataset(
                sources, {"coding": 0.2, "math": 0.8}, global_batch_size=8, seed=42
            )

        expected = collect(create(), 6)
        dataset = create()
        assert collect(dataset, 2, read_workers=2, prefetch_batches=3) == expected[:2]
        checkpoint = root / "dataset.json"
        dataset.save_checkpoint(checkpoint)
        restored = create()
        restored.load_checkpoint(checkpoint)
        assert collect(restored, 4, read_workers=4) == expected[2:]
        print("Exact resume verified with real Lance datasets.")

        for rank in range(2):
            distributed = create()
            distributed.load_checkpoint(checkpoint)
            batches = collect(distributed, 4, rank=rank, world_size=2)
            assert batches == [batch[rank * 4 : (rank + 1) * 4] for batch in expected[2:]]
        print("Changing DP size from 1 to 2 preserved ordered global batches.")

        new_uri = str(root / "coding_v2.lance")
        lance.write_dataset(
            pa.table({"id": range(100, 105), "input_ids": [[5, 6, 7, 8]] * 5}), new_uri
        )
        replacements = dict(restored.sources)
        replacements["coding"] = LanceSource(
            new_uri, token_column="input_ids", tokens_per_sequence=4
        )
        restored.reconfigure(
            sources=replacements, reset_sources={"coding"}, weights={"coding": 0.5, "math": 0.5}
        )
        print("Source replacement began a new phase:", collect(restored, 1)[0])


if __name__ == "__main__":
    main()
