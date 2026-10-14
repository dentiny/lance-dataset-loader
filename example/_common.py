"""Paths and display helpers shared by the feature examples."""

from pathlib import Path

from lance_dataset_loader import TrainingBatch

DATA_DIR = Path(__file__).resolve().parent / "data"


def show_batch(batch: TrainingBatch) -> None:
    print(f"step={batch.step}, rank={batch.rank}/{batch.world_size}, rows={len(batch.rows)}")
    for sample, row in zip(batch.samples, batch.rows, strict=True):
        print(f"  {sample.source_id}: id={row['id']}, pass={sample.pass_index}")
