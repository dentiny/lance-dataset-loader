"""Simulate rank slices and restore the same global stream at a new DP size."""

from _common import DATA_DIR, show_batch

from lance_dataset_loader import LanceSource, MixtureDataset


def create_dataset():
    sources = {
        name: LanceSource(str(DATA_DIR / f"{name}.lance"), version=1) for name in ("coding", "math")
    }
    return MixtureDataset(sources, {"coding": 0.5, "math": 0.5}, global_batch_size=8)


def main():
    state = None
    for world_size in (1, 2, 4):
        print(f"Same global step, world_size={world_size}:")
        combined = []
        for rank in range(world_size):
            dataset = create_dataset()
            with dataset.iter_batches(rank=rank, world_size=world_size) as batches:
                batch = next(batches)
                show_batch(batch)
                combined.extend(
                    (sample.source_id, row["id"])
                    for sample, row in zip(batch.samples, batch.rows, strict=True)
                )
                dataset.commit(batch)
                state = dataset.state_dict()
        print("Concatenated ordered global batch:", combined)
    print("Restore the committed step on two ranks:")
    for rank in range(2):
        restored = create_dataset()
        restored.load_state_dict(state)
        with restored.iter_batches(rank=rank, world_size=2) as batches:
            batch = next(batches)
            show_batch(batch)
            restored.commit(batch)
    print("This is a CPU simulation; real trainers own distributed synchronization.")


if __name__ == "__main__":
    main()
