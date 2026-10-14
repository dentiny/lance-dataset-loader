"""Concurrent reads may run ahead; checkpoint progress waits for commit()."""

from _common import DATA_DIR, show_batch

from lance_dataset_loader import LanceSource, MixtureDataset


def main():
    sources = {
        name: LanceSource(str(DATA_DIR / f"{name}.lance"), version=1) for name in ("coding", "math")
    }
    dataset = MixtureDataset(sources, {"coding": 0.5, "math": 0.5}, global_batch_size=4)
    with dataset.iter_batches(read_workers=4, prefetch_batches=3) as batches:
        batch = next(batches)
        show_batch(batch)
        print("After fetching, next_step:", dataset.state_dict()["progress"]["next_step"])
        dataset.commit(batch)
        print("After training and committing, next_step:", dataset.next_step)
        print(
            "Committed source cursors:",
            dataset.state_dict()["progress"]["source_cursors"],
        )
    print("Worker count affects I/O concurrency, not source selection or row order.")


if __name__ == "__main__":
    main()
