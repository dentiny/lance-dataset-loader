"""Resume the last committed step, including when reads ran ahead."""

import tempfile
from pathlib import Path

from _common import DATA_DIR, show_batch

from lance_dataset_loader import LanceSource, MixtureDataset


def create_dataset():
    sources = {
        name: LanceSource(str(DATA_DIR / f"{name}.lance"), version=1) for name in ("coding", "math")
    }
    return MixtureDataset(sources, {"coding": 0.3, "math": 0.7}, global_batch_size=4)


def main():
    dataset = create_dataset()
    with tempfile.TemporaryDirectory(prefix="lance-checkpoint-") as directory:
        checkpoint = Path(directory) / "data-state.json"
        with dataset.iter_batches(read_workers=2, prefetch_batches=3) as batches:
            for _ in range(2):
                batch = next(batches)
                show_batch(batch)
                dataset.commit(batch)
            uncommitted = next(batches)
            print("Fetched but NOT trained/committed:")
            show_batch(uncommitted)
            dataset.save_checkpoint(checkpoint)
        restored = create_dataset()
        restored.load_checkpoint(checkpoint)
        print("Restored next_step:", restored.next_step)
        with restored.iter_batches() as batches:
            replayed = next(batches)
            print("Resumed batch (replays the uncommitted batch):")
            show_batch(replayed)
            restored.commit(replayed)
    print("Production: save dataset.state_dict() with model and optimizer state.")


if __name__ == "__main__":
    main()
