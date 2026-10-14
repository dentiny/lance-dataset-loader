"""Start a new mixture phase without resetting either source's cursor."""

from _common import DATA_DIR, show_batch

from lance_dataset_loader import LanceSource, MixtureDataset


def main():
    sources = {
        name: LanceSource(str(DATA_DIR / f"{name}.lance"), version=1) for name in ("coding", "math")
    }
    dataset = MixtureDataset(sources, {"coding": 0.2, "math": 0.8}, global_batch_size=4)
    with dataset.iter_batches() as batches:
        batch = next(batches)
        dataset.commit(batch)
    print("Before changing weights:", dataset.state_dict()["progress"])
    dataset.reconfigure(weights={"coding": 0.8, "math": 0.2})
    print("After changing weights:", dataset.state_dict()["progress"])
    print("New weights:", dataset.weights)
    with dataset.iter_batches() as batches:
        batch = next(batches)
        show_batch(batch)
        dataset.commit(batch)


if __name__ == "__main__":
    main()
