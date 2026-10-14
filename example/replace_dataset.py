"""Replace coding data, reset its cursor and retain math's progress."""

from _common import DATA_DIR, show_batch

from lance_dataset_loader import LanceSource, MixtureDataset


def main():
    sources = {
        name: LanceSource(str(DATA_DIR / f"{name}.lance"), version=1) for name in ("coding", "math")
    }
    dataset = MixtureDataset(sources, {"coding": 0.5, "math": 0.5}, global_batch_size=4)
    with dataset.iter_batches() as batches:
        for _ in range(2):
            batch = next(batches)
            dataset.commit(batch)
    print("Before replacement:", dataset.state_dict()["progress"]["source_cursors"])
    replacements = dict(dataset.sources)
    replacements["coding"] = LanceSource(str(DATA_DIR / "coding_v2.lance"), version=1)
    # Close the old iterator first. A changed snapshot requires an explicit reset.
    dataset.reconfigure(sources=replacements, reset_sources={"coding"})
    print("After replacement:", dataset.state_dict()["progress"]["source_cursors"])
    print("Recorded transition:", dataset.state_dict()["transitions"][-1])
    with dataset.iter_batches() as batches:
        batch = next(batches)
        show_batch(batch)
        dataset.commit(batch)


if __name__ == "__main__":
    main()
