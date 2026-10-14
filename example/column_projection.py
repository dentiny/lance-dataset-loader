"""Fetch only the columns required by the trainer."""

from _common import DATA_DIR

from lance_dataset_loader import LanceSource, MixtureDataset


def main():
    source = LanceSource(str(DATA_DIR / "coding.lance"), version=1, columns=["id", "input_ids"])
    dataset = MixtureDataset({"coding": source}, {"coding": 1}, global_batch_size=2)
    with dataset.iter_batches() as batches:
        batch = next(batches)
        print("Requested columns:", source.snapshot.columns)
        for row in batch.rows:
            print(row)
        dataset.commit(batch)
    print("The source's text/source columns were not fetched.")


if __name__ == "__main__":
    main()
