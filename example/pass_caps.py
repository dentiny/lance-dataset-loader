"""Cap source passes and explicitly handle an incomplete final global batch."""

from collections import Counter

from _common import DATA_DIR

from lance_dataset_loader import LanceSource, MixtureDataset


def main():
    sources = {
        name: LanceSource(str(DATA_DIR / f"{name}.lance"), version=1) for name in ("coding", "math")
    }
    for policy in ("drop", "error"):
        dataset = MixtureDataset(
            sources,
            {"coding": 0.5, "math": 0.5},
            global_batch_size=8,
            max_passes={"coding": 1, "math": 1},
            tail_policy=policy,
        )
        counts = Counter()
        try:
            with dataset.iter_batches() as batches:
                for batch in batches:
                    counts.update(sample.source_id for sample in batch.samples)
                    dataset.commit(batch)
        except ValueError as error:
            print(f"tail_policy={policy}: {error}")
        print(f"tail_policy={policy}: committed steps={dataset.next_step}, counts={dict(counts)}")
    print("20 available rows form two full batches of 8 and an incomplete tail of 4.")
    print("After one source exhausts its cap, remaining sources are renormalized.")


if __name__ == "__main__":
    main()
