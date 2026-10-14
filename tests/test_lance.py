import lance
import pyarrow as pa
import pytest

from lance_dataset_loader import LanceSource, MixtureDataset


def test_real_lance_duplicates_and_snapshot_resume(tmp_path):
    uri = str(tmp_path / "tokens.lance")
    written = lance.write_dataset(pa.table({"id": [0, 1], "tokens": [[1, 2], [3, 4]]}), uri)
    source = LanceSource(uri, token_column="tokens", tokens_per_sequence=2)
    assert source.take([1, 0, 1]) == [
        {"id": 1, "tokens": [3, 4]},
        {"id": 0, "tokens": [1, 2]},
        {"id": 1, "tokens": [3, 4]},
    ]
    dataset = MixtureDataset({"a": source}, {"a": 1}, global_batch_size=6, shuffle=False)
    with dataset.iter_batches(prefetch_batches=2, read_workers=3) as iterator:
        batch = next(iterator)
        assert [row["id"] for row in batch.rows] == [0, 1, 0, 1, 0, 1]
        dataset.commit(batch)
    state = dataset.state_dict()
    lance.write_dataset(pa.table({"id": [2], "tokens": [[5, 6]]}), uri, mode="append")
    latest = MixtureDataset({"a": LanceSource(uri)}, {"a": 1}, global_batch_size=6, shuffle=False)
    with pytest.raises(ValueError, match="snapshot mismatch"):
        latest.load_state_dict(state)
    pinned = MixtureDataset(
        {
            "a": LanceSource(
                uri, version=written.version, token_column="tokens", tokens_per_sequence=2
            )
        },
        {"a": 1},
        global_batch_size=6,
        shuffle=False,
    )
    pinned.load_state_dict(state)
    with pinned.iter_batches() as iterator:
        assert [row["id"] for row in next(iterator).rows] == [0, 1, 0, 1, 0, 1]


def test_fixed_token_validation(tmp_path):
    uri = str(tmp_path / "variable.lance")
    lance.write_dataset(pa.table({"tokens": [[1, 2], [3]]}), uri)
    source = LanceSource(uri, token_column="tokens", tokens_per_sequence=2)
    dataset = MixtureDataset(
        {"a": source}, {"a": 1}, global_batch_size=2, shuffle=False, weight_by="tokens"
    )
    before = dataset.state_dict()
    with dataset.iter_batches() as iterator:
        with pytest.raises(ValueError, match="token length"):
            next(iterator)
    assert dataset.state_dict() == before


def test_column_projection(tmp_path):
    uri = str(tmp_path / "columns.lance")
    lance.write_dataset(pa.table({"id": [0, 1], "unused": ["a", "b"]}), uri)
    source = LanceSource(uri, columns=["id"])
    assert source.take([1, 0]) == [{"id": 1}, {"id": 0}]
