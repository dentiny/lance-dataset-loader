import hashlib
import json
from collections import Counter
from dataclasses import replace

import pytest
from conftest import MemorySource

from lance_dataset_loader import MixtureDataset, SourceSnapshot
from lance_dataset_loader.permutation import permute


def make(sources, **kwargs):
    return MixtureDataset(
        sources,
        kwargs.pop("weights", {"coding": 0.2, "math": 0.8}),
        global_batch_size=kwargs.pop("global_batch_size", 12),
        **kwargs,
    )


def collect(dataset, steps, **kwargs):
    batches = []
    with dataset.iter_batches(**kwargs) as iterator:
        for _ in range(steps):
            batch = next(iterator)
            batches.append(batch)
            dataset.commit(batch)
    return batches


@pytest.mark.parametrize("size", [1, 2, 3, 17, 257, 4097])
def test_shuffle_is_bijection(size):
    key = hashlib.sha256(b"test").digest()
    values = [permute(i, size, key) for i in range(size)]
    assert sorted(values) == list(range(size))
    assert values == [permute(i, size, key) for i in range(size)]


def test_seed_and_source_order(sources):
    reference = collect(make(sources), 8)
    reordered = collect(make(dict(reversed(list(sources.items())))), 8)
    assert reference == reordered
    assert reference != collect(make(sources, seed=99), 8)


def test_resume_matches_uninterrupted_run(sources, tmp_path):
    reference = collect(make(sources), 12)
    first = make(sources)
    assert collect(first, 4) == reference[:4]
    path = tmp_path / "state.json"
    first.save_checkpoint(path)
    resumed = make(sources)
    resumed.load_checkpoint(path)
    assert collect(resumed, 8) == reference[4:]


def test_prefetch_and_uncommitted_batch_are_not_checkpointed():
    source = MemorySource(size=100)
    dataset = MixtureDataset({"a": source}, {"a": 1}, global_batch_size=6)
    before = dataset.state_dict()
    with dataset.iter_batches(prefetch_batches=3, read_workers=4) as iterator:
        first = next(iterator)
        assert source.prefetched.wait(5)
        assert dataset.state_dict() == before
        with pytest.raises(RuntimeError, match="commit the previous"):
            next(iterator)
        dataset.commit(first)
        assert dataset.next_step == 1
        assert dataset.state_dict()["progress"]["source_cursors"] == {"a": 6}
        second = next(iterator)
        assert dataset.next_step == 1  # model has not trained the delivered batch yet
    replay = collect(dataset, 1)[0]
    assert replay == second


@pytest.mark.parametrize("world_size", [1, 2, 3, 4, 6, 12])
def test_rank_slices_reconstruct_global_batches(sources, world_size):
    reference = collect(make(sources), 5)
    ranks = [
        collect(
            make(sources), 5, rank=rank, world_size=world_size, read_workers=3, prefetch_batches=2
        )
        for rank in range(world_size)
    ]
    for step, expected in enumerate(reference):
        assert tuple(s for rank in ranks for s in rank[step].samples) == expected.samples
        assert tuple(row for rank in ranks for row in rank[step].rows) == expected.rows


def test_elastic_resume_and_changed_io_workers(sources):
    reference = collect(make(sources), 8)
    old = make(sources)
    collect(old, 3, rank=1, world_size=2, read_workers=1)
    state = old.state_dict()
    ranks = []
    for rank in range(4):
        dataset = make(sources)
        dataset.load_state_dict(state)
        ranks.append(
            collect(dataset, 5, rank=rank, world_size=4, read_workers=5, prefetch_batches=3)
        )
    for i, expected in enumerate(reference[3:]):
        assert tuple(s for rank in ranks for s in rank[i].samples) == expected.samples


def test_pass_caps_cover_every_row_once_per_pass(sources):
    dataset = make(sources, global_batch_size=8, max_passes={"coding": 2, "math": 2})
    refs = []
    with dataset.iter_batches(prefetch_batches=2) as iterator:
        for batch in iterator:
            refs.extend(batch.samples)
            dataset.commit(batch)
    assert len(refs) == 80
    for name, source in sources.items():
        for pass_index in (0, 1):
            indices = [
                r.row_index for r in refs if r.source_id == name and r.pass_index == pass_index
            ]
            assert sorted(indices) == list(range(source.snapshot.row_count))


@pytest.mark.parametrize("tail_policy", ["drop", "error"])
def test_tail_policy_does_not_lose_complete_prefetched_batches(tail_policy):
    dataset = MixtureDataset(
        {"a": MemorySource(size=5)},
        {"a": 1},
        global_batch_size=4,
        max_passes={"a": 1},
        tail_policy=tail_policy,
    )
    with dataset.iter_batches(prefetch_batches=3) as iterator:
        dataset.commit(next(iterator))
        with pytest.raises(StopIteration if tail_policy == "drop" else ValueError):
            next(iterator)
    assert dataset.state_dict()["progress"]["source_cursors"] == {"a": 4}


@pytest.mark.parametrize(
    "override",
    [
        {"seed": 99},
        {"shuffle": False},
        {"global_batch_size": 6},
        {"weights": {"coding": 0.8, "math": 0.2}},
        {"max_passes": {"coding": 2}},
    ],
)
def test_changed_configuration_is_rejected(sources, override):
    checkpoint = make(sources).state_dict()
    with pytest.raises(ValueError, match="mismatch"):
        make(sources, **override).load_state_dict(checkpoint)


def test_changed_snapshot_is_rejected(sources):
    state = make(sources).state_dict()
    changed = dict(sources, coding=MemorySource("coding", 17, version=2))
    with pytest.raises(ValueError, match="snapshot mismatch"):
        make(changed).load_state_dict(state)


def test_replacement_preserves_other_cursors_and_resumes(sources):
    dataset = make(sources)
    collect(dataset, 3)
    before = dataset.state_dict()["progress"]["source_cursors"]
    new_sources = dict(sources, coding=MemorySource("coding", 19, version=2))
    with pytest.raises(ValueError, match="explicit reset"):
        dataset.reconfigure(sources=new_sources)
    dataset.reconfigure(
        sources=new_sources, reset_sources={"coding"}, weights={"coding": 0.7, "math": 0.3}
    )
    state = dataset.state_dict()
    assert state["progress"]["source_cursors"] == {"coding": 0, "math": before["math"]}
    assert state["transitions"][0]["step"] == 3
    reference = collect(dataset, 4)
    resumed = make(new_sources, weights={"coding": 0.7, "math": 0.3})
    resumed.load_state_dict(json.loads(json.dumps(state)))
    assert collect(resumed, 4) == reference


def test_reconfigure_validation_is_transactional(sources):
    dataset = make(sources)
    before = dataset.state_dict()
    with pytest.raises(ValueError):
        dataset.reconfigure(weights={"coding": -1, "math": 1})
    assert dataset.state_dict() == before


def test_storage_failure_does_not_advance_state():
    dataset = MixtureDataset({"a": MemorySource(fail=True)}, {"a": 1}, global_batch_size=2)
    before = dataset.state_dict()
    with dataset.iter_batches(prefetch_batches=2) as iterator:
        with pytest.raises(OSError):
            next(iterator)
    assert dataset.state_dict() == before


def test_duplicate_occurrences_keep_order():
    dataset = MixtureDataset(
        {"a": MemorySource(size=2)}, {"a": 1}, global_batch_size=6, shuffle=False
    )
    batch = collect(dataset, 1)[0]
    assert [row["id"] for row in batch.rows] == [0, 1, 0, 1, 0, 1]
    assert [ref.pass_index for ref in batch.samples] == [0, 0, 1, 1, 2, 2]


def test_sequence_and_fixed_token_weighting():
    sources = {"coding": MemorySource("coding", tokens=8), "math": MemorySource("math", tokens=2)}
    totals = {}
    for mode in ("sequences", "tokens"):
        dataset = make(
            sources,
            weights={"coding": 0.5, "math": 0.5},
            global_batch_size=100,
            weight_by=mode,
            shuffle=False,
        )
        counts = Counter(ref.source_id for b in collect(dataset, 100) for ref in b.samples)
        totals[mode] = counts
    assert 0.47 < totals["sequences"]["coding"] / 10000 < 0.53
    token_counts = {
        k: totals["tokens"][k] * sources[k].snapshot.tokens_per_sequence for k in sources
    }
    assert 0.47 < token_counts["coding"] / sum(token_counts.values()) < 0.53


def test_checkpoint_progress_and_hash_validation(sources):
    dataset = make(sources)
    state = dataset.state_dict()
    state["progress"]["source_cursors"]["coding"] = 1
    with pytest.raises(ValueError, match="aligned"):
        dataset.load_state_dict(state)
    state = dataset.state_dict()
    state["configuration"]["seed"] = 11
    with pytest.raises(ValueError, match="hash"):
        dataset.load_state_dict(state)


def test_illegal_mid_iteration_changes_and_double_commit(sources):
    dataset = make(sources)
    with dataset.iter_batches() as iterator:
        with pytest.raises(RuntimeError):
            dataset.reconfigure(weights=dataset.weights)
        with pytest.raises(RuntimeError):
            dataset.load_state_dict(dataset.state_dict())
        batch = next(iterator)
        dataset.commit(batch)
        with pytest.raises(ValueError):
            dataset.commit(batch)


def test_external_configuration_mutation_is_detected(sources):
    dataset = make(sources)
    dataset.sources["coding"].snapshot = replace(sources["coding"].snapshot, version=9)
    with pytest.raises(RuntimeError, match="mutated"):
        dataset.state_dict()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"world_size": 5},
        {"rank": 2, "world_size": 2},
        {"read_workers": 0},
        {"prefetch_batches": -1},
    ],
)
def test_invalid_reader_topology(sources, kwargs):
    with pytest.raises(ValueError):
        make(sources).iter_batches(**kwargs)


def test_large_virtual_dataset_needs_no_materialized_order():
    class VirtualSource:
        snapshot = SourceSnapshot("virtual://trillion-rows", 1, 10**12)
        read_count = 0

        def take(self, indices):
            self.read_count += len(indices)
            return [{"id": index} for index in indices]

    source = VirtualSource()
    dataset = MixtureDataset({"a": source}, {"a": 1}, global_batch_size=16)
    with dataset.iter_batches(rank=1, world_size=4, prefetch_batches=2) as iterator:
        batch = next(iterator)
        assert len(batch.rows) == 4
        assert len({row["id"] for row in batch.rows}) == 4
        dataset.commit(batch)
    assert source.read_count <= 12  # three rank-local batches, not a scan of the corpus
    assert len(json.dumps(dataset.state_dict())) < 2000


def test_zero_weight_source_is_not_read(sources):
    sources["coding"].fail = True
    dataset = make(sources, weights={"coding": 0, "math": 1})
    batches = collect(dataset, 3, prefetch_batches=2)
    assert all(ref.source_id == "math" for batch in batches for ref in batch.samples)
    assert dataset.state_dict()["progress"]["source_cursors"]["coding"] == 0
