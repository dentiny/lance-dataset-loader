"""Global batch planning, storage reads, and trainer-committed checkpoints."""

import hashlib
import json
import math
import os
import tempfile
from collections import deque
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .permutation import permute
from .sources import Source

_FORMAT_VERSION = 1
_ALGORITHM = "blake2b-choice-feistel8-v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class SampleRef:
    source_id: str
    source_version: int | str
    row_index: int
    pass_index: int
    global_position: int


@dataclass(frozen=True)
class TrainingBatch:
    """One contiguous rank slice; step is zero-based and rows remain unmodified."""

    step: int
    rank: int
    world_size: int
    samples: tuple[SampleRef, ...]
    rows: tuple[Mapping[str, Any], ...]


@dataclass
class _Plan:
    step: int
    samples: tuple[SampleRef, ...]
    cursors: dict[str, int]


class MixtureDataset:
    """Deterministic weighted sampling over named, versioned sources.

    Every rank independently builds the same global sequence of sample references,
    then reads a contiguous rank slice. Weights are sampling probabilities, not
    exact quotas. A source repeats in independently shuffled passes unless capped.

    Call commit(batch) only AFTER successfully training that batch. Checkpoints
    reflect committed batches, including when the iterator prefetches ahead.
    A gradient-accumulation step should consume one configured global batch:
    split its rank-local rows into micro-batches inside the training loop.

    weight_by='tokens' supports verified fixed-length token sequences, not variable
    length/masked loss-token balancing. An exhausted source is removed and remaining
    weights are renormalized. Full global batches only: tail_policy='drop' discards
    an incomplete final batch; 'error' reports it instead.
    """

    def __init__(
        self,
        sources: Mapping[str, Source],
        weights: Mapping[str, float],
        *,
        global_batch_size: int,
        seed: int = 42,
        shuffle: bool = True,
        weight_by: Literal["sequences", "tokens"] = "sequences",
        max_passes: Mapping[str, int | None] | None = None,
        tail_policy: Literal["drop", "error"] = "drop",
    ):
        self.sources = dict(sources)
        self.weights = dict(weights)
        self.global_batch_size = _integer(global_batch_size, "global_batch_size", 1)
        self.seed = _integer(seed, "seed")
        self.shuffle = shuffle
        self.weight_by = weight_by
        self.max_passes = {name: None for name in sources}
        if max_passes is not None:
            if not set(max_passes) <= set(sources):
                raise ValueError("max_passes contains an unknown source")
            self.max_passes.update(max_passes)
        self.tail_policy = tail_policy
        self._validate_configuration()
        self._next_step = 0
        self._phase = 0
        self._cursors = {name: 0 for name in self.sources}
        self._phase_start_step = 0
        self._phase_start_cursors = dict(self._cursors)
        self._transitions: list[dict[str, Any]] = []
        self._iterator: BatchIterator | None = None
        self._pending: tuple[TrainingBatch, _Plan] | None = None
        self._configuration_hash = _fingerprint(self.configuration())

    def _check_configuration(self) -> None:
        if _fingerprint(self.configuration()) != self._configuration_hash:
            raise RuntimeError("configuration was mutated; use reconfigure at a committed boundary")

    def _validate_configuration(self) -> None:
        if not self.sources or any(not isinstance(k, str) or not k for k in self.sources):
            raise ValueError("sources needs at least one non-empty string ID")
        if set(self.weights) != set(self.sources):
            raise ValueError("weights must name exactly the configured sources")
        if any(not math.isfinite(w) or w < 0 for w in self.weights.values()):
            raise ValueError("weights must be finite and non-negative")
        if not math.isfinite(sum(self.weights.values())) or sum(self.weights.values()) <= 0:
            raise ValueError("weights must have a finite positive sum")
        if self.weight_by not in ("sequences", "tokens"):
            raise ValueError("weight_by must be 'sequences' or 'tokens'")
        if not isinstance(self.shuffle, bool):
            raise ValueError("shuffle must be a boolean")
        if self.tail_policy not in ("drop", "error"):
            raise ValueError("tail_policy must be 'drop' or 'error'")
        for name, source in self.sources.items():
            if source.snapshot.row_count > 2**64:
                raise ValueError("row counts larger than uint64 are not supported")
            cap = self.max_passes[name]
            if cap is not None:
                _integer(cap, f"max_passes[{name}]", 1)
            if self.weight_by == "tokens" and source.snapshot.tokens_per_sequence is None:
                raise ValueError(
                    "token weighting requires fixed tokens_per_sequence on every source"
                )

    @property
    def next_step(self) -> int:
        return self._next_step

    def configuration(self) -> dict[str, Any]:
        total = sum(self.weights.values())
        return {
            "algorithm": _ALGORITHM,
            "global_batch_size": self.global_batch_size,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "weight_by": self.weight_by,
            "tail_policy": self.tail_policy,
            "sources": {
                name: {
                    "snapshot": source.snapshot.to_dict(),
                    "weight": self.weights[name] / total,
                    "max_passes": self.max_passes[name],
                }
                for name, source in sorted(self.sources.items())
            },
        }

    def _plan(self, step: int, cursors: dict[str, int]) -> _Plan | None:
        self._check_configuration()
        cursors = dict(cursors)
        samples: list[SampleRef] = []
        for slot in range(self.global_batch_size):
            names, weights = [], []
            for name in sorted(self.sources):
                cap = self.max_passes[name]
                snapshot = self.sources[name].snapshot
                if self.weights[name] == 0 or (
                    cap is not None and cursors[name] >= cap * snapshot.row_count
                ):
                    continue
                names.append(name)
                denominator = snapshot.tokens_per_sequence if self.weight_by == "tokens" else 1
                weights.append(self.weights[name] / denominator)
            if not names:
                if samples and self.tail_policy == "error":
                    raise ValueError("remaining samples do not form a complete global batch")
                return None
            position = step * self.global_batch_size + slot
            draw = hashlib.blake2b(
                _canonical([self.seed, self._phase, position]), digest_size=8
            ).digest()
            target = (int.from_bytes(draw, "little") / 2**64) * sum(weights)
            cumulative = 0.0
            name = names[-1]
            for candidate, weight in zip(names, weights, strict=True):
                cumulative += weight
                if target < cumulative:
                    name = candidate
                    break
            snapshot = self.sources[name].snapshot
            pass_index, row_index = divmod(cursors[name], snapshot.row_count)
            if self.shuffle:
                key = hashlib.blake2b(
                    _canonical([self.seed, name, snapshot.to_dict(), pass_index]), digest_size=32
                ).digest()
                row_index = permute(row_index, snapshot.row_count, key)
            samples.append(SampleRef(name, snapshot.version, row_index, pass_index, position))
            cursors[name] += 1
        return _Plan(step, tuple(samples), cursors)

    def iter_batches(
        self,
        *,
        rank: int = 0,
        world_size: int = 1,
        read_workers: int = 1,
        prefetch_batches: int = 0,
    ) -> "BatchIterator":
        """Create one iterator. Worker threads affect I/O only, never scheduling."""
        _integer(world_size, "world_size", 1)
        _integer(rank, "rank")
        if rank >= world_size or self.global_batch_size % world_size:
            raise ValueError(
                "rank must be in range and global_batch_size must divide by world_size"
            )
        _integer(read_workers, "read_workers", 1)
        _integer(prefetch_batches, "prefetch_batches")
        if self._iterator is not None:
            raise RuntimeError("close the active iterator before creating another")
        self._check_configuration()
        self._iterator = BatchIterator(self, rank, world_size, read_workers, prefetch_batches)
        return self._iterator

    def __iter__(self) -> "BatchIterator":
        return self.iter_batches()

    def commit(self, batch: TrainingBatch) -> None:
        """Acknowledge a successfully trained batch, advancing durable progress."""
        self._check_configuration()
        if self._pending is None or self._pending[0] is not batch:
            raise ValueError("batch is not the outstanding batch of this dataset")
        plan = self._pending[1]
        self._cursors = dict(plan.cursors)
        self._next_step = plan.step + 1
        self._pending = None

    def state_dict(self) -> dict[str, Any]:
        self._check_configuration()
        config = self.configuration()
        return {
            "format_version": _FORMAT_VERSION,
            "configuration": config,
            "configuration_hash": _fingerprint(config),
            "progress": {
                "next_step": self._next_step,
                "phase": self._phase,
                "source_cursors": dict(self._cursors),
                "phase_start_step": self._phase_start_step,
                "phase_start_cursors": dict(self._phase_start_cursors),
            },
            "transitions": deepcopy(self._transitions),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Validate identity/configuration before restoring any progress."""
        self._require_idle()
        self._check_configuration()
        if state.get("format_version") != _FORMAT_VERSION:
            raise ValueError("unsupported checkpoint format")
        config = state["configuration"]
        if _fingerprint(config) != state["configuration_hash"]:
            raise ValueError("checkpoint configuration hash is invalid")
        if config != self.configuration():
            raise ValueError(
                "checkpoint configuration/data snapshot mismatch; restore before reconfiguring"
            )
        progress = state["progress"]
        next_step = _integer(progress["next_step"], "next_step")
        phase = _integer(progress["phase"], "phase")
        start_step = _integer(progress["phase_start_step"], "phase_start_step")
        cursors = dict(progress["source_cursors"])
        starts = dict(progress["phase_start_cursors"])
        if set(cursors) != set(self.sources) or set(starts) != set(self.sources):
            raise ValueError("checkpoint progress must name exactly the configured sources")
        for name in self.sources:
            _integer(cursors[name], "source cursor")
            _integer(starts[name], "phase start cursor")
            cap = self.max_passes[name]
            if cursors[name] < starts[name] or (
                cap is not None and cursors[name] > cap * self.sources[name].snapshot.row_count
            ):
                raise ValueError("checkpoint source cursor is out of range")
        if (
            start_step > next_step
            or sum(cursors[k] - starts[k] for k in cursors)
            != (next_step - start_step) * self.global_batch_size
        ):
            raise ValueError("checkpoint progress is not aligned to completed global batches")
        self._next_step, self._phase, self._phase_start_step = next_step, phase, start_step
        self._cursors, self._phase_start_cursors = cursors, starts
        self._transitions = deepcopy(state.get("transitions", []))

    def save_checkpoint(self, path: str | Path) -> None:
        """Atomically write data state. Include state_dict in the MODEL checkpoint in production."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(self.state_dict(), stream, sort_keys=True, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def load_checkpoint(self, path: str | Path) -> None:
        self.load_state_dict(json.loads(Path(path).read_text()))

    def _require_idle(self) -> None:
        if self._iterator is not None:
            raise RuntimeError("close the active iterator before restoring or reconfiguring")

    def reconfigure(
        self,
        *,
        sources: Mapping[str, Source] | None = None,
        weights: Mapping[str, float] | None = None,
        reset_sources: set[str] | None = None,
    ) -> None:
        """Begin a phase at a committed boundary. Changed sources require explicit reset.

        Stable source IDs are retained; unchanged sources keep their progress.
        Restore the old checkpoint first, then replace data/change weights. No
        configuration is silently overridden by a checkpoint.
        """
        self._require_idle()
        self._check_configuration()
        sources = dict(self.sources if sources is None else sources)
        weights = dict(self.weights if weights is None else weights)
        resets = set(reset_sources or ())
        if set(sources) != set(self.sources) or not resets <= set(sources):
            raise ValueError("reconfigure must retain stable source IDs and valid reset names")
        changed = {k for k in sources if sources[k].snapshot != self.sources[k].snapshot}
        if not changed <= resets:
            raise ValueError("changed data snapshots require explicit reset_sources")
        old_sources, old_weights = self.sources, self.weights
        old_hash = _fingerprint(self.configuration())
        self.sources, self.weights = sources, weights
        try:
            self._validate_configuration()
        except BaseException:
            self.sources, self.weights = old_sources, old_weights
            raise
        self._cursors = {k: 0 if k in resets else v for k, v in self._cursors.items()}
        self._phase += 1
        self._phase_start_step = self._next_step
        self._phase_start_cursors = dict(self._cursors)
        self._transitions.append(
            {
                "step": self._next_step,
                "phase": self._phase,
                "previous_configuration_hash": old_hash,
                "configuration_hash": _fingerprint(self.configuration()),
                "reset_sources": sorted(resets),
            }
        )
        self._configuration_hash = _fingerprint(self.configuration())


class BatchIterator:
    """Bounded speculative reads; use as a context manager to release threads."""

    def __init__(self, dataset, rank, world_size, read_workers, prefetch_batches):
        self.dataset = dataset
        self.rank, self.world_size = rank, world_size
        self._pool = ThreadPoolExecutor(max_workers=read_workers)
        self._depth = prefetch_batches + 1
        self._queue: deque[tuple[_Plan, dict[str, Future]]] = deque()
        self._step = dataset.next_step
        self._cursors = dict(dataset._cursors)
        self._ended = False
        self._closed = False
        self._planning_error: Exception | None = None

    def __iter__(self):
        return self

    def _local_samples(self, plan):
        local_size = self.dataset.global_batch_size // self.world_size
        start = self.rank * local_size
        return plan.samples[start : start + local_size]

    def _fill(self):
        while len(self._queue) < self._depth and not self._ended:
            try:
                plan = self.dataset._plan(self._step, self._cursors)
            except Exception as exc:
                self._planning_error = exc
                self._ended = True
                break
            if plan is None:
                self._ended = True
                break
            grouped: dict[str, list[int]] = {}
            for sample in self._local_samples(plan):
                grouped.setdefault(sample.source_id, []).append(sample.row_index)
            futures = {
                name: self._pool.submit(self.dataset.sources[name].take, indices)
                for name, indices in grouped.items()
            }
            self._queue.append((plan, futures))
            self._step += 1
            self._cursors = plan.cursors

    def __next__(self):
        if self._closed:
            raise StopIteration
        if self.dataset._pending is not None:
            raise RuntimeError(
                "commit the previous batch after training before requesting the next"
            )
        try:
            self._fill()
            if not self._queue:
                if self._planning_error is not None:
                    raise self._planning_error
                self.close()
                raise StopIteration
            plan, futures = self._queue.popleft()
            reads = {name: iter(future.result()) for name, future in futures.items()}
            samples = self._local_samples(plan)
            rows = []
            for sample in samples:
                try:
                    rows.append(next(reads[sample.source_id]))
                except StopIteration as exc:
                    raise RuntimeError("source returned fewer rows than requested") from exc
            sentinel = object()
            if any(next(read, sentinel) is not sentinel for read in reads.values()):
                raise RuntimeError("source returned more rows than requested")
            batch = TrainingBatch(plan.step, self.rank, self.world_size, samples, tuple(rows))
            self.dataset._pending = (batch, plan)
            return batch
        except BaseException:
            self.close()
            raise

    def close(self):
        if self._closed:
            return
        self._closed = True
        for _, futures in self._queue:
            for future in futures.values():
                future.cancel()
        self._queue.clear()
        self._pool.shutdown(wait=True, cancel_futures=True)
        self.dataset._pending = None
        self.dataset._iterator = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
