# lance-dataset-loader

Deterministic, resumable mixtures of versioned Lance datasets for ML training.

## Capabilities

- Named data sources and configurable sampling weights.
- Deterministic source selection and constant-space, per-pass row shuffling.
- Full global batches whose ordered sample occurrences remain identical across data-parallel sizes.
- Batch reads grouped by source, with bounded prefetch and configurable I/O threads.
- Explicit trainer acknowledgement: fetched and prefetched batches never advance a checkpoint until committed.
- JSON-serializable checkpoints binding the data snapshots, source cursors, weights, seed, sampling algorithm and batch size.
- Explicit phase transitions: replace a source/reset its cursor, retain other source progress, or change mixture weights.
- Optional per-source pass caps and an explicit incomplete-tail policy.

## Install and test

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
uv run python examples/train_and_resume.py
```

Python 3.10+ is required. Storage reads use `pylance`, not a LanceDB connection. Core scheduling is independent of PyTorch; a backend may also implement the `Source` protocol.

## Training

One source row is one training sequence. Prepare tokenized/processed data ahead of time; this package does not tokenize, chunk documents or pack variable-length sequences online.

```python
from lance_dataset_loader import LanceSource, MixtureDataset

sources = {
    "coding": LanceSource("/data/coding.lance", version=3, columns=["input_ids"]),
    "math": LanceSource("/data/math.lance", version=7, columns=["input_ids"]),
}
dataset = MixtureDataset(
    sources,
    weights={"coding": 0.2, "math": 0.8},
    global_batch_size=1024,
    seed=42,
)

with dataset.iter_batches(
    rank=rank,
    world_size=dp_world_size,
    read_workers=4,
    prefetch_batches=2,
) as batches:
    for batch in batches:
        # Convert batch.rows to tensors and divide into local micro-batches.
        # Complete forward/backward, gradient accumulation and optimizer.step().
        train_step(batch.rows)
        dataset.commit(batch)  # Only after successfully training this global step.
        if dataset.next_step % 100 == 0:
            save_model_checkpoint(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                dataset=dataset.state_dict(),
            )
```

`rank` and `world_size` refer to **data parallelism**, not tensor/pipeline ranks. Every data-parallel rank executes the same global plan and reads a contiguous slice; concatenating rank slices reconstructs the global batch in order. `global_batch_size` must divide evenly by `world_size`. There is no fixed split count or divisibility constraint involving I/O threads. Each rank's dataset state is identical at a synchronized committed step, so rank 0 may save it after all ranks have completed training.

The loader does not call distributed collectives: the trainer owns process-group setup, synchronization, optimizer state and model checkpoint durability. Keep global batch size unchanged when resuming on another topology. Micro-batch count and gradient accumulation belong to the trainer.

Only one iterator may be active for a dataset. Call `commit` before requesting the next batch. Use the iterator as a context manager; closing it discards outstanding/uncommitted work and releases its thread pool. Reopening replays any uncommitted batch. Do not wrap this iterator in a multiprocessing `torch.utils.data.DataLoader`; its own I/O threads/prefetch and explicit commit boundary provide the supported execution path.

## Checkpoint and restore

`state_dict()` includes:

```text
format_version
configuration + configuration_hash
  algorithm / seed / shuffle / weight_by / global_batch_size / tail_policy
  source ID -> snapshot URI, version, row count, projection, weight, pass cap
progress
  next_step / phase / source cursors / phase-start positions
transitions
  explicit replacement and mix-change boundaries
```

Source cursors count selected occurrences; `cursor // row_count` is the pass number and `cursor % row_count` is the position within its deterministic permutation. No full permutation, token buffer, thread state or prefetch queue is stored.

```python
state = checkpoint["dataset"]
# Reopen the exact source versions recorded in state["configuration"]["sources"].
restored = MixtureDataset(
    sources, weights={"coding": 0.2, "math": 0.8}, global_batch_size=1024, seed=42
)
restored.load_state_dict(state)
# restored.next_step is the first untrained global step.
```

Restoring rejects a different data URI/version, source set, projection, mix, seed, shuffle mode, token lengths, pass caps or global batch size. It never silently overwrites configuration. When `LanceSource(version=None)` is used, the current version is pinned once on opening; reopening "latest" after an append can therefore correctly fail validation. Open the checkpoint's recorded version instead. Keep those Lance versions available; dataset cleanup must not remove snapshots required for recovery.

`save_checkpoint(path)` / `load_checkpoint(path)` write/read atomic JSON **data-state** files for demos and tooling. For production, save the dataset state together with model and optimizer state at the same committed boundary. A separate data file alone is not a complete training checkpoint.

## Replace a dataset or change the mix

Restore the old phase first. Close its iterator, then explicitly start a new phase:

```python
new_sources = dict(dataset.sources)
new_sources["coding"] = LanceSource("/data/coding_v2.lance", version=1, columns=["input_ids"])
dataset.reconfigure(
    sources=new_sources,
    reset_sources={"coding"},
    weights={"coding": 0.3, "math": 0.7},
)
# coding starts at cursor 0; math retains its cursor; global step is unchanged.
```

Changing a snapshot requires an explicit reset. Source IDs remain stable; adding/removing sources is outside the initial API. Changing only weights preserves every source cursor. A transition records its step and old/new configuration hashes. A checkpoint made after the transition must be opened using its new configuration.

## Sequence vs token proportions

`weight_by="sequences"` is the default. Weights control the probability of selecting each source for a sequence occurrence. These are approximate proportions, not exact per-run quotas. With equal valid sequence lengths, sequence and token proportions are equivalent.

For **fixed-length, fully trained** token sequences of different lengths, use:

```python
sources = {
    "coding": LanceSource("coding.lance", token_column="input_ids", tokens_per_sequence=8192),
    "math": LanceSource("math.lance", token_column="input_ids", tokens_per_sequence=2048),
}
dataset = MixtureDataset(
    sources, {"coding": 0.5, "math": 0.5}, global_batch_size=1024, weight_by="tokens"
)
```

Selection probability is proportional to `target_weight / tokens_per_sequence`. Lance reads validate each selected token list's length. This targets equal token contributions in expectation; it does **not** account for padding, masks or variable effective loss-token counts. Variable-length/loss-token balancing and exact quota plans are not implemented.

Sources repeat by default. Each pass has a different deterministic permutation, with each row selected exactly once within that source's pass. Set `max_passes={"coding": 2, "math": 1}` to cap passes. Exhausted sources are removed from sampling; remaining weights are renormalized, so the original mix is not maintained after exhaustion. At final exhaustion, `tail_policy="drop"` (default) omits an incomplete global batch; `"error"` raises after all preceding complete batches have been delivered. Zero-weight sources are not selected.

Read/validation failures propagate; the loader does not silently skip bad rows or advance their training progress. User-provided `Source.take` must preserve requested order and duplicates and must be safe for concurrent read calls.
