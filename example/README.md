# Feature examples

Each feature has its own runnable Python file. Run the commands below from the repository root:

```bash
uv sync
uv run python example/basic_usage.py
```

The small Lance datasets under `example/data/` are committed, so no download or preparation step is required. They contain synthetic text and illustrative integer token IDs, not a real tokenizer's output. Examples only read those snapshots. Checkpoint and physically merged outputs use temporary directories and are cleaned up after each run.

| Python file | Feature |
| --- | --- |
| [basic_usage.py](basic_usage.py) | Open one fixed Lance version, read batches and commit successful training progress. |
| [mix_multiple_datasets.py](mix_multiple_datasets.py) | Mix coding, math and general streams with 20/30/50 sequence probabilities. |
| [merge_multiple_datasets.py](merge_multiple_datasets.py) | Physically concatenate compatible small tables into a temporary new Lance dataset. |
| [deterministic_shuffle.py](deterministic_shuffle.py) | Display identical sample-order fingerprints for the same seed and a different order for a new seed. |
| [checkpoint_resume.py](checkpoint_resume.py) | Save committed progress and replay a fetched but uncommitted batch on resume. |
| [replace_dataset.py](replace_dataset.py) | Replace coding with coding_v2, explicitly reset coding and retain math progress. |
| [change_mix.py](change_mix.py) | Change source weights at a committed phase boundary without resetting source cursors. |
| [token_weighting.py](token_weighting.py) | Compare equal sequence proportions with equal fixed-length token proportions. |
| [distributed_batches.py](distributed_batches.py) | Simulate DP sizes 1, 2 and 4; display rank slices and resume on a different DP size. |
| [pass_caps.py](pass_caps.py) | Limit passes and display both drop/error policies for incomplete final batches. |
| [prefetch.py](prefetch.py) | Configure I/O threads and bounded prefetch while showing trainer-committed progress. |
| [column_projection.py](column_projection.py) | Read only the ID and token columns required for training. |

For any feature, run `uv run python example/<filename>.py`. `_common.py` only holds the data path and output formatting; scheduling/configuration code stays in each feature file. These examples use CPU reads and simulated training acknowledgement, not an actual model or distributed process group.

## Bundled data

All datasets use the same schema: `id: int64`, `source: string`, `text: string`, `input_ids: list<int32>`. Every dataset is pinned to version 1.

| Dataset | Rows | Tokens per sequence | Role |
| --- | ---: | ---: | --- |
| `data/coding.lance/` | 8 | 8 | Original coding source. |
| `data/math.lance/` | 12 | 4 | Shorter sequences for token-weighting examples. |
| `data/general.lance/` | 8 | 4 | A third source for mixing/physical merging. |
| `data/coding_v2.lance/` | 8 | 8 | Replacement coding source with IDs 100–107. |

`data/catalog.json` describes these fixtures. The `.lance` directories include their data files, version manifests and transaction metadata; keep the directories intact when copying them.

To create fresh copies using the same definitions:

```bash
uv run python example/generate_data.py --output-dir /tmp/lance-example-data
```

The generator leaves existing dataset directories untouched. The feature scripts continue to read the bundled `example/data/` directory.

Mixing uses sampling probabilities, not exact quotas. Physical merging does not apply sampling weights. Token weighting uses fixed total sequence lengths, not masked effective loss tokens. The trainer must call `commit(batch)` after successful training and save data state together with model/optimizer state in production.
