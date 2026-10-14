"""Storage adapters. Source reads must preserve order and repeated occurrences."""

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class SourceSnapshot:
    """A stable, serializable data identity; no live connections or credentials."""

    uri: str
    version: int | str
    row_count: int
    columns: tuple[str, ...] | None = None
    tokens_per_sequence: int | None = None
    token_column: str | None = None

    def __post_init__(self):
        if not self.uri or type(self.row_count) is not int or self.row_count <= 0:
            raise ValueError("a source needs a URI and a positive row count")
        if isinstance(self.version, bool) or not isinstance(self.version, (int, str)):
            raise ValueError("a source needs a fixed integer or string version")
        if (isinstance(self.version, int) and self.version <= 0) or self.version == "":
            raise ValueError("source version must be positive or a non-empty string")
        if self.tokens_per_sequence is not None and (
            type(self.tokens_per_sequence) is not int or self.tokens_per_sequence <= 0
        ):
            raise ValueError("tokens_per_sequence must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if self.columns is not None:
            result["columns"] = list(self.columns)
        return result


class Source(Protocol):
    @property
    def snapshot(self) -> SourceSnapshot: ...

    def take(self, indices: Sequence[int]) -> Sequence[Mapping[str, Any]]: ...


class LanceSource:
    """Open one immutable Lance version and read rows by positional index.

    If version is omitted, resolve the current version once at construction.
    Reopen that exact version on resume; do not assume that 'latest' is unchanged.
    Token weighting requires an integer-list token column of the declared fixed length.
    Storage credentials/options belong to the adapter and are not checkpointed.
    """

    def __init__(
        self,
        uri: str,
        *,
        version: int | None = None,
        columns: Sequence[str] | None = None,
        tokens_per_sequence: int | None = None,
        token_column: str | None = None,
        **dataset_options: Any,
    ):
        import lance

        self._dataset = lance.dataset(uri, version=version, **dataset_options)
        self._columns = list(columns) if columns is not None else None
        self._snapshot = SourceSnapshot(
            uri=str(self._dataset.uri),
            version=int(self._dataset.version),
            row_count=self._dataset.count_rows(),
            columns=tuple(columns) if columns is not None else None,
            tokens_per_sequence=tokens_per_sequence,
            token_column=token_column,
        )
        if tokens_per_sequence is not None:
            if token_column is None:
                raise ValueError("fixed token weighting requires token_column")
            if columns is not None and token_column not in columns:
                raise ValueError("columns must include token_column")
            field = self._dataset.schema.field(token_column)
            import pyarrow as pa

            if not (
                pa.types.is_list(field.type)
                or pa.types.is_large_list(field.type)
                or pa.types.is_fixed_size_list(field.type)
            ) or not pa.types.is_integer(field.type.value_type):
                raise ValueError("token_column must contain integer token lists")

    @property
    def snapshot(self) -> SourceSnapshot:
        return self._snapshot

    def take(self, indices: Sequence[int]) -> list[dict[str, Any]]:
        if not indices:
            return []
        rows = self._dataset.take(list(indices), columns=self._columns).to_pylist()
        if len(rows) != len(indices):
            raise RuntimeError("Lance read did not preserve all requested occurrences")
        if self.snapshot.tokens_per_sequence is not None:
            for row in rows:
                tokens = row[self.snapshot.token_column]
                if tokens is None or len(tokens) != self.snapshot.tokens_per_sequence:
                    raise ValueError("token length differs from tokens_per_sequence")
        return rows
