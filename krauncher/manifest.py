# Copyright (c) 2026 Ilya Sergeev. Licensed under the MIT License.

"""DataManifest — the size of the set a task's training loop iterates, for the forecast.

The analyzer sees the task's code and its call arguments, not its data. When
the training set is defined by a file the code reads at run time (a file list,
a CSV or JSON-lines manifest), its size is in that file and the forecast has
to guess. A ``DataManifest`` passed as a call argument carries the size:

    train = DataManifest.from_file("filelists/train_subset_625.txt")   # counted here
    handle = await train_tts(train_list=train, batch_size=18, epochs=2)

    def train_tts(train_list, batch_size, epochs):
        items = open(train_list["name"]).readlines()                     # a plain dict

or states it: ``DataManifest(entries=625, name="train_subset_625.txt")``.

- From a file the count is the base of the set: what the code does on top of
  it (a subset it takes, the negatives it samples) is still applied by the
  analyzer.
- A stated value is the final set the loop iterates per epoch, taken as it
  is — a direct instruction to count so.

The manifest is a dict and travels as one: the task function receives a plain
dict with the same keys. The file itself is not carried — send it with
``files=`` or a data source as usual.
"""

from __future__ import annotations

import csv
import io
import os
from typing import Any

TAG = "data_manifest"
VERSION = 1
FORMATS = ("lines", "csv", "jsonl")
ROLES = ("train", "val")


class DataManifest(dict):
    """Size of a training (or validation) set, stated or counted from a file."""

    def __init__(self, entries: int, *, role: str = "train", name: str | None = None,
                 _source: str = "manual", _format: str | None = None):
        if not isinstance(entries, int) or isinstance(entries, bool) or entries < 1:
            raise ValueError(f"entries must be a positive integer, got {entries!r}")
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}, got {role!r}")
        super().__init__({
            "__krauncher__": TAG, "v": VERSION, "role": role, "entries": entries,
            "entries_source": _source, "name": name, "format": _format,
        })

    @classmethod
    def from_file(cls, path: str | os.PathLike, *, role: str = "train",
                  format: str | None = None, header: bool | None = None) -> "DataManifest":
        """Count the entries of a manifest file.

        *format*: ``"lines"`` (one entry per non-empty line: file lists,
        ``path|text``, ``wav.scp``), ``"csv"`` (rows, quoted fields may span
        lines) or ``"jsonl"`` (one JSON object per non-empty line); by
        default from the extension — ``.csv``, ``.jsonl`` / ``.ndjson``,
        anything else as lines. *header*: a CSV's first row is a header
        (default True for csv, not used otherwise).
        """
        path = os.fspath(path)
        if format is None:
            ext = os.path.splitext(path)[1].lower()
            format = "csv" if ext == ".csv" else "jsonl" if ext in (".jsonl", ".ndjson") else "lines"
        if format not in FORMATS:
            raise ValueError(f"format must be one of {FORMATS}, got {format!r}")
        with open(path, encoding="utf-8", newline="") as f:
            text = f.read()
        if format == "csv":
            rows = [r for r in csv.reader(io.StringIO(text)) if any(c.strip() for c in r)]
            entries = len(rows) - (1 if header is None or header else 0)
        else:
            entries = sum(1 for line in text.splitlines() if line.strip())
        if entries < 1:
            raise ValueError(f"{path}: no entries")
        return cls(entries, role=role, name=os.path.basename(path), _source="file", _format=format)


def is_manifest(value: Any) -> bool:
    """A DataManifest, or the plain dict one travels as."""
    return isinstance(value, dict) and value.get("__krauncher__") == TAG
