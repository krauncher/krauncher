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

It also carries the lengths of the samples, for the memory estimate: the
activations of a sequence model grow with the longest sample of a batch.
Counted from the file, every text field gives its length in characters (max,
p99, mean) — read in the same pass as the count; with ``audio_root=`` the WAV
files the entries name give their durations in seconds, from their headers.
Stated: ``DataManifest(entries=625, lengths={"text_chars": 187,
"audio_seconds": 10.1})``.

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
import json
import math
import os
import wave
from typing import Any

TAG = "data_manifest"
VERSION = 1
FORMATS = ("lines", "csv", "jsonl")
ROLES = ("train", "val")
LENGTHS = ("text_chars", "audio_seconds")
_AUDIO_EXT = (".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a")
_PATH_EXT = _AUDIO_EXT + (".png", ".jpg", ".jpeg", ".npy", ".pt", ".txt", ".json", ".mp4")


def _stats(values: list) -> dict:
    """max, p99 (nearest rank) and mean of a list of numbers."""
    v = sorted(values)
    return {"max": v[-1], "p99": v[min(len(v) - 1, math.ceil(0.99 * len(v)) - 1)],
            "mean": round(sum(v) / len(v), 2)}


def _fields(text: str, format: str, header: bool | None) -> list[list[str]]:
    """The string fields of every entry: the parts of a `path|text` line, the
    columns of a CSV row, the string values of a JSON object."""
    if format == "csv":
        rows = [r for r in csv.reader(io.StringIO(text)) if any(c.strip() for c in r)]
        return rows[1:] if header is None or header else rows
    lines = [line for line in text.splitlines() if line.strip()]
    if format == "jsonl":
        out = []
        for line in lines:
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            out.append([v for v in obj.values() if isinstance(v, str)] if isinstance(obj, dict) else [])
        return out
    return [line.split("|") for line in lines]


def _wav_seconds(path: str) -> tuple[float, int] | None:
    try:
        with wave.open(path, "rb") as w:
            return w.getnframes() / w.getframerate(), w.getframerate()
    except (OSError, wave.Error, EOFError, ZeroDivisionError):
        return None


def _lengths(rows: list[list[str]], audio_root: str | None) -> dict:
    """Sample lengths of the entries: characters of the text fields (the
    longest field), seconds of the WAV files under ``audio_root``."""
    out: dict = {}
    text_lens: list[int] = []
    seconds: list[float] = []
    rates: set[int] = set()
    for row in rows:
        longest = None
        for f in row:
            f = f.strip()
            if f.lower().endswith(_AUDIO_EXT):
                if audio_root is not None:
                    d = _wav_seconds(os.path.join(audio_root, f))
                    if d is not None:
                        seconds.append(d[0])
                        rates.add(d[1])
                continue
            if not f or (" " not in f and f.lower().endswith(_PATH_EXT)):
                continue                      # a path or an empty field, not text
            longest = max(longest or 0, len(f))
        if longest is not None:
            text_lens.append(longest)
    if text_lens:
        out["text_chars"] = _stats(text_lens)
    if seconds:
        out["audio_seconds"] = {k: round(v, 3) for k, v in _stats(seconds).items()}
        if len(rates) == 1:
            out["sample_rate"] = rates.pop()
    return out


class DataManifest(dict):
    """Size of a training (or validation) set, stated or counted from a file."""

    def __init__(self, entries: int, *, role: str = "train", name: str | None = None,
                 lengths: dict | None = None,
                 _source: str = "manual", _format: str | None = None, _lengths_source: str = "manual"):
        if not isinstance(entries, int) or isinstance(entries, bool) or entries < 1:
            raise ValueError(f"entries must be a positive integer, got {entries!r}")
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}, got {role!r}")
        super().__init__({
            "__krauncher__": TAG, "v": VERSION, "role": role, "entries": entries,
            "entries_source": _source, "name": name, "format": _format,
        })
        if lengths:
            norm = {}
            for k, v in lengths.items():
                if k == "sample_rate" and isinstance(v, int) and v > 0:
                    norm[k] = v
                    continue
                if k not in LENGTHS:
                    raise ValueError(f"lengths: {k!r} is not one of {LENGTHS} or sample_rate")
                stats = v if isinstance(v, dict) else {"max": v}
                if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and x > 0
                           for x in stats.values()) or "max" not in stats:
                    raise ValueError(f"lengths: {k} needs a positive max, got {v!r}")
                norm[k] = stats
            self["lengths"] = norm
            self["lengths_source"] = _lengths_source

    @classmethod
    def from_file(cls, path: str | os.PathLike, *, role: str = "train",
                  format: str | None = None, header: bool | None = None,
                  audio_root: str | os.PathLike | None = None) -> "DataManifest":
        """Count the entries of a manifest file.

        *format*: ``"lines"`` (one entry per non-empty line: file lists,
        ``path|text``, ``wav.scp``), ``"csv"`` (rows, quoted fields may span
        lines) or ``"jsonl"`` (one JSON object per non-empty line); by
        default from the extension — ``.csv``, ``.jsonl`` / ``.ndjson``,
        anything else as lines. *header*: a CSV's first row is a header
        (default True for csv, not used otherwise). *audio_root*: the directory
        the audio paths of the entries are relative to — their WAV headers give
        the durations; without it only the text lengths are counted.
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
        lengths = _lengths(_fields(text, format, header),
                           os.fspath(audio_root) if audio_root is not None else None)
        return cls(entries, role=role, name=os.path.basename(path), lengths=lengths or None,
                   _source="file", _format=format, _lengths_source="file")


def is_manifest(value: Any) -> bool:
    """A DataManifest, or the plain dict one travels as."""
    return isinstance(value, dict) and value.get("__krauncher__") == TAG


# ---------------------------------------------------------------------------
# ModelManifest — the facts of a checkpoint the code loads, for the memory
# estimate.
# ---------------------------------------------------------------------------
#
# The analyzer sees the code, not the checkpoint behind a path or a repo id.
# For a model it does not know (a fine-tune, a custom checkpoint) or one whose
# format it cannot infer from the code (a quantized repo), the weights in
# memory and the KV cache per token are in the checkpoint's own files:
#
#   - ``model.safetensors.index.json`` (``metadata.total_size``) or the
#     safetensors header: the bytes of every tensor, i.e. the weights an
#     inference engine puts on the GPU;
#   - ``config.json``: the decoder dims (``text_config`` of a vision-language
#     config), the layer types of a hybrid stack, the KV cache scheme of a
#     compressed-tensors checkpoint;
#   - ``hf_quant_config.json`` (NVIDIA ModelOpt): the KV cache algorithm.
#
#   meta = ModelManifest.from_hf("nvidia/Llama-3.1-8B-Instruct-FP8")    # read here
#   @client.task(data_urls=[...], model_manifest=meta)
#
# or states the values: ``ModelManifest(weights_bytes=9_084_000_000,
# config={"num_hidden_layers": 32, ...})``. The manifest is optional: without
# it the analyzer uses its registry and the code. It travels to the analyzer
# only; the task function does not receive it.

MODEL_TAG = "model_manifest"
# The analyzer-request argument the manifest travels under: not a valid
# identifier for a signature to declare, so it never reaches the task.
MODEL_MANIFEST_KEY = "__krauncher_model__"
MODEL_VERSION = 1
# The config.json keys the analyzer reads, by their transformers names.
CONFIG_KEYS = (
    "num_hidden_layers", "hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim",
    "vocab_size", "tie_word_embeddings", "intermediate_size", "max_position_embeddings",
    "layer_types", "sliding_window", "linear_num_key_heads", "linear_num_value_heads",
    "linear_key_head_dim", "linear_value_head_dim", "linear_conv_kernel_dim", "mamba_ssm_dtype",
)
KV_DTYPES = ("fp8", "fp16", "bf16", "fp32")
_HF_RESOLVE = "https://huggingface.co/{repo}/resolve/{revision}/{file}"


def _config_subset(cfg: dict) -> dict:
    """The keys the analyzer reads, decoder side: a vision-language config keeps
    the decoder under ``text_config``; ``tie_word_embeddings`` may sit on either
    level and the outer one wins, as in transformers."""
    inner = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
    out = {k: inner[k] for k in CONFIG_KEYS if k in inner}
    if "tie_word_embeddings" in cfg:
        out["tie_word_embeddings"] = cfg["tie_word_embeddings"]
    return out


def _kv_dtype(cfg: dict, quant: dict | None) -> str | None:
    """The KV cache dtype the checkpoint declares: ModelOpt ``kv_cache_quant_algo``,
    compressed-tensors ``kv_cache_scheme``; None when it declares none."""
    algo = ((quant or {}).get("quantization") or {}).get("kv_cache_quant_algo")
    if isinstance(algo, str) and algo.upper().startswith("FP8"):
        return "fp8"
    scheme = (cfg.get("quantization_config") or {}).get("kv_cache_scheme")
    if isinstance(scheme, dict) and scheme.get("num_bits") == 8 and scheme.get("type") == "float":
        return "fp8"
    return None


def _safetensors_bytes(read_range) -> int | None:
    """Tensor bytes of one safetensors file from its header: 8 bytes of header
    length, a JSON header whose data_offsets end at the size of the data."""
    head = read_range(0, 7)
    if not head or len(head) < 8:
        return None
    n = int.from_bytes(head, "little")
    try:
        header = json.loads(read_range(8, 8 + n - 1))
    except (ValueError, TypeError):
        return None
    ends = [v["data_offsets"][1] for k, v in header.items() if k != "__metadata__" and "data_offsets" in v]
    return max(ends) if ends else None


# Bytes per element of the safetensors dtypes (the Hub reports parameter counts per dtype).
_ST_BYTES = {"F64": 8, "I64": 8, "U64": 8, "F32": 4, "I32": 4, "U32": 4, "F16": 2, "BF16": 2,
             "I16": 2, "U16": 2, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1, "I8": 1, "U8": 1, "BOOL": 1}


class ModelManifest(dict):
    """Facts of the checkpoint a task loads, stated or read from its files."""

    def __init__(self, *, weights_bytes: int | None = None, config: dict | None = None,
                 kv_cache_dtype: str | None = None, name: str | None = None, _source: str = "manual"):
        if weights_bytes is not None and (not isinstance(weights_bytes, int)
                                          or isinstance(weights_bytes, bool) or weights_bytes < 1):
            raise ValueError(f"weights_bytes must be a positive integer, got {weights_bytes!r}")
        config = dict(config or {})
        unknown = [k for k in config if k not in CONFIG_KEYS]
        if unknown:
            raise ValueError(f"config: {unknown} not among {CONFIG_KEYS}")
        if kv_cache_dtype is not None and kv_cache_dtype not in KV_DTYPES:
            raise ValueError(f"kv_cache_dtype must be one of {KV_DTYPES}, got {kv_cache_dtype!r}")
        if weights_bytes is None and not config and kv_cache_dtype is None:
            raise ValueError("a model manifest needs weights_bytes, config or kv_cache_dtype")
        super().__init__({"__krauncher__": MODEL_TAG, "v": MODEL_VERSION, "source": _source,
                          "name": name, "weights_bytes": weights_bytes, "config": config,
                          "kv_cache_dtype": kv_cache_dtype})

    @classmethod
    def _from_files(cls, read, read_range_of, name: str, source: str,
                    weights_fallback=None) -> "ModelManifest":
        """*read(file)* -> bytes or None; *read_range_of(file)* -> read_range(start, end);
        *weights_fallback()* -> tensor bytes when there is no index (shards named
        otherwise), tried before the single-file header."""
        raw = read("config.json")
        if raw is None:
            raise ValueError(f"{name}: no config.json")
        cfg = json.loads(raw)
        quant_raw = read("hf_quant_config.json")
        quant = json.loads(quant_raw) if quant_raw else None
        weights = None
        index = read("model.safetensors.index.json")
        if index:
            total = (json.loads(index).get("metadata") or {}).get("total_size")
            weights = int(total) if isinstance(total, (int, float)) and total > 0 else None
        if weights is None and weights_fallback is not None:
            weights = weights_fallback()
        if weights is None:
            weights = _safetensors_bytes(read_range_of("model.safetensors"))
        return cls(weights_bytes=weights, config=_config_subset(cfg),
                   kv_cache_dtype=_kv_dtype(cfg, quant), name=name, _source=source)

    @classmethod
    def from_dir(cls, path: str | os.PathLike) -> "ModelManifest":
        """Read a checkpoint directory (the layout ``from_pretrained`` loads)."""
        path = os.fspath(path)

        def read(file):
            p = os.path.join(path, file)
            if not os.path.isfile(p):
                return None
            with open(p, "rb") as f:
                return f.read()

        def read_range_of(file):
            def rr(start, end):
                p = os.path.join(path, file)
                if not os.path.isfile(p):
                    return None
                with open(p, "rb") as f:
                    f.seek(start)
                    return f.read(end - start + 1)
            return rr

        def all_shards():
            # Every safetensors file of the directory, by its header.
            sizes = [_safetensors_bytes(read_range_of(f)) for f in sorted(os.listdir(path))
                     if f.endswith(".safetensors")]
            return sum(sizes) if sizes and None not in sizes else None

        return cls._from_files(read, read_range_of, os.path.basename(path.rstrip("/")), "dir",
                               weights_fallback=all_shards)

    @classmethod
    def from_hf(cls, repo: str, revision: str = "main", token: str | None = None,
                timeout: float = 20.0) -> "ModelManifest":
        """Read a Hugging Face Hub repo: config files and the safetensors index
        or header (two small range requests), not the weights. *token*: for a
        gated or private repo, else ``HF_TOKEN`` from the environment if set."""
        import httpx

        token = token or os.environ.get("HF_TOKEN")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        with httpx.Client(timeout=timeout, headers=headers, follow_redirects=True) as c:
            def url(file):
                return _HF_RESOLVE.format(repo=repo, revision=revision, file=file)

            def read(file):
                r = c.get(url(file))
                if r.status_code == 404:
                    return None
                r.raise_for_status()
                return r.content

            def read_range_of(file):
                def rr(start, end):
                    r = c.get(url(file), headers={"Range": f"bytes={start}-{end}"})
                    return r.content if r.status_code in (200, 206) else None
                return rr

            def hub_params():
                # The Hub's own count from the safetensors headers, per dtype.
                r = c.get(f"https://huggingface.co/api/models/{repo}/revision/{revision}")
                if r.status_code != 200:
                    return None
                per = ((r.json().get("safetensors") or {}).get("parameters") or {})
                if not per or any(k not in _ST_BYTES for k in per):
                    return None
                return sum(n * _ST_BYTES[k] for k, n in per.items())

            return cls._from_files(read, read_range_of, f"{repo}@{revision}", "hf",
                                   weights_fallback=hub_params)


def is_model_manifest(value: Any) -> bool:
    """A ModelManifest, or the plain dict one travels as."""
    return isinstance(value, dict) and value.get("__krauncher__") == MODEL_TAG
