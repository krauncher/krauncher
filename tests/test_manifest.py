# Copyright (c) 2026 Ilya Sergeev. Licensed under the MIT License.

"""DataManifest: the training-set size a call argument carries to the analyzer."""

import json

import pytest

from krauncher import DataManifest, KrauncherClient
from krauncher.analyzer import TaskClassification
from krauncher.KrauncherClient import _kwargs_cache_key


def test_lines_file(tmp_path):
    p = tmp_path / "train_subset.txt"
    p.write_text("wavs/a.wav|one\nwavs/b.wav|two\n\nwavs/c.wav|three\n")
    m = DataManifest.from_file(p)
    assert m == {"__krauncher__": "data_manifest", "v": 1, "role": "train", "entries": 3,
                 "entries_source": "file", "name": "train_subset.txt", "format": "lines"}


def test_csv_header_and_quoted_newline(tmp_path):
    p = tmp_path / "meta.csv"
    p.write_text('path,text\na.wav,"two\nlines"\nb.wav,x\n')
    assert DataManifest.from_file(p)["entries"] == 2
    assert DataManifest.from_file(p, header=False)["entries"] == 3


def test_jsonl_by_extension(tmp_path):
    p = tmp_path / "train_manifest.jsonl"
    p.write_text('{"audio_filepath": "a.wav"}\n{"audio_filepath": "b.wav"}\n')
    m = DataManifest.from_file(p, role="val")
    assert (m["entries"], m["format"], m["role"]) == (2, "jsonl", "val")


def test_stated_value():
    m = DataManifest(entries=625, name="ljs_audio_text_train_subset_625_filelist.txt")
    assert (m["entries"], m["entries_source"], m["format"]) == (625, "manual", None)


@pytest.mark.parametrize("bad", [0, -1, True, 2.5, "625"])
def test_entries_must_be_a_positive_integer(bad):
    with pytest.raises(ValueError):
        DataManifest(entries=bad)


def test_empty_file_and_unknown_role(tmp_path):
    p = tmp_path / "empty.txt"
    p.write_text("\n\n")
    with pytest.raises(ValueError):
        DataManifest.from_file(p)
    with pytest.raises(ValueError):
        DataManifest(entries=1, role="test")


def test_travels_as_a_plain_dict():
    m = DataManifest(entries=625, name="x.txt")
    back = json.loads(json.dumps({"train_list": m}))["train_list"]
    assert type(back) is dict and back == dict(m)


def test_cache_key_follows_the_manifest():
    a = _kwargs_cache_key({"epochs": 2, "train_list": DataManifest(entries=625)})
    b = _kwargs_cache_key({"epochs": 2, "train_list": DataManifest(entries=626)})
    assert a != b
    # other dicts are not sent and do not key the analysis
    assert _kwargs_cache_key({"epochs": 2, "cfg": {"a": 1}}) == (("epochs", 2),)


async def test_the_analyzer_receives_the_manifest(monkeypatch):
    seen: list[dict] = []

    class _Analyzer:
        async def classify(self, code, dataset_mb=None, kwargs=None):
            seen.append(dict(kwargs or {}))
            return TaskClassification(min_vram_gb=8, tier="light", confidence=1.0,
                                      analysis_method="ast", compute_units=1000.0)

    monkeypatch.setattr(KrauncherClient, "_analyzer", property(lambda self: _Analyzer()))
    client = KrauncherClient(api_key="cas_test", estimate_only=True)
    m = DataManifest(entries=625)
    await client._submit("def train(train_list, epochs=2):\n    return 1\n", "train",
                         {"train_list": m}, func_defaults={"epochs": 2}, classification_cache={})
    assert seen[0]["train_list"] == dict(m)


def test_analyzer_kwargs_keep_scalars_and_manifests():
    from krauncher.analyzer import analyzer_kwargs
    m = DataManifest(entries=625)
    out = analyzer_kwargs({"epochs": 2, "train_list": m, "cfg": {"a": 1}, "xs": [1, 2]})
    assert out == {"epochs": 2, "train_list": dict(m)} and type(out["train_list"]) is dict
