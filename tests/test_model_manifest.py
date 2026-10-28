"""ModelManifest: stated values, a checkpoint directory, the analyzer request."""
import json
import struct

import pytest

from krauncher.analyzer import analyzer_kwargs
from krauncher.manifest import MODEL_MANIFEST_KEY, ModelManifest, is_model_manifest


def _safetensors(path, tensors):
    """A minimal safetensors file: {name: (dtype, n_bytes)}."""
    header, off = {}, 0
    for name, (dtype, n) in tensors.items():
        header[name] = {"dtype": dtype, "shape": [n], "data_offsets": [off, off + n]}
        off += n
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * off)


def test_stated_values_validate():
    m = ModelManifest(weights_bytes=1000, config={"num_hidden_layers": 2}, kv_cache_dtype="fp8")
    assert is_model_manifest(m) and m["source"] == "manual"
    with pytest.raises(ValueError):
        ModelManifest()
    with pytest.raises(ValueError):
        ModelManifest(config={"not_a_key": 1})
    with pytest.raises(ValueError):
        ModelManifest(weights_bytes=0)
    with pytest.raises(ValueError):
        ModelManifest(kv_cache_dtype="int3")


def test_from_dir_reads_config_quant_and_index(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "text_config": {"num_hidden_layers": 4, "num_key_value_heads": 2, "head_dim": 64,
                        "layer_types": ["linear_attention"] * 3 + ["full_attention"]},
        "tie_word_embeddings": False, "vision_config": {"depth": 2}}))
    (tmp_path / "hf_quant_config.json").write_text(json.dumps(
        {"quantization": {"quant_algo": "FP8", "kv_cache_quant_algo": "FP8"}}))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": 4242}}))
    m = ModelManifest.from_dir(tmp_path)
    assert m["weights_bytes"] == 4242 and m["kv_cache_dtype"] == "fp8" and m["source"] == "dir"
    assert m["config"]["num_hidden_layers"] == 4 and m["config"]["tie_word_embeddings"] is False
    assert "vision_config" not in m["config"]


def test_from_dir_sums_shard_headers_without_index(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"num_hidden_layers": 2}))
    _safetensors(tmp_path / "layers-0.safetensors", {"a": ("BF16", 100), "b": ("F8_E4M3", 50)})
    _safetensors(tmp_path / "layers-1.safetensors", {"c": ("BF16", 30)})
    assert ModelManifest.from_dir(tmp_path)["weights_bytes"] == 180


def test_compressed_tensors_kv_scheme(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"num_hidden_layers": 2, "quantization_config": {
        "kv_cache_scheme": {"num_bits": 8, "type": "float"}}}))
    assert ModelManifest.from_dir(tmp_path)["kv_cache_dtype"] == "fp8"


def test_rides_to_the_analyzer_as_a_plain_dict():
    m = ModelManifest(weights_bytes=10)
    sent = analyzer_kwargs({"n": 3, MODEL_MANIFEST_KEY: m, "obj": object()})
    assert sent[MODEL_MANIFEST_KEY] == dict(m) and type(sent[MODEL_MANIFEST_KEY]) is dict
    assert "obj" not in sent
