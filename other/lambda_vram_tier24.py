"""VRAM peak of the stand's Lambda tasks of the 24 GB tier, run on Krauncher.

Not a calibration series: kept out of calibration/ so the profile harvester
does not take it into the calibration corpus. Purpose: the actual VRAM
requirement of the stand's Lambda tasks (the stand has only an upper bound
for them, `vram_gb_at_most`: the card the benchmark run completed on; Lambda's
batch sizes grow with card memory, so the peak may be close to the card size;
known_problems.md #3).

Tasks: the stand's Krauncher variants (gpu-forecast-eval,
variants/krauncher/lambdalabs/, at $STAND_ROOT, default ~/src/krauncher_test)
of the 24 GB tier, except resnet50_amp_bs448 (forecast 24 GB, within 10 % of
the bound). Each task runs standalone (no group: a group's worker reuses the
first task's data dir whatever data a later task declares), pinned to a card
of the tier (default RTX 4090, Lambda's anchor of the tier).

The submitted code is the variant's function unchanged plus a wrapper
`vram_<function>` with the same signature, written to a generated module so
the client collects the function as a helper of the wrapper. The wrapper
reads three peaks over the run:
  allocated_gb    torch.cuda.max_memory_allocated (tensors)
  reserved_gb     torch.cuda.max_memory_reserved (PyTorch caching allocator)
  device_used_gb  max of (total - free) from cudaMemGetInfo, polled every
                  50 ms: what nvidia-smi shows, CUDA context and non-PyTorch
                  allocations included; context_gb is its value before the task
and prints them on one line `VRAM_PEAK ...` (also on an OOM or an error).
Where the stand's code does not run with the worker image's libraries, the
measured copy departs from it by PATCHES (the stand itself is not changed);
each departure is commented in the submitted code and printed in the log.

Usage (from cas-client/, CAS_API_KEY in .env):
    python other/lambda_vram_tier24.py [--gpu "RTX 4090"] [--only <task> ...]
"""

import argparse
import asyncio
import importlib.util
import inspect
import os
import sys
import tempfile
import textwrap
from pathlib import Path

from krauncher import KrauncherClient, KrauncherError

STAND = Path(os.environ.get("STAND_ROOT", "~/src/krauncher_test")).expanduser()
VARIANTS = STAND / "variants" / "krauncher" / "lambdalabs"
VRAM_GB = 22          # fits a 24 GB card (the convention of the calibration runners)
BOUND_GB = 24         # the stand's upper bound of the tier
WAIT_TIMEOUT = 4200

# task -> analyzer forecast min_vram_gb (stand checkpoint 2026-10-28_c-c94f0fe2c6df)
TASKS = {
    "bert_base_squad_amp_bs96": 14,
    "bert_large_squad_amp_bs28": 16,
    "ncf_ml20m_bs4278184": 18,
    "tacotron2_ljs625_bs88": 6,
    "transformer_xl_base_wt103_bs24": 5,
    "transformer_xl_large_wt103_bs8": 11,
    "waveglow_ljs625_bs18": 9,
}

# Departures from the stand's code, applied to the measured copy only (the
# stand keeps its tasks unchanged): (substring of the original line, its
# replacement, why). Each is checked to be present, and a comment with the
# reason is inserted above the changed line of the submitted code.
_SQUAD_COLUMNS = (
    'tokenizer(ds["question"], ds["context"],',
    'tokenizer(list(ds["question"]), list(ds["context"]),',
    "datasets 4.x returns a lazy Column for ds[name], which the HF tokenizer "
    "rejects (ValueError: text input must be of type str / list[str]); list() "
    "gives the list[str] the task was written for, the work is unchanged",
)
_NO_APEX = ("NVIDIA apex is not on PyPI and not in the worker image (the worker installs "
            "an unrelated PyPI 'apex' for the import); amp O2 emulated as in calibration "
            "series 64: the same memory (fp16 model and grads, fp32 master weights and "
            "Adam state) and the same work")
_NCF_APEX = [
    ("from apex import amp\n", "\n", _NO_APEX),
    ('''model = NeuMF().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0045, betas=(0.25, 0.5), eps=1e-8)
    criterion = nn.BCEWithLogitsLoss()
    model, optimizer = amp.initialize(model, optimizer, opt_level="O2",
                                      keep_batchnorm_fp32=False, loss_scale="dynamic")''',
     '''model = NeuMF().to(device).half()
    master = [p.detach().clone().float() for p in model.parameters()]
    optimizer = torch.optim.Adam(master, lr=0.0045, betas=(0.25, 0.5), eps=1e-8)
    loss_scale, clean_steps = 2.0 ** 16, 0
    criterion = nn.BCEWithLogitsLoss()''', _NO_APEX),
    ('''loss = criterion(model(users[idx], items[idx]).view(-1), labels[idx]).float()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)''',
     '''loss = criterion(model(users[idx], items[idx]).view(-1).float(), labels[idx])
            (loss * loss_scale).backward()
            grads = [p.grad for p in model.parameters()]
            if torch.stack([g.isfinite().all() for g in grads]).all().item():
                for m, g in zip(master, grads):
                    m.grad = g.float().div_(loss_scale)
                optimizer.step()
                for p, m in zip(model.parameters(), master):
                    p.data.copy_(m.data)
                clean_steps += 1
                if clean_steps % 2000 == 0:
                    loss_scale *= 2.0
            else:
                loss_scale, clean_steps = loss_scale / 2.0, 0
            model.zero_grad(set_to_none=True)''', _NO_APEX),
]
PATCHES = {
    "bert_base_squad_amp_bs96": [_SQUAD_COLUMNS],
    "bert_large_squad_amp_bs28": [_SQUAD_COLUMNS],
    "ncf_ml20m_bs4278184": _NCF_APEX,
}

# Departures from the stand's task options and data sources (measured copy
# only): task -> (OPTIONS overrides, DATA_SOURCES to use or None, why).
# The worker installs every missing package with --no-deps into one --target
# dir, so a package's dependencies must be listed; namespace packages
# (jaraco.*) cannot be (the second one finds jaraco/ present and is skipped),
# hence inflect 7.5.0, the last without jaraco.functools. The worker also
# infers a bare "inflect" from the import and installs it after the list; pip
# --target then skips the inflect/ dir already present, so the 7.5.0 code
# stays (only its metadata reads 7.6.0). The closure was
# found and run (imports, mel filters, pad_center, number_to_words) in the
# worker image cas-worker-vastai-light-mini-cu130.
_LIBROSA = ["librosa", "lazy-loader", "numba", "llvmlite", "msgpack", "pooch"]
_INFLECT = ["inflect==7.5.0", "more-itertools", "typeguard"]
_PIP_WHY = ("the worker installs packages with --no-deps: librosa / inflect "
            "fail to import without their dependencies (lazy_loader, jaraco)")
_WT103 = [dict(name="wikitext-103-v1-smerity", urls=["https://wikitext.smerity.com/wikitext-103-v1.zip"],
               size_gb=0.2)]
_WT103_WHY = ("the stand's and the account's data source 'wikitext-103-v1' points to the "
              "LJSpeech archive, the task reads /data/wikitext-103-v1.zip; a separate "
              "source with the archive the task's docstring names (wikitext.smerity.com)")
OVERRIDES = {
    "tacotron2_ljs625_bs88": (dict(pip=_LIBROSA + _INFLECT), None, _PIP_WHY),
    "waveglow_ljs625_bs18": (dict(pip=_LIBROSA), None, _PIP_WHY),
    # the 48 / 80 GB tiers of the same task (lambda_vram_waveglow.py)
    "waveglow_ljs625_bs32": (dict(pip=_LIBROSA), None, _PIP_WHY),
    "waveglow_ljs625_bs48": (dict(pip=_LIBROSA), None, _PIP_WHY),
    "transformer_xl_base_wt103_bs24": (dict(data=_WT103[0]["name"]), _WT103, _WT103_WHY),
    "transformer_xl_large_wt103_bs8": (dict(data=_WT103[0]["name"]), _WT103, _WT103_WHY),
}

WRAPPER = '''

def vram_{name}({params}):
    """Runs {name} and reports its VRAM peak (lambda_vram_tier24.py)."""
    import threading
    import time

    import torch

    torch.cuda.init()
    free, total = torch.cuda.mem_get_info()
    peak = {{"used": total - free}}
    context = total - free
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            f, _ = torch.cuda.mem_get_info()
            peak["used"] = max(peak["used"], total - f)
            time.sleep(0.05)

    th = threading.Thread(target=poll, daemon=True)
    th.start()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.monotonic()
    gb = 1024 ** 3
    vram = {{}}
    try:
        out = {name}({call})
    finally:
        stop.set()
        th.join()
        f, _ = torch.cuda.mem_get_info()
        peak["used"] = max(peak["used"], total - f)
        vram = {{"gpu": torch.cuda.get_device_name(0),
                "total_gb": round(total / gb, 2),
                "context_gb": round(context / gb, 2),
                "allocated_gb": round(torch.cuda.max_memory_allocated() / gb, 2),
                "reserved_gb": round(torch.cuda.max_memory_reserved() / gb, 2),
                "device_used_gb": round(peak["used"] / gb, 2),
                "run_sec": round(time.monotonic() - t0, 1)}}
        print("VRAM_PEAK " + " ".join(f"{{k}}={{v}}" for k, v in vram.items()), flush=True)
    return {{"vram": vram, "task_output": out}}
'''


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # the client finds the wrapper's helpers through its module
    spec.loader.exec_module(mod)
    return mod


def build(task: str, gen_dir: Path):
    """Variant module and the wrapper task function of a generated module."""
    var = _load(VARIANTS / f"{task}.py", f"variant_{task}")
    fn = var.FUNC
    sig = inspect.signature(fn)
    src = textwrap.dedent(inspect.getsource(fn))
    for old, new, why in PATCHES.get(task, []):
        if src.count(old) != 1:
            raise RuntimeError(f"{task}: patch target not found once in the stand's code: {old!r}")
        line_start = src.rfind("\n", 0, src.index(old)) + 1
        line = src[line_start:src.index("\n", line_start)]
        indent = line[:len(line) - len(line.lstrip())]
        note = f"{indent}# Departure from the stand's task (lambda_vram_tier24.py): {why}\n"
        src = src[:line_start] + note + src[line_start:].replace(old, new, 1)
        print(f"    patch {task}: {old.splitlines()[0]!r} ({old.count(chr(10)) + 1} lines): {why[:60]}...", flush=True)
    src += WRAPPER.format(
        name=fn.__name__, params=", ".join(str(p) for p in sig.parameters.values()),
        call=", ".join(f"{p}={p}" for p in sig.parameters))
    path = gen_dir / f"vram_{task}.py"
    path.write_text(src)
    return var, getattr(_load(path, f"vram_{task}"), f"vram_{fn.__name__}")


def ensure_data_sources(client: KrauncherClient, sources: list[dict]) -> None:
    """Register the task's data sources on this account when missing (as the stand does)."""
    for ds in sources:
        try:
            client.data_source(ds["name"]).info
        except KrauncherError:
            client.data_source(**ds)


async def run_one(client, task: str, gpu: str, gen_dir: Path, forecast: float | None = None,
                  vram_gb: int = VRAM_GB, bound_gb: int = BOUND_GB) -> None:
    """Run one stand task with the peak wrapper on a card of its tier
    (``vram_gb`` asked, ``bound_gb`` the stand's bound; other scripts of
    other/ reuse it for other tiers)."""
    forecast = TASKS.get(task) if forecast is None else forecast
    var, wrapper = build(task, gen_dir)
    over, sources, why = OVERRIDES.get(task, ({}, None, ""))
    if over:
        print(f"    override {task}: {over}: {why}", flush=True)
    ensure_data_sources(client, getattr(var, "DATA_SOURCES", []) if sources is None else sources)
    opts = dict(var.OPTIONS, **over, vram_gb=vram_gb, gpu_name=gpu, stream_stderr=True)
    print(f"--- {task}  forecast {forecast} GB  bound {bound_gb} GB  options {opts}", flush=True)
    try:
        h = await client.task(**opts)(wrapper)(**var.KWARGS)
        r = await h.wait(timeout=WAIT_TIMEOUT)
    except Exception as e:  # keep going with the next task
        print(f"RESULT {task} status=error error={type(e).__name__}: {e}", flush=True)
        return
    if client.estimate_only:  # profile harvest (scripts/harvest_profiles_stand.py): nothing ran
        print(f"RESULT {task} estimate only", flush=True)
        return
    compute = r.execution_time_sec - r.download_sec - r.pip_install_sec
    vram = r.output.get("vram", {}) if isinstance(r.output, dict) else {}
    print(f"RESULT {task} status={r.status} gpu={r.actual_gpu} worker={r.worker_id} "
          f"download={r.download_sec:.1f}s pip={r.pip_install_sec:.1f}s compute={compute:.1f}s "
          f"total={r.execution_time_sec:.1f}s queue={r.queue_wait_sec:.1f}s", flush=True)
    if vram:
        used = vram["device_used_gb"]
        print(f"VRAM {task} device_used={used} GB reserved={vram['reserved_gb']} GB "
              f"allocated={vram['allocated_gb']} GB context={vram['context_gb']} GB | "
              f"forecast {forecast} GB = x{forecast / used:.2f} of device_used", flush=True)
    else:
        # OOM or error: the wrapper printed VRAM_PEAK in the task log before failing
        print(f"VRAM {task} no peak in the output; traceback:\n{r.traceback or r.stderr[-3000:]}", flush=True)
    print(f"OUTPUT {task} {r.output}", flush=True)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="RTX 4090", help="card of the 24 GB tier (gpu_name substring)")
    ap.add_argument("--only", nargs="+", choices=sorted(TASKS), help="run only these tasks")
    args = ap.parse_args()
    client = KrauncherClient()
    if not client.api_key:
        sys.exit("ERROR: Set CAS_API_KEY in .env")
    with tempfile.TemporaryDirectory(prefix="vram_tier24_") as d:
        for task in args.only or TASKS:
            await run_one(client, task, args.gpu, Path(d))


if __name__ == "__main__":
    asyncio.run(main())
