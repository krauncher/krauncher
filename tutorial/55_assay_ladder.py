"""Tutorial 55: The assay of a task, its work and its compute coefficient on every GPU — before running it.

Nothing runs on a GPU here. The analyzer reads your code and issues an assay:
the facts of the payload (what the code does, the model, the run). From an
assay come the estimate — what it needs and how much work it is, measured on
the reference card, in CU — and the ladder: every GPU's coefficient for the
compute phase against the reference card, fastest first. An assay can also be
written by hand, without code: the second part does that.

    Client                                   Analyzer / broker
    ──────                                   ─────────────────
    analyzer.classify(source)           →   analysis (code E2E-encrypted)
    analyzer.assay(job_id)              ←   assay/2: the facts of the payload
    client.estimate(assay)              ←   estimate/1: VRAM, disk, work on the reference card
    client.ladder(assay)                ←   estimate/1 + compute coefficient per GPU

Time on a GPU = compute_ratio x the work's compute phase + its other phases
(warmup, I/O, setup), which do not depend on the GPU. Multiply by the price
you pay. The ladder is available to accounts with ladder access enabled.

Run it from a folder holding your .env (CAS_API_KEY):

    python tutorial/55_assay_ladder.py
"""

import asyncio
import inspect

from krauncher import KrauncherClient, KrauncherError

client = KrauncherClient()


def train(epochs: int = 3, batch_size: int = 128):
    import torch
    import torchvision
    from torch.utils.data import DataLoader

    model = torchvision.models.resnet50(num_classes=10).cuda()
    data = torchvision.datasets.CIFAR10("data", train=True, download=True,
                                        transform=torchvision.transforms.ToTensor())
    loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=4)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(epochs):
        for x, y in loader:
            opt.zero_grad()
            torch.nn.functional.cross_entropy(model(x.cuda()), y.cuda()).backward()
            opt.step()


# The same kind of forecast for a task described by hand: no code. What is not
# stated takes the formula's default; the estimate names it in its hints.
BY_HAND = {
    "meta": {"schema_version": "assay/2"},
    "workload": {
        "type": "llm_inference", "mode": "inference",
        "model": {"id": "meta-llama/Llama-3.1-8B-Instruct", "precision": "bf16", "arch_type": "transformer"},
        "runtime": {"inference_engine": "vllm"},
        "knobs": {"batch_size": 32, "input_tokens": 512, "max_new_tokens": 512},
        "data": {"loads_model_weights": True},
    },
}


def show(out: dict) -> None:
    work, ladder = out["estimate"]["work"], out["ladder"]
    sec_per_cu = work["reference_sec"] / work["reference_cu"]
    ph = {k: v * sec_per_cu for k, v in work["phases_cu"].items()}
    print(f"{'GPU':<28}{'VRAM, GB':>9}{'compute x':>11}{'time, s':>10}")
    print("─" * 58)
    for row in ladder["rows"][:10]:
        warmup = row["warmup_sec"] if row["warmup_sec"] is not None else ph["warmup"]
        sec = row["compute_ratio"] * ph["compute"] + warmup + ph["io"] + ph["setup"]
        print(f"{row['gpu_name']:<28}{row['vram_gb']:>9}{row['compute_ratio']:>11.3f}{sec:>10.1f}")
    print(f"... {len(ladder['rows'])} GPUs, calibration {ladder['meta']['calibration_id']}")


async def main():
    if not client.api_key:
        print("ERROR: Set CAS_API_KEY in .env (run seed_api_key.py first)")
        return

    # The analyzer this account is configured for (address and token come from
    # the broker). classify() returns the analyzer's job id with the result.
    analyzer = client.analyzer()
    c = await analyzer.classify(inspect.getsource(train), kwargs={"epochs": 3, "batch_size": 128})
    assay = await analyzer.assay(c.analyzer_job_id)
    print(f"Assay from code: {assay['workload']['type']}, model {assay['workload']['model']['name']}")

    for name, a in (("from code", assay), ("by hand", BY_HAND)):
        est = await client.estimate(a)
        work = est["work"]
        print(f"\n[{name}] min VRAM {est['requirements']['min_vram_gb']} GB, "
              f"reference card {work['reference_cu']} CU = {work['reference_sec']} s "
              f"(a slow host up to x{work['spread']['factor']})")
        for h in est["hints"]:
            print(f"  hint: state {h['key']} — {h['reason']}")
        try:
            show(await client.ladder(a))
        except KrauncherError as exc:
            print(f"  {exc}")


if __name__ == "__main__":
    asyncio.run(main())
