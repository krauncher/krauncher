"""VRAM peak of the stand's Lambda WaveGlow tasks of the 48 and 80 GB tiers, run on Krauncher.

Not a calibration series: kept out of calibration/ so the profile harvester
does not take it into the calibration corpus. The 24 GB tier (waveglow bs18)
is in lambda_vram_tier24.py; this script runs the two larger batches of the
same task, each on a card of its tier as Lambda did (bs32: a 48 GB card,
bs48: an 80 GB card), with the same peak wrapper, the same departures from
the stand's code (PATCHES / OVERRIDES of lambda_vram_tier24.py) and the same
log lines (VRAM_PEAK, RESULT, VRAM, OUTPUT).

Usage (from cas-client/, CAS_API_KEY in .env):
    python other/lambda_vram_waveglow.py [--gpu "<name>"] [--only <task> ...]
"""

import argparse
import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lambda_vram_tier24 as base  # noqa: E402  (client side only: the submitted code is the generated module)

from krauncher import KrauncherClient  # noqa: E402

# task -> (analyzer forecast min_vram_gb at stand checkpoint 2026-10-28_c-c94f0fe2c6df
# (LLM full prediction, moves between runs), VRAM asked, stand bound, Lambda's anchor card of the tier)
TASKS = {
    "waveglow_ljs625_bs32": (12, 40, 48, "RTX 6000 Ada"),
    "waveglow_ljs625_bs48": (14, 78, 80, "A100"),
}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", help="override the card of every task (gpu_name substring)")
    ap.add_argument("--only", nargs="+", choices=sorted(TASKS), help="run only these tasks")
    args = ap.parse_args()
    client = KrauncherClient()
    if not client.api_key:
        sys.exit("ERROR: Set CAS_API_KEY in .env")
    with tempfile.TemporaryDirectory(prefix="vram_waveglow_") as d:
        for task in args.only or TASKS:
            forecast, vram_gb, bound_gb, gpu = TASKS[task]
            await base.run_one(client, task, args.gpu or gpu, Path(d), forecast=forecast,
                               vram_gb=vram_gb, bound_gb=bound_gb)


if __name__ == "__main__":
    asyncio.run(main())
