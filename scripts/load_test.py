"""Async load runner (plan/phases/phase-7 §7.4).

Sustained concurrent load against a running API; reports latency percentiles. Documented
tool for capacity planning — not part of CI (the CI load check uses TestClient threads).

    python -m scripts.load_test --url http://localhost:8000 --wav clip.wav \
        --concurrency 8 --seconds 60 --token demo:secret
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path


async def _worker(client, url, wav_bytes, headers, stop_at, lats):
    while time.time() < stop_at:
        t0 = time.perf_counter()
        r = await client.post(
            f"{url}/v1/transcribe",
            content=wav_bytes,
            headers={**headers, "content-type": "application/octet-stream"},
        )
        lats.append((time.perf_counter() - t0) * 1000)
        _ = r


async def _run(url, wav, concurrency, seconds, token):
    import httpx  # noqa: PLC0415

    wav_bytes = Path(wav).read_bytes()
    headers = {"authorization": f"Bearer {token.split(':', 1)[1]}"} if token else {}
    lats: list[float] = []
    async with httpx.AsyncClient(timeout=30) as client:
        stop_at = time.time() + seconds
        await asyncio.gather(
            *[_worker(client, url, wav_bytes, headers, stop_at, lats) for _ in range(concurrency)]
        )
    lats.sort()
    if lats:
        n = len(lats)
        print(
            f"requests={n} p50={lats[n // 2]:.0f}ms p95={lats[int(0.95 * n)]:.0f}ms "
            f"p99={lats[min(n - 1, int(0.99 * n))]:.0f}ms"
        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="ARS load test")
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--wav", required=True)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--seconds", type=int, default=60)
    p.add_argument("--token", default="")
    args = p.parse_args(argv)
    asyncio.run(_run(args.url, args.wav, args.concurrency, args.seconds, args.token))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
