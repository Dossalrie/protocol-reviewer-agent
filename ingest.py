from __future__ import annotations

import json
from pathlib import Path

DATA_PATH = Path(__file__).with_name("opportunities.json")
OUTPUT_PATH = Path(__file__).with_name("opportunities_seeded.json")


def load_and_validate() -> list[dict]:
    with DATA_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    opportunities = payload.get("opportunities", [])
    if not opportunities:
        raise ValueError("No opportunities found in opportunities.json")

    return opportunities


if __name__ == "__main__":
    opportunities = load_and_validate()
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump({"opportunities": opportunities}, handle, indent=2)
    print(f"Loaded {len(opportunities)} opportunities from {DATA_PATH}")
    print(f"Wrote validated seed file to {OUTPUT_PATH}")