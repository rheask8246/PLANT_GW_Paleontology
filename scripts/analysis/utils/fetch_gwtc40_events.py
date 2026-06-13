#!/usr/bin/env python3
"""
Download GWTC-4.0 confident-event **default PE** parameters from the GWOSC API and
write a CSV usable with `07b_synthetic_real_validation.py` (`--real-events-csv`).

API: https://gwosc.org/api/v2/docs  
Catalog list: https://gwosc.org/eventapi/html/GWTC-4.0/

Uses only the stdlib (no `requests`). Follows `next` links until all pages are read.

Example:
  python3.11 fetch_gwtc40_gwosc_csv.py -o data/gwtc40_o4a_confident_default_pe.csv

(On some HPC login nodes ``python3`` is 3.6; use ``python3.11`` or ``conda activate plant``.)
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from plant_paths import (  # noqa: E402
    PROJECT_ROOT,
    REPO_ROOT,
    SCRIPTS_DIR,
    ensure_paths,
    find_data_dir,
    find_work_dir,
    load_posterior_network_module,
)

ensure_paths()

import argparse
import sys
import csv
import json
import urllib.request
from typing import Any, Dict, List, Optional

BASE = "https://gwosc.org/api/v2/catalogs/GWTC-4.0/events"
# format=json returns machine-readable JSON; include-default-parameters = table "best" values
QUERY = "format=json&include-default-parameters=true"


def _get_json(url: str) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "PLANT_GW_Paleontology/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _param_map(default_parameters: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for p in default_parameters:
        name = p.get("name")
        if name is None:
            continue
        b = p.get("best")
        out[name] = float(b) if b is not None and isinstance(b, (int, float)) else None
    return out


def main() -> None:
    if sys.version_info < (3, 7):
        print("Need Python ≥ 3.7 (try: python3.11 fetch_gwtc40_gwosc_csv.py ...)", file=sys.stderr)
        sys.exit(1)

    ap = argparse.ArgumentParser(description="GWTC-4.0 GWOSC API → CSV for 07b")
    ap.add_argument("-o", "--output", type=str, required=True, help="Output CSV path")
    args = ap.parse_args()

    url: Optional[str] = f"{BASE}?{QUERY}&page=1"
    rows: List[Dict[str, Any]] = []
    catalog_total: Optional[int] = None
    while url:
        data = _get_json(url)
        if catalog_total is None:
            catalog_total = int(data.get("results_count") or 0)
        for ev in data.get("results", []):
            name = ev.get("name", "")
            pm = _param_map(ev.get("default_parameters") or [])
            m1 = pm.get("mass_1_source")
            m2 = pm.get("mass_2_source")
            xi = pm.get("chi_eff")
            z = pm.get("redshift")
            if m1 is None or m2 is None or xi is None or z is None:
                continue
            rows.append(
                {
                    "event": name,
                    "mass_1_source": m1,
                    "mass_2_source": m2,
                    "chi_eff": xi,
                    "redshift": z,
                }
            )
        nxt = data.get("next")
        url = str(nxt) if nxt else None

    out_path = args.output
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["event", "mass_1_source", "mass_2_source", "chi_eff", "redshift"],
        )
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} events with full mass_1/mass_2/chi_eff/z to {out_path}", flush=True)
    if catalog_total and len(rows) < catalog_total:
        print(
            f"(Skipped {catalog_total - len(rows)} events: missing default PE masses/spin/z in API.)",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
