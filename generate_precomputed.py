"""Erzeugt `precomputed_pools.json` (je Sicherheitsfaktor 10 000 unabhängige Standort-Fenster, SplitMix64) und
`precomputed_boundaries.json` (Pocock-Konstante, O'Brien-Fleming-Konstante und Lan-DeMets-Grenzen für 1 bis 30 Blicke,
rekursive Integration).

    python generate_precomputed.py            # etwa eine Minute"""

import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import seq_constants as C
from seq_boundaries import PRECOMPUTED as BOUNDS_OUT
from seq_boundaries import lan_demets, obf_const, pocock_const
from seq_rng import SplitMix64
from seq_vehicle import window_service_level

OUT = Path(__file__).resolve().parent / "precomputed_pools.json"


def unit_seed(z_index, i):
    return 9_000_011 * (z_index + 1) + 7 * i + 1


def _chunk(args):
    z_index, z, start, stop = args
    return [round(window_service_level(z, SplitMix64(unit_seed(z_index, i))), 6) for i in range(start, stop)]


def _bounds(k):
    c_obf, obf = obf_const(k)
    ld = lan_demets(k)
    return k, {"pocock": round(pocock_const(k), 6), "obf_C": round(c_obf, 6), "obf": [round(float(x), 6) for x in obf],
               "ld": [None if x == float("inf") else round(float(x), 6) for x in ld]}


def main():
    t0 = time.time()
    step = 500
    jobs = [(zi, z, s, min(s + step, C.POOL_SIZE)) for zi, z in enumerate(C.Z_GRID) for s in range(0, C.POOL_SIZE, step)]
    with mp.Pool() as pool:
        parts = pool.map(_chunk, jobs)
        bounds = pool.map(_bounds, list(range(C.K_MIN, C.K_MAX + 1)))
    pools = {f"{z:.1f}": [] for z in C.Z_GRID}
    for (zi, z, _s, _e), values in zip(jobs, parts):
        pools[f"{z:.1f}"].extend(values)
    meta = {"pool_size": C.POOL_SIZE, "window_days": C.WINDOW_DAYS, "burn_days": C.BURN_DAYS, "base_mean": C.BASE_MEAN,
            "disp": C.DISP, "lead": C.LEAD, "z_grid": list(C.Z_GRID)}
    OUT.write_text(json.dumps({"meta": meta, "pools": pools}, separators=(",", ":")), encoding="utf-8")
    BOUNDS_OUT.write_text(json.dumps({"alpha": C.ALPHA, "grid": 0.005, "boundaries": {str(k): v for k, v in bounds}},
                                     separators=(",", ":")), encoding="utf-8")
    print(f"{OUT.name}: {len(C.Z_GRID)} Pools à {C.POOL_SIZE}; {BOUNDS_OUT.name}: {len(bounds)} Blick-Zahlen, "
          f"{time.time() - t0:.0f} s", file=sys.stderr)


if __name__ == "__main__":
    main()
