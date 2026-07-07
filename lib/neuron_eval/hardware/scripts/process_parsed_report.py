"""Condenses a parsed synthesis report into a compact ``scores.json``.

Reads the per-report JSON produced by the report parser and reduces it to a
single logic-cost figure (LUTs plus BRAM/URAM converted to LUT equivalents),
a latency in nanoseconds, and per-scenario dynamic power.
"""

import sys
import json
from pathlib import Path

BRAM_LUT_FACTOR = 10
URAM_LUT_FACTOR = BRAM_LUT_FACTOR * 8
CYCLE_TIME      = 10 # ns

def extract_resource_info(report_json: Path) -> list:
    """Extracts logic, latency and power from a parsed report JSON.

    Returns a dict with keys ``logic``, ``latency`` and ``power``; the logic and
    power fields are ``None`` if the file is missing or not a ``.json``.
    """
    resources = {
        "logic": None,
        "power": None,
    }

    # Check if report is a valid JSON file
    if not (report_json.is_file() and report_json.suffix == ".json"):
        print("WARNING: File does not exist or is not a .json:")
        print(f"    {report_json}")
        return resources

    with open(report_json, "r") as f:
        json_data = json.load(f)

    sum_totalluts = 0
    sum_bram = 0
    sum_uram = 0
    sum_dspblocks = 0

    power_idle  = 0
    power_spike = 0
    power_reset = 0

    latency = 1

    for entry in json_data:
        if entry.get('type') == 'util-h.rpt':
            parsed = entry.get('parsed', {})
            sum_totalluts   = float(parsed.get('totalluts', 0))
            sum_bram        = float(parsed.get('ramb36', 0)) * 2
            sum_bram       += float(parsed.get('ramb18', 0))
            sum_uram        = float(parsed.get('uram', 0))
            sum_dspblocks   = float(parsed.get('dspblocks', 0))
        elif entry.get("type").startswith("power") and entry.get("type").endswith(".rpt"):
            parsed = entry.get('parsed', {})
            if ("idle" in entry.get("type")):
                power_idle  = float(parsed.get("dynamic", {}).get("neuron_0", 0))
            elif ("random-nospike" in entry.get("type")):
                power_spike = float(parsed.get("dynamic", {}).get("neuron_0", 0))
            elif ("random-spike" in entry.get("type")):
                power_reset = float(parsed.get("dynamic", {}).get("neuron_0", 0))
        elif entry.get('type') == 'vivado.log':
            parsed  = entry.get('parsed', {})
            latency = int(parsed.get('run_ip_uut_latency', 0))

    resources["logic"]   = sum_totalluts + sum_bram * BRAM_LUT_FACTOR + sum_uram * URAM_LUT_FACTOR
    resources["latency"] = latency * CYCLE_TIME # latency in ns
    resources["power"]   = {
        "idle": power_idle,
        "spike_in": power_spike,
        "spike_out": power_reset
    }
    return resources


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python process_parsed_report.py <path_to_json>")
        sys.exit(1)
    report_path = Path(sys.argv[1])
    
    scores = extract_resource_info(report_path)
    
    # write to file scores.json
    with open(report_path.parent / "scores.json", "w") as f:
        json.dump(scores, f, indent=4) 
