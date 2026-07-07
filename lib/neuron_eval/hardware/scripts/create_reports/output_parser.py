"""Parses Vivado/Vitis report and log files into a structured JSON summary.

Walks one or more test-run directories, parses the utilization, power and
timing ``.rpt`` files plus the Vivado ``.log`` files found there, and collects
the extracted metrics into a list of report dicts that can be pretty-printed or
written to ``parsed_report.json``.
"""

from pathlib import Path
import os
import sys
import fnmatch
import re


def printwarn(str):
    print("\033[93m{}\033[0m".format(str))


def find_files(directory, pattern):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for filename in fnmatch.filter(files, pattern):
            file_list.append(os.path.join(root, filename))
    return sorted(file_list)


def find_dirs(directory, pattern):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for filename in fnmatch.filter(dirs, pattern):
            file_list.append(os.path.join(root, filename))
    return sorted(file_list)


def parse_rpt_header(content, key):
    match = re.search(f"\| {re.escape(key)} *: (.*)", content, re.IGNORECASE)
    if not match:
        return "?"
    return match.group(1).lower()


def parse_table_get_line(content, key):
    match = re.search(f"^\| *{re.escape(key)} *\|(.*)", content, re.IGNORECASE | re.MULTILINE)
    if not match:
        return None
    return match.group(1).lower().replace(" ", "").split("|")[:-1]


R_TABLE_LINESEP = "^\+-+\+.*"
R_TABLE_LINE = "^\| +\w+ +\|.*"
R_TABLE = f"({R_TABLE_LINESEP}\n)({R_TABLE_LINE}\n)({R_TABLE_LINESEP}\n)"


def find_tables(content):
    iterator = re.finditer(f"{R_TABLE_LINE}$", content, re.MULTILINE)
    for match in iterator:
        print(match.group())


def match_line_groups(regex, content):
    match = re.search(regex, content, re.MULTILINE)
    if not match:
        return None
    return match.groups()


def parse_report_util_h(content):
    data = {}
    line_header = parse_table_get_line(content, "Instance")
    line_content = parse_table_get_line(content, TOP_NAME)

    if not line_header or not line_content:
        return None

    for index, header in enumerate(line_header):
        try:
            data[header] = int(line_content[index])
        except:
            data[header] = line_content[index]

    data["ramb_kb"] = data["ramb36"] * 36 + data["ramb18"] * 18
    data["lutrams_kb"] = data["lutrams"] * 64 / 1000  # 32 for small LUTRAMs, 64 for large ones
    data["ram_kb"] = data["ramb_kb"] + data["lutrams_kb"]
    return data


def parse_report_power(content):
    data = {}
    line_content = parse_table_get_line(content, TOP_NAME)
    line_content_neuron = parse_table_get_line(content, "neuron_0")
    line_content_static = parse_table_get_line(content, "Static Power")
    if not line_content:
        return None

    data["dynamic"] = {}
    data["dynamic"][TOP_NAME] = float(line_content[0])
    data["dynamic"]["neuron_0"] = float(line_content_neuron[0])
    data["static"] = float(line_content_static[0])
    return data


def parse_report_timing(content):
    def getslack(content, slacktype):
        # Note: only the slack value is kept; the matching clock signal name is
        # not currently recorded.
        slack_matches = re.finditer(f"^{slacktype}.*Worst Slack +(.+),", content, re.IGNORECASE | re.MULTILINE)
        if not slack_matches:
            return None
        slack_matches_list = list(slack_matches)
        for slack_match in slack_matches_list:
            strval = slack_match.group(1)
            if not strval.endswith("ns"):
                printwarn(f"parse_report_timing error, not endswith ns: '{strval}'")
            else:
                return float(strval.replace("ns", ""))
        return None

    data = {
        "setup": getslack(content, "Setup"),
        "hold": getslack(content, "Hold"),
    }
    return data


def parse_report(name, fpath):
    report = {}
    report["name"] = name
    type = os.path.basename(fpath).split("_")[-1]
    report["type"] = type

    with open(fpath, "r") as f:
        content = f.read()

    design_state = parse_rpt_header(content, "Design State")
    if design_state == "fully routed":
        design_state = "routed"
    if design_state != "synthesized" and design_state != "routed":
        printwarn(f"unknown design state '{design_state}'")

    report["design_state"] = design_state

    if type == "util-h.rpt":
        report["parsed"] = parse_report_util_h(content)
    elif type.startswith("power") and type.endswith(".rpt"):
        report["parsed"] = parse_report_power(content)
    elif type == "timing-summary.rpt":
        report["parsed"] = parse_report_timing(content)

    return report


def parse_log_common(content):
    data = {}

    groups = match_line_groups("^INFO: .+Design nets matched = ([0-9]+) of ([0-9]+)$", content)
    if groups and len(groups) == 2:
        data["sim_saif_match_p"] = 100 * int(groups[0]) / int(groups[1])
    else:
        data["sim_saif_match_p"] = -1

    groups = match_line_groups("^run_ip_uut_latency (.*)$", content)
    data["run_ip_uut_latency"] = groups[0] if groups else -1

    matches = re.findall("^ERROR: .*$", content, re.MULTILINE)
    data["errors"] = []
    for match in matches:
        data["errors"].append(match)
        print(match)

    matches = re.findall("^WARNING: .*$", content, re.MULTILINE)
    matches += re.findall("^CRITICAL WARNING: .*$", content, re.MULTILINE)
    data["warnings"] = []
    for match in matches:
        data["warnings"].append(match)

    return data


def parse_vivado_log(name, fpath):

    with open(fpath, "r") as f:
        content = f.read()

    global content_parts
    content_parts = content.split("*** Running vivado\n")

    reports = []
    for part in content_parts:
        match = re.search("^Attempting to get a license for feature '([a-zA-Z]+)'", part, re.MULTILINE)
        if not match:
            continue

        report = {}
        report["name"] = name
        type = os.path.basename(fpath).split("_")[-1]
        report["type"] = type

        design_state = match.group(1).lower()
        if design_state == "synthesis":
            design_state = "synthesized"
        elif design_state == "implementation":
            design_state = "routed"
        else:
            printwarn(f"unknown design state '{design_state}'")

        report["design_state"] = design_state
        report["parsed"] = parse_log_common(part)
        reports.append(report)

    return reports


def slack_to_mhz(refclk_mhz, slack_ns):
    refperiod = 1 / (refclk_mhz / 1000)
    print(refperiod)
    period = refperiod - slack_ns
    return (1 / period) * 1000


def find_and_parse_reports(testrun_dirs):
    reports = []

    for testrun_dir in testrun_dirs:
        # Check if dir exists
        target_path = Path(testrun_dir)
        if not target_path.is_dir():
            printwarn("Path is skipped because it does not exist or is no directory:")
            print(f"    {target_path}")
            continue

        name = testrun_dir
        reportfiles = find_files(testrun_dir, "*.rpt") + find_files(testrun_dir, "*.log")
        for reportfile in reportfiles:
            print(reportfile)
            if reportfile.endswith(".rpt"):
                r = parse_report(name, reportfile)
                reports.append(r)
            elif reportfile.endswith(".log"):
                r = parse_vivado_log(name, reportfile)
                reports.extend(r)

    return reports


def main(top_name: str, clk: int, search_folders: list, save_to_file: bool = False) -> str:
    """
    Returns path of generated report summary JSON file or an empty string if no report was
    saved to a file.
    """
    global TOP_NAME
    global REFCLK_MHZ

    TOP_NAME   = top_name
    REFCLK_MHZ = clk

    reports = find_and_parse_reports(search_folders)
    json_path = ""
    if len(reports) > 0:
        if save_to_file:
            import json
            json_path = Path(search_folders[0])/"parsed_report.json"
            with open(json_path, "w") as f:
                json.dump(reports, f, indent=4)
        else:
            import pprint
            pprint.pprint(reports)
    else:
        print("No reports found.")
    return json_path


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python output_parser.py <top_name> <clk> <search_folders>")
        print("Running output_parser with default arguments:")
        print("     top_name:       lif_neuron")
        print("     clk:            100")
        print("     search_folders: [\"outputs\"]")
        main("lif_neuron", 100, ["outputs"])
    else:
        print(f"Running output_parser.py {sys.argv[1]} {sys.argv[2]} {[sys.argv[3]]}")
        main(sys.argv[1], sys.argv[2], [sys.argv[3]], save_to_file=True)