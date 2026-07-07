"""Tests for the HardwareEvaluator (neuron C-code generation and report parsing)."""

import os
import sys

import pytest

pytest.importorskip("torch")

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from lib.neuron_eval.hardware.hardware_evaluator import HardwareEvaluator


def _setup_hardware_evaluator(tmp_path, run_name="testrun"):
    return HardwareEvaluator(run_name, num_devices=0, generation=0, output_main_dir=tmp_path)


def test_cleanup_temp_files_removes_intermediates(tmp_path):
    hw = _setup_hardware_evaluator(tmp_path, run_name="cleanup")
    neuron_dir = hw.output_dir / "neuron_a"

    # Build directory tree with files to keep and remove
    (neuron_dir / "ip" / "ip").mkdir(parents=True)
    (neuron_dir / "ip" / "ip" / "junk.txt").write_text("temp")

    reports = neuron_dir / "reports"
    reports.mkdir(parents=True)
    (reports / "scores.json").write_text("{}")
    (reports / "parsed_report.json").write_text("{}")
    (reports / "extra.txt").write_text("remove me")

    logs = neuron_dir / "logs"
    logs.mkdir(parents=True)
    (logs / "run_hls.log").write_text("keep")
    (logs / "vitis_warn.txt").write_text("keep")
    (logs / "vivado_warn.txt").write_text("keep")
    (logs / "orphan.log").write_text("remove me")

    vivado = neuron_dir / "vivado"
    vivado.mkdir(parents=True)
    (vivado / "neuron_config.vh").write_text("keep")
    (vivado / "tb_neuron.sv").write_text("keep")
    (vivado / "vivado.log").write_text("keep")
    (vivado / "temp.txt").write_text("remove me")

    vitis = neuron_dir / "vitis"
    vitis.mkdir(parents=True)
    (vitis / "artifact.txt").write_text("remove me")

    (neuron_dir / "hls_config.tcl").write_text("remove me")

    hw._cleanup_temp_files(neuron_dir)

    assert not (neuron_dir / "ip" / "ip").exists()
    assert not (reports / "extra.txt").exists()
    assert not (logs / "orphan.log").exists()
    assert not (vivado / "temp.txt").exists()
    assert not vitis.exists()
    assert not (neuron_dir / "hls_config.tcl").exists()

    assert (reports / "scores.json").exists()
    assert (reports / "parsed_report.json").exists()
    assert (logs / "run_hls.log").exists()
    assert (logs / "vitis_warn.txt").exists()
    assert (logs / "vivado_warn.txt").exists()
    assert (vivado / "neuron_config.vh").exists()
    assert (vivado / "tb_neuron.sv").exists()
    assert (vivado / "vivado.log").exists()


def test_copy_failed_neuron_to_error_dir(tmp_path):
    hw = _setup_hardware_evaluator(tmp_path, run_name="copy")
    neuron_name = "neuron_b"
    neuron_dir = hw.output_dir / neuron_name
    (neuron_dir / "reports").mkdir(parents=True)
    (neuron_dir / "reports" / "scores.json").write_text("{}")

    hw._copy_failed_neuron_to_error_dir(neuron_dir, neuron_name)

    copied_dir = hw.error_dir / neuron_name
    assert copied_dir.exists()
    assert (copied_dir / "reports" / "scores.json").exists()
