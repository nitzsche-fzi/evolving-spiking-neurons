"""Shared pytest configuration: the --run-long option and custom markers."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-long",
        action="store_true",
        default=False,
        help="Run tests marked as longrun (slow or compilation-heavy).",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "longrun: mark test as long-running or requiring heavy compilation.",
    )

    config.addinivalue_line(
        "markers",
        "visual: mark test as visual-only (plots, images, slow diagnostics).",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-long"):
        return

    skip_long = pytest.mark.skip(reason="need --run-long option to run")
    for item in items:
        if "longrun" in item.keywords:
            item.add_marker(skip_long)
