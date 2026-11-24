"""Isolated HTTP policy servers."""

from typing import Literal

POLICY_MODULES = {
    "smolvla": "backend.policy_servers.smolvla_server",
    "pi05": "backend.policy_servers.pi05_server",
    "groot": "backend.policy_servers.groot_server",
}
GROOT_SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
GROOT_DEFAULT_SUITE = "libero_object"

TRTPrecision = Literal["bf16", "fp8", "nvfp4"]
TRT_PRECISIONS: tuple[TRTPrecision, ...] = ("bf16", "fp8", "nvfp4")
