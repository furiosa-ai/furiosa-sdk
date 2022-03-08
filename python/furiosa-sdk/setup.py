#!/usr/bin/env python
from typing import Dict, List

from setuptools import setup

version = "0.6.0.dev.0"

EXTRAS_REQUIREMENTS: Dict[str, List[str]] = {
    "server": ["furiosa-server~=" + version],
    "quantizer": ["furiosa-quantizer~=" + version],
    "litmus": ["furiosa-litmus~=" + version],
    "models": ["furiosa-models~=" + version],
}

EXTRAS_REQUIREMENTS["full"] = [
    req for extras_reqs in EXTRAS_REQUIREMENTS.values() for req in extras_reqs
]

if __name__ == "__main__":
    setup(
        version=version,
        extras_require=EXTRAS_REQUIREMENTS,
    )
