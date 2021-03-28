"""Nux - Furiosa Native API Binding"""

import os

log_enabled = os.getenv('FURIOSA_LOG')
if log_enabled is not None and log_enabled == 'true':
    os.environ['RUST_LOG']="info"

from .v1 import LIBNUX

del os
