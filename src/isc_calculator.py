"""Ideal-name alias: Aperiodic-ISC and classic ISC helpers."""
from src.aperiodic_isc import *  # noqa: F401,F403
try:
    from src.aperiodic_isc_analysis import *  # noqa: F401,F403
except ImportError:
    pass
try:
    from src.classic_isc_analysis import *  # noqa: F401,F403
except ImportError:
    pass
