"""Compatibility import for callers that used the pre-upstream sleep name.

The implementation was removed in favour of the pinned public
``core.sleep.run_sleep_cycle`` contract. New code should import
``sleep_adapter`` directly; this module remains a narrow transition shim for
Hermes installations that still resolve the historical module path.
"""

from .sleep_adapter import run_sleep_cycle

__all__ = ["run_sleep_cycle"]
