#!/usr/bin/env python3
"""Run the opt-in deterministic sleep contention benchmark."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plugins.memory.cashew.sleep_benchmark import main  # noqa: E402

if __name__ == "__main__":
    main()
