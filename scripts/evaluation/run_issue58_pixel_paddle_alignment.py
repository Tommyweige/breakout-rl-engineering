"""Run the frozen Issue #58 pixel-only paddle alignment probe."""
import time

_PROCESS_ENTRY_STARTED_AT = time.perf_counter()

from breakout_rl.issue58_pixel_paddle_alignment import cli

if __name__ == "__main__":
    raise SystemExit(cli(process_start=_PROCESS_ENTRY_STARTED_AT))
