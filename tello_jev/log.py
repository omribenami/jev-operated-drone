import logging
import os
import sys
import time

# fixed filename (not timestamped) under logs_dir: mission.py points the video recording's
# top text overlay at this same path, always overwritten with just the latest console line.
OVERLAY_TOP_NAME = "overlay_top.txt"


class _Tee:
    """Duplicates writes to the real console stream and into the mission log file, so
    plain print() output (the CLI's report/probe/calibrate text) lands in the same log
    as the structured logger lines, not just on the terminal. Also keeps overlay_path
    (if given) overwritten with just the latest non-blank line, for burning into video."""

    def __init__(self, stream, fh, overlay_path=None):
        self.stream = stream
        self.fh = fh
        self.overlay_path = overlay_path

    def write(self, data):
        self.stream.write(data)
        self.fh.write(data)
        self.fh.flush()
        if self.overlay_path:
            lines = [l for l in data.splitlines() if l.strip()]
            if lines:
                try:
                    with open(self.overlay_path, "w") as f:
                        f.write(lines[-1])
                except OSError:
                    pass

    def flush(self):
        self.stream.flush()
        self.fh.flush()

    def isatty(self):
        return self.stream.isatty()


def setup(logs_dir: str) -> logging.Logger:
    os.makedirs(logs_dir, exist_ok=True)
    path = os.path.join(logs_dir, time.strftime("mission_%Y%m%d_%H%M%S.log"))
    overlay_path = os.path.join(logs_dir, OVERLAY_TOP_NAME)
    fmt = logging.Formatter("%(asctime)s %(levelname)-5s %(name)s: %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(path)
    fh.setFormatter(fmt)
    root.addHandler(sh)
    root.addHandler(fh)
    tee_file = open(path, "a")
    sys.stdout = _Tee(sys.stdout, tee_file, overlay_path)
    sys.stderr = _Tee(sys.stderr, tee_file, overlay_path)
    log = logging.getLogger("mission")
    log.info("log file: %s", path)
    return log
