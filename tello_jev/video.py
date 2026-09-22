"""Tello H.264 UDP stream -> continuously refreshed JPEG on disk (via ffmpeg).

Only ffmpeg is required; no OpenCV needed to *capture*. `latest_frame()` returns
the newest complete JPEG bytes, or None if the stream is not producing frames.
"""
import logging
import os
import shutil
import subprocess
import time
from typing import Optional

log = logging.getLogger("video")


class VideoStream:
    def __init__(self, port: int, work_dir: str, fps: int = 3, record_path: Optional[str] = None,
                 overlay_top: Optional[str] = None, overlay_bottom: Optional[str] = None,
                 overlay_font: Optional[str] = None):
        self.port = port
        self.dir = work_dir
        self.fps = fps
        self.path = os.path.join(work_dir, "latest.jpg")
        self.record_path = record_path
        # overlay_top/bottom are files ffmpeg re-reads on every frame (drawtext reload=1) and
        # that get overwritten elsewhere with just the latest line -- the run.sh console log
        # at the top, jev's latest decision at the bottom -- so this only ever burns in the
        # single most recent line, not the whole growing log.
        self.overlay_top = overlay_top
        self.overlay_bottom = overlay_bottom
        self.overlay_font = overlay_font
        self.proc: Optional[subprocess.Popen] = None
        os.makedirs(work_dir, exist_ok=True)
        if record_path:
            os.makedirs(os.path.dirname(record_path), exist_ok=True)

    def _overlay_filter(self) -> Optional[str]:
        if not self.overlay_font or not os.path.exists(self.overlay_font):
            if self.overlay_top or self.overlay_bottom:
                log.warning("overlay font %r not found; recording will have no burned-in text", self.overlay_font)
            return None
        parts = []
        common = "reload=1:fontfile=%s:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=6" % self.overlay_font
        for path, y in ((self.overlay_top, "10"), (self.overlay_bottom, "h-th-10")):
            if path:
                open(path, "a").close()  # drawtext errors on a missing file at filter init
                parts.append("drawtext=textfile=%s:%s:x=10:y=%s" % (path, common, y))
        return ",".join(parts) if parts else None

    def start(self):
        ff = shutil.which("ffmpeg")
        if not ff:
            raise RuntimeError("ffmpeg not found on PATH")
        if os.path.exists(self.path):
            os.remove(self.path)
        self.log_path = os.path.join(self.dir, "ffmpeg.log")
        cmd = [
            ff, "-hide_banner", "-loglevel", "warning", "-nostdin",
            # explicit format: without it ffmpeg has to auto-probe the raw H.264-over-UDP
            # stream, which can take longer than the startup window allows.
            "-f", "h264",
            # NOT -fflags nobuffer / -flags low_delay: those disable the reordering buffer,
            # and slightly out-of-order UDP packets (routine over Wi-Fi) then decode as
            # corrupt frames. We only need 3fps snapshots, not true minimum latency.
            "-i", "udp://0.0.0.0:%d?overrun_nonfatal=1&fifo_size=50000000" % self.port,
            # the Tello encodes non-full-range (studio) YUV, which the mjpeg encoder
            # otherwise refuses outright ("Non full-range YUV is non-standard").
            "-vf", "fps=%d" % self.fps, "-q:v", "3", "-strict", "unofficial",
            "-update", "1", "-y", self.path,
        ]
        self._reencoding = False
        if self.record_path:
            overlay = self._overlay_filter()
            if overlay:
                self._reencoding = True
                # drawtext needs decoded pixels, so this output can no longer be a lossless
                # stream copy -- it has to actually decode and re-encode.
                cmd += ["-vf", overlay, "-c:v", "libx264", "-preset", "veryfast", "-strict", "unofficial"]
            else:
                cmd += ["-c:v", "copy"]
            # NOT -movflags +faststart: it requires ffmpeg to rewrite the *entire* file at
            # shutdown to relocate the moov atom to the front, and that rewrite scales with
            # file size -- for a long/heavy recording it can outrun any reasonable shutdown
            # grace period, leaving the moov atom (and so the whole file) never written at
            # all. These are local-only files, never streamed, so faststart buys nothing.
            cmd += ["-an", "-y", self.record_path]
        # a plain PIPE that nothing drains can fill up and block ffmpeg's own writes if it
        # logs enough (e.g. repeated PPS-reference warnings while catching the stream) --
        # a file avoids that deadlock entirely and doubles as a log for post-mortem review.
        self._log_fh = open(self.log_path, "wb")
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=self._log_fh)
        log.info("ffmpeg capture started on udp:%d -> %s%s", self.port, self.path,
                 (" (+ recording -> %s)" % self.record_path) if self.record_path else "")

    def _log_tail(self, n_chars: int = 800) -> str:
        try:
            with open(self.log_path, "rb") as f:
                return f.read().decode(errors="ignore").strip()[-n_chars:]
        except OSError:
            return ""

    def wait_first_frame(self, timeout: float = 15.0) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.latest_frame():
                log.info("first video frame after %.1fs", time.time() - t0)
                return True
            if self.proc and self.proc.poll() is not None:
                log.error("ffmpeg exited early: %s", self._log_tail())
                return False
            time.sleep(0.5)
        log.warning("no video frame within %.0fs; ffmpeg log tail: %s", timeout, self._log_tail())
        return False

    def latest_frame(self, max_age_s: float = 3.0) -> Optional[bytes]:
        try:
            st = os.stat(self.path)
        except FileNotFoundError:
            return None
        if time.time() - st.st_mtime > max_age_s or st.st_size < 2000:
            return None
        with open(self.path, "rb") as f:
            data = f.read()
        # -update writes in place; accept only a complete JPEG
        if data[:2] == b"\xff\xd8" and data[-2:] == b"\xff\xd9":
            return data
        return None

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            # a re-encoded recording (overlays active) needs real time after SIGTERM to
            # flush its encoder and write the mp4 trailer (moov atom) -- killing it too
            # early leaves a file ffmpeg/players can't open at all ("moov atom not found"),
            # not just a truncated tail.
            grace = 10 if getattr(self, "_reencoding", False) else 3
            try:
                self.proc.wait(grace)
            except subprocess.TimeoutExpired:
                log.warning("ffmpeg didn't exit within %ss, killing it (recording may be unplayable)", grace)
                self.proc.kill()
        self.proc = None
        if getattr(self, "_log_fh", None):
            self._log_fh.close()
            self._log_fh = None


class FakeVideoStream(VideoStream):
    """Dry-run: serves a fixed image (or a tiny placeholder JPEG) as the frame."""

    def __init__(self, port, work_dir, fps=3, sample_image: Optional[str] = None, record_path: Optional[str] = None,
                 overlay_top: Optional[str] = None, overlay_bottom: Optional[str] = None,
                 overlay_font: Optional[str] = None):
        super().__init__(port, work_dir, fps, record_path=record_path, overlay_top=overlay_top,
                          overlay_bottom=overlay_bottom, overlay_font=overlay_font)
        self.sample = sample_image

    def start(self):
        log.info("fake video stream (dry-run)")

    def wait_first_frame(self, timeout=15.0):
        return True

    def latest_frame(self, max_age_s=3.0):
        if self.sample and os.path.exists(self.sample):
            with open(self.sample, "rb") as f:
                return f.read()
        # minimal valid JPEG header/footer so downstream code paths run
        return b"\xff\xd8" + b"\x00" * 2100 + b"\xff\xd9"

    def stop(self):
        pass
