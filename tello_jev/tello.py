"""Minimal Tello SDK 2.0 UDP client with a mandatory readiness gate.

Policy (from tello-flight-recovery): never send flight commands until
`command -> ok`. Every command waits for its ack; movement commands get a
longer timeout because the drone acks only after the move completes.
"""
import logging
import socket
import threading
import time
from typing import Dict, Optional

log = logging.getLogger("tello")


class TelloError(RuntimeError):
    pass


class Tello:
    MOVE_TIMEOUT = 12.0
    CMD_TIMEOUT = 5.0

    def __init__(self, ip: str, cmd_port: int, state_port: int, local_port: int):
        self.addr = (ip, cmd_port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("", local_port))
        self.sock.settimeout(self.CMD_TIMEOUT)
        self.state: Dict[str, str] = {}
        self._state_port = state_port
        self._state_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.ready = False

    # --- transport -------------------------------------------------------
    def send(self, cmd: str, timeout: float = None, quiet: bool = False) -> str:
        """Send a command and return its ack ('' if nothing came back)."""
        self.sock.settimeout(timeout or self.CMD_TIMEOUT)
        # drain stale acks so we never pair a late ack with a new command
        self.sock.setblocking(False)
        try:
            while True:
                self.sock.recvfrom(1024)
        except (BlockingIOError, socket.error):
            pass
        self.sock.settimeout(timeout or self.CMD_TIMEOUT)
        self.sock.sendto(cmd.encode(), self.addr)
        try:
            data, _ = self.sock.recvfrom(1024)
            resp = data.decode(errors="ignore").strip()
        except socket.timeout:
            resp = ""
        if not quiet:
            log.info("%-14s -> %s", cmd, resp or "(no response)")
        return resp

    def flight(self, cmd: str, settle: float = 0.8) -> str:
        """Movement command: gated on readiness, longer timeout, error surfaced."""
        if not self.ready:
            raise TelloError("refusing '%s': SDK not acknowledged" % cmd)
        resp = self.send(cmd, timeout=self.MOVE_TIMEOUT)
        if resp.lower().startswith("error") or resp == "":
            raise TelloError("'%s' failed: %s" % (cmd, resp or "no response"))
        time.sleep(settle)
        return resp

    # --- readiness gate --------------------------------------------------
    def wait_ready(self, probes: int, interval: float) -> bool:
        for i in range(1, probes + 1):
            r = self.send("command", timeout=3.0, quiet=True)
            log.info("[probe %d/%d] command -> %s", i, probes, r or "(no response)")
            if r == "ok":
                self.ready = True
                self.start_state_listener()
                return True
            time.sleep(interval)
        return False

    # --- telemetry -------------------------------------------------------
    def start_state_listener(self):
        if self._state_thread:
            return
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind(("", self._state_port))
        s.settimeout(1.0)

        def loop():
            while not self._stop.is_set():
                try:
                    data, _ = s.recvfrom(2048)
                except socket.timeout:
                    continue
                except OSError:
                    break
                for kv in data.decode(errors="ignore").strip().strip(";").split(";"):
                    if ":" in kv:
                        k, v = kv.split(":", 1)
                        self.state[k] = v
            s.close()

        self._state_thread = threading.Thread(target=loop, daemon=True)
        self._state_thread.start()

    def battery(self) -> int:
        if "bat" in self.state:
            try:
                return int(self.state["bat"])
            except ValueError:
                pass
        r = self.send("battery?", quiet=True)
        try:
            return int(r)
        except ValueError:
            return -1

    def height_cm(self) -> int:
        try:
            return int(self.state.get("h", "0"))
        except ValueError:
            return 0

    def telemetry(self) -> Dict[str, str]:
        keys = ("bat", "h", "tof", "yaw", "templ", "temph", "time")
        return {k: self.state.get(k, "?") for k in keys}

    # --- lifecycle -------------------------------------------------------
    def emergency_land(self):
        log.warning("EMERGENCY LAND")
        for _ in range(3):
            if self.send("land", timeout=8.0) == "ok":
                return
            time.sleep(1)
        self.send("emergency", timeout=3.0)

    def close(self):
        self._stop.set()
        try:
            self.sock.close()
        except OSError:
            pass


class FakeTello(Tello):
    """Dry-run stand-in: acks everything, fakes telemetry, touches no network."""

    def __init__(self, *_, **__):
        self.state = {"bat": "83", "h": "0", "tof": "10", "yaw": "0"}
        self.ready = False
        self._h = 0
        self._log = logging.getLogger("tello.fake")

    def send(self, cmd: str, timeout=None, quiet=False) -> str:
        if cmd == "battery?":
            return self.state["bat"]
        if cmd == "takeoff":
            self._h = 80
        elif cmd == "land":
            self._h = 0
        elif cmd.startswith("up "):
            self._h += int(cmd.split()[1])
        elif cmd.startswith("down "):
            self._h -= int(cmd.split()[1])
        self.state["h"] = str(self._h)
        if not quiet:
            self._log.info("%-14s -> ok (dry-run)", cmd)
        time.sleep(0.05)
        return "ok"

    def flight(self, cmd: str, settle: float = 0.8) -> str:
        if not self.ready:
            raise TelloError("refusing '%s': SDK not acknowledged" % cmd)
        return self.send(cmd)

    def wait_ready(self, probes, interval) -> bool:
        self.ready = True
        self._log.info("[probe 1/%d] command -> ok (dry-run)", probes)
        return True

    def start_state_listener(self):
        pass

    def emergency_land(self):
        self._log.warning("EMERGENCY LAND (dry-run)")
        self._h = 0

    def close(self):
        pass
