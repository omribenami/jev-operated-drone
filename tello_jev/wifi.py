"""nmcli helpers: find the TELLO-* AP, connect, and restore the previous network."""
import logging
import subprocess
import time
from typing import Dict, List, Optional

log = logging.getLogger("wifi")


def _run(args: List[str], timeout: int = 30) -> str:
    try:
        p = subprocess.run(args, text=True, capture_output=True, timeout=timeout)
        return (p.stdout + p.stderr).strip()
    except subprocess.TimeoutExpired as exc:
        log.warning("%s timed out after %ss", " ".join(args), timeout)
        return ((exc.stdout or "") + (exc.stderr or "")).strip()


def current_connection(iface: str) -> Optional[str]:
    out = _run(["nmcli", "-t", "-f", "DEVICE,CONNECTION", "dev", "status"])
    for line in out.splitlines():
        parts = line.split(":")
        if parts[0] == iface and len(parts) > 1 and parts[1] not in ("", "--"):
            return parts[1]
    return None


def scan_tello(iface: str, prefix: str, attempts: int = 4) -> List[str]:
    for i in range(attempts):
        _run(["nmcli", "dev", "wifi", "rescan", "ifname", iface])
        time.sleep(3)
        out = _run(["nmcli", "-t", "-f", "SSID", "dev", "wifi", "list", "ifname", iface])
        found = sorted({s for s in out.splitlines() if s.startswith(prefix)})
        log.info("scan %d/%d: %s", i + 1, attempts, found or "no TELLO SSID")
        if found:
            return found
    return []


def connect_tello(cfg_drone: Dict, cfg_wifi: Dict) -> Optional[str]:
    iface = cfg_wifi["iface"]
    found = scan_tello(iface, cfg_drone["ssid_prefix"])
    if not found:
        return None
    ordered = [s for s in cfg_drone["ssid_preferred"] if s in found] + [s for s in found if s not in cfg_drone["ssid_preferred"]]
    for ssid in ordered:
        # a stuck/still-activating attempt from a previous run keeps holding a DHCP lease
        # from the Tello's tiny pool; clear it before asking NetworkManager for a new one.
        _run(["nmcli", "device", "disconnect", iface])
        out = _run(["nmcli", "dev", "wifi", "connect", ssid, "ifname", iface], timeout=45)
        log.info("connect %s: %s", ssid, out.splitlines()[0] if out else "")
        if "successfully activated" in out.lower():
            # NetworkManager's own autoconnect would otherwise race our explicit connect
            # attempts (and each other) for the Tello's tiny DHCP pool on every future run.
            _run(["nmcli", "con", "modify", ssid, "connection.autoconnect", "no"])
            time.sleep(2)
            return ssid
        # our client gave up, but NetworkManager keeps trying to activate in the background
        # (still holding/blocking a DHCP lease) unless told to stop.
        _run(["nmcli", "device", "disconnect", iface])
    return None


def restore(iface: str, previous: Optional[str]):
    if not previous:
        return
    out = _run(["nmcli", "con", "up", previous, "ifname", iface], timeout=45)
    log.info("restore %s: %s", previous, out.splitlines()[0] if out else "")
