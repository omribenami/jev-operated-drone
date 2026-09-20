"""Power the Tello's Tasmota plug via Node-RED inject, falling back to the HA REST API.

Must run BEFORE the host switches to the Tello Wi-Fi (LAN is unreachable after).
"""
import logging
import os
import time
from typing import Dict

import requests

log = logging.getLogger("power")


def _inject(url: str) -> bool:
    try:
        r = requests.post(url, timeout=5)
        log.info("node-red inject %s -> %s", url, r.status_code)
        return r.ok
    except requests.RequestException as exc:
        log.warning("node-red inject failed: %s", exc)
        return False


def _ha_switch(cfg: Dict, service: str) -> bool:
    token = os.environ.get("HA_TOKEN")
    if not token:
        log.warning("HA_TOKEN not set; cannot use HA fallback")
        return False
    headers = {"Authorization": "Bearer " + token}
    base = cfg["ha_url"].rstrip("/")
    try:
        # a wrong/misspelled entity_id otherwise silently "succeeds": the service-call
        # endpoint returns 200 with an empty change list either way, and for a
        # cloud/Bluetooth-bridged actuator (e.g. a button-pressing bot) an empty change
        # list is also normal even when the entity is real, since the physical action
        # is asynchronous. So only check existence, not the immediate response body.
        chk = requests.get("%s/api/states/%s" % (base, cfg["ha_entity"]), headers=headers, timeout=8)
        if chk.status_code == 404:
            log.warning("HA entity %r does not exist", cfg["ha_entity"])
            return False
        r = requests.post(
            "%s/api/services/switch/%s" % (base, service),
            headers=headers,
            json={"entity_id": cfg["ha_entity"]},
            timeout=8,
        )
        log.info("HA switch.%s %s -> %s", service, cfg["ha_entity"], r.status_code)
        return r.ok
    except requests.RequestException as exc:
        log.warning("HA request failed: %s", exc)
        return False


def power_on(cfg: Dict, wait: bool = True) -> bool:
    # HA's switch.tasmota is the confirmed-working path; the node-red inject can return
    # HTTP 200 without actually toggling the plug, so it's kept only as a fallback.
    ok = _ha_switch(cfg, "turn_on") or _inject(cfg["on_url"])
    if not ok:
        return False
    if wait:
        log.info("waiting %ss for Tello to boot and broadcast Wi-Fi", cfg["boot_wait_s"])
        time.sleep(cfg["boot_wait_s"])
    return True


def power_off(cfg: Dict) -> bool:
    return _ha_switch(cfg, "turn_off") or _inject(cfg["off_url"])


def power_cycle(cfg: Dict) -> bool:
    log.info("power-cycling drone (SDK assumed wedged)")
    power_off(cfg)
    time.sleep(8)
    return power_on(cfg, wait=True)
