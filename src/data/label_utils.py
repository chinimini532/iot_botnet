"""
Label utilities for Bot-IoT and N-BaIoT.

Bot-IoT already has label columns (attack, category, subcategory) baked
into every row. N-BaIoT has no label column at all -- the label is
encoded in the filename, so we parse it from the path.
"""

from pathlib import Path


def parse_nbaiot_filename(path: Path) -> dict:
    """
    N-BaIoT filenames look like:
        1.benign.csv
        1.mirai.ack.csv
        1.gafgyt.combo.csv

    Returns a dict with device_id, family, subtype, and a combined label.
    Benign files have no subtype.
    """
    stem = path.stem  # e.g. "1.mirai.ack" or "1.benign"
    parts = stem.split(".")

    device_id = int(parts[0])
    family = parts[1]  # "benign", "mirai", or "gafgyt"
    subtype = parts[2] if len(parts) > 2 else None

    is_benign = family == "benign"
    label = "benign" if is_benign else f"{family}.{subtype}"

    return {
        "device_id": device_id,
        "family": family,
        "subtype": subtype,
        "label": label,
        "is_attack": not is_benign,
    }


# Reference: N-BaIoT device_info.csv mapping (device_id -> device name)
NBAIOT_DEVICE_NAMES = {
    1: "Danmini_Doorbell",
    2: "Ecobee_Thermostat",
    3: "Ennio_Doorbell",
    4: "Philips_B120N10_Baby_Monitor",
    5: "Provision_PT_737E_Security_Camera",
    6: "Provision_PT_838_Security_Camera",
    7: "Samsung_SNH_1011_N_Webcam",
    8: "SimpleHome_XCS7_1002_WHT_Security_Camera",
    9: "SimpleHome_XCS7_1003_WHT_Security_Camera",
}

# Devices 3 and 7 have no Mirai files in the original dataset --
# not missing data, just how N-BaIoT was originally collected.
NBAIOT_DEVICES_WITHOUT_MIRAI = {3, 7}