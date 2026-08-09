import json
from datetime import datetime, timezone
from pathlib import Path

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"


def save_result(name, data):
    OUTPUTS_DIR.mkdir(exist_ok=True)
    payload = {"technique": name, "saved_at": datetime.now(timezone.utc).isoformat(), **data}
    path = OUTPUTS_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
