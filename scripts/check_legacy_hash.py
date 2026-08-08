import hashlib
import json
from pathlib import Path

HASH_FILE = Path(__file__).resolve().parents[1] / "docs" / "legacy_code_hashes.json"


def main():
    items = json.loads(HASH_FILE.read_text(encoding="utf-8"))
    changed = []
    for item in items:
        path = Path(item["path"])
        if not path.exists():
            changed.append((str(path), "missing"))
            continue
        current = hashlib.sha256(path.read_bytes()).hexdigest()
        if current != item["sha256"]:
            changed.append((str(path), "hash_changed"))
    if changed:
        print("legacy hash check failed:")
        for path, reason in changed:
            print(f"- {reason}: {path}")
        raise SystemExit(1)
    print(f"legacy hash check passed: {len(items)} files unchanged")


if __name__ == "__main__":
    main()
