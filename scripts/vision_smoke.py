"""Vision smoke evaluation on tracked samples/ — not a validation report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vision.inference import SCREENING_THRESHOLD, TBScreenModel  # noqa: E402


def main() -> int:
    samples = sorted((ROOT / "samples").glob("*.png"))
    if not samples:
        print("No samples found")
        return 1

    model = TBScreenModel()
    rows = []
    tp = fp = tn = fn = 0
    for path in samples:
        label = "TB" if path.name.startswith("tb") else "NORMAL"
        pred = model.predict(str(path), with_zones=False)
        positive = pred["tb_probability"] >= SCREENING_THRESHOLD
        if label == "TB" and positive:
            tp += 1
        elif label == "TB" and not positive:
            fn += 1
        elif label == "NORMAL" and positive:
            fp += 1
        else:
            tn += 1
        rows.append(
            {
                "file": path.name,
                "label": label,
                "tb_probability": pred["tb_probability"],
                "predicted_positive": positive,
            }
        )

    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    report = {
        "dataset": "samples/ Shenzhen smoke set",
        "n": len(rows),
        "threshold": SCREENING_THRESHOLD,
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "note": "Smoke test only — not held-out clinical validation. Do not report as AUC.",
        "rows": rows,
    }
    out = ROOT / "data" / "eval" / "vision_smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k != "rows"}, indent=2))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
