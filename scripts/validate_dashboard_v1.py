from __future__ import annotations

import json
from pathlib import Path


OUT = Path("outputs/validation")


def add(checks: list[dict], name: str, passed: bool, observed) -> None:
    checks.append({"name": name, "passed": bool(passed), "observed": observed})


def main() -> None:
    checks = []

    dashboard = Path("outputs/dashboard/index.html")
    docs_index = Path("docs/index.html")
    manifest_path = Path("outputs/dashboard/dashboard_manifest.json")

    for path in [dashboard, docs_index, manifest_path]:
        add(checks, f"exists::{path}", path.exists(), str(path))

    html = dashboard.read_text(encoding="utf-8") if dashboard.exists() else ""
    required_phrases = [
        "PulseRank Evidence Dashboard",
        "IPS position-bias correction",
        "display_rank",
        "Offline A/B Simulation",
        "Resume-Safe Claim",
        "Truth boundary",
        "MetaSignal-compatible",
        "Failure/Recovery Defense",
    ]

    for phrase in required_phrases:
        add(checks, f"html_contains::{phrase}", phrase in html, phrase)

    docs_html = docs_index.read_text(encoding="utf-8") if docs_index.exists() else ""
    add(checks, "docs_index_same_core_title", "PulseRank Evidence Dashboard" in docs_html, "docs/index.html")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    add(checks, "manifest_status_pass", manifest.get("status") == "pass", manifest.get("status"))
    add(checks, "manifest_total_artifacts_positive", int(manifest.get("total_json_artifacts", 0)) >= 30, manifest.get("total_json_artifacts"))
    add(checks, "manifest_has_github_pages_path", manifest.get("github_pages_path") == "docs/index.html", manifest.get("github_pages_path"))

    status = "pass" if all(c["passed"] for c in checks) else "fail"

    payload = {
        "artifact": "pulserank_dashboard_validation_v1",
        "status": status,
        "check_count": len(checks),
        "passed_count": sum(1 for c in checks if c["passed"]),
        "checks": checks,
        "evidence_statement": "Validates PulseRank static dashboard and GitHub Pages entrypoint."
    }

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "dashboard_validation_v1.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("pulserank_dashboard_validation_v1 complete")
    print(f"status: {status}")
    print(f"passed_count: {payload['passed_count']}/{payload['check_count']}")
    print(f"wrote {out}")

    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
