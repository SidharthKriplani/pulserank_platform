from __future__ import annotations

import json
from pathlib import Path


REQUIRED_PATHS = [
    "README.md",
    "requirements.txt",
    "pyproject.toml",
    "configs/policy_config.json",
    "docs/PULSERANK_SCOPE.md",
    "docs/PULSERANK_BUILD_GROUPS.md",
    "src/pulserank/__init__.py",
    "src/pulserank/core/constants.py",
    "src/pulserank/simulation/__init__.py",
    "src/pulserank/ranking/__init__.py",
    "src/pulserank/evaluation/__init__.py",
    "src/pulserank/artifacts/__init__.py",
    "outputs/evidence",
    "outputs/reports",
    "outputs/validation",
    "outputs/dashboard"
]


def main() -> None:
    checks = []

    for path in REQUIRED_PATHS:
        checks.append({
            "name": f"path_exists::{path}",
            "passed": Path(path).exists(),
            "observed": path
        })

    config = json.loads(Path("configs/policy_config.json").read_text(encoding="utf-8"))

    config_checks = [
        ("display_rank_required", config["position_bias"]["display_rank_required"] is True),
        ("propensity_floor", config["position_bias"]["propensity_floor"] == 0.01),
        ("seller_gini_threshold", config["reranking"]["seller_gini_threshold"] == 0.65),
        ("attribution_windows", config["attribution"]["sensitivity_windows_days"] == [1, 3, 7]),
        ("temporal_holdout_required", config["promotion_gates"]["temporal_holdout_required"] is True)
    ]

    for name, passed in config_checks:
        checks.append({
            "name": f"config::{name}",
            "passed": passed,
            "observed": name
        })

    readme = Path("README.md").read_text(encoding="utf-8")
    required_readme_phrases = [
        "Production-simulated marketplace recommendation",
        "IPS position-bias correction",
        "delayed conversion attribution",
        "seller/category exposure governance",
        "Claim Boundary",
        "real online A/B test"
    ]

    for phrase in required_readme_phrases:
        checks.append({
            "name": f"readme_contains::{phrase}",
            "passed": phrase in readme,
            "observed": phrase
        })

    status = "pass" if all(c["passed"] for c in checks) else "fail"

    payload = {
        "artifact": "pulserank_scaffold_validation_v1",
        "status": status,
        "check_count": len(checks),
        "passed_count": sum(1 for c in checks if c["passed"]),
        "checks": checks,
        "evidence_statement": "Validates PulseRank Group 0 repo foundation, config, documentation, package scaffold, and truth boundary."
    }

    out = Path("outputs/validation/scaffold_validation_v1.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("pulserank_scaffold_validation_v1 complete")
    print(f"status: {status}")
    print(f"passed_count: {payload['passed_count']}/{payload['check_count']}")
    print(f"wrote {out}")

    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
