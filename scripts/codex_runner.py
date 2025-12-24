# scripts/codex_runner.py
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None


def load_codex_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        print(f"[Codex] Missing codex.yaml at {config_path}", file=sys.stderr)
        sys.exit(1)
    with config_path.open("r") as f:
        content = f.read()

    if yaml is not None:
        return yaml.safe_load(content)

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        print(
            "[Codex] PyYAML is not installed and codex.yaml is not valid JSON. "
            "Install PyYAML or provide codex.yaml in JSON format.",
            file=sys.stderr,
        )
        sys.exit(1)


def collect_tiers_for_combo(cfg: Dict[str, Any], combo: str) -> List[str]:
    combos = cfg.get("combinations", {})
    if combo not in combos:
        print(f"[Codex] Unknown combination: {combo}", file=sys.stderr)
        print(f"Available: {', '.join(combos.keys())}", file=sys.stderr)
        sys.exit(1)
    return combos[combo]["tiers"]


def expand_dependencies(cfg: Dict[str, Any], tiers: List[str]) -> List[str]:
    """
    Ensure tiers are ordered with dependencies satisfied and duplicates removed.
    """
    test_tiers = cfg.get("test_tiers", {})
    ordered: List[str] = []

    def add_tier(tier: str):
        if tier in ordered:
            return
        meta = test_tiers.get(tier)
        if not meta:
            print(f"[Codex] Tier '{tier}' not defined in test_tiers.", file=sys.stderr)
            sys.exit(1)
        for dep in meta.get("depends_on", []):
            add_tier(dep)
        ordered.append(tier)

    for t in tiers:
        add_tier(t)

    return ordered


def build_pytest_cmd(tier_name: str, tier_cfg: Dict[str, Any]) -> List[str]:
    discovery = tier_cfg["discovery"]
    paths = discovery.get("paths", ["tests"])
    markers_any = discovery.get("markers_any", [])

    cmd = [sys.executable, "-m", "pytest"] + paths

    if markers_any:
        # e.g. -m "integration or scenario"
        marker_expr = " or ".join(markers_any)
        cmd.extend(["-m", marker_expr])

    return cmd


def run_tier(tier_name: str, tier_cfg: Dict[str, Any]) -> bool:
    print(f"\n[Codex] === Running tier: {tier_name} ===")
    cmd = build_pytest_cmd(tier_name, tier_cfg)
    print(f"[Codex] Command: {' '.join(cmd)}")
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [str(Path("src").resolve())] + ([env["PYTHONPATH"]] if "PYTHONPATH" in env else [])
        )

        result = subprocess.run(cmd, check=False, env=env)
        success = (result.returncode == 0)
        status = "OK" if success else "FAIL"
        print(f"[Codex] Tier {tier_name}: {status}")
        return success
    except KeyboardInterrupt:
        print("[Codex] Interrupted by user.")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Codex test orchestrator.")
    parser.add_argument(
        "--combo",
        default="standard",
        help="Test combination name (e.g. smoke, standard, full_day, nightly).",
    )
    parser.add_argument(
        "--config",
        default="codex.yaml",
        help="Path to codex.yaml (default: codex.yaml).",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = load_codex_config(config_path)

    # Determine which tiers to run
    requested_tiers = collect_tiers_for_combo(cfg, args.combo)
    tiers_to_run = expand_dependencies(cfg, requested_tiers)

    test_tiers = cfg.get("test_tiers", {})

    print(f"[Codex] Combo '{args.combo}' → tiers: {tiers_to_run}")

    overall_ok = True
    for tier_name in tiers_to_run:
        tier_cfg = test_tiers[tier_name]

        # Allow disabling certain tiers by default (e.g. stress)
        if not tier_cfg.get("default_enabled", True):
            print(f"[Codex] Skipping tier '{tier_name}' (default_enabled = false).")
            continue

        ok = run_tier(tier_name, tier_cfg)
        if not ok:
            overall_ok = False
            # If tier is critical, stop immediately
            if tier_cfg.get("critical", False):
                print(f"[Codex] Critical tier '{tier_name}' failed. Stopping.")
                break

    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
