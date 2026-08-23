from __future__ import annotations

import argparse
import json
from pathlib import Path

from ms_peso.artifacts import save_json
from ms_peso.collection import read_authorization_registry
from ms_peso.uav_capture import (
    audit_uav_capture,
    load_uav_capture_policy,
    read_uav_csv,
)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Valida o contrato de uma coleta UAV sem treinar modelo."
    )
    parser.add_argument("--flights", required=True, type=Path)
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--authorizations", required=True, type=Path)
    parser.add_argument("--policy", default="configs/uav_capture.yaml", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--skip-file-checks", action="store_true")
    parser.add_argument(
        "--output", default="artifacts/uav_capture_audit/report.json", type=Path
    )
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    try:
        policy = load_uav_capture_policy(args.policy)
        report = audit_uav_capture(
            read_uav_csv(args.flights),
            read_uav_csv(args.observations),
            read_authorization_registry(args.authorizations),
            policy,
            flights_manifest_path=args.flights,
            observations_manifest_path=args.observations,
            data_root=args.data_root,
            check_files=not args.skip_file_checks,
        )
        payload = report.to_dict()
    except (FileNotFoundError, ValueError) as exc:
        payload = {
            "status": "rejected",
            "policy": None,
            "summary": None,
            "model_training_ready": False,
            "errors": [str(exc)],
            "warnings": [],
            "limitations": [],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if payload["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
