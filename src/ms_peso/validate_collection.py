from __future__ import annotations

import argparse
import json
from pathlib import Path

from ms_peso.artifacts import save_json
from ms_peso.collection import (
    audit_pilot_collection,
    load_collection_policy,
    read_authorization_registry,
)
from ms_peso.image_quality import load_image_quality_policy
from ms_peso.manifest import read_manifest


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audita uma coleta piloto antes de permitir seu uso."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--authorizations", required=True, type=Path)
    parser.add_argument(
        "--policy", default="configs/pilot_collection.yaml", type=Path
    )
    parser.add_argument(
        "--quality-policy",
        default="configs/image_quality.yaml",
        type=Path,
    )
    parser.add_argument("--image-root", default="data/raw/pilot", type=Path)
    parser.add_argument("--skip-image-checks", action="store_true")
    parser.add_argument(
        "--output",
        default="artifacts/collection_audit/report.json",
        type=Path,
    )
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    try:
        rows = read_manifest(args.manifest)
        authorizations = read_authorization_registry(args.authorizations)
        policy = load_collection_policy(args.policy)
        quality_policy = (
            None
            if args.skip_image_checks
            else load_image_quality_policy(args.quality_policy)
        )
        report = audit_pilot_collection(
            rows,
            authorizations,
            policy,
            manifest_path=args.manifest,
            image_root=args.image_root,
            check_images=not args.skip_image_checks,
            quality_policy=quality_policy,
        )
        payload = report.to_dict()
    except (FileNotFoundError, ValueError) as exc:
        payload = {
            "status": "rejected",
            "policy": None,
            "summary": None,
            "errors": [str(exc)],
            "warnings": [],
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if payload["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
