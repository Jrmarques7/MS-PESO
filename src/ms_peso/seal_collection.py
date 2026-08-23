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
from ms_peso.collection_snapshot import build_collection_snapshot
from ms_peso.image_quality import load_image_quality_policy
from ms_peso.integrity import calculate_sha256
from ms_peso.manifest import read_manifest, write_manifest


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audita, deduplica e sela uma coleta aprovada."
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
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument(
        "--output-report",
        default="artifacts/collection_snapshot/report.json",
        type=Path,
    )
    return parser.parse_args(args)


def _rejection_payload(stage: str, error: str) -> dict[str, object]:
    return {
        "status": "rejected",
        "stage": stage,
        "snapshot_id": None,
        "errors": [error],
        "warnings": [],
    }


def main() -> None:
    args = parse_args()
    payload: dict[str, object]
    try:
        if args.output_manifest.exists():
            raise FileExistsError(
                "O manifesto selado já existe; escolha outro --output-manifest."
            )
        rows = read_manifest(args.manifest)
        authorizations = read_authorization_registry(args.authorizations)
        policy = load_collection_policy(args.policy)
        quality_policy = load_image_quality_policy(args.quality_policy)
        audit = audit_pilot_collection(
            rows,
            authorizations,
            policy,
            manifest_path=args.manifest,
            image_root=args.image_root,
            check_images=True,
            quality_policy=quality_policy,
        )
        if not audit.valid:
            payload = {
                "status": "rejected",
                "stage": "collection_audit",
                "snapshot_id": None,
                "audit": audit.to_dict(),
                "errors": list(audit.errors),
                "warnings": list(audit.warnings),
            }
        else:
            snapshot = build_collection_snapshot(
                rows,
                manifest_path=args.manifest,
                image_root=args.image_root,
                near_duplicate_hamming_distance=(
                    policy.near_duplicate_hamming_distance
                ),
            )
            snapshot_report = snapshot.to_report()
            if not snapshot.valid:
                payload = {
                    "status": "rejected",
                    "stage": "deduplication",
                    "snapshot_id": snapshot.snapshot_id,
                    "audit": audit.to_dict(),
                    "snapshot": snapshot_report,
                    "errors": snapshot_report["errors"],
                    "warnings": snapshot_report["warnings"],
                }
            else:
                write_manifest(snapshot.rows, args.output_manifest)
                payload = {
                    "status": "passed",
                    "stage": "sealed",
                    "snapshot_id": snapshot.snapshot_id,
                    "audit": audit.to_dict(),
                    "snapshot": snapshot_report,
                    "provenance": {
                        "source_manifest_sha256": calculate_sha256(args.manifest),
                        "authorization_registry_sha256": calculate_sha256(
                            args.authorizations
                        ),
                        "collection_policy_sha256": calculate_sha256(args.policy),
                        "quality_policy_sha256": calculate_sha256(
                            args.quality_policy
                        ),
                        "sealed_manifest_path": str(args.output_manifest.resolve()),
                        "sealed_manifest_sha256": calculate_sha256(
                            args.output_manifest
                        ),
                    },
                    "errors": [],
                    "warnings": [
                        *audit.warnings,
                        *snapshot_report["warnings"],
                    ],
                }
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        payload = _rejection_payload("setup", str(exc))

    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output_report, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if payload["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
