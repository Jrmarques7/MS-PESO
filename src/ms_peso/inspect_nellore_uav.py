from __future__ import annotations

import argparse
import json
from pathlib import Path

from ms_peso.importers.nellore_uav import inspect_nellore_uav_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audita uma cópia extraída do NelloreBeefCattleDataset."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", help="Caminho opcional para o inventário JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inventory = inspect_nellore_uav_dataset(args.dataset_root)
    report = json.dumps(inventory.to_dict(), indent=2, ensure_ascii=False)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report + "\n", encoding="utf-8")
        print(f"Inventário gravado em {output_path}")
    else:
        print(report)

    print(f"Imagens UAV brutas: {inventory.raw_images}")
    print(f"Sessões: {len(inventory.sessions)}")
    print(f"Imagens de detecção anotadas: {inventory.detection_images}")
    print(f"Imagens de cocho anotadas: {inventory.feed_bunk_images}")
    print(f"Apto para regressão de peso: {inventory.regression_ready}")


if __name__ == "__main__":
    main()
