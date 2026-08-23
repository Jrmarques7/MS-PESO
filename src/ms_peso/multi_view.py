from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable


def pair_views(
    rows: Iterable[dict[str, str]],
    *,
    primary_view: str,
    secondary_view: str,
) -> list[dict[str, str]]:
    """Combina duas vistas do mesmo animal/evento em uma única amostra."""
    if not primary_view or not secondary_view:
        raise ValueError("As vistas primária e secundária são obrigatórias.")
    if primary_view == secondary_view:
        raise ValueError("As vistas primária e secundária devem ser diferentes.")

    grouped_rows: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        view = row.get("view", "")
        if view in {primary_view, secondary_view}:
            grouped_rows[(row["animal_id"], row["event_id"])].append(row)

    if not grouped_rows:
        raise ValueError("Nenhum par das vistas solicitadas foi encontrado.")

    paired_rows: list[dict[str, str]] = []
    errors: list[str] = []
    for (animal_id, event_id), event_rows in sorted(grouped_rows.items()):
        rows_by_view = {
            view: [row for row in event_rows if row.get("view") == view]
            for view in (primary_view, secondary_view)
        }
        invalid_counts = {
            view: len(view_rows)
            for view, view_rows in rows_by_view.items()
            if len(view_rows) != 1
        }
        if invalid_counts:
            details = ", ".join(
                f"{view}={count}" for view, count in invalid_counts.items()
            )
            errors.append(
                f"{animal_id}/{event_id}: número de vistas inválido ({details})"
            )
            continue

        primary_row = rows_by_view[primary_view][0]
        secondary_row = rows_by_view[secondary_view][0]
        for column in ("animal_id", "event_id", "weight_kg", "split"):
            primary_value = primary_row.get(column, "")
            secondary_value = secondary_row.get(column, "")
            if primary_value != secondary_value:
                errors.append(
                    f"{animal_id}/{event_id}: {column} conflitante entre vistas"
                )

        paired_row = dict(primary_row)
        paired_row["secondary_image_path"] = secondary_row["image_path"]
        paired_row["secondary_view"] = secondary_view
        paired_rows.append(paired_row)

    if errors:
        preview = "\n".join(f"- {error}" for error in errors[:20])
        suffix = "\n- ..." if len(errors) > 20 else ""
        raise ValueError(f"Não foi possível parear as vistas:\n{preview}{suffix}")
    return paired_rows
