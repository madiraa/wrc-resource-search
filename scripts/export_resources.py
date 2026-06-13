#!/usr/bin/env python3
"""
Export WRC resources from SQLite to JSON for the static GitHub Pages site.

Run from the repository root:
    python scripts/export_resources.py
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = REPO_ROOT / "docs" / "data" / "resources.json"
DB_PATH = REPO_ROOT / "wrc_resources.db"

CCSF_REPORTING_RESOURCES = [
    "Title IX Office",
    "CCSF District Police",
]

CRITICAL_CRISIS_RESOURCES = [
    "San Francisco Women Against Rape (SFWAR)",
]


def build_search_text(resource: dict) -> str:
    """Combine fields used by client-side search."""
    parts = [
        resource.get("organization_name") or "",
        resource.get("resource_type") or "",
        resource.get("description") or "",
        resource.get("services") or "",
        resource.get("eligibility") or "",
        resource.get("address") or "",
        resource.get("categories") or "",
        (resource.get("ocr_text") or "")[:500],
    ]
    return " ".join(part for part in parts if part).strip()


def priority_tier(resource: dict) -> int:
    org_name = resource.get("organization_name") or ""
    is_ccsf = resource.get("is_ccsf", False)

    if is_ccsf and any(name in org_name for name in CCSF_REPORTING_RESOURCES):
        return 0
    if org_name in CRITICAL_CRISIS_RESOURCES:
        return 1
    if is_ccsf:
        return 2
    return 3


def export_resources(db_path: Path = DB_PATH, output_path: Path = OUTPUT_PATH) -> dict:
    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found at {db_path}. Run the data pipeline first."
        )

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            r.resource_id,
            r.organization_name,
            r.resource_type,
            r.description,
            r.address,
            r.phone,
            r.email,
            r.website,
            r.hours,
            r.eligibility,
            r.services,
            r.is_current,
            r.last_verified_date,
            i.binder_name,
            i.ocr_text,
            GROUP_CONCAT(rc.category, ', ') AS categories
        FROM resources r
        JOIN images i ON r.image_id = i.image_id
        LEFT JOIN resource_categories rc ON r.resource_id = rc.resource_id
        GROUP BY r.resource_id
        ORDER BY r.organization_name COLLATE NOCASE
        """
    )

    resources = []
    for row in cursor.fetchall():
        resource = dict(row)
        org_name = resource.get("organization_name") or ""
        resource["is_ccsf"] = (
            resource.get("binder_name") == "CCSF_Website" or "(CCSF)" in org_name
        )
        resource["is_crisis_resource"] = org_name in CRITICAL_CRISIS_RESOURCES
        resource["priority_tier"] = priority_tier(resource)
        resource["search_text"] = build_search_text(resource)
        resources.append(resource)

    conn.close()

    current_count = sum(1 for resource in resources if resource.get("is_current"))
    categories = sorted(
        {
            resource["resource_type"]
            for resource in resources
            if resource.get("resource_type")
        }
    )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stats": {
            "total_resources": len(resources),
            "current_resources": current_count,
            "categories": len(categories),
        },
        "categories": categories,
        "resources": resources,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    return payload


if __name__ == "__main__":
    payload = export_resources()
    print(f"Exported {payload['stats']['total_resources']} resources to {OUTPUT_PATH}")
