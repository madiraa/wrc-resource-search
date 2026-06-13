"""Match intake answers to resources via RAG."""

from typing import Any

from intake.outreach import build_search_query


def match_resources(rag, intake: dict[str, Any], top_k: int = 3) -> list[dict[str, Any]]:
    """Return ranked resource matches for an intake."""
    query = build_search_query(intake)
    if not query:
        return []

    results = rag.search(query, top_k=top_k, current_only=True)
    for result in results:
        org_name = result.get("organization_name") or ""
        result["is_ccsf"] = (
            result.get("binder_name") == "CCSF_Website" or "(CCSF)" in org_name
        )
    return results
