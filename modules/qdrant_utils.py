"""
Codi Memory - Qdrant Utilities
Shared helpers for common Qdrant operations.
"""

import logging
from modules.config import qdrant, COLLECTION_NAME

_logger = logging.getLogger(__name__)


def scroll_all(
    scroll_filter=None,
    max_results: int = 500,
    collection: str = None,
    with_payload: bool = True,
    with_vectors: bool = False,
    batch_size: int = 100,
) -> list:
    """Paginated scroll over Qdrant collection.

    Replaces the error-prone pattern:
        points, _ = qdrant.scroll(..., limit=N)  # WRONG: ignores offset

    With a safe, paginated alternative:
        points = scroll_all(filter, max_results=N)

    Args:
        scroll_filter: Qdrant Filter object (optional)
        max_results: Safety cap on total results (default 500)
        collection: Collection name (default COLLECTION_NAME)
        with_payload: Include payload in results (default True)
        with_vectors: Include vectors in results (default False)
        batch_size: Points per scroll call (default 100)

    Returns:
        List of Qdrant point objects
    """
    coll = collection or COLLECTION_NAME
    results = []
    offset = None

    while len(results) < max_results:
        kwargs = {
            "collection_name": coll,
            "limit": min(batch_size, max_results - len(results)),
            "with_payload": with_payload,
            "with_vectors": with_vectors,
            "offset": offset,
        }
        if scroll_filter is not None:
            kwargs["scroll_filter"] = scroll_filter

        pts, next_offset = qdrant.scroll(**kwargs)

        if not pts:
            break

        results.extend(pts)

        if not next_offset:
            break
        offset = next_offset

    return results[:max_results]
