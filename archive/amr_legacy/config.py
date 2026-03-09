"""Central configuration for AMR utilities.

Holds rate limits and retry/backoff settings used by PubMed E-utilities.
Keep values conservative to respect NCBI policy (~3 requests/sec).
"""

from __future__ import annotations

# PubMed E-utilities (Entrez) network settings
EUTILS_RETRIES: int = 3
EUTILS_BACKOFF_FACTOR: int = 5
EUTILS_RATE_LIMIT_DELAY: float = 0.34  # ~3 req/sec

# PubMed efetch endpoint
PUBMED_EFETCH_URL: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

__all__ = [
    "EUTILS_RETRIES",
    "EUTILS_BACKOFF_FACTOR",
    "EUTILS_RATE_LIMIT_DELAY",
    "PUBMED_EFETCH_URL",
]

