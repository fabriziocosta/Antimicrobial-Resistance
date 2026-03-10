"""Central configuration for AMR utilities.

Holds rate limits and retry/backoff settings used by PubMed E-utilities.
Keep values conservative to respect NCBI policy (~3 requests/sec).
"""

from __future__ import annotations

import os

# PubMed E-utilities (Entrez) network settings
EUTILS_API_KEY: str | None = os.getenv("NCBI_API_KEY") or os.getenv("EUTILS_API_KEY")
EUTILS_RETRIES: int = 3
EUTILS_BACKOFF_FACTOR: int = 5
EUTILS_RATE_LIMIT_DELAY: float = 0.11 if EUTILS_API_KEY else 0.34

# PubMed efetch endpoint
PUBMED_EFETCH_URL: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

__all__ = [
    "EUTILS_API_KEY",
    "EUTILS_RETRIES",
    "EUTILS_BACKOFF_FACTOR",
    "EUTILS_RATE_LIMIT_DELAY",
    "PUBMED_EFETCH_URL",
]
