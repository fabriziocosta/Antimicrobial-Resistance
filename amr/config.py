"""Central configuration for AMR utilities.

Holds rate limits and retry/backoff settings used by PubMed E-utilities.
Keep unauthenticated values below NCBI's 3 requests/sec free allowance.
"""

from __future__ import annotations

import os

# PubMed E-utilities (Entrez) network settings
EUTILS_API_KEY: str | None = os.getenv("NCBI_API_KEY") or os.getenv("EUTILS_API_KEY")
OPENROUTER_API_KEY: str | None = os.getenv("OPENROUTER_API_KEY")
EUTILS_RETRIES: int = 3
EUTILS_BACKOFF_FACTOR: int = 5
EUTILS_429_BACKOFF_SECONDS: float = float(os.getenv("EUTILS_429_BACKOFF_SECONDS", "30"))
EUTILS_MAX_BACKOFF_SECONDS: float = float(os.getenv("EUTILS_MAX_BACKOFF_SECONDS", "300"))
EUTILS_RATE_LIMIT_DELAY: float = float(
    os.getenv("EUTILS_RATE_LIMIT_DELAY", "0.11" if EUTILS_API_KEY else "0.5")
)
EUTILS_TIMEOUT_SECONDS: float = float(os.getenv("EUTILS_TIMEOUT_SECONDS", "45"))

# PubMed efetch endpoint
PUBMED_EFETCH_URL: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

__all__ = [
    "EUTILS_API_KEY",
    "OPENROUTER_API_KEY",
    "EUTILS_RETRIES",
    "EUTILS_BACKOFF_FACTOR",
    "EUTILS_429_BACKOFF_SECONDS",
    "EUTILS_MAX_BACKOFF_SECONDS",
    "EUTILS_RATE_LIMIT_DELAY",
    "EUTILS_TIMEOUT_SECONDS",
    "PUBMED_EFETCH_URL",
]
