"""Central configuration for AMR utilities.

Holds rate limits and retry/backoff settings used by PubMed E-utilities.
Keep unauthenticated values below NCBI's 3 requests/sec free allowance.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys


def _openrouter_key_from_keychain() -> str | None:
    """Read the OpenRouter API key from macOS Keychain when configured there.

    Generic-password items use the ``OpenRouter`` service by default; website
    password items use the ``openrouter.ai`` server by default. Set
    ``OPENROUTER_KEYCHAIN_SERVICE`` or
    ``OPENROUTER_KEYCHAIN_INTERNET_SERVER`` (and optionally
    ``OPENROUTER_KEYCHAIN_ACCOUNT``) to match a differently named item. The
    default account is ``OPENROUTER_API_KEY``. The secret is captured by the
    process and is never written to logs.
    """
    if sys.platform != "darwin" or shutil.which("security") is None:
        return None

    service = os.getenv("OPENROUTER_KEYCHAIN_SERVICE", "OpenRouter").strip()
    if not service:
        return None
    account = os.getenv("OPENROUTER_KEYCHAIN_ACCOUNT", "OPENROUTER_API_KEY").strip()
    lookup_specs = [
        ["security", "find-generic-password", "-s", service],
        [
            "security",
            "find-generic-password",
            "-s",
            os.getenv("OPENROUTER_KEYCHAIN_INTERNET_SERVER", "openrouter.ai").strip(),
        ],
        [
            "security",
            "find-internet-password",
            "-s",
            os.getenv("OPENROUTER_KEYCHAIN_INTERNET_SERVER", "openrouter.ai").strip(),
        ],
    ]
    for lookup in lookup_specs:
        commands = []
        if account:
            commands.append([*lookup, "-a", account, "-w"])
        commands.append([*lookup, "-w"])
        for command in commands:
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=30,
                )
            except (OSError, subprocess.TimeoutExpired):
                continue
            if result.returncode == 0:
                key = result.stdout.strip()
                if key:
                    return key
    return None


def _ncbi_key_from_keychain() -> str | None:
    """Read the NCBI E-utilities key from macOS Keychain if configured there."""
    if sys.platform != "darwin" or shutil.which("security") is None:
        return None
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-s", "NCBI", "-a", "NCBI_API_KEY", "-w"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    key = result.stdout.strip()
    return key if result.returncode == 0 and key else None

# PubMed E-utilities (Entrez) network settings
EUTILS_API_KEY: str | None = (
    os.getenv("NCBI_API_KEY") or os.getenv("EUTILS_API_KEY") or _ncbi_key_from_keychain()
)
OPENROUTER_API_KEY: str | None = (
    os.getenv("OPENROUTER_API_KEY") or _openrouter_key_from_keychain()
)
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
