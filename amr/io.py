"""Small I/O helpers shared by pipeline code."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd


def atomic_write_parquet(df: pd.DataFrame, path: str | Path, **kwargs: object) -> None:
    """Write a parquet file via a same-directory temporary file, then replace."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = target.suffix or ".parquet"
    with NamedTemporaryFile(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=suffix,
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        df.to_parquet(tmp_path, **kwargs)
        os.replace(tmp_path, target)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        finally:
            raise


__all__ = ["atomic_write_parquet"]
