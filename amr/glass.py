"""GLASS pathogen-antibiotic list cleaning utilities."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable

import pandas as pd


def _norm_key(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.strip().lower()
    text = text.replace("β", "beta")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _display_text(value: object) -> str:
    return "" if value is None else str(value).strip()


def _build_lookup(df: pd.DataFrame, source_col: str, target_col: str) -> dict[str, str]:
    if source_col not in df.columns or target_col not in df.columns:
        return {}
    if source_col == target_col:
        values = df[source_col].dropna().map(_display_text)
        work = pd.DataFrame({"source": values.map(_norm_key), "target": values})
    else:
        work = df[[source_col, target_col]].dropna().copy()
        work = work.rename(columns={source_col: "source", target_col: "target"})
        work["source"] = work["source"].map(_norm_key)
        work["target"] = work["target"].map(_display_text)
    work = work.loc[work["source"].ne("") & work["target"].ne("")]
    return dict(work.drop_duplicates("source").itertuples(index=False, name=None))


def _expand_pathogen(raw_pathogen: str, pathogen_lookup: dict[str, str], pathogen_universe: Iterable[str] | None) -> list[str]:
    raw = _display_text(raw_pathogen)
    key = _norm_key(raw)
    universe_lookup = (
        {_norm_key(pathogen): _display_text(pathogen) for pathogen in pathogen_universe if _display_text(pathogen)}
        if pathogen_universe is not None
        else {}
    )
    if key in universe_lookup:
        return [universe_lookup[key]]
    if key in pathogen_lookup:
        mapped = pathogen_lookup[key]
        return [universe_lookup.get(_norm_key(mapped), mapped)]

    abbreviation_map = {
        "e coli": "Escherichia coli",
        "k pneumoniae": "Klebsiella pneumoniae",
        "s aureus": "Staphylococcus aureus",
        "s pneumoniae": "Streptococcus pneumoniae",
        "n gonorrhoeae": "Neisseria gonorrhoeae",
    }
    if key in abbreviation_map:
        return [abbreviation_map[key]]

    if key.endswith(" spp") and pathogen_universe is not None:
        genus = key.removesuffix(" spp").strip()
        expanded = [
            pathogen
            for pathogen in sorted({_display_text(p) for p in pathogen_universe if _display_text(p)})
            if _norm_key(pathogen).split(" ")[0:1] == [genus]
        ]
        if expanded:
            return expanded

    return [raw]


def load_and_clean_glass_list(
    glass_xlsx_path: str | Path,
    *,
    bacteria_synonym_path: str | Path,
    antibiotic_synonym_path: str | Path,
    antibiotic_coarsening_path: str | Path,
    output_path: str | Path | None = None,
    pathogen_universe: Iterable[str] | None = None,
    sheet_name: str = "clean",
) -> pd.DataFrame:
    """Load the GLASS Excel list and save canonical pathogen-antibiotic rows."""
    raw_df = pd.read_excel(glass_xlsx_path, sheet_name=sheet_name)
    required = {"species", "antibiotics"}
    missing = required - set(raw_df.columns)
    if missing:
        raise KeyError(f"{glass_xlsx_path} sheet {sheet_name!r} is missing columns: {sorted(missing)}")

    bacteria_synonym_df = pd.read_csv(bacteria_synonym_path)
    antibiotic_synonym_df = pd.read_csv(antibiotic_synonym_path)
    antibiotic_coarsening_df = pd.read_csv(antibiotic_coarsening_path)

    pathogen_lookup = _build_lookup(bacteria_synonym_df, "bacteria synonym", "bacteria name")
    antibiotic_lookup = _build_lookup(antibiotic_synonym_df, "antibiotic synonym", "antibiotic name")
    antibiotic_lookup.update(_build_lookup(antibiotic_coarsening_df, "Antibiotic", "Antibiotic"))
    antibiotic_lookup.update(
        {
            _norm_key("Co-trimoxazole"): "Sulfamethoxazole/trimethoprim",
            _norm_key("Cotrimoxazole"): "Sulfamethoxazole/trimethoprim",
            _norm_key("Imipenem"): "Imipenem/cilastatin",
            _norm_key("oxacilline"): "Oxacillin",
        }
    )

    rows: list[dict[str, object]] = []
    for raw in raw_df[["species", "antibiotics"]].dropna().itertuples(index=False):
        raw_pathogen = _display_text(raw.species)
        raw_antibiotic = _display_text(raw.antibiotics)
        antibiotic_key = _norm_key(raw_antibiotic)
        canonical_antibiotic = antibiotic_lookup.get(antibiotic_key, raw_antibiotic)
        for canonical_pathogen in _expand_pathogen(raw_pathogen, pathogen_lookup, pathogen_universe):
            rows.append(
                {
                    "raw_pathogen": raw_pathogen,
                    "raw_antibiotic": raw_antibiotic,
                    "pathogen": canonical_pathogen,
                    "antibiotic": canonical_antibiotic,
                    "pathogen_standardized": canonical_pathogen != raw_pathogen,
                    "antibiotic_standardized": canonical_antibiotic != raw_antibiotic,
                }
            )

    cleaned_df = pd.DataFrame(rows).drop_duplicates(["pathogen", "antibiotic"]).reset_index(drop=True)
    cleaned_df = cleaned_df.merge(
        antibiotic_coarsening_df,
        left_on="antibiotic",
        right_on="Antibiotic",
        how="left",
    )
    if "Antibiotic" in cleaned_df.columns:
        cleaned_df = cleaned_df.drop(columns=["Antibiotic"])
    cleaned_df = cleaned_df.sort_values(["pathogen", "antibiotic"]).reset_index(drop=True)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned_df.to_csv(output_path, index=False)
    return cleaned_df


__all__ = ["load_and_clean_glass_list"]
