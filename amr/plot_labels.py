"""Readable labels shared by AMR plotting workflows."""


def format_pathogen_antibiotic_label(pathogen: object, antibiotic: object) -> str:
    """Format a pair as an abbreviated binomial plus the full antibiotic name."""
    pathogen_text = str(pathogen).strip()
    pathogen_parts = pathogen_text.split()
    if len(pathogen_parts) >= 2:
        pathogen_text = f"{pathogen_parts[0][0]}. {' '.join(pathogen_parts[1:])}"
    return f"{pathogen_text}<br>{str(antibiotic).strip()}"
