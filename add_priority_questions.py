from copy import deepcopy
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from xml.etree import ElementTree as ET


INPUT = Path("paper/AMR SI first draft.docx")
OUTPUT = INPUT
TMP = INPUT.with_suffix(".tmp.docx")

NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W = f"{{{NS}}}"


def paragraph(style=None, runs=None, page_break_before=False):
    p = ET.Element(W + "p")
    ppr = ET.SubElement(p, W + "pPr")
    if style:
        ps = ET.SubElement(ppr, W + "pStyle")
        ps.set(W + "val", style)
    if page_break_before:
        pb = ET.SubElement(ppr, W + "pageBreakBefore")
        pb.set(W + "val", "1")
    for text, bold in (runs or []):
        r = ET.SubElement(p, W + "r")
        if bold:
            rpr = ET.SubElement(r, W + "rPr")
            ET.SubElement(rpr, W + "b")
        t = ET.SubElement(r, W + "t")
        if text[:1].isspace() or text[-1:].isspace():
            t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        t.text = text
    return p


def main():
    with ZipFile(INPUT, "r") as zin:
        document_xml = zin.read("word/document.xml")
        root = ET.fromstring(document_xml)
        body = root.find(W + "body")
        sectpr = body.find(W + "sectPr")

        # Remove an earlier copy if this script is re-run.
        for p in list(body):
            texts = "".join(t.text or "" for t in p.iter(W + "t")) if p.tag == W + "p" else ""
            if texts == "Questions to answer before submission":
                idx = list(body).index(p)
                while idx < len(list(body)):
                    node = list(body)[idx]
                    if node is sectpr:
                        break
                    body.remove(node)
                break

        section = [
            paragraph("Heading1", [("Questions to answer before submission", False)], page_break_before=True),
            paragraph(None, [("Internal working list for completing the first draft. Resolve these in order because later claims depend on the earlier decisions and evidence.", False)]),
        ]

        questions = [
            ("What is the paper’s single primary research question and central claim?", "Decide whether AMR-SI is presented as a measure of scientific attention only, a forecasting tool for literature trends, or evidence about emerging resistance itself."),
            ("Can the literature-retrieval pipeline be reproduced exactly?", "Specify the search date and query logic, synonym lists, pathogen and antibiotic source lists, date range, title/abstract restrictions, deduplication, and every exclusion step."),
            ("What is the final dataset used for analysis?", "Report the numbers of records retrieved, screened, retained, unique PMIDs, species–antibiotic pairs, years, and exclusions, including the missing placeholders in Results and Figure 1."),
            ("How accurate and reproducible is the LLM screening step?", "Give the model and run details, prompt, sampling or validation design, error rates, and representative false-positive or false-negative examples, including the proposed ‘bad paper’ example."),
            ("Is the AMR-SI model fully specified and internally consistent?", "Resolve notation and terminology, the ridge-penalty parameter, zero handling, log-scale versus back-transformed quantities, inclusion thresholds, and the inconsistent references to 2023 and 2025."),
            ("Does the temporal model justify the 2030 projections?", "Explain the training window, LOESS and fallback rules, boundary-slope extrapolation, validation through 2025, uncertainty bands, and why these are trend projections rather than calibrated epidemiological forecasts."),
            ("Which four contrasting examples demonstrate that AMR-SI recapitulates known trends?", "Choose examples using an explicit rule, describe the biological and publication trends, and add literature citations that support each interpretation without cherry-picking."),
            ("Which pathogen–antibiotic pairs should be prioritised for surveillance in 2030?", "State the exact threshold and ranking rule, show the shortlisted pairs and scores, and test whether the shortlist changes under reasonable sensitivity analyses."),
            ("How should similarities and discrepancies with the WHO GLASS list be interpreted?", "Separate scope and coverage differences from genuine signals, identify plausible additions and false positives, and avoid treating GLASS as a direct gold standard for scientific attention."),
            ("What can the method not infer?", "State clearly that publication attention is not resistance prevalence, incidence, clinical burden, geography, or transmission; address publication lag, strain-level versus species-level evidence, gene versus phenotype, class-only mentions, and unusual pairs such as vancomycin-resistant E. coli."),
            ("Can readers access the materials needed to audit or update the analysis?", "Decide what to provide for the code, search queries, synonym tables, prompts, model settings, screened outputs, figures, and tables, subject to licensing and data-governance constraints."),
            ("Does the Discussion and Conclusion state the novelty and practical use at the right strength?", "Replace placeholders, define how this complements laboratory surveillance, and refine the final message so that ‘from eminence to evidence-based science’ matches the evidence actually presented."),
        ]

        for i, (q, detail) in enumerate(questions, 1):
            section.append(paragraph(None, [(f"{i}. {q} ", True), (detail, False)]))

        insert_at = len(list(body)) - 1 if sectpr is not None else len(list(body))
        for offset, node in enumerate(section):
            body.insert(insert_at + offset, node)

        new_xml = ET.tostring(root, encoding="utf-8", xml_declaration=True)
        with ZipFile(TMP, "w", ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = new_xml if item.filename == "word/document.xml" else zin.read(item.filename)
                zout.writestr(item, data)
    TMP.replace(OUTPUT)


if __name__ == "__main__":
    main()
