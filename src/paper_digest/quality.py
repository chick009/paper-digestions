"""Quality checks for generated reports."""

from __future__ import annotations

from paper_digest.schema import CritiqueSet, DigestReport, QualityCheck


def local_quality_check(report: DigestReport, critique_set: CritiqueSet) -> QualityCheck:
    unsupported_claims: list[str] = []
    weak_sota_claims: list[str] = []
    missing_baselines: list[str] = []
    unclear_formulas: list[str] = []

    if not report.methodology.evidence:
        unsupported_claims.append("Methodology summary has no page-specific evidence.")
    if not report.findings.evidence and report.findings.important_findings:
        unsupported_claims.append("Findings were extracted without explicit evidence references.")
    if critique_set.final_sota_stance == "likely_sota":
        has_baseline = bool(report.findings.baselines or report.findings.reported_results)
        if not has_baseline:
            weak_sota_claims.append("Likely-SOTA stance lacks baselines or reported results.")
    if not report.findings.baselines:
        missing_baselines.append("No baselines were extracted.")
    if report.methodology.important_formulas and not report.explanations:
        unclear_formulas.append(
            "Formulas were identified but no concept explanations were produced."
        )

    revision_notes = unsupported_claims + weak_sota_claims + missing_baselines + unclear_formulas
    return QualityCheck(
        passed=not revision_notes,
        unsupported_claims=unsupported_claims,
        weak_sota_claims=weak_sota_claims,
        missing_baselines=missing_baselines,
        unclear_formulas=unclear_formulas,
        revision_notes=revision_notes,
    )
