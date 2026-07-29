"""
ICLabel-based artifact-control sensitivity analysis.

Does not modify primary derivatives (preprocessed/, epochs/, specparam/, stats/).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from src.eeg_preprocessing import (
    apply_filters,
    drop_reference_and_non_scalp_channels,
    interpolate_bads,
    make_epochs,
    mark_bad_channels_automatic,
    read_raw_eeg,
    resample_if_needed,
    set_montage,
    set_reference,
)
from src.io_utils import (
    attach_usable_epochs,
    ensure_dir,
    exclude_specparam_low_quality,
    get_epochs_fif_path,
    get_raw_fif_path,
    load_analysis_participants,
    load_participants,
    save_csv,
    save_json,
)
from src.psd_utils import compute_subject_psd
from src.qc_utils import flag_invalid_fits, summarize_subject_qc
from src.specparam_utils import fit_subject_specparam
from src.spectral_maturation_analysis import _posterior_channel_mean
from src.stats_utils import model_results_to_row, run_ols

logger = logging.getLogger(__name__)

ICLABEL_CLASS_ORDER = [
    "brain",
    "muscle artifact",
    "eye blink",
    "heart beat",
    "line noise",
    "channel noise",
    "other",
]
REJECT_CLASSES = frozenset({
    "eye blink",
    "muscle artifact",
    "heart beat",
    "line noise",
    "channel noise",
})
ARTIFACT_CLASSES = [c for c in ICLABEL_CLASS_ORDER if c in REJECT_CLASSES]

COVARIATE_FORMULA = " + age_months + C(sex) + IQ_total + usable_epochs"
COVARIATE_FORMULA_EXPONENT = COVARIATE_FORMULA + " + mean_r_squared"
GROUP_TERM = "C(group)[T.TD]"
REMOVED_COMPONENT_COVARIATE = "n_components_removed"


@dataclass
class ICLabelPaths:
    """Separate output paths from primary analysis."""

    threshold_tag: str
    preproc_out: Path
    epochs_out: Path
    qc_dir: Path
    psd_dir: Path
    specparam_dir: Path
    tables_dir: Path
    figures_dir: Path
    reports_dir: Path

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any], threshold: float) -> ICLabelPaths:
        deriv = Path(cfg["paths"]["derivatives_root"])
        outputs = Path(cfg["paths"]["outputs_root"])
        tag = f"threshold_{threshold:.2f}".replace(".", "_")
        base_pre = deriv / "preprocessed" / "iclabel_cleaned" / tag
        return cls(
            threshold_tag=tag,
            preproc_out=base_pre,
            epochs_out=deriv / "epochs" / "iclabel_cleaned" / tag,
            qc_dir=deriv / "qc" / "iclabel" / tag,
            psd_dir=deriv / "psd" / "iclabel_cleaned" / tag,
            specparam_dir=deriv / "specparam" / "iclabel_cleaned" / tag,
            tables_dir=outputs / "tables" / "iclabel_sensitivity",
            figures_dir=outputs / "figures" / "iclabel_sensitivity",
            reports_dir=outputs / "reports",
        )


def threshold_label(threshold: float) -> str:
    return f"{threshold:.2f}"


def default_iclabel_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    ic = dict(cfg.get("iclabel_sensitivity", {}))
    ic.setdefault("random_state", 97)
    ic.setdefault("variance_explained", 0.99)
    ic.setdefault("max_components", 30)
    ic.setdefault("ica_method", "infomax")
    ic.setdefault("prefer_mff_for_ica", True)
    ic.setdefault("ica_highpass_hz", 1.0)
    ic.setdefault("ica_lowpass_hz", 100.0)
    ic.setdefault("retained_artifact_flag", 0.30)
    ic.setdefault("removed_fraction_flag", 0.80)
    return ic


def _n_ica_components(raw: mne.io.BaseRaw, ic_cfg: dict[str, Any]) -> int:
    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    n_eeg = len(picks)
    if n_eeg < 2:
        return 1
    var_frac = float(ic_cfg.get("variance_explained", 0.99))
    max_comp = int(ic_cfg.get("max_components", 30))
    n_from_var = int(round(var_frac * n_eeg))
    return max(1, min(n_eeg - 1, max_comp, n_from_var))


def _get_iclabel_prob_matrix(
    raw_for_label: mne.io.BaseRaw,
    ica: mne.preprocessing.ICA,
) -> np.ndarray:
    """Return (n_components, n_classes) probability matrix."""
    try:
        from mne_icalabel.iclabel import iclabel_label_components

        probs = iclabel_label_components(raw=raw_for_label, ica=ica)
        return np.asarray(probs, dtype=float)
    except Exception:
        from mne_icalabel import label_components

        labeled = label_components(raw_for_label, ica, method="iclabel")
        labels = labeled["labels"]
        y_pred = np.asarray(labeled["y_pred_proba"], dtype=float)
        n_comp = len(labels)
        mat = np.zeros((n_comp, len(ICLABEL_CLASS_ORDER)))
        for i, lab in enumerate(labels):
            if lab in ICLABEL_CLASS_ORDER:
                j = ICLABEL_CLASS_ORDER.index(lab)
                mat[i, j] = y_pred[i]
        return mat


def components_to_dataframe(
    subject_id: str,
    labels: list[str],
    prob_matrix: np.ndarray,
    exclude_idx: list[int],
) -> pd.DataFrame:
    rows = []
    reject_idx = set(exclude_idx)
    for i in range(prob_matrix.shape[0]):
        probs = {ICLABEL_CLASS_ORDER[j]: float(prob_matrix[i, j]) for j in range(len(ICLABEL_CLASS_ORDER))}
        top_j = int(np.argmax(prob_matrix[i]))
        row = {
            "subject_id": subject_id,
            "component": i,
            "label": labels[i] if i < len(labels) else ICLABEL_CLASS_ORDER[top_j],
            "probability": float(prob_matrix[i, top_j]),
            "removed": int(i in reject_idx),
            "brain_prob": probs["brain"],
            "muscle_prob": probs["muscle artifact"],
            "eye_prob": probs["eye blink"],
            "heart_prob": probs["heart beat"],
            "line_noise_prob": probs["line noise"],
            "channel_noise_prob": probs["channel noise"],
            "other_prob": probs["other"],
        }
        rows.append(row)
    return pd.DataFrame(rows)


def select_components_to_remove(
    prob_matrix: np.ndarray,
    threshold: float,
) -> tuple[list[int], list[str]]:
    """Remove if any artifact-class probability >= threshold."""
    exclude: list[int] = []
    labels: list[str] = []
    for i in range(prob_matrix.shape[0]):
        probs = {ICLABEL_CLASS_ORDER[j]: prob_matrix[i, j] for j in range(len(ICLABEL_CLASS_ORDER))}
        labels.append(ICLABEL_CLASS_ORDER[int(np.argmax(prob_matrix[i]))])
        artifact_max = max(probs[c] for c in ARTIFACT_CLASSES)
        if artifact_max >= threshold:
            exclude.append(i)
    return sorted(set(exclude)), labels


def load_raw_for_iclabel(
    subject_id: str,
    row: pd.Series,
    cfg: dict[str, Any],
    ic_cfg: dict[str, Any],
    preproc_dir: Path,
) -> tuple[mne.io.BaseRaw, str, list[str]]:
    """
    Load continuous EEG for ICA fitting.

    Prefer .mff when available; otherwise use existing cleaned raw .fif.
    """
    project_root = Path(cfg["_project_root"])
    eeg_cfg = cfg["eeg"]
    filt = cfg["filter"]
    warnings: list[str] = []

    mff_path = Path(row["raw_EEG_file"])
    if not mff_path.is_absolute():
        mff_path = project_root / mff_path

    use_mff = bool(ic_cfg.get("prefer_mff_for_ica", True)) and mff_path.exists()
    if use_mff:
        raw = read_raw_eeg(mff_path)
        raw, dropped = drop_reference_and_non_scalp_channels(raw, eeg_cfg)
        raw = set_montage(raw, eeg_cfg.get("montage", "GSN-HydroCel-64_1.0"))
        if filt.get("notch_enabled", True) and filt.get("notch_hz"):
            raw.notch_filter(freqs=filt["notch_hz"], picks="eeg", verbose=False)
        hp = float(ic_cfg.get("ica_highpass_hz", 1.0))
        lp = float(ic_cfg.get("ica_lowpass_hz", 100.0))
        raw.filter(l_freq=hp, h_freq=lp, picks="eeg", verbose=False)
        raw = set_reference(raw, cfg.get("reference", {}).get("method", "average"))
        source = "mff_iclabel_pipeline"
        return raw, source, warnings

    clean_path = get_raw_fif_path(preproc_dir, subject_id)
    if clean_path.exists():
        raw = mne.io.read_raw_fif(clean_path, preload=True, verbose=False)
        warnings.append(
            "ICLabel sensitivity uses the available cleaned raw data; preprocessing may "
            "differ from original ICLabel training assumptions."
        )
        if filt.get("lowpass_hz", 45) < 80:
            warnings.append(
                f"Cleaned raw low-pass ({filt.get('lowpass_hz')} Hz) is below ICLabel's "
                "recommended 1–100 Hz band."
            )
        return raw, "cleaned_raw_fif", warnings

    raise FileNotFoundError(
        f"{subject_id}: no .mff ({mff_path}) or cleaned raw ({clean_path})"
    )


def fit_iclabel_ica(
    raw: mne.io.BaseRaw,
    ic_cfg: dict[str, Any],
) -> tuple[mne.preprocessing.ICA, mne.io.BaseRaw]:
    """Fit extended infomax ICA on 1 Hz high-pass copy; return ICA + raw used for labeling."""
    raw_ica = raw.copy().filter(
        l_freq=float(ic_cfg.get("ica_highpass_hz", 1.0)),
        h_freq=None,
        picks="eeg",
        verbose=False,
    )
    n_comp = _n_ica_components(raw_ica, ic_cfg)
    method = ic_cfg.get("ica_method", "infomax")
    fit_params = {"extended": True} if method == "infomax" else None
    ica = mne.preprocessing.ICA(
        n_components=n_comp,
        method=method,
        fit_params=fit_params,
        random_state=int(ic_cfg.get("random_state", 97)),
        max_iter="auto",
    )
    ica.fit(raw_ica, verbose=False)
    return ica, raw_ica


def finalize_raw_after_ica(
    raw: mne.io.BaseRaw,
    ica: mne.preprocessing.ICA,
    exclude: list[int],
    cfg: dict[str, Any],
    source: str,
) -> mne.io.BaseRaw:
    """Apply ICA removal, then match main-analysis filtering for epoching."""
    ica = ica.copy()
    ica.exclude = list(exclude)
    out = raw.copy()
    ica.apply(out, verbose=False)

    if source == "mff_iclabel_pipeline":
        filt = cfg["filter"]
        out = apply_filters(
            out,
            l_freq=filt["highpass_hz"],
            h_freq=filt["lowpass_hz"],
            notch_freq=filt.get("notch_hz"),
            notch_enabled=filt.get("notch_enabled", True),
        )
        out = resample_if_needed(out, cfg["eeg"]["sampling_rate_target"])
        mark_bad_channels_automatic(out)
        out = interpolate_bads(out)
        out = set_reference(out, cfg.get("reference", {}).get("method", "average"))
    return out


def build_subject_qc_record(
    subject_id: str,
    status: str,
    threshold: float,
    comp_df: pd.DataFrame | None,
    usable_epochs: int,
    usable_seconds: float,
    source: str,
    warnings: list[str],
    error_message: str = "",
    retained_artifact_flag: float = 0.30,
) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "subject_id": subject_id,
        "status": status,
        "threshold": threshold,
        "error_message": error_message,
        "data_source": source,
        "warnings": "; ".join(warnings),
        "usable_epochs_after_iclabel": usable_epochs,
        "usable_seconds_after_iclabel": usable_seconds,
    }
    if comp_df is None or comp_df.empty:
        rec.update({
            "n_components_total": np.nan,
            "n_components_removed": np.nan,
            "percent_components_removed": np.nan,
            "n_eye_removed": np.nan,
            "n_muscle_removed": np.nan,
            "n_heart_removed": np.nan,
            "n_line_noise_removed": np.nan,
            "n_channel_noise_removed": np.nan,
            "mean_removed_probability": np.nan,
            "mean_retained_artifact_probability": np.nan,
            "max_retained_artifact_probability": np.nan,
            "flag_removed_components_gt80pct": np.nan,
            "flag_retained_artifact_prob_gt030": np.nan,
        })
        return rec

    n_total = len(comp_df)
    removed = comp_df[comp_df["removed"] == 1]
    retained = comp_df[comp_df["removed"] == 0]
    artifact_cols = [
        "eye_prob", "muscle_prob", "heart_prob", "line_noise_prob", "channel_noise_prob",
    ]

    def _max_artifact(df: pd.DataFrame) -> pd.Series:
        return df[artifact_cols].max(axis=1)

    rec.update({
        "n_components_total": n_total,
        "n_components_removed": int(len(removed)),
        "percent_components_removed": 100.0 * len(removed) / n_total if n_total else np.nan,
        "n_eye_removed": int((removed["label"] == "eye blink").sum()) if len(removed) else 0,
        "n_muscle_removed": int((removed["label"] == "muscle artifact").sum()) if len(removed) else 0,
        "n_heart_removed": int((removed["label"] == "heart beat").sum()) if len(removed) else 0,
        "n_line_noise_removed": int((removed["label"] == "line noise").sum()) if len(removed) else 0,
        "n_channel_noise_removed": int((removed["label"] == "channel noise").sum()) if len(removed) else 0,
        "mean_removed_probability": float(removed["probability"].mean()) if len(removed) else np.nan,
        "mean_retained_artifact_probability": float(_max_artifact(retained).mean()) if len(retained) else np.nan,
        "max_retained_artifact_probability": float(_max_artifact(retained).max()) if len(retained) else np.nan,
        "flag_removed_components_gt80pct": int(rec.get("percent_components_removed", 0) > 80),
        "flag_retained_artifact_prob_gt030": int(
            (rec.get("max_retained_artifact_probability", 0) or 0) > retained_artifact_flag
        ),
    })
    return rec


def process_subject_iclabel(
    subject_id: str,
    row: pd.Series,
    cfg: dict[str, Any],
    paths: ICLabelPaths,
    threshold: float,
    ic_cfg: dict[str, Any],
    overwrite: bool,
    preproc_primary_dir: Path,
) -> dict[str, Any]:
    """ICA + ICLabel + epochs for one subject at one threshold."""
    out_raw = paths.preproc_out / f"{subject_id}_raw_clean_iclabel.fif"
    out_epo = paths.epochs_out / f"{subject_id}_epochs_iclabel.fif"
    out_comp = paths.qc_dir / f"{subject_id}_iclabel_components.csv"
    out_json = paths.qc_dir / f"{subject_id}_iclabel_qc.json"

    if not overwrite and out_raw.exists() and out_epo.exists() and out_comp.exists():
        epochs = mne.read_epochs(out_epo, preload=False, verbose=False)
        comp_df = pd.read_csv(out_comp)
        return build_subject_qc_record(
            subject_id, "ok_cached", threshold, comp_df,
            len(epochs), len(epochs) * cfg["epochs"]["duration_sec"],
            "cached", [],
        )

    warnings: list[str] = []
    try:
        raw, source, warnings = load_raw_for_iclabel(
            subject_id, row, cfg, ic_cfg, preproc_primary_dir,
        )
        ica, raw_for_label = fit_iclabel_ica(raw, ic_cfg)
        prob_matrix = _get_iclabel_prob_matrix(raw_for_label, ica)
        exclude, labels = select_components_to_remove(prob_matrix, threshold)
        comp_df = components_to_dataframe(subject_id, labels, prob_matrix, exclude)

        raw_clean = finalize_raw_after_ica(raw, ica, exclude, cfg, source)
        ep_cfg = cfg["epochs"]
        epochs = make_epochs(
            raw_clean,
            duration=ep_cfg["duration_sec"],
            overlap=ep_cfg.get("overlap_sec", 0.0),
            reject_uv=ep_cfg.get("reject_amplitude_uv", 500.0),
        )

        ensure_dir(paths.preproc_out)
        ensure_dir(paths.epochs_out)
        ensure_dir(paths.qc_dir)
        raw_clean.save(out_raw, overwrite=True, verbose=False)
        epochs.save(out_epo, overwrite=True, verbose=False)
        save_csv(comp_df, out_comp)

        qc_rec = build_subject_qc_record(
            subject_id, "ok", threshold, comp_df,
            len(epochs), len(epochs) * ep_cfg["duration_sec"],
            source, warnings,
            retained_artifact_flag=float(ic_cfg.get("retained_artifact_flag", 0.30)),
        )
        save_json({**qc_rec, "ica_exclude": exclude, "labels": labels}, out_json)
        return qc_rec
    except Exception as exc:
        logger.exception("%s ICLabel failed: %s", subject_id, exc)
        return build_subject_qc_record(
            subject_id, "error", threshold, None, 0, 0.0,
            "", warnings, error_message=str(exc),
        )


def run_specparam_pipeline_iclabel(
    participants: pd.DataFrame,
    cfg: dict[str, Any],
    paths: ICLabelPaths,
    overwrite: bool,
) -> pd.DataFrame:
    """PSD + specparam + subject-level global metrics (iclabel branch only)."""
    min_epochs = int(cfg["epochs"]["min_usable_epochs"])
    global_rows: list[dict[str, Any]] = []

    for _, row in participants.iterrows():
        sid = str(row["subject_id"])
        group = row["group"]
        ep_path = paths.epochs_out / f"{sid}_epochs_iclabel.fif"
        psd_path = paths.psd_dir / f"{sid}_psd.csv"
        if not ep_path.exists():
            continue
        epochs = mne.read_epochs(ep_path, preload=False, verbose=False)
        if len(epochs) < min_epochs:
            logger.warning("%s: %d epochs < %d after ICLabel", sid, len(epochs), min_epochs)
            continue
        if not overwrite and psd_path.exists():
            psd_df = pd.read_csv(psd_path)
        else:
            ensure_dir(paths.psd_dir)
            psd_df = compute_subject_psd(sid, group, ep_path, psd_path, cfg)

        sp_path = paths.specparam_dir / f"{sid}_specparam_channel.csv"
        if not overwrite and sp_path.exists():
            ch_df = pd.read_csv(sp_path)
        else:
            ensure_dir(paths.specparam_dir)
            ch_df = fit_subject_specparam(psd_df, cfg)
            save_csv(ch_df, sp_path)

        qc_cfg = cfg.get("fit_quality", {})
        flagged = flag_invalid_fits(ch_df, qc_cfg)
        valid = flagged[flagged["fit_valid"]]
        global_rows.append({
            "subject_id": sid,
            "group": group,
            "global_exponent": valid["aperiodic_exponent"].mean() if len(valid) else np.nan,
            "global_offset": valid["aperiodic_offset"].mean() if len(valid) else np.nan,
            "mean_r_squared": valid["r_squared"].mean() if len(valid) else np.nan,
            "n_valid_channels": int(flagged["fit_valid"].sum()),
            "invalid_channel_ratio": 1 - len(valid) / len(flagged) if len(flagged) else np.nan,
        })

    return pd.DataFrame(global_rows)


def load_main_cohort(cfg: dict[str, Any]) -> pd.DataFrame:
    """Primary analysis cohort (same as 08_main_group_analysis: N=138)."""
    deriv = Path(cfg["paths"]["derivatives_root"])
    df = load_analysis_participants(cfg)
    roi_path = deriv / "roi" / "specparam_subject_global.csv"
    if roi_path.exists():
        roi = pd.read_csv(roi_path)
        df = df.merge(roi[["subject_id"]].drop_duplicates(), on="subject_id", how="inner")
    df = exclude_specparam_low_quality(df, deriv)
    logger.info("ICLabel primary-matched cohort: N=%d", len(df))
    return df.reset_index(drop=True)


def load_iclabel_stats_cohort(cfg: dict[str, Any]) -> pd.DataFrame:
    """Primary-matched cohort for ICLabel stats-only reruns."""
    try:
        return load_main_cohort(cfg)
    except FileNotFoundError:
        root = Path(cfg["paths"].get("outputs_root", "outputs"))
        if not root.is_absolute():
            root = Path(__file__).resolve().parents[1] / root
        fallback = root / "tables" / "main_cohort_subject_list.csv"
        if not fallback.exists():
            raise
        df = pd.read_csv(fallback)
        df["subject_id"] = df["subject_id"].astype(str)
        deriv = Path(cfg["paths"]["derivatives_root"])
        logger.info("ICLabel stats cohort from %s (N=%d)", fallback.name, len(df))
        return exclude_specparam_low_quality(df, deriv).reset_index(drop=True)


def load_iclabel_channel_specparam(paths: ICLabelPaths) -> pd.DataFrame:
    """Load all channel-level ICLabel specparam fits for one threshold."""
    if not paths.specparam_dir.exists():
        raise FileNotFoundError(f"ICLabel specparam directory not found: {paths.specparam_dir}")
    parts: list[pd.DataFrame] = []
    for path in sorted(paths.specparam_dir.glob("*_specparam_channel.csv")):
        parts.append(pd.read_csv(path))
    if not parts:
        raise FileNotFoundError(f"No ICLabel channel specparam files under {paths.specparam_dir}")
    df = pd.concat(parts, ignore_index=True)
    df["subject_id"] = df["subject_id"].astype(str)
    if "fit_valid" not in df.columns and "r_squared" in df.columns:
        qc = {"min_r_squared": 0.90}
        df = flag_invalid_fits(df, qc)
    return df


def compute_iclabel_posterior_metrics(ch_df: pd.DataFrame) -> pd.DataFrame:
    """Posterior exponent + mean fit quality from ICLabel channel specparam."""
    valid = ch_df[ch_df["fit_valid"]].copy() if "fit_valid" in ch_df.columns else ch_df.copy()
    post = _posterior_channel_mean(valid, "aperiodic_exponent")
    post = post.rename(columns={"aperiodic_exponent": "posterior_exponent"})
    r2 = (
        valid.groupby("subject_id", as_index=False)["r_squared"]
        .mean()
        .rename(columns={"r_squared": "mean_r_squared"})
    )
    return post.merge(r2, on="subject_id", how="left")


def load_iclabel_qc_table(cfg: dict[str, Any], threshold: float) -> pd.DataFrame:
    root = Path(cfg["paths"].get("outputs_root", "outputs"))
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[1] / root
    tag = f"threshold_{threshold:.2f}".replace(".", "_")
    ic_path = root / "tables" / "iclabel_sensitivity" / f"iclabel_qc_summary_{tag}.csv"
    if not ic_path.exists():
        ic_path = root / "tables" / "iclabel_sensitivity" / "iclabel_qc_summary.csv"
    ic = pd.read_csv(ic_path)
    if "threshold" in ic.columns:
        ic = ic[ic["threshold"] == threshold]
    ic["subject_id"] = ic["subject_id"].astype(str)
    ok = ic[ic["status"].astype(str).str.startswith("ok")].copy()
    keep = [
        "subject_id",
        REMOVED_COMPONENT_COVARIATE,
        "percent_components_removed",
        "usable_epochs_after_iclabel",
    ]
    return ok[[c for c in keep if c in ok.columns]]


def build_iclabel_posterior_cohort_df(
    cfg: dict[str, Any],
    cohort_df: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """Merge primary covariates, ICLabel posterior exponent, and QC removal counts."""
    paths = ICLabelPaths.from_cfg(cfg, threshold)
    deriv = Path(cfg["paths"]["derivatives_root"])
    ch_df = load_iclabel_channel_specparam(paths)
    metrics = compute_iclabel_posterior_metrics(ch_df)
    qc = load_iclabel_qc_table(cfg, threshold)

    keep = [
        "subject_id",
        "group",
        "age_months",
        "sex",
        "IQ_total",
        "usable_epochs",
    ]
    base = cohort_df[[c for c in keep if c in cohort_df.columns]].copy()
    base["subject_id"] = base["subject_id"].astype(str)
    base = attach_usable_epochs(base, deriv)
    metric_cols = ["subject_id", "posterior_exponent", "mean_r_squared"]
    df = base.merge(metrics[metric_cols], on="subject_id", how="inner")
    df = df.merge(qc, on="subject_id", how="inner")
    return df.reset_index(drop=True)


def fit_iclabel_removed_component_models(
    df: pd.DataFrame,
    threshold: float,
    outcome: str = "posterior_exponent",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare ICLabel-branch group effects with and without n_components_removed.

    Base model matches the published ICLabel posterior analysis:
    posterior_exponent ~ group + age + sex + IQ + usable_epochs + mean_r_squared
    """
    if outcome not in df.columns or REMOVED_COMPONENT_COVARIATE not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    cov_cols = ["group", "age_months", "sex", "IQ_total", "usable_epochs", "mean_r_squared", outcome]
    cov_cols.append(REMOVED_COMPONENT_COVARIATE)
    sub = df.dropna(subset=[c for c in cov_cols if c in df.columns])
    if len(sub) < 20 or sub["group"].nunique() < 2:
        return pd.DataFrame(), pd.DataFrame()

    specs = [
        (
            f"{outcome}_without_n_removed",
            f"{outcome} ~ C(group){COVARIATE_FORMULA_EXPONENT}",
            ["group", "age_months", "sex", "IQ_total", "usable_epochs", "mean_r_squared", outcome],
        ),
        (
            f"{outcome}_with_n_components_removed",
            f"{outcome} ~ C(group){COVARIATE_FORMULA_EXPONENT} + {REMOVED_COMPONENT_COVARIATE}",
            cov_cols,
        ),
    ]

    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for model_name, formula, cols in specs:
        model_sub = sub.dropna(subset=[c for c in cols if c in sub.columns])
        if len(model_sub) < 20:
            continue
        model = run_ols(formula, model_sub)
        for row in model_results_to_row(model, model_name, outcome, predictors=list(model.params.index)):
            row["threshold"] = threshold
            rows.append(row)
        if GROUP_TERM not in model.params.index:
            continue
        ci = model.conf_int().loc[GROUP_TERM]
        summary_rows.append({
            "outcome": outcome,
            "model": model_name,
            "threshold": threshold,
            "n_total": int(model.nobs),
            "n_ASD": int((model_sub["group"] == "ASD").sum()),
            "n_TD": int((model_sub["group"] == "TD").sum()),
            "coef_TD_vs_ASD": float(model.params[GROUP_TERM]),
            "se": float(model.bse[GROUP_TERM]),
            "ci_low": float(ci[0]),
            "ci_high": float(ci[1]),
            "p": float(model.pvalues[GROUP_TERM]),
            "r_squared": float(model.rsquared),
            "formula": formula,
        })

    return pd.DataFrame(rows), pd.DataFrame(summary_rows)


def run_iclabel_removed_component_sensitivity(
    cfg: dict[str, Any],
    threshold: float = 0.70,
    cohort_df: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Build cohort and fit posterior exponent models ± n_components_removed."""
    cohort = cohort_df if cohort_df is not None else load_iclabel_stats_cohort(cfg)
    analysis_df = build_iclabel_posterior_cohort_df(cfg, cohort, threshold)
    detail, summary = fit_iclabel_removed_component_models(analysis_df, threshold)
    return {
        "analysis_df": analysis_df,
        "detail": detail,
        "summary": summary,
    }


def fit_iclabel_group_models(
    df: pd.DataFrame,
    threshold: float,
    deriv_root: Path,
) -> pd.DataFrame:
    """OLS models matching primary analysis."""
    df = attach_usable_epochs(df, deriv_root)
    rows: list[dict[str, Any]] = []
    for outcome in ("global_exponent", "global_offset"):
        sub = df.dropna(subset=[outcome, "group", "age_months", "sex", "IQ_total", "usable_epochs"])
        if len(sub) < 10:
            continue
        formula = f"{outcome} ~ C(group){COVARIATE_FORMULA}"
        model = run_ols(formula, sub)
        if GROUP_TERM not in model.params.index:
            continue
        asd = sub.loc[sub["group"] == "ASD", outcome]
        td = sub.loc[sub["group"] == "TD", outcome]
        ci = model.conf_int().loc[GROUP_TERM]
        rows.append({
            "outcome": outcome,
            "analysis": "iclabel_artifact_control",
            "threshold": threshold,
            "n_total": int(model.nobs),
            "n_ASD": int((sub["group"] == "ASD").sum()),
            "n_TD": int((sub["group"] == "TD").sum()),
            "coef_TD_vs_ASD": float(model.params[GROUP_TERM]),
            "se": float(model.bse[GROUP_TERM]),
            "ci_low": float(ci[0]),
            "ci_high": float(ci[1]),
            "p": float(model.pvalues[GROUP_TERM]),
            "r_squared": float(model.rsquared),
            "ASD_mean": float(asd.mean()),
            "ASD_sd": float(asd.std()),
            "TD_mean": float(td.mean()),
            "TD_sd": float(td.std()),
        })
    return pd.DataFrame(rows)


def load_primary_group_results(cfg: dict[str, Any]) -> pd.DataFrame:
    deriv = Path(cfg["paths"]["derivatives_root"])
    path = deriv / "stats" / "main_group_analysis.csv"
    if not path.exists():
        raise FileNotFoundError(f"Primary results not found: {path}")
    raw = pd.read_csv(path)
    rows = []
    for outcome in ("global_exponent", "global_offset"):
        sub = raw[(raw["outcome"] == outcome) & (raw["term"] == GROUP_TERM)]
        if sub.empty:
            continue
        r = sub.iloc[0]
        rows.append({
            "analysis": "primary",
            "outcome": outcome,
            "threshold": np.nan,
            "n_total": int(r["n_obs"]),
            "coef_TD_vs_ASD": float(r["coef"]),
            "ci_low": float(r["ci_low"]),
            "ci_high": float(r["ci_high"]),
            "p": float(r["pvalue"]),
        })
    return pd.DataFrame(rows)


def interpret_comparison(
    primary_coef: float,
    ic_coef: float,
    primary_p: float,
    ic_p: float,
    n_ic: int,
) -> str:
    if n_ic < 10 or np.isnan(ic_coef):
        return "insufficient data"
    if np.sign(primary_coef) != np.sign(ic_coef) or primary_coef == 0:
        return "direction inconsistent"
    if primary_p < 0.05 and ic_p < 0.05:
        return "direction consistent, significant"
    if np.sign(primary_coef) == np.sign(ic_coef):
        return "direction consistent, weaker evidence"
    return "insufficient data"


def build_comparison_table(
    primary_df: pd.DataFrame,
    iclabel_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for outcome in ("global_exponent", "global_offset"):
        p_row = primary_df[primary_df["outcome"] == outcome]
        if p_row.empty:
            continue
        pr = p_row.iloc[0]
        for _, ir in iclabel_df[iclabel_df["outcome"] == outcome].iterrows():
            rows.append({
                "analysis": f"iclabel_threshold_{ir['threshold']:.2f}",
                "outcome": outcome,
                "n_total": int(ir["n_total"]),
                "coef_TD_vs_ASD": float(ir["coef_TD_vs_ASD"]),
                "ci_low": float(ir["ci_low"]),
                "ci_high": float(ir["ci_high"]),
                "p": float(ir["p"]),
                "interpretation": interpret_comparison(
                    float(pr["coef_TD_vs_ASD"]),
                    float(ir["coef_TD_vs_ASD"]),
                    float(pr["p"]),
                    float(ir["p"]),
                    int(ir["n_total"]),
                ),
            })
        rows.append({
            "analysis": "primary",
            "outcome": outcome,
            "n_total": int(pr["n_total"]),
            "coef_TD_vs_ASD": float(pr["coef_TD_vs_ASD"]),
            "ci_low": float(pr["ci_low"]),
            "ci_high": float(pr["ci_high"]),
            "p": float(pr["p"]),
            "interpretation": "primary reference",
        })
    return pd.DataFrame(rows)


def plot_forest_comparison(
    primary_df: pd.DataFrame,
    iclabel_results: pd.DataFrame,
    out_base: Path,
) -> None:
    """Forest plot: primary vs ICLabel thresholds."""
    ensure_dir(out_base.parent)
    plot_rows: list[dict[str, Any]] = []

    for outcome in ("global_exponent", "global_offset"):
        pr = primary_df[primary_df["outcome"] == outcome]
        if not pr.empty:
            plot_rows.append({
                "label": f"Primary — {outcome}",
                "coef": pr.iloc[0]["coef_TD_vs_ASD"],
                "ci_low": pr.iloc[0]["ci_low"],
                "ci_high": pr.iloc[0]["ci_high"],
            })
        for _, ir in iclabel_results[iclabel_results["outcome"] == outcome].iterrows():
            plot_rows.append({
                "label": f"ICLabel {ir['threshold']:.2f} — {outcome}",
                "coef": ir["coef_TD_vs_ASD"],
                "ci_low": ir["ci_low"],
                "ci_high": ir["ci_high"],
            })

    if not plot_rows:
        return

    df = pd.DataFrame(plot_rows)
    y_pos = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * len(df))))
    ax.axvline(0, color="0.5", linewidth=1, linestyle="-", zorder=0)
    ax.errorbar(
        df["coef"], y_pos,
        xerr=[df["coef"] - df["ci_low"], df["ci_high"] - df["coef"]],
        fmt="o", color="0.15", ecolor="0.25", capsize=3, markersize=6,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["label"])
    ax.set_xlabel("β (TD − ASD) with 95% CI")
    ax.set_title("Artifact-control sensitivity: primary vs ICLabel cleaning")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_base.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_manuscript_snippets(
    paths: ICLabelPaths,
    qc_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    iclabel_models: pd.DataFrame,
) -> None:
    ok = qc_summary[qc_summary["status"].str.startswith("ok")]
    mean_removed = ok["n_components_removed"].mean() if len(ok) else np.nan
    mean_pct = ok["percent_components_removed"].mean() if len(ok) else np.nan
    mean_ret = ok["mean_retained_artifact_probability"].mean() if len(ok) else np.nan

    exp_row = iclabel_models[iclabel_models["outcome"] == "global_exponent"]
    interp = "directionally consistent"
    conclusion_en = (
        "The TD–ASD exponent effect remained directionally consistent after automated "
        "ICLabel-based artifact removal, suggesting that the primary finding was unlikely "
        "to be solely driven by residual ocular or myogenic artifacts."
    )
    if not exp_row.empty:
        comp = comparison[
            (comparison["outcome"] == "global_exponent")
            & (comparison["analysis"].str.startswith("iclabel"))
        ]
        exp_comp = comp[
            (comp["outcome"] == "global_exponent")
            & (comp["analysis"].str.startswith("iclabel"))
        ]
        use_row = exp_comp.iloc[-1] if len(exp_comp) else None
        if use_row is not None:
            coef = float(use_row["coef_TD_vs_ASD"])
            pval = float(use_row["p"])
        if use_row is not None and (
            "inconsistent" in str(use_row["interpretation"])
            or pval >= 0.05
        ):
            interp = "attenuated (weaker evidence)"
            conclusion_en = (
                "The group effect was attenuated after ICLabel-based artifact removal "
                f"(β = {coef:.3f}, p = {pval:.3f}), suggesting that residual artifacts may "
                "partly contribute to the primary effect and should be considered in interpretation."
            )

    thr = exp_row["threshold"].iloc[0] if len(exp_row) else 0.80
    coef = exp_row["coef_TD_vs_ASD"].iloc[0] if len(exp_row) else np.nan
    pval = exp_row["p"].iloc[0] if len(exp_row) else np.nan

    text = f"""# ICLabel 敏感性分析 — 可粘贴稿件段落

> 自动生成；请勿覆盖主分析。分析名称：**artifact-control sensitivity analysis**（伪迹控制敏感性分析）。

## Methods（中文）

为检验残留眼动或肌电伪迹是否影响非周期 exponent 的组间效应，我们在不改动主分析流水线的前提下，另行开展 **伪迹控制敏感性分析**。对每名被试的连续 EEG 重新进行 extended infomax ICA（`random_state=97`），并使用 **mne-icalabel**（ICLabel）对独立成分进行自动分类。当某成分在眼动、肌电、心电、工频或通道噪声类别上的概率 ≥ {thr:.2f} 时予以剔除；脑活动与其他类别成分保留。随后在 `derivatives/preprocessed/iclabel_cleaned/` 与 `derivatives/epochs/iclabel_cleaned/` 下重新分段、估计 Welch PSD 并拟合 specparam（fixed 模式，1–40 Hz）。在通过主分析相同纳入与 specparam QC 标准的队列（*N* = 138）上，拟合与主分析一致的 OLS 模型：`global_exponent` / `global_offset` ~ 组别 + 年龄 + 性别 + IQ + 可用 epoch 数。本分析 **未** 使用 HAPPE 或 MARA。

## Methods（英文模板）

To address whether residual ocular or myogenic artifacts could influence the aperiodic exponent group effect, we conducted an **artifact-control sensitivity analysis** separate from the primary pipeline. Continuous EEG was re-processed with extended infomax ICA (`random_state=97`) and independent component labels from **mne-icalabel** (ICLabel). Components with artifact-class probabilities ≥ {thr:.2f} (eye blink, muscle artifact, heart beat, line noise, channel noise) were removed; brain and other components were retained. Epochs, Welch PSD, and specparam (fixed mode, 1–40 Hz) were recomputed under `derivatives/*/iclabel_cleaned/` without altering primary outputs. The same OLS models as the primary analysis (`global_exponent` / `global_offset` ~ group + age + sex + IQ + usable epochs) were fit on the primary cohort (*N* = 138 after identical QC rules). This procedure is **not** HAPPE or MARA.

## Results（中文）

ICLabel 处理成功 {len(ok)} 名被试；平均剔除 {mean_removed:.1f} 个 ICA 成分（占 {mean_pct:.1f}%），保留成分的平均最大伪迹类概率为 {mean_ret:.3f}。在概率阈值 {thr:.2f} 下，global aperiodic exponent 的 TD − ASD 效应为 β = {coef:.3f}，*p* = {pval:.3f}（*n* = {int(exp_row['n_total'].iloc[0]) if len(exp_row) else 'NA'}），主分析为 β = 0.079，*p* = .012。敏感性分析与主分析在效应方向上 **{'一致' if 'consistent' in interp else '不一致或减弱'}**。

## Results（英文模板）

ICLabel processing succeeded in {len(ok)} subjects (mean removed components = {mean_removed:.1f}, mean removed fraction = {mean_pct:.1f}%, mean max retained artifact probability among kept components = {mean_ret:.3f}). After ICLabel cleaning (threshold = {thr:.2f}), the global exponent TD − ASD effect was β = {coef:.3f}, *p* = {pval:.3f} (*n* = {int(exp_row['n_total'].iloc[0]) if len(exp_row) else 'NA'}), compared with the primary β = 0.079, *p* = .012. The sensitivity analysis was **{interp}** with the primary direction.

{conclusion_en}

## Discussion（中文）

ICLabel 敏感性分析用于评估：在自动剔除高概率眼动/肌电相关成分后，非周期 exponent 的 ASD–TD 差异是否仍然存在。若与主分析方向一致，则支持组间差异不太可能仅由 ICLabel 可检测的残留伪迹驱动；若效应减弱或反转，则应在讨论中更谨慎地表述，并考虑补充人工 ICA 复核。该分析 **不替代** 主分析预处理流程。

## Discussion（英文模板）

The artifact-control sensitivity analysis using ICLabel was intended to test robustness of the aperiodic exponent finding to automated ocular and myogenic artifact removal. Agreement with the primary effect would support interpretation that group differences in 1/f slope were not solely driven by residual artifacts detectable by ICLabel; divergence would warrant more conservative wording and further manual ICA review.

## Limitations（中文）

ICLabel 在原始训练中假设平均参考、extended infomax 及约 1–100 Hz 数据；本敏感性分支若复用已低通至 45 Hz 的清洗数据，分类性能可能受影响。本研究 **未** 运行 HAPPE 或 MARA。ICLabel 在儿童 EGI 睁眼静息数据上的自动标注应谨慎解释。本分析仅为敏感性补充，不能替代主分析推断。

## Limitations（英文模板）

ICLabel was trained under specific preprocessing assumptions (average reference, extended infomax, approximately 1–100 Hz data). Our sensitivity branch may use cleaned continuous data that differ from those assumptions (e.g., lower low-pass if reloaded from prior preprocessing). ICLabel is not HAPPE, MARA, or HAPPE-style automated pipelines unless explicitly run. Automatic labels on pediatric EGI resting-state data should be interpreted cautiously. This analysis supplements but does not replace the primary preprocessing and main inferential results.
"""
    out = paths.reports_dir / "iclabel_text_for_manuscript.md"
    out.write_text(text, encoding="utf-8")
    logger.info("Wrote %s", out)


def write_summary_report(
    paths: ICLabelPaths,
    qc_summary: pd.DataFrame,
    iclabel_models: pd.DataFrame,
    comparison: pd.DataFrame,
    n_success: int,
    n_fail: int,
) -> None:
    ok = qc_summary[qc_summary["status"].str.startswith("ok")]
    lines = [
        "# ICLabel artifact-control sensitivity report",
        "",
        "## 1. Rationale",
        "Test whether residual ocular/myogenic artifacts influence the global aperiodic exponent group effect.",
        "",
        "## 2. Method",
        "- **Tool**: mne-icalabel (ICLabel), extended infomax ICA",
        "- **Not**: HAPPE, MARA (not run unless explicitly added later)",
        "- **Reject classes** (prob ≥ threshold): eye blink, muscle, heart, line noise, channel noise",
        "- **Keep**: brain, other",
        f"- **Threshold tag**: `{paths.threshold_tag}`",
        "",
        "## 3. QC summary",
        f"- Success: {n_success}, Failed: {n_fail}",
        f"- Mean components removed: {ok['n_components_removed'].mean():.2f}" if len(ok) else "- No successful subjects",
        f"- Mean % removed: {ok['percent_components_removed'].mean():.2f}" if len(ok) else "",
        f"- Mean retained artifact prob: {ok['mean_retained_artifact_probability'].mean():.3f}" if len(ok) else "",
        "",
        "## 4. Main model comparison",
        "",
        iclabel_models.to_string(index=False) if len(iclabel_models) else "_No models_",
        "",
        "## 5. vs Primary",
        "",
        comparison.to_string(index=False) if len(comparison) else "_No comparison_",
        "",
        "## 6. Limitations",
        "- Sensitivity only; primary pipeline unchanged",
        "- ICLabel training assumptions may not fully match pediatric EGI preprocessing",
        "",
    ]
    out = paths.reports_dir / "iclabel_sensitivity_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", out)
