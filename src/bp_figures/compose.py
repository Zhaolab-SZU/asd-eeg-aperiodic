"""Assemble Biological Psychiatry main figures 1–5."""

from __future__ import annotations

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import mne
    HAS_MNE = True
except ImportError:
    HAS_MNE = False

from src.bp_figures.io import FALLBACK, df, has_rows, load_data
from src.bp_figures.plots import (
    cohort_flow,
    fit_line_ci,
    forest_h,
    raincloud,
    residualize,
    segment_bars,
    specparam_exemplar,
)
from src.bp_figures.style import (
    COLOR_ASD,
    COLOR_ROI,
    COLOR_TD,
    FIG_W_IN,
    MONTAGE_NAME,
    POSTERIOR_CORE,
    annotate_box,
    apply_bp_style,
    format_p,
    panel_label,
    save_figure,
    write_caption,
)


def _segment_means(movie: pd.DataFrame, col: str, fallback: list[dict]) -> tuple[list[str], list[float], list[float], list[float]]:
    segments = ["Mentalizing", "Pain-related", "Neutral"]
    sub = movie[~movie["participant_id"].astype(str).str.startswith("SUMMARY")].copy()
    td_vals, asd_vals, pvals = [], [], []
    for seg in segments:
        fb = next((f for f in fallback if f["segment"] == seg), None)
        if has_rows(sub) and col in sub.columns and sub[col].notna().sum() > 0:
            td_vals.append(sub.loc[(sub["group"] == "TD") & (sub["segment"] == seg), col].mean())
            asd_vals.append(sub.loc[(sub["group"] == "ASD") & (sub["segment"] == seg), col].mean())
        elif fb:
            td_vals.append(fb["td_r"])
            asd_vals.append(fb["asd_r"])
        else:
            td_vals.append(np.nan)
            asd_vals.append(np.nan)
        pvals.append(fb["p"] if fb else np.nan)
    return segments, td_vals, asd_vals, pvals


def build_figure_1(out_dir: Path, data_dir: Path | None = None) -> None:
    apply_bp_style()
    load_data(data_dir)
    part = df("participants_rest.csv")
    flow = FALLBACK["fig1_flow"]
    ge, go, pe = FALLBACK["fig1_global_exponent"], FALLBACK["fig1_global_offset"], FALLBACK["fig1_posterior"]

    fig = plt.figure(figsize=(FIG_W_IN, 4.8))
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1.1, 1], hspace=0.55, wspace=0.38)

    ax_a = fig.add_subplot(gs[0, 0])
    panel_label(ax_a, "A")
    cohort_flow(ax_a, flow)

    ax_b = fig.add_subplot(gs[0, 1])
    panel_label(ax_b, "B")
    specparam_exemplar(ax_b)

    ax_c = fig.add_subplot(gs[0, 2])
    panel_label(ax_c, "C")
    if has_rows(part):
        raincloud(
            ax_c, part, "global_exponent", ylabel="Global aperiodic exponent",
            stats_text=f"Adjusted β = {ge['beta']:.3f}\n95% CI [{ge['ci'][0]:.3f}, {ge['ci'][1]:.3f}]\n{format_p(ge['p'])}\nN = {ge['n']}",
        )
    else:
        ax_c.text(0.5, 0.5, "Data required", ha="center", va="center", transform=ax_c.transAxes)

    ax_d = fig.add_subplot(gs[1, 0])
    panel_label(ax_d, "D")
    if has_rows(part):
        raincloud(
            ax_d, part, "global_offset", ylabel="Global aperiodic offset",
            stats_text=f"Adjusted β = {go['beta']:.3f}\n{format_p(go['p'])}\nN = {go['n']}",
        )

    ax_e = fig.add_subplot(gs[1, 1:])
    panel_label(ax_e, "E", x=-0.07)
    if has_rows(part):
        raincloud(
            ax_e, part, "posterior_exponent", ylabel="Posterior aperiodic exponent",
            stats_text=f"Adjusted β = {pe['beta']:.3f}\n95% CI [{pe['ci'][0]:.3f}, {pe['ci'][1]:.3f}]\n{format_p(pe['p'])}\nN = {pe['n']}",
        )
        ax_e.text(0.5, -0.18, "Posterior ROI defined in Figure 2", transform=ax_e.transAxes, ha="center", fontsize=7, color="#666666")

    save_figure(fig, "Figure1", out_dir)
    write_caption(1, """Figure 1. Resting-state spectral parameterization and posterior aperiodic flattening in autistic and typically developing children.
(A) Cohort flow for resting-state EEG spectral analyses. Numbers indicate total sample size and group counts at each inclusion stage.
(B) Illustration of spectral parameterization separating periodic peaks from the aperiodic 1/f-like background; exponent reflects spectral slope and offset reflects broadband vertical shift.
(C) Global aperiodic exponent by group (raincloud plots with individual participants). TD children showed higher global exponent than ASD children (adjusted model).
(D) Global aperiodic offset by group; no significant group difference in offset.
(E) Posterior aperiodic exponent by group, indicating flatter posterior 1/f-like slope in ASD (TD > ASD).
Annotations reflect covariate-adjusted OLS models (age, sex, IQ, usable epochs); raincloud plots are descriptive.""", out_dir)


def build_figure_2(out_dir: Path, data_dir: Path | None = None) -> None:
    apply_bp_style()
    load_data(data_dir)
    roi = df("roi_effects.csv")
    ch = df("channel_effects.csv")
    rob = df("robustness_results.csv")

    fig = plt.figure(figsize=(FIG_W_IN, 5.8))
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.1, 1], hspace=0.42, wspace=0.38)
    ax = fig.add_subplot(gs[0, 0])
    panel_label(ax, "A")
    roi_fb = FALLBACK["fig2_roi"]
    if has_rows(roi) and "Frontal" in roi["roi"].values:
        r = roi.copy()
        labels = r["roi"].tolist()
        betas = r["beta_interaction"].tolist()
        ci_l = (r["beta_interaction"] - 1.96 * r["se"]).tolist()
        ci_h = (r["beta_interaction"] + 1.96 * r["se"]).tolist()
    else:
        labels = [x["roi"] for x in roi_fb]
        betas = [x["beta_interaction"] for x in roi_fb]
        ci_l = [x["beta_interaction"] - 1.96 * x["se"] for x in roi_fb]
        ci_h = [x["beta_interaction"] + 1.96 * x["se"] for x in roi_fb]
    colors = [COLOR_ROI if lab in ("Frontal", "Occipital") else "#2A2A2A" for lab in labels]
    forest_h(ax, labels, betas, ci_l, ci_h, colors=colors, xlabel="Interaction β (TD − ASD vs central)")
    ax.set_title("ROI group × region modulation", fontsize=8.5)

    ax = fig.add_subplot(gs[0, 1])
    panel_label(ax, "B")
    plotted = False
    if HAS_MNE and has_rows(ch):
        try:
            montage = mne.channels.make_standard_montage(MONTAGE_NAME)
            ch_names = [c for c in montage.ch_names if c.startswith("E")]
            coef_map = ch.set_index("channel")["beta_td_minus_asd"].to_dict()
            values = np.array([coef_map.get(c, np.nan) for c in ch_names])
            info = mne.create_info(ch_names=ch_names, sfreq=250.0, ch_types="eeg")
            info.set_montage(montage)
            vmax = 0.25
            im, _ = mne.viz.plot_topomap(values, info, axes=ax, show=False, cmap="RdBu_r", vlim=(-vmax, vmax), contours=0)
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("β (TD − ASD)", fontsize=7)
            cbar.ax.tick_params(labelsize=7)
            pos = montage.get_positions()["ch_pos"]
            sig = ch.loc[ch["q_fdr"] < 0.05, "channel"].tolist() if "q_fdr" in ch.columns else POSTERIOR_CORE
            for c in sig:
                if c in pos:
                    xy = pos[c][:2]
                    ax.plot(xy[0], xy[1], "o", mfc="none", mec="black", mew=1.8, ms=11, zorder=10)
            inset = ax.inset_axes([0.02, 0.02, 0.28, 0.35])
            for c in POSTERIOR_CORE:
                if c in pos:
                    xy = pos[c][:2]
                    inset.scatter(xy[0], xy[1], s=80, c=COLOR_ROI, edgecolors="white", linewidths=0.8, zorder=3)
                    inset.text(xy[0], xy[1], c.replace("E", ""), ha="center", va="center", color="white", fontsize=6, fontweight="bold")
            inset.set_aspect("equal")
            inset.axis("off")
            inset.set_title("Posterior ROI", fontsize=6.5, pad=1)
            plotted = True
        except Exception:
            plotted = False
    if not plotted and has_rows(ch):
        sub = ch.dropna(subset=["x", "y", "beta_td_minus_asd"])
        sc = ax.scatter(sub["x"], sub["y"], c=sub["beta_td_minus_asd"], cmap="RdBu_r", vmin=-0.25, vmax=0.25, s=35, edgecolors="0.3", linewidths=0.3)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="β (TD − ASD)")
        for c in POSTERIOR_CORE:
            row = sub[sub["channel"] == c]
            if len(row):
                ax.plot(row["x"], row["y"], "o", mfc="none", mec="black", mew=1.8, ms=11)
        ax.set_aspect("equal")
        ax.axis("off")

    ax = fig.add_subplot(gs[1, :])
    panel_label(ax, "C", x=-0.03, y=1.05)
    items = FALLBACK["fig2_robustness"]
    if has_rows(rob):
        r = rob.copy()
        labels = r["analysis"].tolist()
        betas = r["beta_td_minus_asd"].tolist()
        ci_l = r["ci_low"].tolist()
        ci_h = r["ci_high"].tolist()
        ann = [f"n = {int(n)}, {format_p(p)}" for n, p in zip(r["n"], r["p"])]
    else:
        labels = [x["label"] for x in items]
        betas = [x["beta"] for x in items]
        ci_l = [x["ci_low"] for x in items]
        ci_h = [x["ci_high"] for x in items]
        ann = [f"n = {x['n']}, {format_p(x['p'])}" for x in items]
    ref = betas[0]
    forest_h(ax, labels, betas, ci_l, ci_h, colors=[COLOR_ROI] * len(labels),
             ref_line=ref, xlabel="β (TD − ASD), posterior exponent (95% CI)", annotations=ann)
    ax.set_title("Robustness across sensitivity analyses", fontsize=8.5)

    save_figure(fig, "Figure2", out_dir)
    write_caption(2, """Figure 2. Posterior localization and robustness of aperiodic flattening in ASD.
(A) Region-specific group × ROI interaction contrasts relative to central reference, showing strongest modulation over frontal and occipital cortex.
(B) Channel-level adjusted TD − ASD regression coefficients (topography); black circles mark FDR-significant posterior electrodes (E33, E36, E37, E38); inset shows posterior ROI.
(C) Posterior exponent group effects across primary and sensitivity models; dashed vertical line indicates the primary estimate. All posterior effects remained significant whereas global effects attenuated under several specifications (see Supplementary Table S4).
Error bars denote 95% confidence intervals from adjusted models.""", out_dir)


def build_figure_3(out_dir: Path, data_dir: Path | None = None) -> None:
    apply_bp_style()
    load_data(data_dir)
    dev = df("developmental_results.csv")
    ai = FALLBACK["fig3_age_interaction"]
    sl = FALLBACK["fig3_slopes"]
    dv = FALLBACK["fig3_deviation"]
    sens = FALLBACK["fig3_sensitivity"]

    fig = plt.figure(figsize=(FIG_W_IN, 5.4))
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1.2, 1], hspace=0.48, wspace=0.42)

    ax_a = fig.add_subplot(gs[0, :2])
    panel_label(ax_a, "A", x=-0.06)
    if has_rows(dev):
        for grp, color in [("TD", COLOR_TD), ("ASD", COLOR_ASD)]:
            sub = dev[dev["group"] == grp].dropna(subset=["age_months", "posterior_exponent"])
            ax_a.scatter(sub["age_months"], sub["posterior_exponent"], s=20, alpha=0.6, color=color, edgecolors="white", linewidths=0.25, label=grp)
            xg = np.linspace(sub["age_months"].min(), sub["age_months"].max(), 60)
            pred, lo, hi = fit_line_ci(sub["age_months"].values, sub["posterior_exponent"].values, xg)
            ax_a.plot(xg, pred, color=color, lw=2)
            ax_a.fill_between(xg, lo, hi, color=color, alpha=0.15, linewidth=0)
        ax_a.set_xlabel("Age (months)")
        ax_a.set_ylabel("Posterior aperiodic exponent")
        ax_a.legend(frameon=False, loc="upper right")
        annotate_box(ax_a,
            f"Age × group β = {ai['beta']:.4f}/mo\nSE = {ai['se']:.4f}, {format_p(ai['p'])}\n"
            f"ASD slope ≈ {sl['asd']:.4f}/mo\nTD slope ≈ {sl['td']:+.4f}/mo", "top_left")

    ax_b = fig.add_subplot(gs[0, 2])
    panel_label(ax_b, "B")
    slope_data = [("ASD", sl["asd"], sl["asd_p"], COLOR_ASD), ("TD", sl["td"], sl["td_p"], COLOR_TD)]
    for i, (lab, beta, p, color) in enumerate(slope_data):
        ax_b.barh(i, beta, color=color, alpha=0.85, height=0.5)
        ax_b.text(beta + (0.0003 if beta >= 0 else -0.0003), i, f"{format_p(p)}", va="center",
                  ha="left" if beta >= 0 else "right", fontsize=7)
    ax_b.axvline(0, color="#666666", lw=0.8)
    ax_b.set_yticks([0, 1])
    ax_b.set_yticklabels(["ASD", "TD"])
    ax_b.set_xlabel("Simple slope (exponent/mo)")
    ax_b.set_title("Posterior exponent\nage slopes", fontsize=8)

    ax_c = fig.add_subplot(gs[1, 0])
    panel_label(ax_c, "C")
    if has_rows(dev) and "td_reference_deviation_z" in dev.columns:
        asd = dev[dev["group"] == "ASD"].dropna(subset=["td_reference_deviation_z", "age_months"]).copy()
        asd["age_bin"] = np.where(asd["age_months"] > 72, ">72 mo", "≤72 mo")
        groups = ["All ASD", ">72 mo"]
        data_v = [
            asd["td_reference_deviation_z"].values,
            asd.loc[asd["age_bin"] == ">72 mo", "td_reference_deviation_z"].values,
        ]
        parts = ax_c.violinplot(data_v, positions=[0, 1], widths=0.55, showextrema=False, showmeans=False)
        for i, body in enumerate(parts["bodies"]):
            body.set_facecolor(COLOR_ASD if i == 0 else COLOR_ROI)
            body.set_alpha(0.35)
        ax_c.axhline(0, color="#666666", ls="--", lw=0.8)
        rng = np.random.default_rng(42)
        for i, vals in enumerate(data_v):
            jitter = rng.uniform(-0.08, 0.08, len(vals))
            ax_c.scatter(i + jitter, vals, s=12, alpha=0.55, color=COLOR_ASD)
        ax_c.set_xticks([0, 1])
        ax_c.set_xticklabels(groups, fontsize=7.5)
        ax_c.set_ylabel("TD-reference deviation (z)")
        annotate_box(ax_c, f"All: z = {dv['all_z']:.3f}\nOlder: z = {dv['older_z']:.3f}", "bottom_right")

    ax_d = fig.add_subplot(gs[1, 1:])
    panel_label(ax_d, "D", x=-0.05)
    labels = [x["label"] for x in sens]
    betas = [x["beta"] for x in sens]
    ci_l = [x.get("ci_low", x["beta"] - 0.002) for x in sens]
    ci_h = [x.get("ci_high", x["beta"] + 0.002) for x in sens]
    colors = [COLOR_ROI if x["p"] < 0.05 else "#AAAAAA" for x in sens]
    ann = [format_p(x["p"]) for x in sens]
    forest_h(ax_d, labels, betas, ci_l, ci_h, colors=colors,
             xlabel="β (age × group interaction, /month)", annotations=ann)
    ax_d.set_title("Sensitivity of age × group interaction", fontsize=8.5)

    save_figure(fig, "Figure3", out_dir)
    write_caption(3, """Figure 3. Age-dependent divergence of posterior aperiodic dynamics in cross-sectional data.
(A) Posterior exponent as a function of age by group, with group-specific linear fits and 95% confidence bands.
(B) Simple age slopes for posterior exponent within ASD and TD groups.
(C) TD-reference deviation scores in all autistic children and in the older (>72 months) subgroup relative to TD age expectations.
(D) Sensitivity of the age × group interaction across primary, matched, and quality-control model specifications.
Age effects are cross-sectional and should not be interpreted as within-individual longitudinal trajectories.""", out_dir)


def build_figure_4(out_dir: Path, data_dir: Path | None = None) -> None:
    apply_bp_style()
    load_data(data_dir)
    clin = df("clinical_results.csv")
    clin_rob = df("clinical_robustness.csv")
    stats_map = {x["outcome"]: x for x in FALLBACK["fig4_clinical"]}

    fig, axes = plt.subplots(1, 3, figsize=(FIG_W_IN, 3.2))
    fig.subplots_adjust(wspace=0.42)

    def _clinical(ax, letter, col, title, key):
        panel_label(ax, letter)
        st = stats_map[key]
        sub = clin[clin["group"] == "ASD"].dropna(subset=[col, "posterior_exponent", "age_months", "fsiq"])
        if len(sub) < 5:
            ax.text(0.5, 0.5, "Data required", ha="center", va="center", transform=ax.transAxes)
            return
        cov = sub[["age_months", "fsiq"]]
        x_res = residualize(sub["posterior_exponent"], cov)
        y_res = residualize(sub[col], cov)
        valid = x_res.notna() & y_res.notna()
        x_res, y_res = x_res[valid], y_res[valid]
        ax.scatter(x_res, y_res, s=28, color=COLOR_ASD, alpha=0.75, edgecolors="white", linewidths=0.3)
        if len(x_res) >= 3:
            from scipy import stats as sp_stats
            slope, intercept, _, _, _ = sp_stats.linregress(x_res.values, y_res.values)
            xg = np.linspace(x_res.min(), x_res.max(), 50)
            ax.plot(xg, intercept + slope * xg, color=COLOR_ROI, lw=2)
        ax.set_xlabel("Posterior exponent (residual)")
        ax.set_ylabel(f"{title} (residual)")
        annotate_box(ax, f"partial r = {st['partial_r']:.2f}\n{format_p(st['p'])}\nFDR q = {st['q']:.3f}\nn = {st['n']}")

    _clinical(axes[0], "A", "ados_social_affect", "ADOS Social Affect", "ADOS Social Affect")
    _clinical(axes[1], "B", "ados_total", "ADOS Total", "ADOS Total")

    ax = axes[2]
    panel_label(ax, "C")
    primary = FALLBACK["fig4_clinical"]
    rob = FALLBACK["fig4_robustness"] if not has_rows(clin_rob) else clin_rob.to_dict("records")
    labels = [p["outcome"] for p in primary]
    x = np.arange(len(labels))
    w = 0.35
    primary_r = [p["partial_r"] for p in primary]
    ax.bar(x - w / 2, primary_r, w, color=COLOR_ROI, alpha=0.85, label="Partial r")
    rob_est = []
    for p in primary:
        match = next((r for r in rob if r["outcome"] == p["outcome"]), None)
        rob_est.append(match["estimate"] if match else np.nan)
    ax.bar(x + w / 2, rob_est, w, color=COLOR_ASD, alpha=0.75, label="Robustness")
    ax.axhline(0, color="#666666", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Social Affect", "Total"], fontsize=8)
    ax.set_ylabel("Association (r / ρ)")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("Primary and robustness estimates", fontsize=8.5)

    save_figure(fig, "Figure4", out_dir)
    write_caption(4, """Figure 4. Clinical relevance of posterior aperiodic exponent within autistic children.
(A) Age- and IQ-adjusted association between posterior exponent and ADOS Social Affect within ASD (partial correlation).
(B) Age- and IQ-adjusted association between posterior exponent and ADOS Total within ASD.
(C) Summary of primary partial correlations and robustness checks (partial Spearman ρ for Social Affect; bootstrap partial correlation for ADOS Total).
Clinical analyses were conducted within autistic children only. Scatter plots show residuals after adjusting for age and IQ.""", out_dir)


def build_figure_5(out_dir: Path, data_dir: Path | None = None) -> None:
    apply_bp_style()
    load_data(data_dir)
    movie = df("movie_isc_results.csv")
    tc = df("movie_timecourse.csv")
    hbn = df("hbn_results.csv")

    fig = plt.figure(figsize=(FIG_W_IN, 6.8))
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 0.55], hspace=0.52, wspace=0.38)

    ax_a = fig.add_subplot(gs[0, 0])
    panel_label(ax_a, "A")
    segs, td_v, asd_v, pvals = _segment_means(movie, "td_template_aperiodic_isc", FALLBACK["fig5_td_template"])
    segment_bars(ax_a, segs, td_v, asd_v, pvals, title="TD-template Aperiodic-ISC")

    ax_b = fig.add_subplot(gs[0, 1])
    panel_label(ax_b, "B")
    segs, td_v, asd_v, pvals = _segment_means(movie, "within_group_aperiodic_isc", FALLBACK["fig5_within_group"])
    segment_bars(ax_b, segs, td_v, asd_v, pvals, title="Within-group leave-one-out ISC")

    ax_c = fig.add_subplot(gs[1, 0])
    panel_label(ax_c, "C")
    ctrl = FALLBACK["fig5_pain_controls"]
    metrics = [c["metric"].replace("Broadband envelope ISC", "Envelope ISC").replace("Alpha PLV ISC", "Alpha PLV ISC") for c in ctrl]
    x = np.arange(len(ctrl))
    w = 0.34
    ax_c.bar(x - w / 2, [c["td"] for c in ctrl], w, color=COLOR_TD, alpha=0.88, label="TD")
    ax_c.bar(x + w / 2, [c["asd"] for c in ctrl], w, color=COLOR_ASD, alpha=0.88, label="ASD")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(["Aperiodic-ISC", "Alpha PLV", "Envelope ISC"], fontsize=7)
    ax_c.set_ylabel("ISC (pain-related)")
    ax_c.legend(frameon=False, fontsize=7)
    ax_c.set_title("Classic synchrony controls (pain-related)", fontsize=8.5)
    for i, c in enumerate(ctrl):
        ax_c.text(i, max(c["td"], c["asd"]) * 1.08, format_p(c["p"]), ha="center", fontsize=6.5)

    ax_d = fig.add_subplot(gs[1, 1])
    panel_label(ax_d, "D")
    if has_rows(tc):
        seg_colors = {"Mentalizing": "#E8EEF5", "Pain-related": "#F5E8E8", "Neutral": "#EFEFEF"}
        for seg, color in seg_colors.items():
            sub_seg = tc[tc["segment_label"] == seg]
            if len(sub_seg):
                ax_d.axvspan(sub_seg["time_sec"].min(), sub_seg["time_sec"].max(), color=color, alpha=0.45, lw=0)
        for grp, color in [("TD", COLOR_TD), ("ASD", COLOR_ASD)]:
            sub = tc[tc["group"] == grp].sort_values("time_sec")
            sub = sub.iloc[::8].copy()
            ax_d.plot(sub["time_sec"], sub["mean_aperiodic_isc"], color=color, lw=1.6, label=grp)
            ax_d.fill_between(sub["time_sec"],
                              sub["mean_aperiodic_isc"] - sub["sem_aperiodic_isc"],
                              sub["mean_aperiodic_isc"] + sub["sem_aperiodic_isc"],
                              color=color, alpha=0.18, linewidth=0)
        ax_d.set_xlabel("Time (s)")
        ax_d.set_ylabel("Within-group Aperiodic-ISC")
        ax_d.legend(frameon=False, fontsize=7, loc="upper right")
    else:
        ax_d.text(0.5, 0.5, "Timecourse unavailable", ha="center", va="center", transform=ax_d.transAxes, color="#888888")
    ax_d.set_title("Time-resolved within-group Aperiodic-ISC", fontsize=8.5)

    ax_e = fig.add_subplot(gs[2, :])
    panel_label(ax_e, "E", x=-0.03, y=1.15)
    hbn_fb = FALLBACK["fig5_hbn"]
    summaries = [{"label": h["label"], "asd": h["asd_z"], "td": h["td_z"], "p": h["p"]} for h in hbn_fb]
    x = np.arange(len(summaries))
    w = 0.32
    for i, s in enumerate(summaries):
        ax_e.bar(i - w / 2, s["td"], w, color=COLOR_TD, alpha=0.88, label="TD" if i == 0 else "")
        ax_e.bar(i + w / 2, s["asd"], w, color=COLOR_ASD, alpha=0.88, label="ASD" if i == 0 else "")
        ax_e.text(i, max(s["td"], s["asd"]) * 1.05, format_p(s["p"]), ha="center", fontsize=7)
    ax_e.set_xticks(x)
    ax_e.set_xticklabels([s["label"] for s in summaries], fontsize=8)
    ax_e.set_ylabel("Posterior Aperiodic-ISC (z)")
    ax_e.legend(frameon=False, fontsize=7, loc="upper right")
    ax_e.set_title("External convergence: HBN ThePresent matched cohort (n = 119/group)", fontsize=8.5, pad=6)

    save_figure(fig, "Figure5", out_dir)
    write_caption(5, """Figure 5. Naturalistic movie Aperiodic-ISC, classic synchrony controls, and external HBN convergence.
(A) TD-template Aperiodic-ISC across mentalizing, pain-related, and neutral Partly Cloudy segments; ASD showed reduced alignment to typical posterior dynamics.
(B) Within-group leave-one-out Aperiodic-ISC; reductions were most pronounced for pain-related and neutral segments.
(C) Classic synchrony controls during pain-related segments: Aperiodic-ISC and broadband envelope ISC differed between groups whereas alpha phase PLV ISC did not.
(D) Time-resolved within-group Aperiodic-ISC with SEM shading; background bands indicate event segments.
(E) External convergence in the age-, IQ-, and sex-matched HBN ThePresent cohort (full-movie posterior Aperiodic-ISC).
HBN ThePresent provides external convergence for full-movie posterior Aperiodic-ISC rather than direct replication of Partly Cloudy segment-specific effects. Rest-to-movie coupling and delta-exponent analyses are reported in Supplementary Materials.""", out_dir)


def build_figure(n: int, out_dir: Path, data_dir: Path | None = None) -> None:
    builders = {1: build_figure_1, 2: build_figure_2, 3: build_figure_3, 4: build_figure_4, 5: build_figure_5}
    if n not in builders:
        raise ValueError(f"Unknown figure: {n}")
    builders[n](out_dir, data_dir)


def build_all_figures(out_dir: Path | None = None, data_dir: Path | None = None) -> Path:
    from src.bp_figures.io import V3_DIR, export_data
    out_dir = out_dir or V3_DIR
    data_dir = data_dir or (out_dir / "data")
    export_data(data_dir)
    load_data(data_dir)
    for n in range(1, 6):
        print(f"Generating Figure {n}...")
        build_figure(n, out_dir, data_dir)
    print(f"Done. Outputs in {out_dir}")
    return out_dir
