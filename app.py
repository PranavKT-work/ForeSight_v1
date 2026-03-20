"""
CardioGuard — Heart Failure Risk Assessment
Feature importance and top-15 selection driven by YOUR SHAP values,
not XGBoost's built-in gain metric.

Required files alongside app.py:
  model.pkl              — trained XGBoost model
  feature_names.json     — ordered list of all 246 feature names
  feature_medians.json   — per-feature median from training set
  shap_summary.csv       — two columns: 'feature' and 'mean_abs_shap'
                           (one row per feature, sorted or unsorted)
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import json

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ForeSight V1 · Heart Failure Readmission Risk",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0d1117;
    color: #c9d1d9;
}
section[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #21262d;
}
.main .block-container { padding: 2rem 2.5rem 3rem; max-width: 1300px; }

.risk-high {
    background: rgba(248,81,73,0.12); border: 1.5px solid #f85149;
    color: #f85149; border-radius: 8px; padding: 1.1rem 1.5rem;
    font-size: 1.5rem; font-weight: 600; text-align: center;
}
.risk-low {
    background: rgba(63,185,80,0.10); border: 1.5px solid #3fb950;
    color: #3fb950; border-radius: 8px; padding: 1.1rem 1.5rem;
    font-size: 1.5rem; font-weight: 600; text-align: center;
}
.prob-number {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 3.2rem; font-weight: 600; line-height: 1;
}
.prob-label {
    font-size: 0.75rem; letter-spacing: 0.1em;
    text-transform: uppercase; color: #8b949e; margin-top: 0.3rem;
}
.section-header {
    font-size: 0.68rem; font-weight: 600; letter-spacing: 0.14em;
    text-transform: uppercase; color: #58a6ff;
    margin: 1.4rem 0 0.6rem; padding-bottom: 0.4rem;
    border-bottom: 1px solid #21262d;
}
.shap-source-badge {
    display: inline-block;
    background: rgba(188,140,255,0.12); border: 1px solid #bc8cff;
    color: #bc8cff; border-radius: 4px; padding: 0.15rem 0.6rem;
    font-size: 0.70rem; font-weight: 600; letter-spacing: 0.08em;
}
.gain-source-badge {
    display: inline-block;
    background: rgba(210,153,34,0.12); border: 1px solid #d29922;
    color: #d29922; border-radius: 4px; padding: 0.15rem 0.6rem;
    font-size: 0.70rem; font-weight: 600; letter-spacing: 0.08em;
}
.ok-badge {
    background: rgba(63,185,80,0.08); border: 1px solid #3fb95040;
    color: #3fb950; border-radius: 6px; padding: 0.5rem 0.9rem;
    font-size: 0.80rem; margin: 0.4rem 0;
}
.missing-badge {
    background: rgba(248,81,73,0.10); border: 1px solid #f8514940;
    color: #f85149; border-radius: 6px; padding: 0.5rem 0.9rem;
    font-size: 0.80rem; margin: 0.4rem 0;
}
.disclaimer {
    background: rgba(187,128,9,0.08); border-left: 3px solid #d29922;
    border-radius: 0 6px 6px 0; padding: 0.75rem 1rem;
    font-size: 0.78rem; color: #b0832a; margin-top: 1rem;
}
.chart-caption {
    font-size: 0.72rem; color: #8b949e; margin-top: 0.3rem; font-style: italic;
}
.stButton button { min-height: 48px !important; font-size: 0.95rem !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model(path="model.pkl"):
    return joblib.load(path) if os.path.exists(path) else None


@st.cache_data
def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


@st.cache_data
def load_shap_summary(path="shap_summary.csv") -> pd.DataFrame | None:
    """
    Load the SHAP summary CSV.
    Expected columns: 'feature', 'mean_abs_shap'
    Rows = your top 20 (or all) features, any order.
    Returns a DataFrame sorted by mean_abs_shap descending.
    """
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    # Normalise column names defensively
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    # Accept common column name variants
    rename_map = {
        "mean_shap": "mean_abs_shap",
        "shap_value": "mean_abs_shap",
        "importance": "mean_abs_shap",
        "mean_absolute_shap": "mean_abs_shap",
        "feat": "feature",
        "feature_name": "feature",
    }
    df = df.rename(columns=rename_map)
    if "feature" not in df.columns or "mean_abs_shap" not in df.columns:
        return None
    df["mean_abs_shap"] = df["mean_abs_shap"].abs()
    return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


# ── Load everything ────────────────────────────────────────────────────────────
model        = load_model("model.pkl")
medians      = load_json("feature_medians.json") or {}
feature_names = load_json("feature_names.json")
shap_df      = load_shap_summary("shap_summary.csv")

if model is None:
    st.error("❌  `model.pkl` not found. Place it alongside `app.py`.")
    st.stop()

if feature_names is None:
    try:
        feature_names = list(model.get_booster().feature_names)
    except Exception:
        st.error("❌  Cannot determine feature names. Provide `feature_names.json`.")
        st.stop()

N_FEATURES = len(feature_names)


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE RANKING — SHAP takes priority over XGBoost gain
# ══════════════════════════════════════════════════════════════════════════════

def get_ranked_features(n: int = 15) -> tuple[list[str], str]:
    """
    Returns (top_n_feature_names, source_label).

    Priority:
      1. shap_summary.csv  — your pre-computed SHAP ranking  ← BEST
      2. model.feature_importances_  — XGBoost gain fallback ← FALLBACK
    """
    if shap_df is not None:
        # Use your SHAP values — filter to only features the model knows
        valid = shap_df[shap_df["feature"].isin(feature_names)]
        top = valid["feature"].tolist()[:n]
        return top, "shap"
    else:
        # Fallback: XGBoost built-in gain importance
        try:
            scores = model.feature_importances_
            idx = np.argsort(scores)[::-1][:n]
            return [feature_names[i] for i in idx], "gain"
        except Exception:
            return feature_names[:n], "gain"


top_features, importance_source = get_ranked_features(n=15)


# ══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def draw_gauge(prob: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4, 2.2), subplot_kw={"aspect": "equal"})
    fig.patch.set_facecolor("#161b22")
    ax.set_facecolor("#161b22")
    theta = np.linspace(np.pi, 0, 300)
    ax.plot(np.cos(theta), np.sin(theta), color="#21262d", linewidth=18, solid_capstyle="round")
    fill_theta = np.linspace(np.pi, np.pi - prob * np.pi, 300)
    color = "#f85149" if prob >= 0.5 else "#3fb950"
    ax.plot(np.cos(fill_theta), np.sin(fill_theta), color=color, linewidth=18, solid_capstyle="round")
    needle_angle = np.pi - prob * np.pi
    ax.annotate("", xy=(0.65 * np.cos(needle_angle), 0.65 * np.sin(needle_angle)),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="#e6edf3", lw=1.8, mutation_scale=12))
    ax.add_patch(plt.Circle((0, 0), 0.07, color="#e6edf3", zorder=5))
    ax.text(-1.05, -0.2, "LOW",  fontsize=7, color="#3fb950", fontfamily="monospace", fontweight="bold", ha="center")
    ax.text( 1.05, -0.2, "HIGH", fontsize=7, color="#f85149", fontfamily="monospace", fontweight="bold", ha="center")
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-0.35, 1.15); ax.axis("off")
    plt.tight_layout(pad=0)
    return fig


def draw_shap_global(top_n: int = 20) -> plt.Figure:
    """
    Bar chart of mean |SHAP| values — the standard SHAP summary bar plot.
    Uses your shap_summary.csv if available, otherwise falls back to XGBoost gain.
    """
    if shap_df is not None:
        plot_df = (shap_df[shap_df["feature"].isin(feature_names)]
                   .head(top_n)
                   .sort_values("mean_abs_shap", ascending=True))
        xlabel = "Mean |SHAP value|  (average impact on model output)"
        title_suffix = "SHAP-based"
    else:
        scores = model.feature_importances_
        feat_imp = pd.DataFrame({"feature": feature_names, "mean_abs_shap": scores})
        plot_df = feat_imp.sort_values("mean_abs_shap", ascending=True).tail(top_n)
        xlabel = "XGBoost Gain Importance  (fallback — SHAP not loaded)"
        title_suffix = "Gain-based (fallback)"

    fig, ax = plt.subplots(figsize=(6, max(3.5, len(plot_df) * 0.32)))
    fig.patch.set_facecolor("#161b22")
    ax.set_facecolor("#161b22")

    # Gradient color — brighter = more important
    norm = plt.Normalize(plot_df["mean_abs_shap"].min(), plot_df["mean_abs_shap"].max())
    cmap = mcolors.LinearSegmentedColormap.from_list("shap", ["#58a6ff", "#bc8cff"])
    colors = [cmap(norm(v)) for v in plot_df["mean_abs_shap"]]

    ax.barh(plot_df["feature"], plot_df["mean_abs_shap"],
            color=colors, height=0.58, edgecolor="none")
    ax.set_xlabel(xlabel, fontsize=7.5, color="#8b949e", labelpad=8)
    ax.tick_params(colors="#8b949e", labelsize=7.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#21262d", linewidth=0.8)
    ax.yaxis.set_tick_params(length=0)
    fig.tight_layout(pad=1.5)
    return fig, title_suffix


def draw_shap_waterfall(input_df: pd.DataFrame, risk_prob: float) -> plt.Figure:
    """
    Patient-level SHAP-style waterfall chart.

    With shap_summary.csv: scales mean |SHAP| by how far this patient's
    value deviates from the training median, giving a directional,
    patient-specific contribution estimate.

    This is an approximation — for exact per-patient SHAP values you would
    call shap.TreeExplainer on the model at prediction time (add 'shap'
    to requirements.txt). The chart below is clearly labelled as an estimate.
    """
    try:
        if shap_df is not None:
            valid = shap_df[shap_df["feature"].isin(feature_names)].head(20)
            feats = valid["feature"].tolist()
            shap_weights = valid["mean_abs_shap"].values
        else:
            scores = model.feature_importances_
            idx = np.argsort(scores)[::-1][:20]
            feats = [feature_names[i] for i in idx]
            shap_weights = scores[idx]

        # Direction: if patient value > median → pushes toward whichever class
        # the feature is positively correlated with (approximated by sign of
        # deviation × overall risk direction)
        contributions = []
        for feat, weight in zip(feats, shap_weights):
            median_val = float(medians.get(feat, 0.0))
            patient_val = float(input_df[feat].iloc[0])
            deviation = patient_val - median_val
            # Normalise deviation to [-1, 1] range using a soft sigmoid
            dev_norm = np.tanh(deviation / (abs(median_val) + 1e-6))
            direction = 1 if risk_prob >= 0.5 else -1
            contributions.append(weight * dev_norm * direction)

        contrib_df = pd.DataFrame({
            "feature": feats,
            "contribution": contributions,
        }).sort_values("contribution")

        fig, ax = plt.subplots(figsize=(6, max(3.5, len(contrib_df) * 0.36)))
        fig.patch.set_facecolor("#161b22")
        ax.set_facecolor("#161b22")

        colors = ["#f85149" if v > 0 else "#3fb950" for v in contrib_df["contribution"]]
        ax.barh(contrib_df["feature"], contrib_df["contribution"],
                color=colors, height=0.58, edgecolor="none")

        ax.axvline(0, color="#8b949e", linewidth=0.8)
        ax.set_xlabel("← Reduces predicted risk  |  Increases predicted risk →",
                      fontsize=7.5, color="#8b949e", labelpad=8)
        ax.tick_params(colors="#8b949e", labelsize=7.5)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color="#21262d", linewidth=0.8)
        ax.yaxis.set_tick_params(length=0)
        fig.tight_layout(pad=1.5)
        return fig
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🫀 ForeSight V1")
    st.markdown(
        f"<div style='font-size:0.78rem;color:#8b949e;margin-top:-0.5rem;margin-bottom:0.5rem;'>"
        f"Clinical Decision Support · <b style='color:#58a6ff;'>{N_FEATURES} features</b></div>",
        unsafe_allow_html=True,
    )

    # SHAP status badge in sidebar
    if shap_df is not None:
        st.markdown(
            f"<div style='margin-bottom:1rem;'>"
            f"<span class='shap-source-badge'>✦ SHAP · {len(shap_df)} features ranked</span></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='margin-bottom:1rem;'>"
            "<span class='gain-source-badge'>⚠ XGBoost Gain (no SHAP file)</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div class='section-header'>Input Mode</div>", unsafe_allow_html=True)
    mode = st.radio(
        "input_mode",
        ["📂  Upload Patient CSV", "✏️  Manual (Top Features)"],
        label_visibility="collapsed",
    )

    upload_df = None
    manual_overrides = {}
    selected_patient_idx = 0

    # ── Mode A: CSV Upload ─────────────────────────────────────────────────────
    if mode == "📂  Upload Patient CSV":
        st.markdown("<div class='section-header'>Patient File</div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload CSV (EHR export)", type=["csv"],
            help="One row per patient. Column headers must match training feature names.",
        )
        if uploaded_file:
            upload_df = pd.read_csv(uploaded_file)
            for drop_col in ["DEATH_EVENT", "death_event", "label", "target", "outcome"]:
                if drop_col in upload_df.columns:
                    upload_df = upload_df.drop(columns=[drop_col])

            n_pts = len(upload_df)
            st.success(f"✔ Loaded **{n_pts}** patient record{'s' if n_pts > 1 else ''}.")
            if n_pts > 1:
                selected_patient_idx = st.selectbox(
                    "Select patient",
                    options=list(range(n_pts)),
                    format_func=lambda i: f"Patient #{i + 1}",
                )
            missing_cols = [f for f in feature_names if f not in upload_df.columns]
            if missing_cols:
                st.markdown(
                    f"<div class='missing-badge'>⚠ {len(missing_cols)} missing → using medians.<br>"
                    f"<span style='font-size:0.72rem;'>{', '.join(missing_cols[:5])}{'...' if len(missing_cols) > 5 else ''}</span></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown("<div class='ok-badge'>✔ All feature columns present.</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-header'>SHAP Top-8 Quick Overrides</div>", unsafe_allow_html=True)
        st.caption("Highest-impact features per SHAP analysis:")
        for feat in top_features[:8]:
            manual_overrides[feat] = st.number_input(
                feat.replace("_", " ").title(),
                value=float(medians.get(feat, 0.0)),
                key=f"ov_{feat}", format="%.4g",
            )

    # ── Mode B: Manual Top-15 ──────────────────────────────────────────────────
    else:
        src_label = "SHAP-ranked" if importance_source == "shap" else "Gain-ranked (fallback)"
        st.markdown(f"<div class='section-header'>Top 15 Features ({src_label})</div>",
                    unsafe_allow_html=True)
        st.caption(f"Remaining {N_FEATURES - 15} features use training-set medians.")

        for feat in top_features:
            median_val = float(medians.get(feat, 0.0))
            if median_val in [0.0, 1.0]:
                manual_overrides[feat] = st.selectbox(
                    feat.replace("_", " ").title(),
                    options=[0, 1], index=int(median_val),
                    format_func=lambda x: "Yes (1)" if x == 1 else "No (0)",
                    key=f"m_{feat}",
                )
            else:
                manual_overrides[feat] = st.number_input(
                    feat.replace("_", " ").title(),
                    value=median_val, key=f"m_{feat}", format="%.4g",
                )

    st.markdown("---")
    st.button("⚡  Run Risk Assessment", use_container_width=True, type="primary")


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE VECTOR BUILD  (3-layer merge)
# ══════════════════════════════════════════════════════════════════════════════
def build_input_vector() -> pd.DataFrame:
    row = {feat: float(medians.get(feat, 0.0)) for feat in feature_names}  # Layer 1
    if upload_df is not None and mode == "📂  Upload Patient CSV":          # Layer 2
        patient_row = upload_df.iloc[selected_patient_idx]
        for feat in feature_names:
            if feat in patient_row.index:
                row[feat] = float(patient_row[feat])
    for feat, val in manual_overrides.items():                              # Layer 3
        row[feat] = float(val)
    return pd.DataFrame([row], columns=feature_names)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PANEL
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## Heart Failure · Risk Assessment")
exp_color = "#bc8cff" if importance_source == "shap" else "#d29922"
exp_label = "SHAP values ✦" if importance_source == "shap" else "XGBoost Gain ⚠"

st.markdown(
    f"<div style='color:#8b949e;font-size:0.85rem;margin-top:-0.8rem;margin-bottom:1.2rem;'>"
    f"Prototype Clinical Decision Support · "
    f"<span style='font-family:monospace;color:#58a6ff;'>{N_FEATURES} features</span> · "
    f"Explainability: <span style='font-family:monospace;color:{exp_color};'>"
    f"{exp_label}</span> · "
    f"Not for clinical use</div>",
    unsafe_allow_html=True,
)

input_df = build_input_vector()
proba = model.predict_proba(input_df)[0]
risk_prob = float(proba[1])
is_high_risk = risk_prob >= 0.5

# ── Row 1: Verdict + Gauge + Probability ──────────────────────────────────────
c1, c2, c3 = st.columns([2.2, 2, 1.2])

with c1:
    st.markdown("<div style='font-size:0.70rem;font-weight:600;letter-spacing:0.12em;text-transform:uppercase;color:#8b949e;margin-bottom:0.6rem;'>Risk Classification</div>", unsafe_allow_html=True)
    if is_high_risk:
        st.markdown("<div class='risk-high'>⚠ HIGH RISK<br><span style='font-size:0.82rem;font-weight:400;'>Mortality risk elevated — clinical review advised</span></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='risk-low'>✔ LOW RISK<br><span style='font-size:0.82rem;font-weight:400;'>Within acceptable range — continue monitoring</span></div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.progress(risk_prob, text=f"Risk Score: {risk_prob:.1%}")

with c2:
    st.markdown("<div style='font-size:0.70rem;font-weight:600;letter-spacing:0.12em;text-transform:uppercase;color:#8b949e;margin-bottom:0.6rem;'>Risk Gauge</div>", unsafe_allow_html=True)
    fig_g = draw_gauge(risk_prob)
    st.pyplot(fig_g, use_container_width=True); plt.close(fig_g)

with c3:
    pcolor = "#f85149" if is_high_risk else "#3fb950"
    st.markdown(f"""
    <div style='text-align:center;padding-top:0.4rem;'>
        <div class='prob-number' style='color:{pcolor};'>{risk_prob:.0%}</div>
        <div class='prob-label'>High-Risk<br>Probability</div>
        <div style='margin-top:1rem;font-size:0.75rem;color:#8b949e;'>
            Low risk<br>
            <span style='color:#3fb950;font-family:monospace;font-weight:600;'>{(1-risk_prob):.0%}</span>
        </div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Row 2: SHAP Global Bar + SHAP Patient Waterfall ───────────────────────────
c_left, c_right = st.columns(2)

with c_left:
    fig_shap, title_sfx = draw_shap_global(top_n=20)
    st.markdown(
        f"<div style='font-size:0.70rem;font-weight:600;letter-spacing:0.12em;text-transform:uppercase;color:#8b949e;margin-bottom:0.2rem;'>"
        f"Global Feature Importance · Top 20</div>"
        f"<div class='chart-caption'>Source: {title_sfx} — mean |SHAP| averaged over training set</div>",
        unsafe_allow_html=True,
    )
    st.pyplot(fig_shap, use_container_width=True); plt.close(fig_shap)

with c_right:
    fig_wf = draw_shap_waterfall(input_df, risk_prob)
    source_note = "SHAP weight × patient deviation from median" if importance_source == "shap" else "Gain weight × deviation (SHAP not loaded)"
    st.markdown(
        f"<div style='font-size:0.70rem;font-weight:600;letter-spacing:0.12em;text-transform:uppercase;color:#8b949e;margin-bottom:0.2rem;'>"
        f"Patient-level Feature Drivers</div>"
        f"<div class='chart-caption'>Estimated directional contributions — {source_note}</div>",
        unsafe_allow_html=True,
    )
    if fig_wf:
        st.pyplot(fig_wf, use_container_width=True); plt.close(fig_wf)

# ── Row 3: SHAP ranking table ─────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("✦  SHAP Feature Ranking Table", expanded=False):
    if shap_df is not None:
        display_shap = shap_df[shap_df["feature"].isin(feature_names)].copy()
        display_shap.index = range(1, len(display_shap) + 1)
        display_shap.index.name = "Rank"
        display_shap.columns = ["Feature", "Mean |SHAP|"]
        st.dataframe(display_shap, use_container_width=True, height=400)
        st.caption("Mean |SHAP| = average absolute impact on model log-odds output across training set. Higher = more influential globally.")
    else:
        st.info("No `shap_summary.csv` found. Add it to see SHAP-based rankings.")

with st.expander("📋  Top 20 Active Feature Values (This Patient)", expanded=False):
    top20 = [shap_df["feature"].iloc[i] for i in range(min(20, len(shap_df)))] if shap_df is not None \
            else feature_names[:20]
    top20 = [f for f in top20 if f in feature_names]
    st.dataframe(
        pd.DataFrame({"Feature": top20,
                      "Patient Value": [round(input_df[f].iloc[0], 4) for f in top20],
                      "Training Median": [round(float(medians.get(f, 0)), 4) for f in top20],
                      "Δ from Median": [round(input_df[f].iloc[0] - float(medians.get(f, 0)), 4) for f in top20],
                      }).set_index("Feature"),
        use_container_width=True, height=400,
    )

with st.expander(f"🗂  All {N_FEATURES} Feature Values", expanded=False):
    full = input_df.T.copy(); full.columns = ["Value"]
    full.index.name = "Feature"; full["Value"] = full["Value"].round(4)
    st.dataframe(full, use_container_width=True, height=500)

# ── Disclaimer ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class='disclaimer'>
⚕ <strong>Clinical Disclaimer:</strong> This tool is a research prototype only.
Patient-level contribution charts are directional estimates based on SHAP weights and deviation
from training medians — not exact per-patient Shapley values. Features not manually entered
are filled from training-set medians. Always apply clinical judgement.
</div>
""", unsafe_allow_html=True)
