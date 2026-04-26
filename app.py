"""
╔═══════════════════════════════════════════════════════════════════╗
║  SURGE SAVVY — Dengue Surge Prediction Dashboard                 ║
║  PS 08 · Disease Prediction using XGBoost + Climate Data          ║
╚═══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_absolute_error,
    r2_score,
)
import warnings

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Surge Savvy — Dengue Prediction",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  CUSTOM CSS — Dark premium theme
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
<style>
/* ── Import Google Font ──────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Root variables ──────────────────────────────────────────────────────── */
:root {
    --bg-primary: #0a0e17;
    --bg-secondary: #111827;
    --bg-card: #1a2035;
    --bg-card-hover: #1f2747;
    --accent-blue: #3b82f6;
    --accent-purple: #8b5cf6;
    --accent-cyan: #06b6d4;
    --accent-emerald: #10b981;
    --accent-amber: #f59e0b;
    --accent-red: #ef4444;
    --accent-pink: #ec4899;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --border: #1e293b;
    --glow-blue: 0 0 20px rgba(59,130,246,0.3);
    --glow-purple: 0 0 20px rgba(139,92,246,0.3);
}

/* ── Global ──────────────────────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif !important;
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0a0e17 0%, #0f172a 50%, #0a0e17 100%) !important;
}

/* ── Header ──────────────────────────────────────────────────────────────── */
[data-testid="stHeader"] {
    background: rgba(10, 14, 23, 0.8) !important;
    backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--border);
}

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #111827 100%) !important;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--text-primary) !important;
}

[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li,
[data-testid="stSidebar"] label {
    color: var(--text-secondary) !important;
}

/* ── Metrics ─────────────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px 24px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}

[data-testid="stMetric"]:hover {
    border-color: var(--accent-blue);
    box-shadow: var(--glow-blue);
    transform: translateY(-2px);
}

[data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

[data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-weight: 700 !important;
    font-size: 1.8rem !important;
}

/* ── Cards / Expanders ───────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    overflow: hidden;
}

[data-testid="stExpander"] summary {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
}

/* ── Tabs ─────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: var(--bg-secondary);
    border-radius: 12px;
    padding: 4px;
    border: 1px solid var(--border);
}

.stTabs [data-baseweb="tab"] {
    border-radius: 10px !important;
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    padding: 10px 20px !important;
    transition: all 0.2s ease;
}

.stTabs [data-baseweb="tab"]:hover {
    color: var(--text-primary) !important;
    background: rgba(59,130,246,0.1);
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple)) !important;
    color: white !important;
    font-weight: 600 !important;
}

.stTabs [data-baseweb="tab-highlight"] {
    display: none;
}

.stTabs [data-baseweb="tab-border"] {
    display: none;
}

/* ── Buttons ──────────────────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple)) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 28px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(59,130,246,0.3);
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 25px rgba(59,130,246,0.5) !important;
}

/* ── Sliders ──────────────────────────────────────────────────────────────── */
.stSlider label {
    color: var(--text-secondary) !important;
}

.stSlider [data-baseweb="slider"] div {
    background: var(--accent-blue) !important;
}

/* ── Select boxes ─────────────────────────────────────────────────────────── */
.stSelectbox label, .stNumberInput label {
    color: var(--text-secondary) !important;
}

/* ── Dataframe ────────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
}

/* ── Markdown headings ────────────────────────────────────────────────────── */
h1, h2, h3 {
    color: var(--text-primary) !important;
}

p, li {
    color: var(--text-secondary) !important;
}

/* ── Hero Title ───────────────────────────────────────────────────────────── */
.hero-title {
    font-size: 3rem;
    font-weight: 900;
    background: linear-gradient(135deg, #3b82f6, #8b5cf6, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0;
    line-height: 1.1;
}

.hero-sub {
    font-size: 1.1rem;
    color: var(--text-muted);
    margin-top: 8px;
    font-weight: 400;
}

/* ── Stat Cards ───────────────────────────────────────────────────────────── */
.stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.stat-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
}

.stat-card:hover {
    border-color: var(--accent-blue);
    box-shadow: var(--glow-blue);
    transform: translateY(-3px);
}

.stat-value {
    font-size: 2rem;
    font-weight: 800;
    color: var(--text-primary);
    margin-bottom: 4px;
}

.stat-label {
    font-size: 0.8rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 500;
}

/* ── Risk badges ──────────────────────────────────────────────────────────── */
.risk-low { background: linear-gradient(135deg, #065f46, #10b981); color: white; padding: 6px 16px; border-radius: 999px; font-weight: 600; font-size: 0.85rem; display: inline-block; }
.risk-medium { background: linear-gradient(135deg, #92400e, #f59e0b); color: white; padding: 6px 16px; border-radius: 999px; font-weight: 600; font-size: 0.85rem; display: inline-block; }
.risk-high { background: linear-gradient(135deg, #9a3412, #f97316); color: white; padding: 6px 16px; border-radius: 999px; font-weight: 600; font-size: 0.85rem; display: inline-block; }
.risk-critical { background: linear-gradient(135deg, #991b1b, #ef4444); color: white; padding: 6px 16px; border-radius: 999px; font-weight: 600; font-size: 0.85rem; display: inline-block; animation: pulse 2s ease-in-out infinite; }

@keyframes pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.4); }
    50% { box-shadow: 0 0 20px 5px rgba(239,68,68,0.2); }
}

/* ── Divider ──────────────────────────────────────────────────────────────── */
.gradient-divider {
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent-blue), var(--accent-purple), transparent);
    margin: 24px 0;
    border: none;
}

/* ── Footer ───────────────────────────────────────────────────────────────── */
.footer {
    text-align: center;
    padding: 30px 0;
    color: var(--text-muted);
    font-size: 0.8rem;
    border-top: 1px solid var(--border);
    margin-top: 40px;
}
</style>
""",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(26,32,53,0.6)",
    font=dict(family="Inter", color="#94a3b8"),
    title_font=dict(color="#f1f5f9", size=16),
    xaxis=dict(gridcolor="rgba(30,41,59,0.8)", zerolinecolor="rgba(30,41,59,0.8)"),
    yaxis=dict(gridcolor="rgba(30,41,59,0.8)", zerolinecolor="rgba(30,41,59,0.8)"),
    margin=dict(l=40, r=40, t=60, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
)


def classify_risk(x):
    """Classify dengue case count into risk levels."""
    if x < 3000:
        return "Low"
    elif x < 6000:
        return "Medium"
    elif x < 10000:
        return "High"
    else:
        return "Critical"


def risk_color(level):
    return {
        "Low": "#10b981",
        "Medium": "#f59e0b",
        "High": "#f97316",
        "Critical": "#ef4444",
    }.get(level, "#94a3b8")


def risk_badge(level):
    css_class = f"risk-{level.lower()}"
    return f'<span class="{css_class}">{level}</span>'


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING & MODEL TRAINING (cached)
# ══════════════════════════════════════════════════════════════════════════════


@st.cache_data
def load_data():
    """Load and return the raw dengue dataset."""
    import os
    if os.path.exists("dataset/final.csv"):
        df = pd.read_csv("dataset/final.csv")
    else:
        df = pd.read_csv("final.csv")
    return df


@st.cache_resource
def train_model(df):
    """Preprocess data, engineer features, train XGBoost, return everything."""
    # ── Preprocessing ────────────────────────────────────────────────────
    drop_cols = ["serial", "conditions", "stations", "labels", "snow", "snowdepth"]
    clean = df.drop(columns=[c for c in drop_cols if c in df.columns])
    clean = clean.ffill().fillna(clean.median(numeric_only=True))

    # ── Feature Engineering ──────────────────────────────────────────────
    clean["lag_1_cases"] = clean["cases"].shift(1)
    clean["lag_2_cases"] = clean["cases"].shift(2)
    clean["roll7_cases"] = clean["cases"].rolling(7).mean()
    clean["roll7_temp"] = clean["temp"].rolling(7).mean()
    clean["roll7_humid"] = clean["humidity"].rolling(7).mean()
    clean["climate_risk_idx"] = (
        0.35 * clean["humidity"]
        + 0.30 * clean["temp"]
        + 0.20 * clean["precip"]
        + 0.15 * clean["uvindex"]
    )
    clean["temp_range"] = clean["tempmax"] - clean["tempmin"]
    clean = clean.dropna().reset_index(drop=True)

    # ── Split ────────────────────────────────────────────────────────────
    feature_cols = [c for c in clean.columns if c != "cases"]
    X = clean[feature_cols]
    y = clean["cases"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    # ── Train ────────────────────────────────────────────────────────────
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # ── Predict ──────────────────────────────────────────────────────────
    predictions = np.clip(model.predict(X_test), 0, None)
    mape = mean_absolute_percentage_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # ── Feature importance ───────────────────────────────────────────────
    importance_df = (
        pd.DataFrame({"Feature": feature_cols, "Importance": model.feature_importances_})
        .sort_values("Importance", ascending=False)
        .head(14)
    )

    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "predictions": predictions,
        "feature_cols": feature_cols,
        "clean": clean,
        "mape": mape,
        "mae": mae,
        "r2": r2,
        "importance_df": importance_df,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Load data ────────────────────────────────────────────────────────
    raw = load_data()
    res = train_model(raw)

    # ── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🦟 Surge Savvy")
        st.markdown("---")
        st.markdown(
            """
**Disease Prediction System**  
Powered by XGBoost & Climate Data

---
**Dataset Info**
"""
        )
        st.markdown(f"- **Records:** {len(raw):,}")
        st.markdown(f"- **Features:** {raw.shape[1]}")
        st.markdown(f"- **Target:** Dengue Cases")
        st.markdown("---")
        st.markdown(
            """
**Risk Levels**
- 🟢 **Low** — < 3,000 cases
- 🟡 **Medium** — 3,000–6,000
- 🟠 **High** — 6,000–10,000
- 🔴 **Critical** — > 10,000
"""
        )
        st.markdown("---")
        st.markdown(
            '<p style="color:#64748b;font-size:0.75rem;">Kaggle Dataset: siddhvr/dengue-prediction</p>',
            unsafe_allow_html=True,
        )

    # ── Hero Header ──────────────────────────────────────────────────────
    st.markdown(
        """
<div style="text-align:center; padding: 20px 0 10px 0;">
    <div class="hero-title">Surge Savvy</div>
    <div class="hero-sub">AI-Powered Dengue Surge Prediction &amp; Climate Risk Analysis</div>
</div>
<div class="gradient-divider"></div>
""",
        unsafe_allow_html=True,
    )

    # ── KPI Row ──────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f"""<div class="stat-card">
            <div class="stat-value">{(1 - res['mape']) * 100:.1f}%</div>
            <div class="stat-label">Model Accuracy</div>
        </div>""",
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""<div class="stat-card">
            <div class="stat-value">{res['r2']:.3f}</div>
            <div class="stat-label">R² Score</div>
        </div>""",
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"""<div class="stat-card">
            <div class="stat-value">{res['mae']:.0f}</div>
            <div class="stat-label">MAE (cases)</div>
        </div>""",
            unsafe_allow_html=True,
        )
    with k4:
        avg_cases = raw["cases"].mean()
        st.markdown(
            f"""<div class="stat-card">
            <div class="stat-value">{avg_cases:,.0f}</div>
            <div class="stat-label">Avg Dengue Cases</div>
        </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ─────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 Dashboard", "🔍 EDA", "🎯 Predictions", "🧠 Model Insights", "⚡ Live Predictor"]
    )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  TAB 1: DASHBOARD
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab1:
        st.markdown("### Actual vs Predicted Dengue Cases")

        y_test = res["y_test"]
        preds = res["predictions"]

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Time-Series Comparison", "Scatter — Prediction Accuracy"),
            horizontal_spacing=0.08,
        )

        fig.add_trace(
            go.Scatter(
                x=list(range(len(y_test))),
                y=y_test.values,
                name="Actual",
                mode="lines+markers",
                line=dict(color="#3b82f6", width=2),
                marker=dict(size=5),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=list(range(len(preds))),
                y=preds,
                name="Predicted",
                mode="lines+markers",
                line=dict(color="#f97316", width=2, dash="dash"),
                marker=dict(size=5, symbol="x"),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=y_test.values,
                y=preds,
                mode="markers",
                name="Predictions",
                marker=dict(color="#8b5cf6", size=8, opacity=0.6, line=dict(width=1, color="white")),
            ),
            row=1,
            col=2,
        )
        lim = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
        fig.add_trace(
            go.Scatter(
                x=lim,
                y=lim,
                mode="lines",
                name="Perfect Fit",
                line=dict(color="#ef4444", dash="dash", width=1.5),
            ),
            row=1,
            col=2,
        )

        fig.update_layout(**PLOTLY_LAYOUT, height=420, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

        # ── Risk distribution ────────────────────────────────────────────
        st.markdown("### Risk Level Distribution")
        risks = [classify_risk(p) for p in preds]
        risk_counts = pd.Series(risks).value_counts()
        order = ["Low", "Medium", "High", "Critical"]
        risk_counts = risk_counts.reindex([r for r in order if r in risk_counts.index], fill_value=0)

        colors_map = {"Low": "#10b981", "Medium": "#f59e0b", "High": "#f97316", "Critical": "#ef4444"}

        col_left, col_right = st.columns([1, 1])
        with col_left:
            fig_bar = go.Figure(
                go.Bar(
                    x=risk_counts.index,
                    y=risk_counts.values,
                    marker_color=[colors_map[r] for r in risk_counts.index],
                    text=risk_counts.values,
                    textposition="outside",
                    textfont=dict(size=14, color="#f1f5f9"),
                )
            )
            fig_bar.update_layout(**PLOTLY_LAYOUT, height=350, title="Prediction Risk Breakdown")
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_right:
            fig_pie = go.Figure(
                go.Pie(
                    labels=risk_counts.index,
                    values=risk_counts.values,
                    marker=dict(colors=[colors_map[r] for r in risk_counts.index]),
                    hole=0.55,
                    textinfo="percent+label",
                    textfont=dict(size=12),
                )
            )
            fig_pie.update_layout(**PLOTLY_LAYOUT, height=350, title="Risk Share (%)")
            st.plotly_chart(fig_pie, use_container_width=True)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  TAB 2: EDA
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab2:
        st.markdown("### Exploratory Data Analysis")

        eda_col1, eda_col2 = st.columns(2)

        with eda_col1:
            # Distribution of cases
            fig_dist = go.Figure(
                go.Histogram(
                    x=raw["cases"],
                    nbinsx=30,
                    marker_color="#3b82f6",
                    marker_line_color="#1e293b",
                    marker_line_width=1,
                    opacity=0.85,
                )
            )
            fig_dist.update_layout(**PLOTLY_LAYOUT, height=350, title="Distribution of Dengue Cases")
            st.plotly_chart(fig_dist, use_container_width=True)

        with eda_col2:
            # Labels distribution
            if "labels" in raw.columns:
                label_counts = raw["labels"].value_counts()
                fig_labels = go.Figure(
                    go.Bar(
                        x=label_counts.index,
                        y=label_counts.values,
                        marker_color=["#10b981", "#f59e0b", "#ef4444"][: len(label_counts)],
                        text=label_counts.values,
                        textposition="outside",
                        textfont=dict(color="#f1f5f9"),
                    )
                )
                fig_labels.update_layout(**PLOTLY_LAYOUT, height=350, title="Original Risk Label Distribution")
                st.plotly_chart(fig_labels, use_container_width=True)

        # ── Correlation Heatmap ──────────────────────────────────────────
        st.markdown("### Feature Correlation Heatmap")
        num_cols = raw.select_dtypes(include=[np.number]).drop(columns=["serial"], errors="ignore")
        corr = num_cols.corr()

        fig_heatmap = go.Figure(
            go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                colorscale="RdBu_r",
                zmid=0,
                text=np.round(corr.values, 2),
                texttemplate="%{text}",
                textfont=dict(size=8),
            )
        )
        fig_heatmap.update_layout(**PLOTLY_LAYOUT, height=550, title="Correlation Matrix")
        st.plotly_chart(fig_heatmap, use_container_width=True)

        # ── Scatter: temp vs cases ───────────────────────────────────────
        st.markdown("### Climate vs Dengue Cases")
        scatter_col1, scatter_col2, scatter_col3 = st.columns(3)

        with scatter_col1:
            fig_s1 = px.scatter(
                raw,
                x="temp",
                y="cases",
                color="cases",
                color_continuous_scale="turbo",
                title="Temperature vs Cases",
                opacity=0.6,
            )
            fig_s1.update_layout(**PLOTLY_LAYOUT, height=320)
            st.plotly_chart(fig_s1, use_container_width=True)

        with scatter_col2:
            fig_s2 = px.scatter(
                raw,
                x="humidity",
                y="cases",
                color="cases",
                color_continuous_scale="turbo",
                title="Humidity vs Cases",
                opacity=0.6,
            )
            fig_s2.update_layout(**PLOTLY_LAYOUT, height=320)
            st.plotly_chart(fig_s2, use_container_width=True)

        with scatter_col3:
            fig_s3 = px.scatter(
                raw,
                x="precip",
                y="cases",
                color="cases",
                color_continuous_scale="turbo",
                title="Precipitation vs Cases",
                opacity=0.6,
            )
            fig_s3.update_layout(**PLOTLY_LAYOUT, height=320)
            st.plotly_chart(fig_s3, use_container_width=True)

        # ── Raw data preview ─────────────────────────────────────────────
        with st.expander("📋 View Raw Dataset"):
            st.dataframe(raw, use_container_width=True, height=300)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  TAB 3: PREDICTIONS TABLE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab3:
        st.markdown("### Prediction Results")

        results_df = pd.DataFrame(
            {
                "Actual Cases": y_test.values.astype(int),
                "Predicted Cases": preds.round(0).astype(int),
                "Difference": (preds - y_test.values).round(0).astype(int),
                "Error %": ((abs(preds - y_test.values) / y_test.values) * 100).round(1),
                "Risk Level": [classify_risk(p) for p in preds],
            }
        )

        st.dataframe(
            results_df.style.map(
                lambda x: (
                    "color: #10b981"
                    if x == "Low"
                    else (
                        "color: #f59e0b"
                        if x == "Medium"
                        else "color: #f97316" if x == "High" else "color: #ef4444" if x == "Critical" else ""
                    )
                ),
                subset=["Risk Level"],
            ),
            use_container_width=True,
            height=500,
        )

        st.markdown("### Risk Classification Summary")
        risk_html = ""
        for i, (_, row) in enumerate(results_df.iterrows()):
            risk_html += f'<span style="margin-right:8px;">{risk_badge(row["Risk Level"])}</span>'
            if (i + 1) % 10 == 0:
                risk_html += "<br><br>"

        st.markdown(risk_html, unsafe_allow_html=True)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  TAB 4: MODEL INSIGHTS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab4:
        st.markdown("### Feature Importance")

        imp = res["importance_df"]

        fig_imp = go.Figure(
            go.Bar(
                x=imp["Importance"],
                y=imp["Feature"],
                orientation="h",
                marker=dict(
                    color=imp["Importance"],
                    colorscale="Viridis",
                    line=dict(width=0),
                ),
                text=imp["Importance"].round(4),
                textposition="outside",
                textfont=dict(size=11, color="#f1f5f9"),
            )
        )
        fig_imp.update_layout(
            **PLOTLY_LAYOUT,
            height=480,
            title="Top Features Driving Dengue Prediction",
            yaxis=dict(autorange="reversed", gridcolor="rgba(30,41,59,0.8)"),
        )
        st.plotly_chart(fig_imp, use_container_width=True)

        # ── Model summary ────────────────────────────────────────────────
        st.markdown("### Model Configuration")
        config_data = {
            "Parameter": [
                "Algorithm",
                "Estimators",
                "Learning Rate",
                "Max Depth",
                "Subsample",
                "Col Sample by Tree",
                "Min Child Weight",
                "Gamma",
                "Train Samples",
                "Test Samples",
            ],
            "Value": [
                "XGBoost Regressor",
                300,
                0.05,
                6,
                0.85,
                0.8,
                3,
                0.1,
                len(res["X_train"]),
                len(res["X_test"]),
            ],
        }
        st.dataframe(pd.DataFrame(config_data), use_container_width=True, hide_index=True)

        # ── Residual plot ────────────────────────────────────────────────
        st.markdown("### Residual Analysis")
        residuals = y_test.values - preds
        fig_resid = go.Figure(
            go.Scatter(
                x=preds,
                y=residuals,
                mode="markers",
                marker=dict(color="#8b5cf6", size=7, opacity=0.6, line=dict(width=1, color="white")),
            )
        )
        fig_resid.add_hline(y=0, line_dash="dash", line_color="#ef4444", line_width=1.5)
        fig_resid.update_layout(
            **PLOTLY_LAYOUT,
            height=350,
            title="Residuals (Actual − Predicted)",
            xaxis_title="Predicted Cases",
            yaxis_title="Residual",
        )
        st.plotly_chart(fig_resid, use_container_width=True)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  TAB 5: LIVE PREDICTOR
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab5:
        st.markdown("### Real-Time Dengue Case Predictor")
        st.markdown(
            '<p style="color:#64748b;">Adjust climate parameters to predict dengue case count and risk level.</p>',
            unsafe_allow_html=True,
        )

        inp_col1, inp_col2, inp_col3 = st.columns(3)

        with inp_col1:
            temp_val = st.slider("🌡 Temperature (°C)", 20.0, 45.0, 33.0, 0.5)
            tempmax_val = st.slider("🌡 Max Temp (°C)", 25.0, 48.0, 37.0, 0.5)
            tempmin_val = st.slider("🌡 Min Temp (°C)", 15.0, 35.0, 25.0, 0.5)
            feelslike_val = st.slider("🤒 Feels Like (°C)", 20.0, 50.0, 35.0, 0.5)
            feelslikemax_val = st.slider("🤒 Feels Like Max", 25.0, 55.0, 40.0, 0.5)
            feelslikemin_val = st.slider("🤒 Feels Like Min", 18.0, 35.0, 26.0, 0.5)

        with inp_col2:
            humidity_val = st.slider("💧 Humidity (%)", 30.0, 100.0, 75.0, 1.0)
            precip_val = st.slider("🌧 Precipitation (mm)", 0.0, 150.0, 20.0, 1.0)
            precipprob_val = st.slider("🌧 Precip Probability (%)", 0.0, 100.0, 45.0, 1.0)
            precipcover_val = st.slider("🌧 Precip Cover", 0.0, 30.0, 3.0, 0.5)
            dew_val = st.slider("💨 Dew Point (°C)", 10.0, 30.0, 22.0, 0.5)
            windspeed_val = st.slider("🌬 Wind Speed (km/h)", 0.0, 50.0, 15.0, 0.5)

        with inp_col3:
            winddir_val = st.slider("🧭 Wind Direction (°)", 0.0, 360.0, 180.0, 5.0)
            pressure_val = st.slider("📊 Sea Level Pressure", 995.0, 1025.0, 1008.0, 0.5)
            cloud_val = st.slider("☁ Cloud Cover (%)", 0.0, 100.0, 50.0, 1.0)
            solar_val = st.slider("☀ Solar Radiation", 50.0, 350.0, 210.0, 5.0)
            solarenergy_val = st.slider("⚡ Solar Energy", 5.0, 30.0, 18.0, 0.5)
            visibility_val = st.slider("👁 Visibility (km)", 1.0, 25.0, 4.0, 0.5)
            uvindex_val = st.slider("🔆 UV Index", 1.0, 11.0, 7.0, 0.5)

        if st.button("🚀 Predict Dengue Risk", use_container_width=True):
            # Build input matching feature_cols
            input_dict = {
                "tempmax": tempmax_val,
                "tempmin": tempmin_val,
                "temp": temp_val,
                "feelslikemax": feelslikemax_val,
                "feelslikemin": feelslikemin_val,
                "feelslike": feelslike_val,
                "dew": dew_val,
                "humidity": humidity_val,
                "precip": precip_val,
                "precipprob": precipprob_val,
                "precipcover": precipcover_val,
                "windspeed": windspeed_val,
                "winddir": winddir_val,
                "sealevelpressure": pressure_val,
                "cloudcover": cloud_val,
                "visibility": visibility_val,
                "solarradiation": solar_val,
                "solarenergy": solarenergy_val,
                "uvindex": uvindex_val,
            }

            # Engineered features — use reasonable defaults
            avg_cases = raw["cases"].mean()
            input_dict["lag_1_cases"] = avg_cases
            input_dict["lag_2_cases"] = avg_cases
            input_dict["roll7_cases"] = avg_cases
            input_dict["roll7_temp"] = temp_val
            input_dict["roll7_humid"] = humidity_val
            input_dict["climate_risk_idx"] = (
                0.35 * humidity_val + 0.30 * temp_val + 0.20 * precip_val + 0.15 * uvindex_val
            )
            input_dict["temp_range"] = tempmax_val - tempmin_val

            # Ensure column order matches training
            input_df = pd.DataFrame([input_dict])[res["feature_cols"]]
            prediction_val = max(0, float(res["model"].predict(input_df)[0]))
            risk_val = classify_risk(prediction_val)

            st.markdown("<br>", unsafe_allow_html=True)

            # Result cards
            r1, r2, r3 = st.columns(3)
            with r1:
                st.markdown(
                    f"""<div class="stat-card">
                    <div class="stat-value" style="color:#3b82f6;">{prediction_val:,.0f}</div>
                    <div class="stat-label">Predicted Cases</div>
                </div>""",
                    unsafe_allow_html=True,
                )
            with r2:
                st.markdown(
                    f"""<div class="stat-card">
                    <div class="stat-value" style="color:{risk_color(risk_val)};">{risk_val}</div>
                    <div class="stat-label">Risk Level</div>
                </div>""",
                    unsafe_allow_html=True,
                )
            with r3:
                cri = 0.35 * humidity_val + 0.30 * temp_val + 0.20 * precip_val + 0.15 * uvindex_val
                st.markdown(
                    f"""<div class="stat-card">
                    <div class="stat-value" style="color:#8b5cf6;">{cri:.1f}</div>
                    <div class="stat-label">Climate Risk Index</div>
                </div>""",
                    unsafe_allow_html=True,
                )

            # Risk gauge
            st.markdown("<br>", unsafe_allow_html=True)
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=prediction_val,
                    title={"text": "Dengue Case Prediction", "font": {"size": 16, "color": "#f1f5f9"}},
                    number={"font": {"size": 36, "color": "#f1f5f9"}},
                    gauge=dict(
                        axis=dict(range=[0, 15000], tickcolor="#64748b"),
                        bar=dict(color=risk_color(risk_val)),
                        bgcolor="rgba(26,32,53,0.6)",
                        bordercolor="#1e293b",
                        steps=[
                            dict(range=[0, 3000], color="rgba(16,185,129,0.2)"),
                            dict(range=[3000, 6000], color="rgba(245,158,11,0.2)"),
                            dict(range=[6000, 10000], color="rgba(249,115,22,0.2)"),
                            dict(range=[10000, 15000], color="rgba(239,68,68,0.2)"),
                        ],
                        threshold=dict(line=dict(color="#ef4444", width=3), thickness=0.8, value=prediction_val),
                    ),
                )
            )
            fig_gauge.update_layout(**PLOTLY_LAYOUT, height=320)
            st.plotly_chart(fig_gauge, use_container_width=True)

    # ── Footer ───────────────────────────────────────────────────────────
    st.markdown(
        """
<div class="footer">
    <strong>Surge Savvy</strong> — Disease Prediction System (PS 08)<br>
    Powered by XGBoost · Streamlit · Plotly<br>
    Dataset: <a href="https://www.kaggle.com/datasets/siddhvr/dengue-prediction" style="color:#3b82f6;">Kaggle — Dengue Prediction</a>
</div>
""",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
