# ─────────────────────────────────────────────────────────────────
# SA Bank Trust Score — Streamlit Dashboard
# Phase 2 of the SA Bank Trust Score project
#
# This dashboard makes the Trust Score analysis from the Phase 1
# notebook interactive and accessible to general consumers.
#
# Author:  Lindiwe Songelwa
# GitHub:  https://github.com/Lindiwe-22/SA-Bank-Trust-Score
# ─────────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# Must be the first Streamlit call in the script.
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SA Bank Trust Score",
    page_icon="🏦",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# Brand colours and chart styling defined once and reused
# throughout the dashboard for visual consistency.
# ─────────────────────────────────────────────────────────────────
BANK_COLORS = {
    "Standard Bank": "#1565C0",
    "FNB":           "#E65100",
    "Absa":          "#C62828",
    "Nedbank":       "#2E7D32",
    "Capitec":       "#6A1B9A",
    "TymeBank":      "#00838F"
}

# Light theme for all matplotlib charts
CHART_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.labelcolor":  "#333333",
    "xtick.color":      "#333333",
    "ytick.color":      "#333333",
    "text.color":       "#333333",
    "axes.titlecolor":  "#333333",
    "axes.edgecolor":   "#dddddd",
    "grid.color":       "#dddddd",
}
for k, v in CHART_STYLE.items():
    plt.rcParams[k] = v


# ─────────────────────────────────────────────────────────────────
# DATA LOADING
# Cached so data is only read from disk once per session.
# ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    complaints = pd.read_csv("data/complaints.csv")
    sanctions  = pd.read_csv("data/sanctions.csv")
    sentiment  = pd.read_csv("data/sentiment.csv")
    return complaints, sanctions, sentiment


@st.cache_data
def build_scores(complaints, sanctions, sentiment):
    """
    Replicates the Trust Score model from the Phase 1 notebook.
    Normalises each dimension to 0-10 and applies weights.
    Cached so scores are only recalculated when data changes.
    """

    def normalise(series, invert=False):
        mn, mx = series.min(), series.max()
        if mx == mn:
            return pd.Series([5.0] * len(series), index=series.index)
        n = (series - mn) / (mx - mn) * 10
        return (10 - n) if invert else n

    df = complaints[[
        "bank",
        "referral_conversion_rate_pct",
        "cases_decided_consumer_favour_pct",
        "formal_cases_2021",
        "formal_cases_2022",
        "formal_cases_2023"
    ]].copy()

    df = df.merge(
        sentiment[["bank", "dataeq_net_sentiment_pct", "sagaci_satisfaction_2025"]],
        on="bank"
    )

    sanctions_total = (
        sanctions.groupby("bank")["penalty_zar"]
        .sum().reset_index()
        .rename(columns={"penalty_zar": "total_penalty_zar"})
    )
    df = df.merge(sanctions_total, on="bank", how="left").fillna(0)

    df["score_resolution"] = normalise(df["referral_conversion_rate_pct"], invert=True)
    df["score_favour"]     = normalise(df["cases_decided_consumer_favour_pct"])
    df["score_sanctions"]  = normalise(df["total_penalty_zar"], invert=True)
    df["score_sentiment"]  = normalise(
        (df["dataeq_net_sentiment_pct"] + df["sagaci_satisfaction_2025"]) / 2
    )

    df["trust_score"] = (
        df["score_resolution"] * 0.30 +
        df["score_favour"]     * 0.25 +
        df["score_sanctions"]  * 0.25 +
        df["score_sentiment"]  * 0.20
    )

    return df.sort_values("trust_score", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────
def trust_label(score):
    if score >= 7:   return "🟢 HIGH TRUST"
    elif score >= 4: return "🟡 MEDIUM TRUST"
    else:            return "🔴 LOW TRUST"


def trust_color(score):
    if score >= 7:   return "#2E7D32"
    elif score >= 4: return "#F9A825"
    else:            return "#C62828"


def leaderboard_chart(df):
    """
    Horizontal bar chart of all six banks ranked by Trust Score.
    Used in the overview section.
    """
    df_plot = df.sort_values("trust_score", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("white")

    ax.axvspan(0, 4,  alpha=0.06, color="red",    label="Low Trust")
    ax.axvspan(4, 7,  alpha=0.06, color="yellow", label="Medium Trust")
    ax.axvspan(7, 10, alpha=0.06, color="green",  label="High Trust")

    colors = [BANK_COLORS[b] for b in df_plot["bank"]]
    bars   = ax.barh(df_plot["bank"], df_plot["trust_score"],
                     color=colors, edgecolor="#dddddd", height=0.55)

    for bar, val in zip(bars, df_plot["trust_score"]):
        ax.text(val + 0.08, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}/10", va="center", fontsize=11, fontweight="bold",
                color="#333333")

    ax.set_xlim(0, 11.5)
    ax.set_xlabel("Trust Score (out of 10)", fontsize=11, color="#333333")
    ax.set_title("Overall Trust Score Ranking", fontsize=13, pad=12, color="#333333")
    ax.legend(facecolor="white", edgecolor="#dddddd", labelcolor="#333333")
    ax.grid(axis="x", alpha=0.3)

    return fig


def comparison_bar(banks, values, title, xlabel, invert_note=""):
    """
    Simple two-bank comparison bar chart.
    Used in the Compare section for each individual dimension.
    """
    fig, ax = plt.subplots(figsize=(5, 3))
    fig.patch.set_facecolor("#0d1117")

    colors = [BANK_COLORS[b] for b in banks]
    bars   = ax.bar(banks, values, color=colors, edgecolor="#30363d", width=0.45)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.02,
                f"{val:.1f}", ha="center", fontsize=12,
                fontweight="bold", color="white")

    ax.set_title(f"{title}{invert_note}", fontsize=11, pad=10, color="white")
    ax.set_xlabel(xlabel, fontsize=9, color="white")
    ax.set_ylim(0, max(values) * 1.25)
    ax.grid(axis="y", alpha=0.3)

    return fig


# ─────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────
complaints, sanctions, sentiment = load_data()
df = build_scores(complaints, sanctions, sentiment)


# ═════════════════════════════════════════════════════════════════
# SECTION 1 — HEADER
# ═════════════════════════════════════════════════════════════════
st.markdown("""
<h1 style='text-align:center; color:white;'>🏦 South African Bank Trust Score</h1>
<p style='text-align:center; color:#8b949e; font-size:17px;'>
    Helping South African consumers choose a bank based on evidence, not marketing.
</p>
<hr style='border-color:#30363d;'>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
col1.metric("Banks Analysed", "6")
col2.metric("Data Sources", "5 Official Sources")
col3.metric("Years of Data", "2021 – 2025")

st.markdown("<br>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════
# SECTION 2 — BANK RANKINGS WITH STAR SCORES
# ═════════════════════════════════════════════════════════════════
st.markdown("## 🏦 Bank Rankings")
st.caption("Click a bank to see its detailed score breakdown.")

def star_rating(score):
    """Convert a 0-10 score to a 0-5 star display."""
    stars_filled = int(round(score / 2))
    stars_empty  = 5 - stars_filled
    return "★" * stars_filled + "☆" * stars_empty

DIMENSION_DESCRIPTIONS = {
    "Complaint Resolution": (
        "score_resolution",
        "Measures how often a customer complaint is resolved by the bank before it escalates "
        "to a formal case at the Ombudsman. A higher score means the bank handles issues "
        "quickly and fairly at the first point of contact — a sign of a responsive and "
        "customer-friendly institution."
    ),
    "Consumer Favour Rate": (
        "score_favour",
        "Of the complaints that did reach the Ombudsman, this measures how often the ruling "
        "was decided in the consumer's favour rather than the bank's. A higher score reflects "
        "a bank whose internal processes are more likely to be fair to customers."
    ),
    "Regulatory Record": (
        "score_sanctions",
        "Tracks the total financial penalties issued against each bank by the South African "
        "Reserve Bank Prudential Authority for regulatory non-compliance between 2022 and 2025. "
        "A higher score means fewer or no financial penalties — indicating stronger compliance "
        "and governance standards."
    ),
    "Consumer Sentiment": (
        "score_sentiment",
        "Combines social media net sentiment (DataEQ 2024 — 3 million posts analysed) and "
        "independent consumer satisfaction survey results (Sagaci Research 2025). A higher "
        "score means consumers speak more positively about the bank both online and in surveys."
    ),
}

# Initialise session state for selected bank
if "selected_logo_bank" not in st.session_state:
    st.session_state.selected_logo_bank = df["bank"].tolist()[0]

# CSS for vibrate on hover
st.markdown("""
<style>
@keyframes vibrate {
    0%   { transform: translate(0); }
    20%  { transform: translate(-2px, 2px); }
    40%  { transform: translate(-2px, -2px); }
    60%  { transform: translate(2px, 2px); }
    80%  { transform: translate(2px, -2px); }
    100% { transform: translate(0); }
}
.bank-badge:hover {
    animation: vibrate 0.3s linear infinite;
    opacity: 0.85;
    cursor: pointer;
}
.bank-badge {
    border-radius: 10px;
    padding: 14px 6px;
    text-align: center;
    margin-bottom: 6px;
    transition: opacity 0.2s;
}
</style>
""", unsafe_allow_html=True)

# Logo badges as Streamlit buttons styled to look like badges
logo_cols = st.columns(6)
for i, (_, row) in enumerate(df.iterrows()):
    with logo_cols[i]:
        color = BANK_COLORS[row["bank"]]
        st.markdown(f"""
        <div class='bank-badge' style='background:{color};'>
            <div style='font-size:11px; font-weight:bold; color:white; line-height:1.3;'>{row["bank"]}</div>
            <div style='font-size:10px; color:rgba(255,255,255,0.8); margin-top:4px;'>#{i+1}</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button(row["bank"], key=f"badge_btn_{i}", use_container_width=True):
            st.session_state.selected_logo_bank = row["bank"]

selected_logo_bank = st.session_state.selected_logo_bank

logo_row = df[df["bank"] == selected_logo_bank].iloc[0]
lcolor   = BANK_COLORS[selected_logo_bank]

st.markdown(f"""
<div style='
    background:#f8f9fa;
    border:1px solid #e0e0e0;
    border-radius:10px;
    padding:24px;
    margin-top:8px;
'>
    <div style='font-size:20px; font-weight:bold; color:{lcolor}; margin-bottom:4px;'>
        {selected_logo_bank}
    </div>
    <div style='font-size:26px; font-weight:bold; color:#333; margin-bottom:4px;'>
        Overall Trust Score: {logo_row["trust_score"]:.1f}/10
    </div>
    <div style='font-size:18px; color:#f9a825; margin-bottom:20px; letter-spacing:2px;'>
        {star_rating(logo_row["trust_score"])}
    </div>
""", unsafe_allow_html=True)

for dim_name, (field, description) in DIMENSION_DESCRIPTIONS.items():
    score = logo_row[field]
    st.markdown(f"""
    <div style='margin-bottom:18px; padding-bottom:18px; border-bottom:1px solid #e0e0e0;'>
        <div style='font-size:15px; font-weight:bold; color:#333;'>{dim_name}</div>
        <div style='font-size:22px; color:#f9a825; letter-spacing:2px; margin:4px 0;'>
            {star_rating(score)}
            <span style='font-size:14px; color:#555; margin-left:8px;'>{score:.1f} / 10</span>
        </div>
        <div style='font-size:13px; color:#666; margin-top:6px; line-height:1.6;'>
            {description}
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<hr style='border-color:#30363d;'>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════
# SECTION — TRUST vs SENTIMENT SCATTER
# ═════════════════════════════════════════════════════════════════
st.markdown("## 📈 Trust Score vs Consumer Sentiment")
st.caption(
    "Does how consumers feel about a bank reflect its overall trust score? "
    "Each dot represents one of the six banks. The trend line shows the correlation."
)

def trust_sentiment_chart(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x = df["score_sentiment"]
    y = df["trust_score"]

    # Trend line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x.min() - 0.5, x.max() + 0.5, 100)
    ax.plot(x_line, p(x_line), "--", color="#aaaaaa", linewidth=1.5, label="Trend line")

    # Plot each bank
    for _, row in df.iterrows():
        color = BANK_COLORS[row["bank"]]
        ax.scatter(row["score_sentiment"], row["trust_score"],
                   color=color, s=180, zorder=5, edgecolors="white", linewidths=1.5)
        ax.annotate(
            row["bank"],
            xy=(row["score_sentiment"], row["trust_score"]),
            xytext=(8, 4),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
            color=color
        )

    ax.set_xlabel("Consumer Sentiment Score (0–10)", fontsize=12, color="#333333")
    ax.set_ylabel("Overall Trust Score (0–10)", fontsize=12, color="#333333")
    ax.set_title(
        "Trust Score vs Consumer Sentiment — South African Banks\n"
        "Combined DataEQ Social Media Sentiment & Sagaci Satisfaction Survey",
        fontsize=13, pad=15, color="#333333"
    )
    ax.tick_params(colors="#333333")
    ax.spines["bottom"].set_color("#dddddd")
    ax.spines["left"].set_color("#dddddd")
    ax.spines["top"].set_color("#dddddd")
    ax.spines["right"].set_color("#dddddd")
    ax.grid(alpha=0.3, color="#dddddd")
    ax.legend(facecolor="white", edgecolor="#dddddd", labelcolor="#333333")
    ax.set_xlim(x.min() - 1, x.max() + 1.5)
    ax.set_ylim(y.min() - 1, y.max() + 1)

    return fig

st.pyplot(trust_sentiment_chart(df))
st.markdown("<hr style='border-color:#30363d;'>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════
# SECTION 2 — OVERALL LEADERBOARD
# ═════════════════════════════════════════════════════════════════
st.markdown("## 🏆 Overall Trust Score Leaderboard")
st.caption(
    "Weighted across complaint resolution (30%), consumer favour rate (25%), "
    "regulatory sanctions (25%), and consumer sentiment (20%)."
)

# Leaderboard chart
st.pyplot(leaderboard_chart(df))

st.markdown("<br>", unsafe_allow_html=True)

# Score cards — one per bank, ranked
st.markdown("### Score Breakdown")
cols = st.columns(3)
for i, (_, row) in enumerate(df.iterrows()):
    with cols[i % 3]:
        color = BANK_COLORS[row["bank"]]
        score = row["trust_score"]
        st.markdown(f"""
        <div style='
            background:#f8f9fa;
            border:1px solid {color};
            border-radius:10px;
            padding:16px;
            margin-bottom:12px;
            text-align:center;
        '>
            <div style='font-size:13px; color:#888888;'>#{i+1}</div>
            <div style='font-size:18px; font-weight:bold; color:{color};'>{row["bank"]}</div>
            <div style='font-size:32px; font-weight:bold; color:#333333;'>{score:.1f}<span style='font-size:16px;color:#888888;'>/10</span></div>
            <div style='font-size:13px; color:{trust_color(score)};'>{trust_label(score)}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<hr style='border-color:#30363d;'>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════
# SECTION 3 — BANK EXPLORER
# Select a single bank and explore its full breakdown
# ═════════════════════════════════════════════════════════════════
st.markdown("## 🔍 Bank Explorer")
st.caption("Select a bank to see its full Trust Score breakdown and performance radar.")

selected_bank = st.selectbox(
    "Choose a bank:",
    options=df["bank"].tolist(),
    key="explorer_bank"
)

row = df[df["bank"] == selected_bank].iloc[0]
color = BANK_COLORS[selected_bank]

# Bank header
st.markdown(f"""
<div style='
    background:#f8f9fa;
    border-left:5px solid {color};
    padding:16px 20px;
    border-radius:6px;
    margin-bottom:16px;
'>
    <span style='font-size:22px; font-weight:bold; color:{color};'>{selected_bank}</span>
    &nbsp;&nbsp;
    <span style='font-size:20px; color:#333333; font-weight:bold;'>{row["trust_score"]:.1f} / 10</span>
    &nbsp;&nbsp;
    <span style='font-size:15px; color:{trust_color(row["trust_score"])};'>{trust_label(row["trust_score"])}</span>
</div>
""", unsafe_allow_html=True)

# Dimension scores
left = st.container()

with left:
    st.markdown("#### Dimension Scores")

    dimensions = [
        ("Complaint Resolution",  row["score_resolution"],
         f"Referral conversion rate: {row['referral_conversion_rate_pct']}% "
         f"(lower is better — industry avg 52%)"),
        ("Consumer Favour Rate",  row["score_favour"],
         f"{row['cases_decided_consumer_favour_pct']}% of formal cases decided in consumer's favour"),
        ("Regulatory Record",     row["score_sanctions"],
         f"Total penalties: R{row['total_penalty_zar']/1_000_000:.1f}M (2022–2025)"),
        ("Consumer Sentiment",    row["score_sentiment"],
         f"Social media: {row['dataeq_net_sentiment_pct']}% net · "
         f"Survey satisfaction: {row['sagaci_satisfaction_2025']}%"),
    ]

    for dim_name, score, detail in dimensions:
        bar_pct = int(score * 10)
        bar_color = trust_color(score)
        st.markdown(f"""
        <div style='margin-bottom:14px;'>
            <div style='display:flex; justify-content:space-between; margin-bottom:4px;'>
                <span style='color:#333333; font-size:14px;'>{dim_name}</span>
                <span style='color:#333333; font-weight:bold;'>{score:.1f}/10</span>
            </div>
            <div style='background:#e0e0e0; border-radius:4px; height:10px;'>
                <div style='width:{bar_pct}%; background:{bar_color}; height:10px; border-radius:4px;'></div>
            </div>
            <div style='color:#666666; font-size:12px; margin-top:4px;'>{detail}</div>
        </div>
        """, unsafe_allow_html=True)

# Complaint trend
st.markdown("#### Complaint Volume Trend (2021–2023)")
trend_cols = st.columns(3)
trend_cols[0].metric("2021 Cases", f"{int(row['formal_cases_2021']):,}")
trend_cols[1].metric("2022 Cases", f"{int(row['formal_cases_2022']):,}")
trend_cols[2].metric(
    "2023 Cases",
    f"{int(row['formal_cases_2023']):,}",
    delta=f"{int(row['formal_cases_2023']) - int(row['formal_cases_2022']):+,} vs 2022"
)

st.markdown("<hr style='border-color:#30363d;'>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════
# SECTION — OMBUDSMAN CALLOUT
# ═════════════════════════════════════════════════════════════════
st.markdown("""
<div style='
    background:#f0f7ff;
    border:1px solid #90caf9;
    border-left:5px solid #1565C0;
    border-radius:8px;
    padding:20px 24px;
    margin-top:40px;
    margin-bottom:24px;
'>
    <div style='font-size:16px; font-weight:bold; color:#1565C0; margin-bottom:8px;'>
        🏛️ Have a complaint about your bank?
    </div>
    <div style='font-size:14px; color:#333333; line-height:1.7;'>
        The <b>Ombudsman for Banking Services (OBS)</b> is a free, independent service that 
        helps South African consumers resolve disputes with their banks. If your bank has not 
        resolved your complaint satisfactorily, you can escalate it to the OBS at no cost.
    </div>
    <div style='margin-top:12px;'>
        <a href='https://www.obs.org.za' target='_blank' style='
            background:#1565C0;
            color:white;
            padding:8px 18px;
            border-radius:6px;
            text-decoration:none;
            font-size:13px;
            font-weight:bold;
        '>Visit the OBS Website →</a>
    </div>
</div>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════
# SECTION 5 — FOOTER
# ═════════════════════════════════════════════════════════════════
st.markdown("""
<div style='text-align:center; color:#8b949e; font-size:13px; padding:10px;'>
    <b>Data Sources:</b> OBS Annual Report 2023 · NFO Annual Report 2024 ·
    SARB Prudential Authority 2022–2025 · DataEQ SA Banking Index 2024 · Sagaci Research 2025
    <br><br>
    Built by <b>Lindiwe Songelwa</b> ·
    <a href='https://github.com/Lindiwe-22/SA-Bank-Trust-Score' style='color:#58a6ff;'>GitHub</a> ·
    <a href='https://www.linkedin.com/in/lindiwe-songelwa' style='color:#58a6ff;'>LinkedIn</a>
</div>

<div style='
    margin-top:24px;
    background:#161b22;
    border:1px solid #30363d;
    border-radius:8px;
    padding:18px 24px;
    color:#8b949e;
    font-size:12px;
    text-align:left;
'>
    <b style='color:white; font-size:13px;'>📋 Terms of Use</b><br><br>
    The information presented on this dashboard is compiled from publicly available official sources including the 
    Ombudsman for Banking Services (OBS), the National Financial Ombud (NFO), the South African Reserve Bank (SARB) 
    Prudential Authority, DataEQ, and Sagaci Research. It is intended solely for <b>educational and consumer 
    awareness purposes</b>.<br><br>
    This dashboard does not constitute financial advice, and no part of its content should be interpreted as a 
    recommendation to open, close, or transfer any banking product. Trust scores are calculated using a weighted 
    model developed independently by the author and do not represent the official views of any regulator, 
    institution, or data provider.<br><br>
    While every effort has been made to ensure accuracy, the author makes no warranties — express or implied — 
    regarding the completeness or fitness of this information for any particular purpose. The author accepts no 
    liability for decisions made on the basis of this dashboard.<br><br>
    All data, scoring methodologies, visual designs, and written content on this dashboard are the original work 
    of <b>Lindiwe Songelwa</b> © 2025. Reproduction or redistribution without written permission is prohibited.
</div>
""", unsafe_allow_html=True)