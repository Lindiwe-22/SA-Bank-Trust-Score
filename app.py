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

# Dark theme for all matplotlib charts
CHART_STYLE = {
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "axes.labelcolor":  "white",
    "xtick.color":      "white",
    "ytick.color":      "white",
    "text.color":       "white",
    "axes.titlecolor":  "white",
    "axes.edgecolor":   "#30363d",
    "grid.color":       "#30363d",
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


def radar_chart(row, title_color):
    """
    Draw a single radar chart for one bank showing all
    four dimension scores. Returns a matplotlib figure.
    """
    categories = [
        "Complaint\nResolution",
        "Consumer\nFavour",
        "Regulatory\nRecord",
        "Public\nSentiment"
    ]
    N      = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    values = [
        row["score_resolution"],
        row["score_favour"],
        row["score_sanctions"],
        row["score_sentiment"]
    ]
    values += values[:1]

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(projection="polar"))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    ax.plot(angles, values, "o-", linewidth=2, color=title_color)
    ax.fill(angles, values, alpha=0.25, color=title_color)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=8, color="white")
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2", "4", "6", "8", "10"], size=7, color="#8b949e")
    ax.grid(color="#30363d", linewidth=0.5)
    ax.tick_params(colors="white")

    return fig


def leaderboard_chart(df):
    """
    Horizontal bar chart of all six banks ranked by Trust Score.
    Used in the overview section.
    """
    df_plot = df.sort_values("trust_score", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0d1117")

    ax.axvspan(0, 4,  alpha=0.06, color="red",    label="Low Trust")
    ax.axvspan(4, 7,  alpha=0.06, color="yellow", label="Medium Trust")
    ax.axvspan(7, 10, alpha=0.06, color="green",  label="High Trust")

    colors = [BANK_COLORS[b] for b in df_plot["bank"]]
    bars   = ax.barh(df_plot["bank"], df_plot["trust_score"],
                     color=colors, edgecolor="#30363d", height=0.55)

    for bar, val in zip(bars, df_plot["trust_score"]):
        ax.text(val + 0.08, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}/10", va="center", fontsize=11, fontweight="bold",
                color="white")

    ax.set_xlim(0, 11.5)
    ax.set_xlabel("Trust Score (out of 10)", fontsize=11, color="white")
    ax.set_title("Overall Trust Score Ranking", fontsize=13, pad=12, color="white")
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white")
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
            background:#161b22;
            border:1px solid {color};
            border-radius:10px;
            padding:16px;
            margin-bottom:12px;
            text-align:center;
        '>
            <div style='font-size:13px; color:#8b949e;'>#{i+1}</div>
            <div style='font-size:18px; font-weight:bold; color:{color};'>{row["bank"]}</div>
            <div style='font-size:32px; font-weight:bold; color:white;'>{score:.1f}<span style='font-size:16px;color:#8b949e;'>/10</span></div>
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
    background:#161b22;
    border-left:5px solid {color};
    padding:16px 20px;
    border-radius:6px;
    margin-bottom:16px;
'>
    <span style='font-size:22px; font-weight:bold; color:{color};'>{selected_bank}</span>
    &nbsp;&nbsp;
    <span style='font-size:20px; color:white; font-weight:bold;'>{row["trust_score"]:.1f} / 10</span>
    &nbsp;&nbsp;
    <span style='font-size:15px; color:{trust_color(row["trust_score"])};'>{trust_label(row["trust_score"])}</span>
</div>
""", unsafe_allow_html=True)

# Two columns: dimension scores + radar chart
left, right = st.columns([1.2, 1])

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
                <span style='color:white; font-size:14px;'>{dim_name}</span>
                <span style='color:white; font-weight:bold;'>{score:.1f}/10</span>
            </div>
            <div style='background:#30363d; border-radius:4px; height:10px;'>
                <div style='width:{bar_pct}%; background:{bar_color}; height:10px; border-radius:4px;'></div>
            </div>
            <div style='color:#8b949e; font-size:12px; margin-top:4px;'>{detail}</div>
        </div>
        """, unsafe_allow_html=True)

with right:
    st.markdown("#### Performance Radar")
    st.pyplot(radar_chart(row, color))

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
# SECTION 4 — BANK COMPARISON
# Select two banks and compare them side by side
# ═════════════════════════════════════════════════════════════════
st.markdown("## ⚖️ Compare Two Banks")
st.caption("Select any two banks to compare their performance across every dimension.")

bank_list = df["bank"].tolist()

c1, c2 = st.columns(2)
bank_a = c1.selectbox("Bank A:", options=bank_list, index=0, key="compare_a")
bank_b = c2.selectbox("Bank B:", options=bank_list, index=1, key="compare_b")

if bank_a == bank_b:
    st.warning("Please select two different banks to compare.")
else:
    row_a = df[df["bank"] == bank_a].iloc[0]
    row_b = df[df["bank"] == bank_b].iloc[0]

    # Overall score comparison header
    h1, mid, h2 = st.columns([2, 1, 2])
    with h1:
        st.markdown(f"""
        <div style='text-align:center; background:#161b22; border:1px solid {BANK_COLORS[bank_a]};
             border-radius:10px; padding:16px;'>
            <div style='font-size:20px; font-weight:bold; color:{BANK_COLORS[bank_a]};'>{bank_a}</div>
            <div style='font-size:36px; font-weight:bold; color:white;'>{row_a["trust_score"]:.1f}/10</div>
            <div style='color:{trust_color(row_a["trust_score"])};'>{trust_label(row_a["trust_score"])}</div>
        </div>
        """, unsafe_allow_html=True)
    with mid:
        st.markdown("<div style='text-align:center; padding-top:40px; font-size:24px; color:#8b949e;'>vs</div>",
                    unsafe_allow_html=True)
    with h2:
        st.markdown(f"""
        <div style='text-align:center; background:#161b22; border:1px solid {BANK_COLORS[bank_b]};
             border-radius:10px; padding:16px;'>
            <div style='font-size:20px; font-weight:bold; color:{BANK_COLORS[bank_b]};'>{bank_b}</div>
            <div style='font-size:36px; font-weight:bold; color:white;'>{row_b["trust_score"]:.1f}/10</div>
            <div style='color:{trust_color(row_b["trust_score"])};'>{trust_label(row_b["trust_score"])}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Side-by-side dimension comparison charts
    st.markdown("#### Dimension by Dimension")
    dim_comparisons = [
        ("Complaint Resolution Score",  "Score (0–10)", "score_resolution", "\n(higher = fewer escalations)"),
        ("Consumer Favour Score",       "Score (0–10)", "score_favour",     ""),
        ("Regulatory Record Score",     "Score (0–10)", "score_sanctions",  "\n(higher = fewer penalties)"),
        ("Consumer Sentiment Score",    "Score (0–10)", "score_sentiment",  ""),
    ]

    dim_cols = st.columns(2)
    for i, (title, xlabel, field, note) in enumerate(dim_comparisons):
        with dim_cols[i % 2]:
            fig = comparison_bar(
                [bank_a, bank_b],
                [row_a[field], row_b[field]],
                title, xlabel, note
            )
            st.pyplot(fig)

    # Side-by-side radar charts
    st.markdown("#### Performance Radar")
    rad1, rad2 = st.columns(2)
    with rad1:
        st.markdown(f"<p style='text-align:center; color:{BANK_COLORS[bank_a]};'><b>{bank_a}</b></p>",
                    unsafe_allow_html=True)
        st.pyplot(radar_chart(row_a, BANK_COLORS[bank_a]))
    with rad2:
        st.markdown(f"<p style='text-align:center; color:{BANK_COLORS[bank_b]};'><b>{bank_b}</b></p>",
                    unsafe_allow_html=True)
        st.pyplot(radar_chart(row_b, BANK_COLORS[bank_b]))

    # Plain language verdict
    st.markdown("#### Verdict")
    winner = bank_a if row_a["trust_score"] > row_b["trust_score"] else bank_b
    winner_row = row_a if winner == bank_a else row_b
    loser  = bank_b if winner == bank_a else bank_a
    margin = abs(row_a["trust_score"] - row_b["trust_score"])

    if margin < 0.5:
        verdict = f"**{bank_a}** and **{bank_b}** are very closely matched. Either is a reasonable choice — review the dimension breakdown above to decide based on what matters most to you."
    else:
        verdict = (
            f"Based on verified data, **{winner}** scores higher than **{loser}** "
            f"by {margin:.1f} points. "
            f"{winner} performs particularly well on "
        )
        best_dim = max(
            [("complaint resolution", winner_row["score_resolution"]),
             ("consumer favour rate", winner_row["score_favour"]),
             ("regulatory record",    winner_row["score_sanctions"]),
             ("consumer sentiment",   winner_row["score_sentiment"])],
            key=lambda x: x[1]
        )
        verdict += f"**{best_dim[0]}** ({best_dim[1]:.1f}/10)."

    st.info(verdict)


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