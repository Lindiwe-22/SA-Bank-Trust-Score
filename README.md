# 🏦 SA Bank Trust Score — Consumer Protection Intelligence

> A data-driven trust scoring system that helps South African consumers choose a bank based on verified evidence, not marketing.

---

## 🔗 Demo Link

👉 [View Live Streamlit Dashboard](https://sa-bank-trust-score.streamlit.app/)

---

## 📋 Table of Contents

- [Business Understanding](#business-understanding)
- [Technologies](#technologies)
- [Setup](#setup)
- [Approach](#approach)
- [Status](#status)
- [Credits](#credits)

---

## 💼 Business Understanding

South African consumers are increasingly exposed to banking failures — unresolved complaints, regulatory violations, fraud, and poor service. Yet most consumers choose a bank based on advertising, not evidence.

This project changes that.

Using verified data from official regulatory bodies, this system builds a transparent **Trust Score** for each of South Africa's major retail banks — giving consumers, journalists, and researchers a data-driven basis for one of the most important financial decisions they will make.

Banks are scored across four verified dimensions:

| Dimension | What It Measures |
|---|---|
| 📋 **Complaint Resolution** | How quickly banks resolve complaints before Ombudsman escalation |
| ⚖️ **Consumer Favour Rate** | % of formal cases decided in the consumer's favour |
| 🔒 **Regulatory Record** | Total financial penalties from the SARB Prudential Authority |
| 💬 **Consumer Sentiment** | Social media net sentiment + consumer satisfaction surveys |

Each bank receives a final Trust Score out of 10:

| Score | Rating |
|---|---|
| 🟢 **7–10** | High Trust — strong track record across all dimensions |
| 🟡 **4–7** | Medium Trust — mixed performance |
| 🔴 **0–4** | Low Trust — significant concerns across multiple dimensions |

---

## 🛠️ Technologies

**Languages & Environment**

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

**Data & Analysis**

![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge&logo=python&logoColor=white)

**Deployment**

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

| Category | Tools |
|---|---|
| **Data & Analysis** | Python, Pandas, NumPy, Matplotlib, Seaborn |
| **Notebook** | Jupyter |
| **Dashboard** | Streamlit Cloud |

---

## ⚙️ Setup
```bash
# Clone the repository
git clone https://github.com/Lindiwe-22/SA-Bank-Trust-Score.git
cd SA-Bank-Trust-Score

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit dashboard locally
streamlit run app.py

# Or open the analysis notebook
jupyter notebook notebooks/bank_trust_score.ipynb
```

---

## 🔍 Approach

### Phase 1 — Data Collection & Verification
All data was sourced exclusively from verified, publicly available official sources. No estimates or unverified third-party data was used. Sources include the OBS Annual Report 2023, NFO Annual Report 2024, SARB Prudential Authority sanctions register, DataEQ SA Banking Index 2024, and Sagaci Research 2025.

### Phase 2 — Exploratory Data Analysis
Each data dimension was analysed independently — complaint volumes across three years, referral-to-formal-case conversion rates, penalty amounts per bank, and dual-source sentiment scores. Key findings include Capitec's high complaint volume relative to market share, FNB's industry-leading complaint resolution rate, and Capitec's R56.25M regulatory penalty being the largest single sanction in the dataset.

### Phase 3 — Trust Score Model
Each dimension was normalised to a 0–10 scale using min-max normalisation, with metrics where lower is better (conversion rate, penalties) inverted. The four normalised scores were combined using the following weights:

| Dimension | Weight | Rationale |
|---|---|---|
| Complaint Resolution | 30% | Most direct measure of how a bank treats customers |
| Consumer Favour Rate | 25% | Shows whether the Ombudsman agreed the consumer was wronged |
| Regulatory Record | 25% | Legal compliance is foundational to trust |
| Consumer Sentiment | 20% | Real-world perception matters but is more subjective |

### Phase 4 — Streamlit Dashboard
An interactive single-page dashboard was built and deployed allowing consumers to explore the full leaderboard, investigate any single bank's breakdown across all four dimensions, and compare two banks side by side with a plain language verdict.

| Section | Description |
|---|---|
| 🏆 **Leaderboard** | All six banks ranked by Trust Score with colour-coded trust zones |
| 🔍 **Bank Explorer** | Select one bank — full dimension breakdown, progress bars, radar chart, complaint trend |
| ⚖️ **Compare** | Select two banks — side-by-side scores, radar charts, and plain language verdict |

---

## 📌 Status

![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

Phase 1 (notebook) and Phase 2 (dashboard) are complete and deployed. Planned future work:

- Phase 3 — Automated data pipeline with scheduled updates when new OBS/NFO reports are published
- Expanded bank coverage including African Bank and Discovery Bank
- Integration of FSCA enforcement actions as an additional regulatory dimension

---

## 🙏 Credits

**Developed by Lindiwe Songelwa — Data Scientist | Developer | Insight Creator**

| Platform | Link |
|---|---|
| 💼 LinkedIn | [Lindiwe S.](https://za.linkedin.com/in/lindiwe-songelwa) |
| 🌐 Portfolio | [Lindiwe-22.github.io](https://lindiwe-22.github.io/Portfolio-Website/) |
| 🏅 Credly | [Lindiwe Songelwa – Badges](https://www.credly.com/users/samnkelisiwe-lindiwe-songelwa) |
| 🚀 Live App | [Streamlit Dashboard](https://banktrustscore.streamlit.app/) |
| 📧 Email | [sl.songelwa@hotmail.co.za](mailto:sl.songelwa@hotmail.co.za) |

---

## 📊 Data Sources

| Source | Used For |
|---|---|
| [OBS Annual Report 2023](https://nfosa.co.za) | Formal complaint counts, conversion rates, consumer favour rates |
| [NFO Annual Report 2024](https://nfosa.co.za) | Updated resolution data |
| [SARB Prudential Authority](https://www.resbank.co.za) | Regulatory financial penalties 2022–2025 |
| [DataEQ SA Banking Index 2024](https://dataeq.com) | Social media net sentiment |
| [Sagaci Research 2025](https://sagaciresearch.com) | Consumer satisfaction survey scores |

---

*© 2026 Lindiwe Songelwa. For educational and consumer advocacy purposes. Not financial advice.*
