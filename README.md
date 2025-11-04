# 🌍 Streamlit ESG Analyzer

The **ESG Analyzer** is a lightweight, multilingual web app built with **Streamlit** that allows users to upload and analyze sustainability reports (PDFs).  
It automatically extracts, processes, and scores environmental, social, and governance (ESG) disclosures, producing comparable dashboards and risk insights across multiple reports.

---

## 🚀 Features

- **Multi-language support**: Automatically detects report language and optionally translates to English.  
- **Fast PDF extraction**: Uses *PyMuPDF* for efficient text parsing.  
- **Seed-based ESG scoring**: Evaluates ESG subtopics (E, S, G pillars) using transparent keyword coverage metrics.  
- **Risk signal detection**: Highlights physical, regulatory, reputational, and compliance risks.  
- **Interactive dashboards**: Generates dynamic comparisons using *Plotly* charts and *Streamlit* components.  
- **Customizable scoring weights**: Adjust pillar weights directly from the sidebar.  
- **Exportable results**: Download structured outputs as **CSV** or **JSON**.  
- **Caching for performance**: Streamlit’s cache system ensures faster repeated analyses.

---

## 🧩 How It Works

1. **Upload** one or more PDF sustainability or ESG reports.  
2. The app:
   - Extracts text from the PDFs.
   - Detects the report language and translates if necessary.
   - Calculates subtopic coverage for ESG areas using keyword seeds.
   - Computes E, S, G pillar scores, overall ESG score, and confidence level.
   - Identifies risk mentions based on predefined lexicons.
3. **Visualize results** interactively with comparison dashboards and heatmaps.  
4. **Export results** for further processing.

---

## ⚙️ Installation

```bash
# 1. Clone this repository
git clone https://github.com/yourusername/esg-analyzer.git
cd esg-analyzer

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
