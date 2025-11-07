# 💎 Smart Finance Assistant — RAG + AI Edition

An intelligent **personal finance management system** powered by **AI, LSTM forecasting, and Retrieval-Augmented Generation (RAG)** — built with **Streamlit**, **PyTorch**, and **SentenceTransformers**.

This app helps users visualize expenses, predict future spending, detect high-cost months, and query transactions in natural language — all locally and securely.

---

## 🚀 Features

### 🧠 AI-Powered Capabilities
- **RAG-based Query Engine** — Ask questions like:
  - “What were my biggest expenses in food and drink?”
  - “Which month had the highest expenses?”
  - “What did I spend in June?”
- Uses **SentenceTransformers + FAISS** for fast semantic retrieval.
- Smart **local summarizer** computes totals, max spending, and category insights.

### 🔮 Predictive Analysis
- Uses an **LSTM neural network** to forecast your **next month’s expenses**.
- Learns from your monthly spending trends.

### 📊 Visualization Dashboard
- Monthly expense trends and category-wise pie charts.
- Auto-calculates **average monthly spend** and **savings goal progress**.
- Built using **Plotly Express** for interactivity.

### 💡 Smart Insights
- Identifies top spending categories.
- Tracks progress toward savings goals.
- Highlights overspending behavior.

### 🔒 Privacy-First
- Your uploaded data is used **only in-memory** (via Streamlit session state).
- **No CSVs or personal data** are stored locally or uploaded to GitHub.

---

## 🧩 Architecture Overview


User Uploads CSV → Data Cleaning → Visualization + Forecasting
↓
RAG Embeddings (MiniLM)
↓
FAISS Vector Retrieval
↓
Local Summarization Engine (Smart RAG)


---

## 🗂️ Dataset Format

Your uploaded CSV must include these columns (case-insensitive):

| Column | Description | Example |
|--------|--------------|----------|
| `Date` | Transaction date | `2023-06-15` |
| `Transaction Description` | Description of transaction | `Dinner with friends` |
| `Category` | Expense category | `Food & Drink` |
| `Amount` | Amount spent | `1245.50` |
| `Type` | Type (Expense, Credit, etc.) | `Expense` |

---

## 🛠️ Tech Stack

| Component | Technology |
|------------|-------------|
| Frontend | [Streamlit](https://streamlit.io/) |
| ML Forecast | PyTorch (LSTM) |
| NLP Model | SentenceTransformers (`all-MiniLM-L6-v2`) |
| Vector Search | FAISS |
| Visualization | Plotly |
| Data Processing | Pandas, NumPy |
| Deployment | Streamlit Cloud / GitHub |

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/DAMINIKATAKAM/smart-finance-assistant.git
cd smart-finance-assistant


Example Queries for RAG
“What were my biggest expenses in food and drink?”
“What were my biggest expenses in June?”
“What were my biggest expenses in travel in March?”

### 2️⃣ Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt

### 4️⃣ Run the App
```bash
Streamlit run app.py

