# app.py — Smart Finance Assistant (Final Intelligent Edition)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import date, timedelta
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer
import faiss
from calendar import month_name

# ---------- CONFIG ----------
st.set_page_config(page_title="Smart Finance Assistant", layout="wide")
EMB_DIM = 384

# ---------- UTILITIES ----------
def make_corpus(df):
    texts = []
    for _, r in df.iterrows():
        date_str = pd.to_datetime(r['Date']).strftime('%Y-%m-%d')
        texts.append(
            f"On {date_str}, {r['Type']} of ₹{r['Amount']} in {r['Category']}. "
            f"Desc: {r['Transaction Description']}"
        )
    return texts

# ---------- MODEL: LSTM FORECAST ----------
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def forecast_expenses(expense_df):
    df = expense_df.groupby(pd.Grouper(key='Date', freq='M'))['Amount'].sum().reset_index()
    if len(df) < 4:
        return None, "Need at least 4 months of expense data."

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    X, y = [], []
    for i in range(3, len(scaled)):
        X.append(scaled[i-3:i])
        y.append(scaled[i])
    X_t = torch.tensor(np.array(X), dtype=torch.float32)
    y_t = torch.tensor(np.array(y), dtype=torch.float32)

    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(100):
        model.train()
        optimizer.zero_grad()
        out = model(X_t)
        loss = criterion(out, y_t)
        loss.backward()
        optimizer.step()

    last_seq = torch.tensor(scaled[-3:].reshape(1, 3, 1), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        pred = model(last_seq).numpy().flatten()
    pred_value = scaler.inverse_transform(pred.reshape(-1, 1))[0][0]

    next_month = df['Date'].max() + pd.offsets.MonthBegin(1)
    df_forecast = pd.concat([df, pd.DataFrame({'Date': [next_month], 'Amount': [pred_value]})])
    return df_forecast, pred_value

# ---------- RAG ----------
@st.cache_resource
def get_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def build_faiss_index(texts):
    model = get_embed_model()
    embs = model.encode(texts, convert_to_numpy=True)
    index = faiss.IndexFlatL2(EMB_DIM)
    index.add(embs.astype(np.float32))
    return model, index, embs

def rag_search(query, model, index, texts, k=5):
    q_emb = model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb.astype(np.float32), k)
    return [texts[i] for i in I[0]]

# ---------- CSV HANDLER ----------
def normalize_columns(df):
    mapping = {
        "date": "Date",
        "transaction description": "Transaction Description",
        "description": "Transaction Description",
        "details": "Transaction Description",
        "category": "Category",
        "amount": "Amount",
        "value": "Amount",
        "price": "Amount",
        "type": "Type",
        "transaction type": "Type"
    }
    new_cols = {}
    for c in df.columns:
        clean = c.strip().lower()
        if clean in mapping:
            new_cols[c] = mapping[clean]
    return df.rename(columns=new_cols)

def normalize_dates(df):
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=False)
    return df.dropna(subset=['Date'])

# ---------- SIDEBAR ----------
st.title("💎 Smart Finance Assistant — Final Intelligent Edition")

if "transactions" not in st.session_state:
    st.session_state.transactions = pd.DataFrame(columns=["Date", "Transaction Description", "Category", "Amount", "Type"])

st.sidebar.header("Upload Transactions (CSV)")
uploaded_file = st.sidebar.file_uploader("Upload your transactions CSV", type=["csv"])

if uploaded_file:
    try:
        new_df = pd.read_csv(uploaded_file)
        new_df = normalize_columns(new_df)
        new_df = normalize_dates(new_df)
        required = {"Date", "Transaction Description", "Category", "Amount", "Type"}
        missing = required - set(new_df.columns)
        if missing:
            st.sidebar.error(f"Missing columns: {', '.join(missing)}")
        else:
            st.session_state.transactions = new_df.reset_index(drop=True)
            st.sidebar.success("✅ File uploaded and active for this session.")
            st.sidebar.dataframe(st.session_state.transactions.head())
    except Exception as e:
        st.sidebar.error(f"Upload error: {e}")

df = st.session_state.transactions
if df.empty:
    st.info("Upload your transaction CSV to get started.")
    st.stop()

st.sidebar.header("Financial Inputs")
income = st.sidebar.number_input("Monthly Income (₹)", min_value=0.0)
goal = st.sidebar.number_input("Savings Goal (₹)", min_value=0.0)
deadline = st.sidebar.date_input("Goal Deadline", value=date.today() + timedelta(days=30))

# ---------- TABS ----------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔮 Forecast", "💡 Insights", "🧠 RAG Assistant"])

# --- TAB 1: Dashboard ---
with tab1:
    st.subheader("📊 Expense Dashboard")

    expense_keywords = ["expense", "debit", "withdrawal", "payment", "spent", "purchase"]
    expense_df = df[df["Type"].str.lower().str.contains("|".join(expense_keywords), na=False)].copy()

    if expense_df.empty:
        st.warning("No expense-type transactions found.")
    else:
        expense_df['Month'] = expense_df['Date'].dt.to_period('M').dt.to_timestamp()
        monthly = expense_df.groupby('Month', as_index=False)['Amount'].sum().sort_values('Month')
        cat = expense_df.groupby('Category', as_index=False)['Amount'].sum().sort_values('Amount', ascending=False)

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.bar(monthly, x='Month', y='Amount', title="📅 Monthly Expense Trend", text_auto=".2s"), use_container_width=True)
        with c2:
            st.plotly_chart(px.pie(cat, values='Amount', names='Category', title="💸 Category-wise Expense Split"), use_container_width=True)

        st.markdown("### 📊 Monthly Expense Breakdown")
        monthly['Amount'] = monthly['Amount'].map(lambda x: f"₹{x:,.2f}")
        st.dataframe(monthly)

# --- TAB 2: Forecast ---
with tab2:
    st.subheader("🔮 Next Month Expense Prediction (LSTM)")
    fc_df, pred = forecast_expenses(expense_df)
    if fc_df is None:
        st.warning(pred)
    else:
        st.success(f"Predicted next month's expense: ₹{pred:,.2f}")
        st.line_chart(fc_df.set_index('Date')['Amount'])

# --- TAB 3: Insights ---
with tab3:
    st.subheader("💡 AI-Generated Insights")

    expense_df['Month'] = expense_df['Date'].dt.to_period('M').dt.to_timestamp()
    monthly_sum = expense_df.groupby('Month', as_index=False)['Amount'].sum().sort_values('Month')

    top_spend = expense_df.groupby('Category')['Amount'].sum().sort_values(ascending=False).head(3)
    st.write(f"💰 **Top Spending Categories:** {', '.join(top_spend.index)}")

    avg_monthly_spend = monthly_sum['Amount'].mean()
    last_month_spend = monthly_sum.iloc[-1]['Amount'] if not monthly_sum.empty else 0

    st.metric("Average Monthly Spend", f"₹{avg_monthly_spend:,.2f}")
    st.metric("Last Month’s Total Expenses", f"₹{last_month_spend:,.2f}")

    savings = income - last_month_spend
    progress = (savings / goal * 100) if goal > 0 else 0
    progress = max(0, min(progress, 100))
    st.progress(progress / 100.0)
    st.caption(f"Savings Goal Progress: {progress:.2f}%")

    if savings < 0:
        st.warning("⚠️ You’ve spent more than your income this month!")

    st.markdown("### 📆 Recent Monthly Expenses")
    recent = monthly_sum.tail(6).copy()
    recent['Amount'] = recent['Amount'].map(lambda x: f"₹{x:,.2f}")
    st.dataframe(recent)

# --- TAB 4: RAG Assistant ---
with tab4:
    st.subheader("🧠 Ask Your Finance Assistant")

    texts = make_corpus(expense_df)
    model, index, embs = build_faiss_index(texts)
    q = st.text_input("Ask anything about your transactions...")

    def detect_month_from_query(query):
        """Detect month name from user query (e.g. 'June')."""
        query_lower = query.lower()
        for m in month_name:
            if m and m.lower() in query_lower:
                return m
        return None

    def local_summary(hits, full_df, user_query):
        total = 0
        max_amt = 0
        max_desc = ""
        cat = None
        count = 0

        for t in hits:
            if "in" in t:
                possible_cat = t.split("in")[1].split(".")[0].strip()
                if not cat:
                    cat = possible_cat

        q_lower = user_query.lower()
        key_terms = ["food", "drink", "travel", "rent", "shopping", "entertainment", "utilities", "investment", "education"]
        matched_terms = [term for term in key_terms if term in q_lower]

        detected_month = detect_month_from_query(user_query)
        if detected_month:
            month_num = list(month_name).index(detected_month)
            full_df['MonthNum'] = full_df['Date'].dt.month
            month_df = full_df[full_df['MonthNum'] == month_num]
        else:
            month_df = full_df.copy()

        if matched_terms:
            filter_terms = "|".join(matched_terms)
            df_filtered = month_df[month_df["Category"].str.lower().str.contains(filter_terms, na=False)]
        elif cat:
            df_filtered = month_df[month_df["Category"].str.lower().str.contains(cat.lower(), na=False)]
        else:
            df_filtered = month_df

        if df_filtered.empty:
            return "No matching transactions found."

        # 'Which month had highest spending?'
        if "which month" in q_lower and "spend" in q_lower:
            month_totals = full_df.groupby(full_df["Date"].dt.month)["Amount"].sum()
            best_month = month_totals.idxmax()
            best_value = month_totals.max()
            month_name_best = month_name[best_month]
            return f"📆 Your highest spending month was **{month_name_best}**, with total expenses of ₹{best_value:,.2f}."

        total = df_filtered["Amount"].sum()
        max_row = df_filtered.loc[df_filtered["Amount"].idxmax()]
        max_amt = max_row["Amount"]
        max_desc = f"On {max_row['Date'].strftime('%Y-%m-%d')}, Expense of ₹{max_amt:,.2f} in {max_row['Category']}. Desc: {max_row['Transaction Description']}"

        summary = f"Found {len(df_filtered)} transactions"
        if matched_terms:
            summary += f" in {matched_terms[0].capitalize()}"
        if detected_month:
            summary += f" during {detected_month}"
        summary += f".\nTotal spending: ₹{total:,.2f}."
        summary += f"\nHighest single expense: ₹{max_amt:,.2f}."
        summary += f"\nTop transaction: {max_desc}"
        return summary

    if st.button("Ask") and q.strip():
        hits = rag_search(q, model, index, texts)
        st.markdown("### 🔍 Retrieved Results")
        for r in hits:
            st.write("-", r)

        summary = local_summary(hits, expense_df, q)
        st.markdown("### 🧾 Summary")
        st.info(summary)

