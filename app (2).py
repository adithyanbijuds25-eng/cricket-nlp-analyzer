import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import pickle
from collections import Counter

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Cricket NLP Analyzer",
    page_icon="🏏",
    layout="wide"
)

# ── CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .block-container { padding-top: 2rem; }
    h1 { color: #00d4aa; font-family: 'Georgia', serif; }
    h2, h3 { color: #e0e0e0; }
    .result-box {
        background: #1e2130;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #00d4aa;
    }
    .metric-card {
        background: #1e2130;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        border: 1px solid #2d3250;
    }
    .positive { border-left-color: #4caf50 !important; }
    .negative { border-left-color: #f44336 !important; }
    .stTextArea textarea { background: #1e2130; color: #e0e0e0; border: 1px solid #00d4aa; }
    .stButton > button {
        background: linear-gradient(135deg, #00d4aa, #0096c7);
        color: white;
        border: none;
        border-radius: 8px;
        font-size: 16px;
        padding: 10px 30px;
        width: 100%;
        font-weight: bold;
    }
    .stButton > button:hover { opacity: 0.9; transform: translateY(-1px); }
    .tag {
        display: inline-block;
        background: #2d3250;
        color: #00d4aa;
        padding: 3px 10px;
        border-radius: 20px;
        margin: 3px;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# ── Load models (cached) ───────────────────────────────────────
@st.cache_resource
def load_spacy():
    import spacy
    try:
        return spacy.load("en_core_web_sm")
    except:
        os.system("python -m spacy download en_core_web_sm")
        import spacy
        return spacy.load("en_core_web_sm")

@st.cache_resource
def load_sentiment():
    from transformers import pipeline
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

@st.cache_resource
def load_classifier():
    """Train classifier on the CSV data if available, else return None."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split

    # Try to find CSVs
    csv_files = []
    for name in ["train.csv", "test.csv", "validation.csv"]:
        if os.path.exists(name):
            csv_files.append(name)

    if not csv_files:
        return None, None, None, None

    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, header=None)
            dfs.append(df)
        except:
            pass

    if not dfs:
        return None, None, None, None

    df_raw = pd.concat(dfs, ignore_index=True)

    # Parse — same logic as notebook
    records = []
    for _, row in df_raw.iterrows():
        try:
            text = str(row.iloc[0])
            parts = text.split("|")
            if len(parts) >= 6:
                records.append({
                    "play_type": parts[0].strip(),
                    "commentary": parts[-1].strip()
                })
            else:
                records.append({"play_type": "unknown", "commentary": text})
        except:
            pass

    df = pd.DataFrame(records)
    df = df[df["commentary"].str.len() > 10].dropna()
    df["commentary"] = df["commentary"].str.lower().str.replace(r"[^a-z\s]", " ", regex=True).str.strip()
    df = df[df["play_type"] != "unknown"]

    if len(df) < 100:
        return None, None, None, None

    le = LabelEncoder()
    df["label"] = le.fit_transform(df["play_type"])

    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X = tfidf.fit_transform(df["commentary"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)

    return tfidf, model, le, round(acc * 100, 2)

# ── Venue list ─────────────────────────────────────────────────
VENUES = [
    "wankhede", "eden gardens", "lords", "lord's", "oval", "gabba",
    "mcg", "scg", "headingley", "old trafford", "chepauk",
    "chinnaswamy", "feroz shah kotla", "narendra modi stadium",
    "dharamsala", "mohali", "pune", "rajkot", "vizag"
]

# ── Analyze function ───────────────────────────────────────────
def analyze(text, nlp, sentiment_pipe, tfidf, model, le):
    results = {}

    # Clean
    cleaned = re.sub(r"\s+", " ", text.strip())

    # NER
    doc = nlp(cleaned)
    players = list(set([e.text for e in doc.ents if e.label_ == "PERSON"]))
    orgs = list(set([e.text for e in doc.ents if e.label_ in ("ORG", "GPE")]))
    venues = [v.title() for v in VENUES if v in cleaned.lower()]
    results["players"] = players
    results["teams"] = orgs
    results["venues"] = venues

    # Sentiment
    try:
        out = sentiment_pipe(cleaned[:512])[0]
        results["sentiment"] = out["label"]
        results["sentiment_score"] = round(out["score"] * 100, 1)
    except:
        results["sentiment"] = "NEUTRAL"
        results["sentiment_score"] = 50.0

    # Classification
    if model is not None:
        vec = tfidf.transform([cleaned.lower()])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec).max()
        results["play_type"] = le.inverse_transform([pred])[0]
        results["play_confidence"] = round(prob * 100, 1)
    else:
        results["play_type"] = "Model not loaded"
        results["play_confidence"] = 0

    return results

# ── UI ─────────────────────────────────────────────────────────
st.markdown("# 🏏 Cricket Commentary NLP Analyzer")
st.markdown("**NLP Project** — Named Entity Recognition · Sentiment Analysis · Play Type Classification")
st.markdown("---")

# Sidebar — team info
with st.sidebar:
    st.markdown("### 👥 Team")
    for name, role in [
        ("Adhi", "Data & Preprocessing"),
        ("Khaise", "Named Entity Recognition"),
        ("Meera", "Sentiment Analysis"),
        ("Indrajith", "Classification & Demo"),
    ]:
        st.markdown(f"**{name}** — {role}")

    st.markdown("---")
    st.markdown("### 📊 Project Stats")
    st.metric("Dataset Rows", "89,338")
    st.metric("Classifier Accuracy", "90.06%")
    st.metric("Sentiment Confidence", "93.4%")
    st.metric("NER Coverage", "100%")
    st.markdown("---")
    st.markdown("### 🔧 Tech Stack")
    st.markdown("- spaCy `en_core_web_sm`")
    st.markdown("- DistilBERT (SST-2)")
    st.markdown("- TF-IDF + Logistic Regression")
    st.markdown("- Streamlit")

# Load models
with st.spinner("Loading NLP models... (first load takes ~1 min)"):
    nlp = load_spacy()
    sentiment_pipe = load_sentiment()
    tfidf, clf_model, le, clf_acc = load_classifier()

if clf_model is not None:
    st.success(f"✅ All models loaded! Classifier accuracy: {clf_acc}%")
else:
    st.warning("⚠️ CSV files not found — classifier unavailable. NER and Sentiment still work!")

st.markdown("---")

# Tabs
tab1, tab2 = st.tabs(["🎯 Analyze Commentary", "📋 Sample Outputs"])

with tab1:
    st.markdown("### Paste any cricket commentary below")

    # Quick examples
    st.markdown("**Try one of these:**")
    examples = {
        "Six 🏏": "Rohit Sharma launches it over long-on for a massive six! The ball lands in the second tier at Eden Gardens. What a shot!",
        "Wicket ❌": "Bumrah bowls a perfect yorker and crashes into the stumps! Clean bowled! England lose their top batsman for just 12 runs.",
        "Four 🔥": "Kohli drives beautifully through the covers for a magnificent four! The crowd erupts at Wankhede as India take control.",
        "Dot ball 😐": "Anderson bowls a tight line just outside off stump. Kohli leaves it alone, no run taken. Excellent disciplined bowling.",
        "Wide 😬": "Down the leg side, way outside off stump, the umpire raises his hand immediately. Wide called. Poor delivery.",
    }
    cols = st.columns(len(examples))
    selected = ""
    for i, (label, text) in enumerate(examples.items()):
        if cols[i].button(label):
            selected = text

    user_input = st.text_area(
        "Commentary text",
        value=selected,
        height=120,
        placeholder="e.g. Virat Kohli drives through covers for a beautiful four at Wankhede!",
        label_visibility="collapsed"
    )

    if st.button("🔍 Analyze"):
        if not user_input.strip():
            st.warning("Please enter some commentary text first!")
        else:
            with st.spinner("Analyzing..."):
                res = analyze(user_input, nlp, sentiment_pipe, tfidf, clf_model, le)

            st.markdown("---")
            st.markdown("### Results")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("#### 🔍 Named Entities")
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                if res["players"]:
                    st.markdown("**Players detected:**")
                    st.markdown(" ".join([f'<span class="tag">{p}</span>' for p in res["players"]]), unsafe_allow_html=True)
                else:
                    st.markdown("*No players detected*")

                if res["teams"]:
                    st.markdown("**Teams / Countries:**")
                    st.markdown(" ".join([f'<span class="tag">{t}</span>' for t in res["teams"]]), unsafe_allow_html=True)

                if res["venues"]:
                    st.markdown("**Venues:**")
                    st.markdown(" ".join([f'<span class="tag">{v}</span>' for v in res["venues"]]), unsafe_allow_html=True)

                if not res["players"] and not res["teams"] and not res["venues"]:
                    st.markdown("*No entities found in this text*")
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                sentiment_class = "positive" if res["sentiment"] == "POSITIVE" else "negative"
                emoji = "😊" if res["sentiment"] == "POSITIVE" else "😞"
                st.markdown("#### 💬 Sentiment")
                st.markdown(f'<div class="result-box {sentiment_class}">', unsafe_allow_html=True)
                st.markdown(f"**{emoji} {res['sentiment']}**")
                st.progress(res["sentiment_score"] / 100)
                st.markdown(f"Confidence: **{res['sentiment_score']}%**")
                st.markdown('</div>', unsafe_allow_html=True)

            with col3:
                st.markdown("#### 📂 Play Type")
                if clf_model is not None:
                    st.markdown('<div class="result-box">', unsafe_allow_html=True)
                    play_emojis = {
                        "no run": "⚫", "run": "🏃", "four": "4️⃣",
                        "six": "6️⃣", "out": "❌", "wide": "↔️",
                        "no ball": "🚫", "bye": "🟡", "leg bye": "🦵"
                    }
                    emoji_p = play_emojis.get(res["play_type"], "🏏")
                    st.markdown(f"**{emoji_p} {res['play_type'].upper()}**")
                    st.progress(res["play_confidence"] / 100)
                    st.markdown(f"Confidence: **{res['play_confidence']}%**")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.info("Classifier unavailable — upload CSV files to enable.")

with tab2:
    st.markdown("### Sample Outputs from the Project")
    samples = [
        {
            "commentary": "Kohli drives beautifully through the covers for a magnificent four! The crowd erupts at Wankhede.",
            "players": ["Kohli"], "venues": ["Wankhede"],
            "sentiment": "POSITIVE", "confidence": "100%", "play_type": "four", "play_conf": "84.8%"
        },
        {
            "commentary": "Down the leg side, way outside off stump, the umpire raises his hand. Wide called.",
            "players": [], "venues": [],
            "sentiment": "NEGATIVE", "confidence": "99.8%", "play_type": "wide", "play_conf": "89.7%"
        },
        {
            "commentary": "Bumrah bowls a perfect yorker and crashes into the stumps! Clean bowled!",
            "players": ["Bumrah"], "venues": [],
            "sentiment": "POSITIVE", "confidence": "97.2%", "play_type": "out", "play_conf": "91.3%"
        },
    ]

    for s in samples:
        with st.expander(f"🏏 `{s['commentary'][:60]}...`"):
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**Players:** {', '.join(s['players']) if s['players'] else 'None'}")
            c1.markdown(f"**Venues:** {', '.join(s['venues']) if s['venues'] else 'None'}")
            c2.markdown(f"**Sentiment:** {s['sentiment']} ({s['confidence']})")
            c3.markdown(f"**Play Type:** {s['play_type']} ({s['play_conf']})")

st.markdown("---")
st.markdown("<center><small>Cricket NLP Analyzer · NLP Course Project · Adhi · Khaise · Meera · Indrajith</small></center>", unsafe_allow_html=True)
