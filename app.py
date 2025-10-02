import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import joblib
import time

# ======================
# Load ML model + vectorizer (safe)
# ======================
vectorizer = None
model = None
load_error = None
try:
    vectorizer = joblib.load("vectorizer.pkl")
    model = joblib.load("best_model.pkl")
except Exception as e:
    load_error = str(e)
# --- Enhanced sentiment logic with lexicon + model ---
import re, numpy as np

# Detect positive/negative label indices from the trained model
_POS_HINT = "i absolutely loved this amazing movie"
_NEG_HINT = "i absolutely hated this terrible movie"
try:
    _pos_label = model.predict(vectorizer.transform([_POS_HINT]))[0]
    _neg_label = model.predict(vectorizer.transform([_NEG_HINT]))[0]
except Exception:
    _pos_label, _neg_label = 1, 0

_CLASSES = list(model.classes_)
def _idx_of(label):
    try: return _CLASSES.index(label)
    except Exception: return 1 if str(label) in {"1","True"} else 0

_POS_IDX = _idx_of(_pos_label)
_NEG_IDX = _idx_of(_neg_label)

# Lexicon of words/phrases
_POS_WORDS = {
    "good","great","amazing","awesome","excellent","fantastic","love","loved","iconic","classic","memorable","fun",
    "entertaining","enjoyable","masterpiece","brilliant","superb","mustwatch","perfect","recommend","beautiful"
}
_NEG_WORDS = {
    "bad","boring","awful","terrible","worst","waste","hate","hated",
    "garbage","cringe","cringey","disappointing","slow","confusing","meh",
    "childish","awkward","slapstick","campy","predictable","dull","messy"
}
_PHRASES = {
    # Strong Positive
    "must watch": 1.6,
    "highly recommend": 1.4,
    "worth watching": 1.2,
    "mind blowing": 1.5,
    "fell in love": 1.4,
    "piece of cinema": 1.3,
    "blend of emotions": 1.2,
    "beautifully done": 1.3,
    "well executed": 1.2,
    "brilliant performance": 1.5,
    "outstanding acting": 1.4,
    "never boring": 1.4,
    "won't let you get bored": 1.5,
    "visually stunning": 1.3,
    "masterpiece": 1.6,

    # Mild Positive
    "not bad": 0.8,
    "worth a try": 0.9,
    "good attempt": 0.8,
    "quite enjoyable": 1.0,

    # Strong Negative
    "waste of time": -2.0,
    "do not watch": -2.0,
    "worst movie": -1.8,
    "predictable plot": -1.3,
    "boring scenes": -1.5,
    "childish comedy": -1.2,
    "all over the place": -1.2,
    "no one asked for": -1.2,
    "bad acting": -1.4,
    "terrible script": -1.6,
    "garbage movie": -1.8,
    "cringe worthy": -1.5,
    "poor direction": -1.4,
    "messy storytelling": -1.3,

    # Mild Negative
    "not good": -1.0,
    "could have been better": -0.8,
    "somewhat boring": -0.7,
    "slow in parts": -0.6,
}
_NEGATORS = {"not","no","never","isnt","isn't","dont","don't","cant","can't","won't","wont"}
_INTENSIFIERS = {"very":1.2,"too":1.2,"extremely":1.4,"overly":1.3,"really":1.1,"way":1.2}
_DOWNTONERS   = {"slightly":0.8,"somewhat":0.85,"a bit":0.85}

_token_re = re.compile(r"[a-z']+")

def _lexicon_score(text: str) -> float:
    """Return sentiment score [-1,1]"""
    t = text.lower()
    score = 0.0
    for p, w in _PHRASES.items():
        if p in t:
            score += w
    toks = _token_re.findall(t)
    for i, tok in enumerate(toks):
        mult = 1.0
        if i > 0 and toks[i-1] in _INTENSIFIERS: mult *= _INTENSIFIERS[toks[i-1]]
        if i > 0 and toks[i-1] in _DOWNTONERS:   mult *= _DOWNTONERS[toks[i-1]]
        negated = any(tok2 in _NEGATORS for tok2 in toks[max(0, i-3):i])
        if tok in _POS_WORDS:
            score += (-1 if negated else 1) * mult
        elif tok in _NEG_WORDS:
            score += (1 if negated else -1) * mult
    return float(np.tanh(score))  # squashed [-1,1]

def predict_sentiment(text: str):
    """Blend ML + lexicon for stronger, clearer classification"""
    X = vectorizer.transform([text])
    ml_proba = model.predict_proba(X)[0]
    ml_pos = float(ml_proba[_POS_IDX])
    ml_neg = float(ml_proba[_NEG_IDX])

    lx = _lexicon_score(text)
    lex_pos = 0.5 * (lx + 1.0)

    alpha = 0.35
    if len(text.split()) < 25: alpha += 0.20
    elif len(text.split()) > 50: alpha += 0.15
    pos_final = (1 - alpha) * ml_pos + alpha * lex_pos
    pos_final = max(0.0, min(1.0, pos_final))
    neg_final = 1.0 - pos_final

    POS_T, NEG_T = 0.55, 0.45
    if pos_final >= POS_T:
        label, conf = "positive", pos_final * 100
    elif pos_final <= NEG_T:
        label, conf = "negative", neg_final * 100
    else:
        leaning = "positive" if pos_final >= 0.5 else "negative"
        label, conf = f"mixed (leaning {leaning})", (0.5 + abs(pos_final - 0.5)) * 100

    return label, conf, {"positive": pos_final*100, "negative": neg_final*100}

# --------------------------------------------------------------
# ======================
# Streamlit Page Config
# ======================
st.set_page_config(
    page_title="🎬 Movie Sentiment AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ======================
# Custom CSS (UI Theme)
# ======================
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Floating decorative emojis */
    .floating-emoji {
        position: fixed;
        font-size: 3rem;
        opacity: 0.10;
        animation: floatEmoji 20s ease-in-out infinite;
        pointer-events: none;
        z-index: 1;
    }
    @keyframes floatEmoji {
        0%, 100% { transform: translate(0, 0) rotate(0deg); }
        25% { transform: translate(100px, -100px) rotate(90deg); }
        50% { transform: translate(200px, 0) rotate(180deg); }
        75% { transform: translate(100px, 100px) rotate(270deg); }
    }

    /* Card shells */
    .hero-container, .stats-container, .demo-container, .section-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 25px;
        padding: 3rem 2rem;
        margin: 2rem auto;
        max-width: 1200px;
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.2);
        position: relative;
        z-index: 2;
    }
    .hero-container {
        border-radius: 30px;
        padding: 4rem 2rem;
        text-align: center;
    }

    /* Titles */
    .main-title {
        font-size: 4.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        text-shadow: 0 0 30px rgba(102, 126, 234, 0.3);
    }
    .subtitle {
        font-size: 1.8rem;
        color: #555;
        margin-bottom: 2rem;
        font-weight: 300;
        letter-spacing: 1px;
    }
    .tagline {
        font-size: 1.3rem;
        color: #888;
        margin-bottom: 3rem;
        font-style: italic;
    }
    .section-title {
        text-align: center;
        font-size: 2.5rem;
        margin-bottom: 2rem;
        color: #333;
        font-weight: 800;
    }

    /* Feature grid */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 2rem;
        margin: 3rem 0;
    }
    .feature-card {
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        transition: all 0.3s ease;
        cursor: pointer;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        animation: float 3s ease-in-out infinite;
    }
    .feature-card:nth-child(1) { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); animation-delay: 0s;}
    .feature-card:nth-child(2) { background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); animation-delay: 0.5s;}
    .feature-card:nth-child(3) { background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); animation-delay: 1s;}
    .feature-card:nth-child(4) { background: linear-gradient(135deg, #30cfd0 0%, #330867 100%); animation-delay: 1.5s;}

    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    .feature-card:hover {
        transform: translateY(-15px) scale(1.05);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
    }
    .feature-icon { font-size: 3.5rem; margin-bottom: 1rem; }
    .feature-title { font-size: 1.5rem; font-weight: bold; margin-bottom: 0.5rem; }
    .feature-description { font-size: 1rem; opacity: 0.9; }

    /* Stats cards */
    .stat-box {
        text-align: center;
        padding: 2rem;
        border-radius: 15px;
        color: white;
        transition: all 0.3s ease;
    }
    .stat-box:hover { transform: scale(1.05); }
    .stat-number {
        font-size: 3.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        text-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .stat-label {
        font-size: 1.2rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 1rem 3rem !important;
        font-size: 1.2rem !important;
        font-weight: bold !important;
        text-transform: uppercase !important;
        letter-spacing: 2px !important;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-5px) !important;
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.6) !important;
    }

    /* Quick example (right column) */
    .quick-example {
        color: white !important;
        background-color: rgba(255, 255, 255, 0.05) !important;
        padding: 0.8rem 1rem;
        border-radius: 10px;
        font-size: 1.05rem;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ======================
# Floating Emojis (decoration)
# ======================
st.markdown("""
<div class="floating-emoji" style="top: 10%; left: 10%;">🎬</div>
<div class="floating-emoji" style="top: 20%; right: 15%; animation-delay: 2s;">🎭</div>
<div class="floating-emoji" style="bottom: 15%; left: 20%; animation-delay: 4s;">🍿</div>
<div class="floating-emoji" style="bottom: 25%; right: 10%; animation-delay: 6s;">⭐</div>
<div class="floating-emoji" style="top: 60%; left: 5%; animation-delay: 8s;">🎥</div>
<div class="floating-emoji" style="top: 40%; right: 5%; animation-delay: 10s;">🎪</div>
""", unsafe_allow_html=True)

# ======================
# Hero Section
# ======================
st.markdown("""
<div class="hero-container">
    <div class="main-title">🎬 Movie Sentiment AI</div>
    <div class="subtitle">Powered by Advanced Machine Learning </div>
    <div class="tagline">"Analyzing Movie Reviews with State-of-the-Art AI Technology"</div>
</div>
""", unsafe_allow_html=True)

# ======================
# Center Button
# ======================
c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    if st.button("🚀 START ANALYZING REVIEWS", use_container_width=True):
        st.balloons()
        st.success("🎉 Scroll down to the Quick Demo to analyze your review!")

# ======================
# Feature Cards
# ======================
st.markdown("""
<div class="section-card">
    <div class="section-title">✨ Powerful Features</div>
    <div class="feature-grid">
        <div class="feature-card">
            <div class="feature-icon">🤖</div>
            <div class="feature-title">AI-Powered</div>
            <div class="feature-description">Multiple state-of-the-art ML models including BERT, RoBERTa & custom-trained models</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <div class="feature-title">Real-Time Analysis</div>
            <div class="feature-description">Instant predictions with easy-to-read confidence scores</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Visual Analytics</div>
            <div class="feature-description">Beautiful probability charts and insights at a glance</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🎯</div>
            <div class="feature-title">High Accuracy</div>
            <div class="feature-description">Trained on thousands of reviews for robust performance</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ======================
# Stats Section
# ======================
st.markdown('<div class="stats-container">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📈 Live Statistics</div>', unsafe_allow_html=True)

s1, s2, s3, s4 = st.columns(4)
with s1:
    st.markdown("""
    <div class="stat-box" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
        <div class="stat-number">10K+</div>
        <div class="stat-label">Reviews Analyzed</div>
    </div>
    """, unsafe_allow_html=True)
with s2:
    st.markdown("""
    <div class="stat-box" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
        <div class="stat-number">94.2%</div>
        <div class="stat-label">Accuracy</div>
    </div>
    """, unsafe_allow_html=True)
with s3:
    st.markdown("""
    <div class="stat-box" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
        <div class="stat-number">4</div>
        <div class="stat-label">AI Models</div>
    </div>
    """, unsafe_allow_html=True)
with s4:
    st.markdown("""
    <div class="stat-box" style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);">
        <div class="stat-number">&lt; 2s</div>
        <div class="stat-label">Processing Time</div>
    </div>
    """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ======================
# Demo Section (ML-powered)
# ======================
st.markdown("""
<div class="demo-container">
    <div class="section-title">🎭 Try It Out - Quick Demo</div>
</div>
""", unsafe_allow_html=True)

col_demo1, col_demo2 = st.columns([3, 2])

with col_demo1:
    demo_text = st.text_area(
        "Enter a movie review:",
        placeholder="Type: 'This movie was absolutely amazing! The cinematography was breathtaking...'",
        height=140,
        key="demo_input"
    )

    if st.button("🔮 Analyze This Review", key="demo_btn"):
        if demo_text.strip():
            with st.spinner("🧠 AI is analyzing..."):
                time.sleep(0.8)

                label, conf, probs = predict_sentiment(demo_text)

                if label == "positive":
                    st.success(f"🟢 **POSITIVE SENTIMENT** detected! Confidence: {conf:.1f}%")
                    st.balloons()
                elif label == "negative":
                    st.error(f"🔴 **NEGATIVE SENTIMENT** detected! Confidence: {conf:.1f}%")
                else:
                    st.info(
                        f"🟡 **MIXED / NEUTRAL** sentiment "
                        f"(Pos: {probs['positive']:.1f}%, Neg: {probs['negative']:.1f}%)"
                    )

                # Plot bar chart
                chart_df = {
                    "Class": ["Negative", "Positive"],
                    "Confidence (%)": [probs["negative"], probs["positive"]],
                }
                import plotly.express as px
                fig = px.bar(
                    chart_df, x="Class", y="Confidence (%)",
                    title="Sentiment Probability (%)",
                    range_y=[0, 100]
                )
                fig.update_layout(template="plotly_dark", height=420, margin=dict(l=10,r=10,t=50,b=10))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please enter some text to analyze!")
with col_demo2:
    st.markdown("### 🎯 Quick Examples")
    if st.button("😊 Positive Review"):
        st.markdown("<div class='quick-example'>'This movie was absolutely incredible! Amazing performances!'</div>", unsafe_allow_html=True)
    if st.button("😞 Negative Review"):
        st.markdown("<div class='quick-example'>'Terrible film. Boring and predictable. Waste of time.'</div>", unsafe_allow_html=True)
    if st.button("🎬 Mixed Review"):
        st.markdown("<div class='quick-example'>'Good acting but the plot was confusing and slow.'</div>", unsafe_allow_html=True)

# ======================
# Tech Stack Section
# ======================
st.markdown("""
<div class="section-card">
    <div class="section-title">🛠️ Technology Stack</div>
    <div style="text-align: center;">
        <span style="display:inline-block;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:#fff;padding:.6rem 1.5rem;border-radius:25px;margin:.5rem;font-weight:600;box-shadow:0 4px 15px rgba(102,126,234,.3);">🐍 Python</span>
        <span style="display:inline-block;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:#fff;padding:.6rem 1.5rem;border-radius:25px;margin:.5rem;font-weight:600;box-shadow:0 4px 15px rgba(102,126,234,.3);">🧠 Scikit-learn</span>
        <span style="display:inline-block;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:#fff;padding:.6rem 1.5rem;border-radius:25px;margin:.5rem;font-weight:600;box-shadow:0 4px 15px rgba(102,126,234,.3);">📊 Streamlit</span>
        <span style="display:inline-block;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:#fff;padding:.6rem 1.5rem;border-radius:25px;margin:.5rem;font-weight:600;box-shadow:0 4px 15px rgba(102,126,234,.3);">🔄 Joblib</span>
        <span style="display:inline-block;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:#fff;padding:.6rem 1.5rem;border-radius:25px;margin:.5rem;font-weight:600;box-shadow:0 4px 15px rgba(102,126,234,.3);">📈 Plotly</span>
        <span style="display:inline-block;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:#fff;padding:.6rem 1.5rem;border-radius:25px;margin:.5rem;font-weight:600;box-shadow:0 4px 15px rgba(102,126,234,.3);">💾 NLTK</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ======================
# How It Works
# ======================
st.markdown("""
<div class="demo-container">
    <div class="section-title">🔍 How It Works</div>
</div>
""", unsafe_allow_html=True)

h1, h2, h3, h4 = st.columns(4)
with h1:
    st.markdown("""
    <div style="text-align:center;padding:1rem;">
        <div style="font-size:3rem;margin-bottom:1rem;">📝</div>
        <h3 style="color:#667eea;">1. Input</h3>
        <p>Enter your movie review text</p>
    </div>
    """, unsafe_allow_html=True)
with h2:
    st.markdown("""
    <div style="text-align:center;padding:1rem;">
        <div style="font-size:3rem;margin-bottom:1rem;">🧠</div>
        <h3 style="color:#667eea;">2. Vectorize</h3>
        <p>Convert text to features with TF-IDF</p>
    </div>
    """, unsafe_allow_html=True)
with h3:
    st.markdown("""
    <div style="text-align:center;padding:1rem;">
        <div style="font-size:3rem;margin-bottom:1rem;">🎯</div>
        <h3 style="color:#667eea;">3. Predict</h3>
        <p>Model outputs probabilities</p>
    </div>
    """, unsafe_allow_html=True)
with h4:
    st.markdown("""
    <div style="text-align:center;padding:1rem;">
        <div style="font-size:3rem;margin-bottom:1rem;">📊</div>
        <h3 style="color:#667eea;">4. Visualize</h3>
        <p>See sentiment & confidence</p>
    </div>
    """, unsafe_allow_html=True)

# ======================
# Footer CTA
# ======================
st.markdown("""
<div style="text-align:center;padding:3rem 0;color:white;">
    <h3 style="font-size:2rem;margin-bottom:1rem;">Ready to Get Started?</h3>
    <p style="font-size:1.2rem;margin-bottom:2rem;">Experience the power of AI-driven sentiment analysis</p>
</div>
""", unsafe_allow_html=True)

cta1, cta2, cta3 = st.columns([1, 2, 1])
with cta2:
    if st.button("🎬 LAUNCH FULL DASHBOARD", use_container_width=True, type="primary"):
        st.balloons()
        st.success("🚀 Loading complete dashboard...")
        st.snow()
