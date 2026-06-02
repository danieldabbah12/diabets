"""
🫁 בדיקת ריאות — ניתוח נתונים קליניים
ממשק פשוט וידידותי. לצורכי הדגמה בלבד.
"""

import streamlit as st
import numpy as np
import pandas as pd
import time

st.set_page_config(
    page_title="בדיקת ריאות",
    page_icon="🫁",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rubik:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Rubik', sans-serif;
    background-color: #f5f7fa;
    color: #2d3748;
    direction: rtl;
}
#MainMenu, footer, header { visibility: hidden; }

.hero {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 24px;
    padding: 2.5rem 2rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 20px 60px rgba(102,126,234,0.3);
}
.hero-icon  { font-size: 3.5rem; margin-bottom: 0.4rem; }
.hero-title { font-size: 2.2rem; font-weight: 700; color: white; margin-bottom: 0.5rem; }
.hero-sub   { font-size: 1rem; color: rgba(255,255,255,0.85); line-height: 1.7; }

.card {
    background: white;
    border-radius: 18px;
    padding: 1.6rem 1.8rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.06);
    border: 1px solid #e8ecf4;
    margin-bottom: 1.2rem;
}
.card-title { font-size: 1.05rem; font-weight: 600; color: #4a5568; margin-bottom: 1rem; }

.result-good {
    background: linear-gradient(135deg, #d4f5e9, #c6f0e0);
    border: 2px solid #48bb78;
    border-radius: 20px; padding: 2rem; text-align: center; margin-top: 1rem;
    box-shadow: 0 8px 30px rgba(72,187,120,0.2);
}
.result-bad {
    background: linear-gradient(135deg, #fff5f5, #fed7d7);
    border: 2px solid #fc8181;
    border-radius: 20px; padding: 2rem; text-align: center; margin-top: 1rem;
    box-shadow: 0 8px 30px rgba(252,129,129,0.2);
}
.result-emoji { font-size: 3.5rem; margin-bottom: 0.6rem; }
.result-title-good { font-size: 1.9rem; font-weight: 700; color: #276749; margin-bottom: 0.5rem; }
.result-title-bad  { font-size: 1.9rem; font-weight: 700; color: #c53030; margin-bottom: 0.5rem; }
.result-msg { font-size: 0.95rem; line-height: 1.7; color: #4a5568; }

.disclaimer {
    background: #fffbeb; border: 1px solid #f6e05e;
    border-radius: 12px; padding: 0.9rem 1.1rem;
    font-size: 0.82rem; color: #744210;
    margin-top: 1.2rem; text-align: center; line-height: 1.6;
}

.stButton > button {
    width: 100%; border-radius: 14px !important;
    font-family: 'Rubik', sans-serif !important;
    font-weight: 600 !important; font-size: 1.05rem !important;
    padding: 0.75rem !important;
}
</style>
""", unsafe_allow_html=True)

# ── HERO ──────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-icon">🫁</div>
    <div class="hero-title">ניתוח בדיקת ריאות</div>
    <div class="hero-sub">הזינו את הנתונים הקליניים וקבלו הערכת סיכון מיידית</div>
</div>
""", unsafe_allow_html=True)

# ── MODEL ─────────────────────────────────────────────────────────
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def predict(data):
    score = (
        -4.5
        + 0.045  * data["age"]
        + 0.30   * data["smoking_years"]
        + 0.022  * max(data["fev1_pct"] - 80, 0) * -1   # FEV1 נמוך = סיכון
        + 0.018  * data["fvc_pct"] * -1
        + 0.25   * data["wheeze"]
        + 0.20   * data["cough_months"]
        + 0.15   * data["dyspnea"]
        + 0.30   * data["family_history"]
        + 0.40   * data["hemoptysis"]
    )
    p_concern = float(sigmoid(score))
    p_healthy = 1 - p_concern
    return p_healthy, p_concern


# ── FORM ──────────────────────────────────────────────────────────
st.markdown('<div class="card"><div class="card-title">👤 פרטים אישיים</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
age          = col1.slider("גיל", 20, 90, 50)
smoking_yrs  = col2.slider("שנות עישון", 0, 50, 0)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card"><div class="card-title">🔬 נתוני ספירומטריה</div>', unsafe_allow_html=True)
col3, col4 = st.columns(2)
fev1_pct = col3.slider("FEV1 — % מהצפוי", 30, 120, 85,
                        help="נפח אוויר בנשיפה מאולצת בשנייה הראשונה")
fvc_pct  = col4.slider("FVC — % מהצפוי", 40, 120, 90,
                        help="קיבולת ריאות כוללת")
fev1_fvc = round(fev1_pct / fvc_pct, 2) if fvc_pct > 0 else 0
st.metric("יחס FEV1/FVC", f"{fev1_fvc:.2f}",
          help="מתחת ל-0.70 עשוי להצביע על חסימה")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card"><div class="card-title">🩺 תסמינים</div>', unsafe_allow_html=True)
col5, col6 = st.columns(2)
wheeze       = col5.selectbox("שריקות בנשימה?",     ["לא","כן"]) == "כן"
hemoptysis   = col6.selectbox("שיעול דם?",           ["לא","כן"]) == "כן"
col7, col8 = st.columns(2)
dyspnea      = col7.selectbox("קוצר נשימה?",         ["לא","כן"]) == "כן"
family_hist  = col8.selectbox("היסטוריה משפחתית?",   ["לא","כן"]) == "כן"
cough_months = st.slider("משך שיעול כרוני (חודשים)", 0, 24, 0)
st.markdown('</div>', unsafe_allow_html=True)

analyze = st.button("🔍  נתח נתונים", type="primary")

# ── RESULTS ───────────────────────────────────────────────────────
if analyze:
    data = dict(
        age=age, smoking_years=smoking_yrs,
        fev1_pct=fev1_pct, fvc_pct=fvc_pct,
        wheeze=int(wheeze), hemoptysis=int(hemoptysis),
        dyspnea=int(dyspnea), family_history=int(family_hist),
        cough_months=cough_months / 4,
    )

    with st.spinner("מנתח נתונים..."):
        bar = st.progress(0)
        time.sleep(0.3); bar.progress(50)
        p_healthy, p_concern = predict(data)
        time.sleep(0.3); bar.progress(100)
        time.sleep(0.2); bar.empty()

    is_healthy = p_healthy >= 0.60

    # ── תוצאה ראשית ──
    if is_healthy:
        st.markdown(f"""
        <div class="result-good">
            <div class="result-emoji">✅</div>
            <div class="result-title-good">תקין</div>
            <div class="result-msg">הנתונים הקליניים אינם מצביעים על ממצאים חריגים.<br>המשיכו לשמור על אורח חיים בריא.</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-bad">
            <div class="result-emoji">⚠️</div>
            <div class="result-title-bad">מומלץ לבדוק</div>
            <div class="result-msg">זוהו נתונים שכדאי לבדוק לעומק.<br>אנו ממליצים לפנות לרופא ריאות לבדיקה נוספת.</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── גרף 1: הסתברות תקין vs חשוד ──
    st.markdown("**📊 הסתברות לפי קטגוריה**")
    prob_df = pd.DataFrame({
        "קטגוריה": ["תקין ✅", "מומלץ לבדוק ⚠️"],
        "הסתברות (%)": [round(p_healthy * 100, 1), round(p_concern * 100, 1)]
    }).set_index("קטגוריה")
    st.bar_chart(prob_df, color=["#48bb78"], height=200)

    # ── גרף 2: רדאר — פרופיל סיכון לפי גורמים ──
    st.markdown("**📈 פרופיל גורמי הסיכון שלך**")

    risk_factors = {
        "עישון":            min(smoking_yrs / 30, 1.0),
        "FEV1 נמוך":        max(0, (85 - fev1_pct) / 55),
        "FVC נמוך":         max(0, (85 - fvc_pct) / 45),
        "תסמינים":          (int(wheeze) + int(dyspnea) + int(hemoptysis) + min(cough_months/12,1)) / 4,
        "גיל":              (age - 20) / 70,
        "היסטוריה משפחתית": float(family_hist),
    }

    rf_df = pd.DataFrame({
        "גורם סיכון": list(risk_factors.keys()),
        "עוצמה (0–1)": [round(v, 2) for v in risk_factors.values()]
    }).set_index("גורם סיכון")

    st.bar_chart(rf_df, color=["#fc8181"], height=260)

    # ── גרף 3: ספירומטריה vs נורמה ──
    st.markdown("**🫁 ספירומטריה — הנתונים שלך מול הנורמה**")
    spiro_df = pd.DataFrame({
        "מדד": ["FEV1", "FVC"],
        "הנתונים שלך (%)": [fev1_pct, fvc_pct],
        "נורמה (%)":        [80, 80],
    }).set_index("מדד")
    st.bar_chart(spiro_df, color=["#667eea", "#cbd5e0"], height=220)

    st.markdown("""
    <div class="disclaimer">
        ⚠️ <b>שימו לב:</b> תוצאה זו מיועדת לצורכי הדגמה בלבד ואינה מהווה אבחון רפואי.
        בכל מקרה של חשש יש לפנות לרופא מוסמך.
    </div>
    """, unsafe_allow_html=True)
