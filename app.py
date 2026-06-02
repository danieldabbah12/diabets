import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="DiabetesIQ · מנוע חיזוי סיכון",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg: #0a0e1a;
    --surface: #111827;
    --surface2: #1a2235;
    --border: rgba(99,179,237,0.12);
    --accent: #38bdf8;
    --accent2: #818cf8;
    --accent3: #34d399;
    --danger: #f87171;
    --warn: #fbbf24;
    --text: #e2e8f0;
    --muted: #64748b;
    --font-head: 'Syne', sans-serif;
    --font-body: 'DM Sans', sans-serif;
}

html, body, [class*="css"] {
    font-family: var(--font-body);
    color: var(--text);
    background-color: var(--bg);
}

.stApp {
    background-color: var(--bg);
    background-image:
        linear-gradient(rgba(56,189,248,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(56,189,248,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
}

#MainMenu, footer, header {visibility: hidden;}

.stTabs [data-baseweb="tab-list"] {
    background: var(--surface);
    border-radius: 16px;
    padding: 6px;
    border: 1px solid var(--border);
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    font-family: var(--font-head);
    font-weight: 600;
    font-size: 0.85rem;
    color: var(--muted);
    padding: 10px 22px;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: white !important;
}

[data-testid="metric-container"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 18px 20px;
}
[data-testid="metric-container"] label {
    color: var(--muted);
    font-size: 0.78rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: var(--font-head);
    font-size: 2rem;
    font-weight: 800;
    color: var(--accent);
}

.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    color: white;
    border: none;
    border-radius: 12px;
    font-family: var(--font-head);
    font-weight: 700;
    padding: 14px 32px;
    font-size: 1rem;
    width: 100%;
    transition: all 0.2s;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(56,189,248,0.35);
}

.risk-box {
    border-radius: 18px;
    padding: 28px 32px;
    text-align: center;
    margin: 16px 0;
    animation: fadeUp 0.4s ease;
}
.risk-high {
    background: linear-gradient(135deg, rgba(248,113,113,0.12), rgba(239,68,68,0.06));
    border: 1px solid rgba(248,113,113,0.35);
}
.risk-low {
    background: linear-gradient(135deg, rgba(52,211,153,0.12), rgba(16,185,129,0.06));
    border: 1px solid rgba(52,211,153,0.35);
}
.risk-title { font-family: var(--font-head); font-size: 1.8rem; font-weight: 800; margin-bottom: 8px; }
.risk-title-high { color: #f87171; }
.risk-title-low  { color: #34d399; }
.risk-sub { color: var(--muted); font-size: 0.92rem; line-height: 1.5; }

.prob-bar-wrap {
    background: var(--surface2);
    border-radius: 999px;
    height: 14px;
    overflow: hidden;
    margin: 14px 0 8px;
}
.prob-bar-fill { height: 100%; border-radius: 999px; }
.prob-bar-high { background: linear-gradient(90deg, #f87171, #ef4444); }
.prob-bar-low  { background: linear-gradient(90deg, #34d399, #10b981); }

.factor-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 10px 16px;
    margin-bottom: 8px;
}

.section-label {
    font-family: var(--font-head);
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 4px;
    font-weight: 700;
}
.section-title {
    font-family: var(--font-head);
    font-size: 1.6rem;
    font-weight: 800;
    color: var(--text);
    margin-bottom: 20px;
}

@keyframes fadeUp {
    from { opacity:0; transform: translateY(12px); }
    to   { opacity:1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)


# ── DATA ──────────────────────────────────────────────────────────
@st.cache_data
def generate_data(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    regions = ["תל אביב","ירושלים","חיפה","באר שבע","נצרת",
               "ראשון לציון","פתח תקווה","אשדוד","חולון","בת ים"]
    region_w = [0.20,0.15,0.12,0.08,0.07,0.09,0.08,0.06,0.08,0.07]
    region   = rng.choice(regions, n, p=region_w)
    age      = rng.integers(20, 80, n)
    bmi      = rng.normal(27, 5, n).clip(16, 50)
    glucose  = rng.normal(100, 25, n).clip(60, 280)
    bp       = rng.normal(75, 12, n).clip(45, 130)
    insulin  = rng.normal(80, 40, n).clip(0, 300)
    skin     = rng.normal(23, 9, n).clip(5, 60)
    preg     = rng.integers(0, 12, n)
    dpf      = rng.uniform(0.08, 2.5, n)
    smoker   = rng.choice([0,1], n, p=[0.72,0.28])
    activity = rng.choice(["נמוכה","בינונית","גבוהה"], n, p=[0.35,0.40,0.25])

    log_odds = (
        -6.0 + 0.04*age + 0.10*(bmi-25) + 0.025*(glucose-90)
        + 0.008*insulin + 0.30*dpf + 0.25*smoker
        + np.where(activity=="נמוכה", 0.4, np.where(activity=="בינונית", 0.1, -0.3))
        + rng.normal(0, 0.5, n)
    )
    prob     = 1 / (1 + np.exp(-log_odds))
    diabetes = (rng.random(n) < prob).astype(int)

    return pd.DataFrame({
        "גיל": age, "אזור": region,
        "BMI": bmi.round(1), "גלוקוז": glucose.round(1),
        "לחץ_דם": bp.round(1), "אינסולין": insulin.round(1),
        "עובי_עור": skin.round(1), "הריונות": preg,
        "DPF": dpf.round(3), "מעשן": smoker,
        "פעילות_גופנית": activity, "סוכרת": diabetes,
    })


# מודל לוגיסטי פשוט — ללא sklearn
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def simple_predict(row):
    """חישוב הסתברות פשוט מבוסס מאפיינים ידועים מהספרות"""
    score = (
        -6.5
        + 0.042  * row["גיל"]
        + 0.11   * (row["BMI"] - 25)
        + 0.028  * (row["גלוקוז"] - 90)
        + 0.006  * row["אינסולין"]
        + 0.35   * row["DPF"]
        + 0.28   * row["מעשן"]
        + {"נמוכה": 0.45, "בינונית": 0.1, "גבוהה": -0.35}.get(row["פעילות_גופנית"], 0)
    )
    return float(sigmoid(score))

df = generate_data()
df["סיכון"] = df.apply(simple_predict, axis=1)
diabetic = df[df["סוכרת"]==1]
healthy  = df[df["סוכרת"]==0]

# ── HEADER ───────────────────────────────────────────────────────
st.markdown("""
<div style="padding:36px 0 24px; text-align:center;">
  <div style="display:inline-block; background:rgba(56,189,248,0.1);
              border:1px solid rgba(56,189,248,0.25); color:#38bdf8;
              border-radius:999px; padding:4px 16px; font-size:0.78rem;
              font-weight:600; letter-spacing:0.07em; text-transform:uppercase; margin-bottom:10px;">
    🩺 &nbsp;Medical AI · סימולציה בלבד
  </div>
  <h1 style="font-family:'Syne',sans-serif; font-size:2.8rem; font-weight:800;
             background:linear-gradient(135deg,#38bdf8,#818cf8);
             -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:8px 0 4px;">
    DiabetesIQ
  </h1>
  <p style="color:#64748b; font-size:0.95rem; margin:0;">
    מנוע חיזוי סיכון לסוכרת מסוג 2 · דאטה סינטטי
  </p>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📊  סטטיסטיקות ותובנות", "🤖  מנוע חיזוי אישי"])


# ══════════════════════════════════════════════════════
# TAB 1
# ══════════════════════════════════════════════════════
with tab1:

    # KPIs
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("סה\"כ מטופלים",    f"{len(df):,}")
    c2.metric("שכיחות סוכרת",     f"{df['סוכרת'].mean()*100:.1f}%")
    c3.metric("גיל ממוצע (חולים)", f"{diabetic['גיל'].mean():.0f}")
    c4.metric("BMI ממוצע (חולים)", f"{diabetic['BMI'].mean():.1f}")

    st.markdown("---")

    # ── גיל + אזור ──
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-label">התפלגות לפי קבוצת גיל</div>', unsafe_allow_html=True)
        df["קבוצת_גיל"] = pd.cut(df["גיל"], bins=range(20,85,10),
                                  labels=["20-30","30-40","40-50","50-60","60-70","70-80"])
        age_chart = df.groupby(["קבוצת_גיל","סוכרת"]).size().unstack(fill_value=0)
        age_chart.columns = ["בריא","סוכרתי"]
        st.bar_chart(age_chart, color=["#38bdf8","#f87171"])

    with col2:
        st.markdown('<div class="section-label">שכיחות סוכרת לפי אזור (%)</div>', unsafe_allow_html=True)
        region_rate = (
            df.groupby("אזור")["סוכרת"]
            .mean().mul(100).round(1)
            .sort_values(ascending=False)
            .rename("שכיחות %")
        )
        st.bar_chart(region_rate, color="#818cf8")

    st.markdown("---")

    # ── BMI + גלוקוז ──
    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="section-label">BMI ממוצע לפי קבוצת גיל וסטטוס</div>', unsafe_allow_html=True)
        bmi_chart = df.groupby(["קבוצת_גיל","סוכרת"])["BMI"].mean().unstack()
        bmi_chart.columns = ["בריא","סוכרתי"]
        st.line_chart(bmi_chart, color=["#38bdf8","#f87171"])

    with col4:
        st.markdown('<div class="section-label">גלוקוז ממוצע לפי קבוצת גיל וסטטוס</div>', unsafe_allow_html=True)
        glu_chart = df.groupby(["קבוצת_גיל","סוכרת"])["גלוקוז"].mean().unstack()
        glu_chart.columns = ["בריא","סוכרתי"]
        st.line_chart(glu_chart, color=["#38bdf8","#f87171"])

    st.markdown("---")

    # ── פעילות גופנית + עישון ──
    col5, col6 = st.columns(2)

    with col5:
        st.markdown('<div class="section-label">שכיחות סוכרת לפי פעילות גופנית (%)</div>', unsafe_allow_html=True)
        act_rate = (
            df.groupby("פעילות_גופנית")["סוכרת"]
            .mean().mul(100).round(1)
            .reindex(["נמוכה","בינונית","גבוהה"])
            .rename("שכיחות %")
        )
        st.bar_chart(act_rate, color="#34d399")

    with col6:
        st.markdown('<div class="section-label">שכיחות סוכרת: מעשנים vs לא מעשנים (%)</div>', unsafe_allow_html=True)
        smoke_rate = (
            df.groupby("מעשן")["סוכרת"]
            .mean().mul(100).round(1)
        )
        smoke_rate.index = ["לא מעשן","מעשן"]
        smoke_rate.name = "שכיחות %"
        st.bar_chart(smoke_rate, color="#fbbf24")

    st.markdown("---")

    # ── חשיבות פיצ'רים ──
    st.markdown('<div class="section-label">חשיבות פיצ\'רים במודל</div>', unsafe_allow_html=True)
    fi = pd.Series(
        model.feature_importances_,
        index=feature_cols + ["פעילות_גופנית"]
    ).sort_values(ascending=False).rename("חשיבות")
    st.bar_chart(fi, color="#818cf8")

    # AUC badge
    st.markdown(f"""
    <div style="text-align:center; margin-top:20px; padding:18px;
                background:#111827; border:1px solid rgba(99,179,237,0.12);
                border-radius:14px;">
      <span style="color:#64748b; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.1em;">
        דיוק המודל (AUC-ROC)
      </span><br>
      <span style="font-family:'Syne',sans-serif; font-size:2.4rem; font-weight:800;
                   background:linear-gradient(135deg,#38bdf8,#818cf8);
                   -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
        {auc_score:.3f}
      </span>
      <span style="color:#64748b; font-size:0.9rem;"> / 1.000</span>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# TAB 2
# ══════════════════════════════════════════════════════
with tab2:

    st.markdown("""
    <div style="margin-bottom:24px;">
      <div class="section-label">הזן את הנתונים שלך</div>
      <div class="section-title">חישוב סיכון אישי לסוכרת</div>
      <p style="color:#64748b; margin-top:-12px; font-size:0.9rem;">
        הזז את הסליידרים לפי הערכים האישיים שלך וקבל הערכת סיכון מיידית
      </p>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1.3, 1])

    with left:
        st.markdown("**📋 נתונים ביומטריים**")
        age_in     = st.slider("גיל", 18, 80, 40)
        bmi_in     = st.slider("BMI", 15.0, 50.0, 25.0, 0.1)
        glucose_in = st.slider("גלוקוז בצום (mg/dL)", 60, 280, 100)
        bp_in      = st.slider("לחץ דם דיאסטולי (mmHg)", 40, 130, 72)
        insulin_in = st.slider("אינסולין (μU/mL)", 0, 300, 80)

        st.markdown("**🧬 נתונים נוספים**")
        skin_in    = st.slider("עובי קפל עור (mm)", 5, 60, 23)
        preg_in    = st.slider("מספר הריונות", 0, 12, 0)
        dpf_in     = st.slider("DPF — היסטוריה משפחתית", 0.08, 2.5, 0.47, 0.01)

        ca, cb = st.columns(2)
        smoker_in   = ca.selectbox("מעשן/ת?", ["לא","כן"])
        activity_in = cb.selectbox("פעילות גופנית", ["נמוכה","בינונית","גבוהה"])

    with right:
        amap = {"נמוכה":0,"בינונית":1,"גבוהה":2}
        X_user = np.array([[
            age_in, bmi_in, glucose_in, bp_in, insulin_in,
            skin_in, preg_in, dpf_in,
            1 if smoker_in=="כן" else 0,
            amap[activity_in]
        ]])
        row_dict = {
            "גיל": age_in, "BMI": bmi_in, "גלוקוז": glucose_in,
            "אינסולין": insulin_in, "DPF": dpf_in,
            "מעשן": 1 if smoker_in=="כן" else 0,
            "פעילות_גופנית": activity_in,
        }
        prob = simple_predict(row_dict)
        pct  = prob * 100

        if prob >= 0.65:
            risk_label, risk_class, title_class, bar_class = "סיכון גבוה","risk-high","risk-title-high","prob-bar-high"
            emoji  = "⚠️"
            advice = "מומלץ לפנות לרופא לבדיקת דם ולשנות הרגלים."
        elif prob >= 0.40:
            risk_label, risk_class, title_class, bar_class = "סיכון בינוני","risk-high","risk-title-high","prob-bar-high"
            emoji  = "🔶"
            advice = "קיים סיכון מסוים. כדאי לדון עם הרופא המשפחה."
        else:
            risk_label, risk_class, title_class, bar_class = "סיכון נמוך","risk-low","risk-title-low","prob-bar-low"
            emoji  = "✅"
            advice = "פרופיל בסיכון נמוך. המשך לשמור על אורח חיים בריא!"

        st.markdown(f"""
        <div class="risk-box {risk_class}">
          <div style="font-size:2.8rem;">{emoji}</div>
          <div class="risk-title {title_class}">{risk_label}</div>
          <div class="prob-bar-wrap">
            <div class="prob-bar-fill {bar_class}" style="width:{pct:.1f}%"></div>
          </div>
          <div style="font-size:1.6rem; font-family:'Syne',sans-serif; font-weight:800;
                      color:#e2e8f0; margin-bottom:6px;">
            {pct:.1f}% הסתברות לסוכרת
          </div>
          <div class="risk-sub">{advice}</div>
        </div>
        """, unsafe_allow_html=True)

        # Progress bars per factor
        st.markdown("**📌 גורמי סיכון עיקריים:**")

        factors = {
            "גלוקוז": (glucose_in, 60, 280, 125),
            "BMI":    (bmi_in,    15, 50,  30),
            "DPF":    (dpf_in,    0.08, 2.5, 1.0),
            "גיל":    (age_in,    18, 80,  55),
            "לחץ דם": (bp_in,    40, 130,  90),
        }

        for name, (val, lo, hi, threshold) in factors.items():
            norm   = (val - lo) / (hi - lo)
            danger = val > threshold
            color  = "#f87171" if danger else "#34d399"
            flag   = "⬆️" if danger else "✅"
            st.markdown(f"""
            <div style="margin-bottom:10px;">
              <div style="display:flex; justify-content:space-between;
                          font-size:0.85rem; color:#94a3b8; margin-bottom:4px;">
                <span>{flag} {name}</span><span style="color:{color}; font-weight:600;">{val}</span>
              </div>
              <div style="background:#1a2235; border-radius:999px; height:8px; overflow:hidden;">
                <div style="width:{norm*100:.1f}%; height:100%; border-radius:999px;
                            background:{color}; transition:width 0.5s;"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; margin-top:28px; padding:14px;
                border:1px solid rgba(251,191,36,0.2); border-radius:12px;
                background:rgba(251,191,36,0.05);">
      <span style="color:#fbbf24; font-size:0.82rem;">
        ⚠️ <strong>אזהרה:</strong> אפליקציה זו מבוססת על דאטה סינטטי ומיועדת לדמונסטרציה בלבד.
        אין להשתמש בה לצרכים רפואיים.
      </span>
    </div>
    """, unsafe_allow_html=True)
