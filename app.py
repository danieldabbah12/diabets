import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ────────────────────────────────────────────────
st.set_page_config(
    page_title="DiabetesIQ · מנוע חיזוי סיכון",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── GLOBAL CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* Root variables */
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

/* Global reset */
html, body, [class*="css"] {
    font-family: var(--font-body);
    color: var(--text);
    background-color: var(--bg);
}

/* Background grid */
.stApp {
    background-color: var(--bg);
    background-image:
        linear-gradient(rgba(56,189,248,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(56,189,248,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
}

/* Hide streamlit chrome */
#MainMenu, footer, header {visibility: hidden;}

/* Tabs */
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

/* Slider */
.stSlider [data-baseweb="slider"] {
    padding: 0;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 18px 20px;
}
[data-testid="metric-container"] label {
    color: var(--muted);
    font-size: 0.78rem;
    font-family: var(--font-body);
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: var(--font-head);
    font-size: 2rem;
    font-weight: 800;
    color: var(--accent);
}

/* Buttons */
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
    cursor: pointer;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(56,189,248,0.35);
}

/* Selectbox / number_input */
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}

/* Divider */
hr {
    border-color: var(--border);
    margin: 24px 0;
}

/* Info box */
.info-pill {
    display: inline-block;
    background: rgba(56,189,248,0.1);
    border: 1px solid rgba(56,189,248,0.25);
    color: var(--accent);
    border-radius: 999px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 8px;
}

/* Risk result box */
.risk-box {
    border-radius: 18px;
    padding: 30px 36px;
    text-align: center;
    margin-top: 20px;
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
.risk-title {
    font-family: var(--font-head);
    font-size: 1.8rem;
    font-weight: 800;
    margin-bottom: 8px;
}
.risk-title-high { color: #f87171; }
.risk-title-low  { color: #34d399; }
.risk-sub {
    color: var(--muted);
    font-size: 0.92rem;
    line-height: 1.5;
}

/* Progress bar custom */
.prob-bar-wrap {
    background: var(--surface2);
    border-radius: 999px;
    height: 12px;
    overflow: hidden;
    margin: 16px 0 8px;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.6s ease;
}
.prob-bar-high { background: linear-gradient(90deg, #f87171, #ef4444); }
.prob-bar-low  { background: linear-gradient(90deg, #34d399, #10b981); }

@keyframes fadeUp {
    from { opacity:0; transform: translateY(12px); }
    to   { opacity:1; transform: translateY(0); }
}

/* Sidebar hidden */
[data-testid="collapsedControl"] { display: none; }

/* Section headers */
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
    line-height: 1.2;
}
</style>
""", unsafe_allow_html=True)


# ─── SYNTHETIC DATA GENERATION ──────────────────────────────────
@st.cache_data
def generate_data(n=2000, seed=42):
    rng = np.random.default_rng(seed)

    regions = ["תל אביב", "ירושלים", "חיפה", "באר שבע", "נצרת", "ראשון לציון",
               "פתח תקווה", "אשדוד", "חולון", "בת ים"]
    region_weights = [0.20, 0.15, 0.12, 0.08, 0.07, 0.09, 0.08, 0.06, 0.08, 0.07]

    region = rng.choice(regions, n, p=region_weights)
    age    = rng.integers(20, 80, n)
    bmi    = rng.normal(27, 5, n).clip(16, 50)
    glucose = rng.normal(100, 25, n).clip(60, 280)
    bp      = rng.normal(75, 12, n).clip(45, 130)
    insulin = rng.normal(80, 40, n).clip(0, 300)
    skin_thickness = rng.normal(23, 9, n).clip(5, 60)
    pregnancies = rng.integers(0, 12, n)
    dpf     = rng.uniform(0.08, 2.5, n)  # diabetes pedigree function
    smoker  = rng.choice([0, 1], n, p=[0.72, 0.28])
    activity = rng.choice(["נמוכה", "בינונית", "גבוהה"], n, p=[0.35, 0.40, 0.25])

    # Logistic probability
    log_odds = (
        -6.0
        + 0.04  * age
        + 0.10  * (bmi - 25)
        + 0.025 * (glucose - 90)
        + 0.008 * insulin
        + 0.30  * dpf
        + 0.25  * smoker
        + np.where(activity == "נמוכה", 0.4, np.where(activity == "בינונית", 0.1, -0.3))
        + rng.normal(0, 0.5, n)
    )
    prob = 1 / (1 + np.exp(-log_odds))
    diabetes = (rng.random(n) < prob).astype(int)

    df = pd.DataFrame({
        "גיל": age,
        "אזור": region,
        "BMI": bmi.round(1),
        "גלוקוז": glucose.round(1),
        "לחץ_דם": bp.round(1),
        "אינסולין": insulin.round(1),
        "עובי_עור": skin_thickness.round(1),
        "הריונות": pregnancies,
        "DPF": dpf.round(3),
        "מעשן": smoker,
        "פעילות_גופנית": activity,
        "סוכרת": diabetes,
    })
    return df


@st.cache_resource
def train_model(df):
    feature_cols = ["גיל","BMI","גלוקוז","לחץ_דם","אינסולין","עובי_עור","הריונות","DPF","מעשן"]
    activity_map = {"נמוכה": 0, "בינונית": 1, "גבוהה": 2}
    X = df[feature_cols].copy()
    X["פעילות_גופנית"] = df["פעילות_גופנית"].map(activity_map)
    y = df["סוכרת"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.08, max_depth=4, random_state=42)
    model.fit(X_train_s, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(X_test_s)[:,1])
    return model, scaler, feature_cols, auc


# ─── PLOTLY THEME ───────────────────────────────────────────────
COLORS = {
    "accent": "#38bdf8",
    "accent2": "#818cf8",
    "danger": "#f87171",
    "success": "#34d399",
    "warn": "#fbbf24",
    "bg": "#111827",
    "grid": "rgba(255,255,255,0.04)",
    "text": "#94a3b8",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color=COLORS["text"], size=12),
    title_font=dict(family="Syne, sans-serif", color="#e2e8f0", size=15, weight=700),
    xaxis=dict(gridcolor=COLORS["grid"], linecolor="rgba(0,0,0,0)", tickfont=dict(color=COLORS["text"])),
    yaxis=dict(gridcolor=COLORS["grid"], linecolor="rgba(0,0,0,0)", tickfont=dict(color=COLORS["text"])),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=COLORS["text"])),
    margin=dict(l=10, r=10, t=40, b=10),
)


# ─── LOAD DATA & MODEL ──────────────────────────────────────────
df = generate_data()
model, scaler, feature_cols, auc_score = train_model(df)
diabetic = df[df["סוכרת"] == 1]
healthy  = df[df["סוכרת"] == 0]


# ─── HEADER ─────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 40px 0 30px; text-align:center;">
  <div class="info-pill">🩺 &nbsp;Medical AI · סימולציה בלבד</div>
  <h1 style="font-family:'Syne',sans-serif; font-size:2.8rem; font-weight:800;
             background:linear-gradient(135deg,#38bdf8,#818cf8); -webkit-background-clip:text;
             -webkit-text-fill-color:transparent; margin:10px 0 6px;">
    DiabetesIQ
  </h1>
  <p style="color:#64748b; font-size:1rem; margin:0;">
    מנוע חיזוי סיכון לסוכרת מסוג 2 · מבוסס דאטה סינטטי
  </p>
</div>
""", unsafe_allow_html=True)

# ─── TABS ───────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📊  סטטיסטיקות ותובנות", "🤖  מנוע חיזוי אישי"])


# ══════════════════════════════════════════════════════════════════
# TAB 1 — STATISTICS
# ══════════════════════════════════════════════════════════════════
with tab1:

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    prevalence = df["סוכרת"].mean() * 100
    avg_age_d  = diabetic["גיל"].mean()
    avg_bmi_d  = diabetic["BMI"].mean()
    col1.metric("סה\"כ מטופלים", f"{len(df):,}")
    col2.metric("שכיחות סוכרת", f"{prevalence:.1f}%")
    col3.metric("גיל ממוצע (חולים)", f"{avg_age_d:.0f}")
    col4.metric("BMI ממוצע (חולים)", f"{avg_bmi_d:.1f}")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── ROW 1: Age distribution + Region map ──
    r1c1, r1c2 = st.columns([1.2, 1])

    with r1c1:
        st.markdown('<div class="section-label">התפלגות לפי גיל</div>', unsafe_allow_html=True)

        age_bins = pd.cut(df["גיל"], bins=range(20, 85, 5))
        age_grp = df.groupby([age_bins, "סוכרת"]).size().unstack(fill_value=0)
        age_grp.index = [str(i) for i in age_grp.index]

        fig_age = go.Figure()
        fig_age.add_bar(
            name="ללא סוכרת", x=age_grp.index, y=age_grp.get(0, 0),
            marker_color=COLORS["accent"], opacity=0.7
        )
        fig_age.add_bar(
            name="עם סוכרת", x=age_grp.index, y=age_grp.get(1, 0),
            marker_color=COLORS["danger"]
        )
        fig_age.update_layout(**PLOTLY_LAYOUT, barmode="stack",
                              title="מספר מקרים לפי קבוצת גיל",
                              xaxis_title="קבוצת גיל", yaxis_title="מספר אנשים")
        st.plotly_chart(fig_age, use_container_width=True)

    with r1c2:
        st.markdown('<div class="section-label">שכיחות לפי אזור</div>', unsafe_allow_html=True)

        region_stats = df.groupby("אזור").agg(
            total=("סוכרת","count"),
            diabetic=("סוכרת","sum")
        )
        region_stats["rate"] = region_stats["diabetic"] / region_stats["total"] * 100
        region_stats = region_stats.sort_values("rate", ascending=True)

        fig_region = go.Figure(go.Bar(
            x=region_stats["rate"],
            y=region_stats.index,
            orientation="h",
            marker=dict(
                color=region_stats["rate"],
                colorscale=[[0,"#38bdf8"],[0.5,"#818cf8"],[1,"#f87171"]],
                showscale=False,
            ),
            text=[f"{v:.1f}%" for v in region_stats["rate"]],
            textposition="outside",
            textfont=dict(color="#e2e8f0", size=11)
        ))
        fig_region.update_layout(**PLOTLY_LAYOUT,
                                  title="% סוכרת לפי אזור מגורים",
                                  xaxis_title="שכיחות (%)",
                                  xaxis=dict(gridcolor=COLORS["grid"], range=[0, region_stats["rate"].max()+5]))
        st.plotly_chart(fig_region, use_container_width=True)

    # ── ROW 2: BMI violin + Glucose box ──
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        st.markdown('<div class="section-label">BMI לפי סטטוס</div>', unsafe_allow_html=True)
        fig_bmi = go.Figure()
        for status, label, color in [(0,"בריא",COLORS["accent"]), (1,"סוכרתי",COLORS["danger"])]:
            sub = df[df["סוכרת"]==status]["BMI"]
            fig_bmi.add_trace(go.Violin(
                y=sub, name=label,
                box_visible=True, meanline_visible=True,
                fillcolor=color, opacity=0.6,
                line_color=color,
            ))
        fig_bmi.update_layout(**PLOTLY_LAYOUT, title="התפלגות BMI", violinmode="group")
        st.plotly_chart(fig_bmi, use_container_width=True)

    with r2c2:
        st.markdown('<div class="section-label">גלוקוז לפי קבוצת גיל</div>', unsafe_allow_html=True)
        df_tmp = df.copy()
        df_tmp["קבוצת_גיל"] = pd.cut(df["גיל"], bins=[20,35,50,65,80],
                                      labels=["20-35","35-50","50-65","65-80"])
        fig_glu = go.Figure()
        for status, label, color in [(0,"בריא",COLORS["accent"]), (1,"סוכרתי",COLORS["danger"])]:
            sub = df_tmp[df_tmp["סוכרת"]==status]
            fig_glu.add_trace(go.Box(
                x=sub["קבוצת_גיל"].astype(str), y=sub["גלוקוז"],
                name=label, marker_color=color, line_color=color, fillcolor=color,
                opacity=0.65, boxmean=True,
            ))
        fig_glu.update_layout(**PLOTLY_LAYOUT, title="גלוקוז לפי גיל וסטטוס", boxmode="group",
                               xaxis_title="קבוצת גיל", yaxis_title="גלוקוז (mg/dL)")
        st.plotly_chart(fig_glu, use_container_width=True)

    # ── ROW 3: Correlation heatmap + Activity pie ──
    r3c1, r3c2 = st.columns([1.4, 1])

    with r3c1:
        st.markdown('<div class="section-label">קורלציות בין פיצ\'רים</div>', unsafe_allow_html=True)
        num_cols = ["גיל","BMI","גלוקוז","לחץ_דם","אינסולין","DPF","סוכרת"]
        corr = df[num_cols].corr().round(2)
        fig_heat = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index,
            colorscale=[[0,"#818cf8"],[0.5,"#111827"],[1,"#f87171"]],
            text=corr.values, texttemplate="%{text}",
            textfont=dict(size=10, color="white"),
            zmin=-1, zmax=1,
        ))
        fig_heat.update_layout(**PLOTLY_LAYOUT, title="מטריצת קורלציות")
        st.plotly_chart(fig_heat, use_container_width=True)

    with r3c2:
        st.markdown('<div class="section-label">פעילות גופנית לפי סטטוס</div>', unsafe_allow_html=True)
        act_d = diabetic["פעילות_גופנית"].value_counts()
        act_h = healthy["פעילות_גופנית"].value_counts()

        fig_act = make_subplots(rows=1, cols=2, specs=[[{"type":"pie"},{"type":"pie"}]],
                                subplot_titles=["סוכרתיים","בריאים"])
        fig_act.add_trace(go.Pie(
            labels=act_d.index, values=act_d.values,
            hole=0.5, marker_colors=[COLORS["danger"],"#fb923c",COLORS["warn"]],
            textfont=dict(color="white", size=11),
        ), 1, 1)
        fig_act.add_trace(go.Pie(
            labels=act_h.index, values=act_h.values,
            hole=0.5, marker_colors=[COLORS["accent"],COLORS["accent2"],COLORS["success"]],
            textfont=dict(color="white", size=11),
        ), 1, 2)
        fig_act.update_layout(**PLOTLY_LAYOUT, title="התפלגות פעילות גופנית",
                               showlegend=True)
        st.plotly_chart(fig_act, use_container_width=True)

    # ── ROW 4: Feature importance ──
    st.markdown('<div class="section-label">חשיבות פיצ\'רים במודל</div>', unsafe_allow_html=True)
    fi = pd.Series(model.feature_importances_,
                   index=feature_cols + ["פעילות_גופנית"]).sort_values(ascending=True)
    fig_fi = go.Figure(go.Bar(
        x=fi.values, y=fi.index, orientation="h",
        marker=dict(
            color=fi.values,
            colorscale=[[0,COLORS["accent2"]],[1,COLORS["accent"]]],
        ),
        text=[f"{v:.3f}" for v in fi.values],
        textposition="outside",
        textfont=dict(color="#e2e8f0", size=11),
    ))
    fig_fi.update_layout(**PLOTLY_LAYOUT, title="חשיבות כל משתנה לחיזוי סוכרת",
                          xaxis_title="Feature Importance",
                          height=320)
    st.plotly_chart(fig_fi, use_container_width=True)

    # Model accuracy badge
    st.markdown(f"""
    <div style="text-align:center; padding: 16px;
                background: var(--surface); border:1px solid var(--border); border-radius:14px;">
      <span style="font-family:'Syne',sans-serif; font-size:0.8rem; color:var(--muted); letter-spacing:0.1em; text-transform:uppercase;">
        דיוק המודל (AUC-ROC)
      </span><br>
      <span style="font-family:'Syne',sans-serif; font-size:2.2rem; font-weight:800;
                   background:linear-gradient(135deg,#38bdf8,#818cf8);
                   -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
        {auc_score:.3f}
      </span>
      <span style="color:var(--muted); font-size:0.85rem;"> / 1.000</span>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# TAB 2 — PREDICTION ENGINE
# ══════════════════════════════════════════════════════════════════
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

        age_in   = st.slider("גיל", 18, 80, 40, help="גילך בשנים")
        bmi_in   = st.slider("BMI (מדד מסת גוף)", 15.0, 50.0, 25.0, 0.1,
                             help="BMI = משקל(ק\"ג) / גובה²(מ')")
        glucose_in = st.slider("רמת גלוקוז בצום (mg/dL)", 60, 280, 100,
                               help="רמת סוכר בדם לאחר 8 שעות צום")
        bp_in    = st.slider("לחץ דם דיאסטולי (mmHg)", 40, 130, 72,
                             help="הערך הנמוך בבדיקת לחץ הדם")
        insulin_in = st.slider("אינסולין (μU/mL)", 0, 300, 80,
                               help="רמת אינסולין בסרום")

        st.markdown("**🧬 נתונים נוספים**")

        skin_in  = st.slider("עובי קפל עור (mm)", 5, 60, 23,
                             help="עובי קפל עור ב-Triceps")
        preg_in  = st.slider("מספר הריונות", 0, 12, 0)
        dpf_in   = st.slider("היסטוריה משפחתית (DPF)", 0.08, 2.5, 0.47, 0.01,
                             help="ציון המשקלל היסטוריה גנטית — ערך גבוה = סיכון גנטי גבוה יותר")

        col_a, col_b = st.columns(2)
        smoker_in   = col_a.selectbox("מעשן/ת?", ["לא", "כן"])
        activity_in = col_b.selectbox("פעילות גופנית", ["נמוכה", "בינונית", "גבוהה"])

    with right:
        st.markdown("**🤖 תוצאת המודל**")

        # Build input vector
        activity_map = {"נמוכה": 0, "בינונית": 1, "גבוהה": 2}
        X_user = np.array([[
            age_in, bmi_in, glucose_in, bp_in, insulin_in,
            skin_in, preg_in, dpf_in,
            1 if smoker_in == "כן" else 0,
            activity_map[activity_in]
        ]])
        X_user_s = scaler.transform(X_user)
        prob = model.predict_proba(X_user_s)[0][1]
        pct  = prob * 100

        # Determine risk level
        if prob >= 0.65:
            risk_label = "סיכון גבוה"
            risk_class = "risk-high"
            title_class = "risk-title-high"
            bar_class = "prob-bar-high"
            emoji = "⚠️"
            advice = "מומלץ מאוד לפנות לרופא לבדיקת דם מקיפה ולהתחיל שינויים באורח החיים."
        elif prob >= 0.40:
            risk_label = "סיכון בינוני"
            risk_class = "risk-high"
            title_class = "risk-title-high"
            bar_class = "prob-bar-high"
            emoji = "🔶"
            advice = "קיים סיכון מסוים. כדאי לדון עם הרופא המשפחה ולשפר הרגלי תזונה."
        else:
            risk_label = "סיכון נמוך"
            risk_class = "risk-low"
            title_class = "risk-title-low"
            bar_class = "prob-bar-low"
            emoji = "✅"
            advice = "הפרופיל שלך מצביע על סיכון נמוך. המשך לשמור על אורח חיים בריא!"

        st.markdown(f"""
        <div class="risk-box {risk_class}">
          <div style="font-size:2.8rem; margin-bottom:8px;">{emoji}</div>
          <div class="risk-title {title_class}">{risk_label}</div>
          <div class="prob-bar-wrap">
            <div class="prob-bar-fill {bar_class}" style="width:{pct:.1f}%"></div>
          </div>
          <div style="font-size:1.6rem; font-family:'Syne',sans-serif; font-weight:800;
                      color:#e2e8f0; margin-bottom:4px;">
            {pct:.1f}% הסתברות לסוכרת
          </div>
          <div class="risk-sub">{advice}</div>
        </div>
        """, unsafe_allow_html=True)

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": "%", "font": {"size": 32, "family": "Syne", "color": "#e2e8f0"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": COLORS["text"],
                         "tickfont": {"color": COLORS["text"]}},
                "bar": {"color": COLORS["danger"] if prob >= 0.4 else COLORS["success"],
                        "thickness": 0.25},
                "bgcolor": COLORS["bg"],
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40],   "color": "rgba(52,211,153,0.12)"},
                    {"range": [40, 65],  "color": "rgba(251,191,36,0.12)"},
                    {"range": [65, 100], "color": "rgba(248,113,113,0.12)"},
                ],
                "threshold": {"line": {"color": "white", "width": 2}, "value": pct}
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans", color=COLORS["text"]),
            height=220, margin=dict(l=20, r=20, t=20, b=10)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Contributing factors
        st.markdown("**📌 גורמי סיכון בולטים עבורך:**")
        factors = []
        if glucose_in > 125: factors.append(("🔴 גלוקוז גבוה", f"{glucose_in} mg/dL"))
        if bmi_in > 30:      factors.append(("🔴 BMI גבוה", f"{bmi_in}"))
        if age_in > 55:      factors.append(("🟡 גיל מבוגר", f"{age_in} שנים"))
        if dpf_in > 1.0:     factors.append(("🟡 היסטוריה משפחתית", f"DPF {dpf_in}"))
        if smoker_in == "כן": factors.append(("🟠 עישון", "פעיל"))
        if activity_in == "נמוכה": factors.append(("🟡 פעילות נמוכה", ""))
        if not factors:      factors.append(("✅ אין גורמי סיכון בולטים", ""))

        for label, val in factors:
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:center;
                        background:var(--surface2); border:1px solid var(--border);
                        border-radius:10px; padding:10px 16px; margin-bottom:8px;">
              <span style="font-size:0.9rem;">{label}</span>
              <span style="color:var(--muted); font-size:0.85rem;">{val}</span>
            </div>
            """, unsafe_allow_html=True)

    # Disclaimer
    st.markdown("""
    <div style="text-align:center; margin-top:32px; padding:16px;
                border:1px solid rgba(251,191,36,0.2); border-radius:12px;
                background:rgba(251,191,36,0.05);">
      <span style="color:#fbbf24; font-size:0.82rem;">
        ⚠️ &nbsp;<strong>אזהרה:</strong> אפליקציה זו מבוססת על דאטה סינטטי ומיועדת לדמונסטרציה בלבד.
        אין להשתמש בה לצרכים רפואיים. פנה/י לרופא מוסמך לייעוץ מקצועי.
      </span>
    </div>
    """, unsafe_allow_html=True)
