"""
GlucoVibes — Predictor de pico glucémico postprandial
App de demostración para el Reto Empresarial 2026

Ejecutar con:
    streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# ============================================================
# CONSTANTES
# ============================================================

BASE_DIR      = Path(__file__).parent
MODELS_DIR    = BASE_DIR / "glucovibes" / "clean_data" / "models"
DATASET_PATH  = BASE_DIR / "glucovibes" / "clean_data" / "modeling_dataset_pico.csv"
FOOD_COMP_PATH = BASE_DIR / "glucovibes" / "data" / "food_composition.csv"
MEAL_ITEM_PATH = BASE_DIR / "glucovibes" / "data" / "meal_item.csv"

BRAND_COLOR  = "#00d4a0"   # verde vibrante sobre fondo oscuro
WARN_COLOR   = "#fbbf24"   # amarillo cálido
DANGER_COLOR = "#f87171"   # rojo suave
BG_DARK      = "#0d1117"
BG_CARD      = "#161b22"
BG_SIDEBAR   = "#13181f"
BORDER_COLOR = "#30363d"
TEXT_PRIMARY = "#e6edf3"
TEXT_MUTED   = "#8b949e"

MEAL_PERIOD_HOURS = {"Desayuno": 8, "Media mañana": 11, "Almuerzo": 14, "Merienda": 17, "Cena": 21}
MEAL_PERIOD_ENC   = {"Desayuno": 0, "Media mañana": 3,  "Almuerzo": 2,  "Merienda": 5,  "Cena": 1}
MEAL_TYPE_ENC     = {"Desayuno": 3, "Media mañana": 2,  "Almuerzo": 2,  "Merienda": 5,  "Cena": 1}

CGM_SLOPE_MAP = {
    "↓↓  Bajando rápido": -0.40,
    "↓   Bajando":        -0.15,
    "→   Estable":         0.007,
    "↑   Subiendo":        0.15,
    "↑↑  Subiendo rápido": 0.40,
}

SPORT_INTENSITY = {"Suave (caminar, yoga)": 1.0, "Moderado (ciclismo, natación)": 2.0, "Intenso (correr, HIIT)": 3.0}

# Comidas predefinidas para la demo: {nombre: [(alimento, gramos), ...]}
PRESET_MEALS = {
    "🥑 Desayuno saludable": [
        ("Huevo De Gallina Frito",   120),
        ("Aguacate",                  80),
        ("Pan Integral",              60),
        ("Leche Semidesnatada Pasteurizada", 200),
    ],
    "🍝 Almuerzo clásico": [
        ("Pasta Alimenticia Integral Hervida", 200),
        ("Pan Blanco De Barra",                60),
        ("Aceite De Oliva Virgen Extra",        15),
    ],
    "🍕 Cena informal": [
        ("Pizza 4 Quesos", 300),
        ("Coca-Cola",      330),
    ],
    "🥗 Comida mediterránea": [
        ("Pollo Pechuga Plancha",      150),
        ("Arroz Hervido",              150),
        ("Brocoli Hervido",            120),
        ("Aceite De Oliva Virgen Extra", 10),
    ],
}


# ============================================================
# CARGA DE DATOS (cacheados)
# ============================================================

@st.cache_resource(show_spinner=False)
def load_models():
    model_peak = joblib.load(MODELS_DIR / "model_peak_value.pkl")
    calibrator = joblib.load(MODELS_DIR / "calibrator_peak_value.pkl")
    feat_peak  = joblib.load(MODELS_DIR / "feature_set_peak_value.pkl")
    model_ttp  = joblib.load(MODELS_DIR / "model_ttp_binary.pkl")
    feat_ttp   = joblib.load(MODELS_DIR / "feature_set_ttp.pkl")
    return model_peak, calibrator, feat_peak, model_ttp, feat_ttp


@st.cache_data(show_spinner=False)
def load_food_db():
    """
    Carga la base de datos de alimentos.
    Devuelve un DataFrame indexado por nombre con macros por 100g,
    ordenado por frecuencia de uso en el dataset real.
    """
    food = pd.read_csv(FOOD_COMP_PATH, low_memory=False, encoding="utf-8")
    food = food.rename(columns={
        "food_esp_name":            "nombre",
        "carbohydrate_gram_value":  "carbs_100g",
        "fat_grams_value":          "fat_100g",
        "protein_gram_value":       "protein_100g",
        "fibre_gram_value":         "fibre_100g",
        "atri_high_glycemic_index": "high_gi",
        "atri_ultraprocessed":      "ultraprocessed",
        "saturated_fatty_acid":     "saturated_fat_100g",
        "portion":                  "porcion_default",
    })

    # Filtrar filas con datos nutricionales mínimos
    food = food.dropna(subset=["nombre", "carbs_100g", "fat_100g", "protein_100g"])
    food["nombre"] = food["nombre"].str.strip().str.title()
    food["high_gi"] = food["high_gi"].fillna(0).astype(float).clip(0, 1)
    food["ultraprocessed"] = food["ultraprocessed"].fillna(0).astype(float).clip(0, 1)
    food["fibre_100g"] = food["fibre_100g"].fillna(0.0)
    food["saturated_fat_100g"] = food["saturated_fat_100g"].fillna(0.0)
    food["porcion_default"] = pd.to_numeric(food["porcion_default"], errors="coerce").fillna(100.0).clip(5, 500)

    # Ordenar por frecuencia en el dataset de comidas
    try:
        items = pd.read_csv(MEAL_ITEM_PATH, usecols=["food_name"])
        freq = items["food_name"].str.strip().str.title().value_counts()
        food["freq"] = food["nombre"].map(freq).fillna(0)
        food = food.sort_values("freq", ascending=False)
    except Exception:
        pass

    food = food.drop_duplicates(subset=["nombre"]).reset_index(drop=True)
    return food[["nombre", "carbs_100g", "fat_100g", "protein_100g",
                 "fibre_100g", "high_gi", "ultraprocessed",
                 "saturated_fat_100g", "porcion_default"]].copy()


@st.cache_data(show_spinner=False)
def load_medians():
    df = pd.read_csv(DATASET_PATH)
    medians = df.median(numeric_only=True).to_dict()
    medians["cgm_slope_best"] = medians.get("cgm_slope_60m", 0.007)
    medians["cgm_delta_best"] = medians.get("cgm_delta_60m", 0.0)
    medians["cgm_slope_90m"]  = medians.get("cgm_slope_60m", 0.007)
    medians["cgm_delta_90m"]  = medians.get("cgm_delta_60m", 0.0)
    medians["cgm_std_90m"]    = medians.get("cgm_std_30m",   2.5)

    for col, enc_col in [
        ("meal_period", "meal_period_enc"),
        ("meal_type",   "meal_type_enc"),
        ("sport_prior_intensity", "sport_prior_intensity_enc"),
    ]:
        if col in df.columns and enc_col not in medians:
            le = LabelEncoder()
            medians[enc_col] = float(np.median(
                le.fit_transform(df[col].fillna("unknown").astype(str))
            ))

    if "fat_carb_ratio_meal" not in medians:
        medians["fat_carb_ratio_meal"] = float(
            (df["total_fat"] / df["total_carbs"].replace(0, np.nan)).median()
        )
    if "gi_fiber_interaction" not in medians:
        fc = df["fibre_carb_ratio"].fillna(0).clip(0, 1)
        medians["gi_fiber_interaction"] = float((df["n_high_gi"] * (1 - fc)).median())
    if "sport_2h_insulin_proxy" not in medians:
        score = df.get("sport_intensity_score", pd.Series(0, index=df.index)).fillna(0)
        dur   = df.get("sport_2h_duration",     pd.Series(0, index=df.index)).fillna(0)
        medians["sport_2h_insulin_proxy"] = float((dur * score).median())

    medians.setdefault("sport_intensity_score", 2.0)
    medians.setdefault("n_readings_pre_60m",    12.0)
    medians.setdefault("hour_of_day",           14.0)
    medians.setdefault("day_of_week",            3.0)
    medians.setdefault("is_weekend",             0.0)
    return medians


# ============================================================
# CÁLCULO DE MACROS DESDE ALIMENTOS SELECCIONADOS
# ============================================================

def compute_meal_macros(food_db: pd.DataFrame, selections: dict) -> dict:
    """
    selections: {nombre: gramos}
    Devuelve totales nutricionales de la comida.
    """
    totals = dict(carbs=0.0, fat=0.0, protein=0.0, fibre=0.0,
                  n_items=0, n_high_gi=0, n_ultraprocessed=0, avg_sat_fat=0.0)
    sat_fats = []
    rows = food_db.set_index("nombre")

    for name, grams in selections.items():
        if name not in rows.index or grams <= 0:
            continue
        r  = rows.loc[name]
        f  = grams / 100.0
        totals["carbs"]    += r["carbs_100g"]   * f
        totals["fat"]      += r["fat_100g"]     * f
        totals["protein"]  += r["protein_100g"] * f
        totals["fibre"]    += r["fibre_100g"]   * f
        totals["n_items"]  += 1
        if r["high_gi"] >= 0.5:
            totals["n_high_gi"] += 1
        if r["ultraprocessed"] >= 0.5:
            totals["n_ultraprocessed"] += 1
        sat_fats.append(r["saturated_fat_100g"] * f)

    totals["avg_sat_fat"] = float(np.mean(sat_fats)) if sat_fats else 0.0
    return totals


# ============================================================
# INGENIERÍA DE FEATURES
# ============================================================

def build_feature_vector(
    glucose_preprandial, total_carbs, total_fat, total_protein,
    total_fibre, n_high_gi, n_ultraprocessed, avg_sat_fat,
    cgm_slope, sport_2h_dur, sport_6h_dur, sport_intensity,
    meal_period_label, n_items, medians, feat_peak, feat_ttp,
):
    total_calories = max(total_protein * 4 + total_fat * 9 + total_carbs * 4, 1.0)

    pct_cal_carbs   = (total_carbs * 4   / total_calories) * 100
    pct_cal_protein = (total_protein * 4 / total_calories) * 100
    pct_cal_fat     = (total_fat * 9     / total_calories) * 100

    fibre_carb_ratio   = (total_fibre / total_carbs) if total_carbs > 0 else 0.0
    net_carbs          = max(0.0, total_carbs - total_fibre)
    fat_delay_score    = total_fat * total_carbs
    effective_gi_score = n_high_gi * (1.0 - pct_cal_fat / 100.0)
    carbs_per_item     = (total_carbs / n_items) if n_items > 0 else total_carbs
    pct_high_gi_items  = (n_high_gi  / n_items) if n_items > 0 else 0.0

    meal_period_enc = float(MEAL_PERIOD_ENC[meal_period_label])
    meal_type_enc   = float(MEAL_TYPE_ENC[meal_period_label])
    hour_of_day     = float(MEAL_PERIOD_HOURS[meal_period_label])

    cgm_slope_best = cgm_slope
    cgm_delta_best = cgm_slope * 60.0
    cgm_slope_30m  = cgm_slope
    cgm_delta_30m  = cgm_slope * 30.0
    cgm_slope_90m  = cgm_slope
    cgm_delta_90m  = cgm_slope * 90.0
    cgm_std_30m    = medians.get("cgm_std_30m", 2.5)
    cgm_std_90m    = medians.get("cgm_std_90m", 2.5)

    cgm_tir_pre          = 100.0 if 70 <= glucose_preprandial <= 140 else 0.0
    cgm_pct_above_target = 100.0 if glucose_preprandial > 140 else 0.0

    sport_intensity_score  = sport_intensity
    sport_2h_insulin_proxy = sport_2h_dur * sport_intensity_score

    fat_carb_ratio_meal  = (total_fat / total_carbs) if total_carbs > 0 else 0.0
    gi_fiber_interaction = n_high_gi * (1.0 - min(1.0, fibre_carb_ratio))

    overrides = {
        "glucose_preprandial":    glucose_preprandial,
        "total_calories":         total_calories,
        "total_protein":          total_protein,
        "total_fat":              total_fat,
        "total_carbs":            total_carbs,
        "total_fibre":            total_fibre,
        "n_items":                float(n_items),
        "pct_cal_carbs":          pct_cal_carbs,
        "pct_cal_protein":        pct_cal_protein,
        "pct_cal_fat":            pct_cal_fat,
        "fibre_carb_ratio":       fibre_carb_ratio,
        "n_high_gi":              float(n_high_gi),
        "n_ultraprocessed":       float(n_ultraprocessed),
        "avg_saturated_fat":      avg_sat_fat,
        "net_carbs":              net_carbs,
        "fat_delay_score":        fat_delay_score,
        "effective_gi_score":     effective_gi_score,
        "carbs_per_item":         carbs_per_item,
        "pct_high_gi_items":      pct_high_gi_items,
        "hour_of_day":            hour_of_day,
        "meal_period_enc":        meal_period_enc,
        "meal_type_enc":          meal_type_enc,
        "cgm_slope_best":         cgm_slope_best,
        "cgm_delta_best":         cgm_delta_best,
        "cgm_slope_30m":          cgm_slope_30m,
        "cgm_delta_30m":          cgm_delta_30m,
        "cgm_std_30m":            cgm_std_30m,
        "cgm_slope_90m":          cgm_slope_90m,
        "cgm_delta_90m":          cgm_delta_90m,
        "cgm_std_90m":            cgm_std_90m,
        "cgm_tir_pre":            cgm_tir_pre,
        "cgm_pct_above_target":   cgm_pct_above_target,
        "n_readings_pre_60m":     12.0,
        "sport_2h_duration":      float(sport_2h_dur),
        "sport_6h_duration":      float(sport_6h_dur),
        "sport_intensity_score":  sport_intensity_score,
        "sport_2h_insulin_proxy": sport_2h_insulin_proxy,
        "fat_carb_ratio_meal":    fat_carb_ratio_meal,
        "gi_fiber_interaction":   gi_fiber_interaction,
    }

    row = dict(medians)
    row.update(overrides)
    X_peak = pd.DataFrame([{f: row.get(f, 0.0) for f in feat_peak}])
    X_ttp  = pd.DataFrame([{f: row.get(f, 0.0) for f in feat_ttp}])
    return X_peak, X_ttp


# ============================================================
# PREDICCIÓN
# ============================================================

def predict_glucose_peak(X_peak, X_ttp, model_peak, calibrator, feat_peak, model_ttp, feat_ttp):
    peak_raw        = model_peak.predict(X_peak[feat_peak])
    peak_calibrated = float(calibrator.predict(peak_raw)[0])
    ttp_proba       = float(model_ttp.predict_proba(X_ttp[feat_ttp])[0, 1])

    if ttp_proba > 0.65:
        ttp_label, ttp_minutes = "Tardío  (>75 min)", 90
    elif ttp_proba < 0.35:
        ttp_label, ttp_minutes = "Rápido  (<45 min)", 35
    else:
        ttp_label, ttp_minutes = "Incierto  (~60 min)", 60

    if peak_calibrated < 140:
        classification, class_color, class_emoji = "Normal",       BRAND_COLOR,  "✅"
    elif peak_calibrated < 180:
        classification, class_color, class_emoji = "Elevado",      WARN_COLOR,   "⚠️"
    else:
        classification, class_color, class_emoji = "Muy elevado",  DANGER_COLOR, "🔴"

    amplitude = peak_calibrated - float(X_peak["glucose_preprandial"].iloc[0])
    return {
        "peak_value": peak_calibrated, "ttp_proba": ttp_proba,
        "ttp_label": ttp_label,        "ttp_minutes": ttp_minutes,
        "amplitude": amplitude,        "classification": classification,
        "class_color": class_color,    "class_emoji": class_emoji,
    }


# ============================================================
# CURVA GLUCÉMICA
# ============================================================

def build_glucose_curve(glucose_preprandial, peak_value, ttp_minutes, pre_min=30, recovery_min=120):
    amplitude = peak_value - glucose_preprandial
    t_pre  = np.linspace(-pre_min, 0, 30)
    g_pre  = np.full_like(t_pre, glucose_preprandial)

    t_rise   = np.linspace(0, ttp_minutes, max(20, ttp_minutes))
    k        = 8.0 / max(ttp_minutes, 1)
    sig      = 1.0 / (1.0 + np.exp(-k * (t_rise - ttp_minutes * 0.45)))
    sig_norm = (sig - sig[0]) / max(sig[-1] - sig[0], 1e-9)
    g_rise   = glucose_preprandial + amplitude * sig_norm

    lam       = -np.log(0.1) / recovery_min
    t_rec_rel = np.linspace(0, recovery_min, recovery_min)
    g_rec     = peak_value + (glucose_preprandial - peak_value) * (1.0 - np.exp(-lam * t_rec_rel))

    return (
        np.concatenate([t_pre, t_rise[1:], t_rec_rel[1:] + ttp_minutes]),
        np.concatenate([g_pre, g_rise[1:], g_rec[1:]]),
    )


# ============================================================
# GRÁFICO PLOTLY
# ============================================================

def build_plotly_figure(time_axis, glucose, peak_value, ttp_minutes, glucose_preprandial):
    y_min = max(50, glucose_preprandial - 20)
    y_max = max(220, peak_value + 35)

    fig = go.Figure()
    fig.add_hrect(y0=70,  y1=140,       fillcolor="rgba(0,179,134,0.07)", line_width=0, layer="below")
    fig.add_hrect(y0=140, y1=180,       fillcolor="rgba(245,158,11,0.08)", line_width=0, layer="below")
    fig.add_hrect(y0=180, y1=y_max+50,  fillcolor="rgba(239,68,68,0.08)",  line_width=0, layer="below")

    fig.add_hline(y=140, line_dash="dot", line_color="rgba(245,158,11,0.5)", line_width=1.5,
                  annotation_text="140 mg/dL", annotation_position="right",
                  annotation_font=dict(size=11, color=WARN_COLOR))
    fig.add_hline(y=180, line_dash="dot", line_color="rgba(239,68,68,0.5)", line_width=1.5,
                  annotation_text="180 mg/dL", annotation_position="right",
                  annotation_font=dict(size=11, color=DANGER_COLOR))
    fig.add_vline(x=0, line_dash="dash", line_color="rgba(100,100,100,0.4)", line_width=1.5,
                  annotation_text="🍽  Comida", annotation_position="top right",
                  annotation_font=dict(size=12, color="gray"))

    fig.add_trace(go.Scatter(
        x=time_axis, y=glucose, mode="lines", name="Curva de glucosa",
        line=dict(color=BRAND_COLOR, width=3.5),
        hovertemplate="<b>%{x:.0f} min</b>: %{y:.0f} mg/dL<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[ttp_minutes], y=[peak_value], mode="markers+text",
        name=f"Pico: {peak_value:.0f} mg/dL",
        marker=dict(size=15, color=DANGER_COLOR, symbol="star", line=dict(color="white", width=1.5)),
        text=[f"  {peak_value:.0f} mg/dL"], textposition="middle right",
        textfont=dict(size=14, color=DANGER_COLOR),
        hovertemplate=f"Pico: {peak_value:.0f} mg/dL  |  {ttp_minutes} min<extra></extra>",
    ))

    x_max = time_axis[-1]
    fig.add_annotation(x=x_max-8, y=105,  text="Normal",      showarrow=False, font=dict(color=BRAND_COLOR,  size=10), xanchor="right")
    if y_max > 140: fig.add_annotation(x=x_max-8, y=160, text="Elevado",     showarrow=False, font=dict(color=WARN_COLOR,   size=10), xanchor="right")
    if y_max > 180: fig.add_annotation(x=x_max-8, y=min(196,y_max-12), text="Muy elevado", showarrow=False, font=dict(color=DANGER_COLOR, size=10), xanchor="right")

    fig.update_layout(
        title=dict(
            text="Curva glucémica postprandial (simulada)",
            font=dict(size=15, color="#e6edf3"),
        ),
        xaxis_title="Minutos desde la comida",
        yaxis_title="Glucosa (mg/dL)",
        xaxis=dict(
            range=[time_axis[0], x_max],
            color="#8b949e", gridcolor="#21262d", zerolinecolor="#30363d",
        ),
        yaxis=dict(
            range=[y_min, y_max],
            color="#8b949e", gridcolor="#21262d", zerolinecolor="#30363d",
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(color="#e6edf3"),
        ),
        hovermode="x unified",
        paper_bgcolor="#161b22",
        plot_bgcolor="#0d1117",
        font=dict(color="#e6edf3"),
        height=420,
        margin=dict(r=90, t=55),
    )
    return fig


# ============================================================
# INTERFAZ PRINCIPAL
# ============================================================

def main():
    st.set_page_config(
        page_title="GlucoVibes · Demo",
        page_icon="🩸",
        layout="wide",
    )

    st.markdown(f"""
    <style>
    /* ── Base ── */
    .stApp, .stApp > header {{
        background-color: {BG_DARK} !important;
    }}
    .main .block-container {{
        background-color: {BG_DARK};
        padding-top: 2rem;
    }}

    /* ── Texto global ── */
    .stApp p, .stApp span, .stApp label,
    .stApp div, .stApp li, .stApp a,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li {{
        color: {TEXT_PRIMARY} !important;
    }}
    .stApp h1, .stApp h2, .stApp h3, .stApp h4 {{
        color: {TEXT_PRIMARY} !important;
    }}

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {{
        background-color: {BG_SIDEBAR} !important;
        border-right: 1px solid {BORDER_COLOR};
    }}
    [data-testid="stSidebar"] * {{
        color: {TEXT_PRIMARY} !important;
    }}
    [data-testid="stSidebar"] .stMarkdown p {{
        color: {TEXT_MUTED} !important;
    }}

    /* ── Sliders ── */
    [data-testid="stSlider"] label,
    [data-testid="stSlider"] p {{
        color: {TEXT_PRIMARY} !important;
        font-weight: 500;
    }}
    [data-testid="stSlider"] [data-testid="stTickBar"] {{
        color: {TEXT_MUTED} !important;
    }}

    /* ── Selectbox y multiselect ── */
    [data-testid="stSelectbox"] label,
    [data-testid="stMultiSelect"] label {{
        color: {TEXT_PRIMARY} !important;
        font-weight: 500;
    }}
    [data-testid="stSelectbox"] > div > div,
    [data-testid="stMultiSelect"] > div > div {{
        background-color: {BG_CARD} !important;
        border-color: {BORDER_COLOR} !important;
        color: {TEXT_PRIMARY} !important;
    }}

    /* ── Number inputs ── */
    [data-testid="stNumberInput"] input {{
        background-color: {BG_CARD} !important;
        border-color: {BORDER_COLOR} !important;
        color: {TEXT_PRIMARY} !important;
    }}
    [data-testid="stNumberInput"] label {{
        color: {TEXT_PRIMARY} !important;
    }}

    /* ── Metric cards ── */
    div[data-testid="metric-container"] {{
        background: {BG_CARD};
        border: 1px solid {BORDER_COLOR};
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }}
    div[data-testid="metric-container"] label {{
        color: {TEXT_MUTED} !important;
        font-size: 0.82em !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {{
        color: {TEXT_PRIMARY} !important;
        font-weight: 700;
    }}
    div[data-testid="metric-container"] [data-testid="stMetricDelta"] {{
        color: {TEXT_MUTED} !important;
    }}

    /* ── Botones ── */
    [data-testid="stButton"] button {{
        background-color: {BG_CARD};
        border: 1px solid {BORDER_COLOR};
        color: {TEXT_PRIMARY} !important;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.15s;
    }}
    [data-testid="stButton"] button:hover {{
        background-color: #21262d;
        border-color: {BRAND_COLOR};
    }}
    [data-testid="stButton"][data-baseweb="button"] button[kind="primary"],
    button[kind="primary"] {{
        background-color: {BRAND_COLOR} !important;
        color: #0d1117 !important;
        border: none !important;
        font-weight: 700;
    }}
    button[kind="primary"]:hover {{
        background-color: #00e8b0 !important;
    }}

    /* ── Divisores ── */
    hr {{
        border-color: {BORDER_COLOR} !important;
        margin: 1rem 0;
    }}

    /* ── Alertas adaptadas al dark mode ── */
    [data-testid="stAlert"] {{
        background-color: {BG_CARD} !important;
        border-color: {BORDER_COLOR} !important;
    }}
    [data-testid="stAlert"] p {{
        color: {TEXT_PRIMARY} !important;
    }}

    /* ── Expander ── */
    [data-testid="stExpander"] {{
        background-color: {BG_CARD};
        border: 1px solid {BORDER_COLOR} !important;
        border-radius: 10px;
    }}
    [data-testid="stExpander"] summary {{
        color: {TEXT_PRIMARY} !important;
    }}
    [data-testid="stExpander"] summary:hover {{
        color: {BRAND_COLOR} !important;
    }}

    /* ── Toggle ── */
    [data-testid="stToggle"] label {{
        color: {TEXT_PRIMARY} !important;
    }}

    /* ── Radio ── */
    [data-testid="stRadio"] label,
    [data-testid="stRadio"] p {{
        color: {TEXT_PRIMARY} !important;
    }}

    /* ── Progress ── */
    [data-testid="stProgress"] > div > div {{
        background-color: {BG_CARD} !important;
    }}

    /* ── Tabla/markdown ── */
    table {{ border-collapse: collapse; width: 100%; }}
    th {{ background-color: #21262d; color: {TEXT_PRIMARY} !important; }}
    td {{ color: {TEXT_PRIMARY} !important; border-color: {BORDER_COLOR} !important; }}

    /* ── Fila de alimento ── */
    .food-row {{
        padding: 7px 10px;
        border-bottom: 1px solid {BORDER_COLOR};
        color: {TEXT_PRIMARY};
        font-size: 0.95em;
    }}

    /* ── Spinner ── */
    [data-testid="stSpinner"] p {{ color: {TEXT_MUTED} !important; }}

    /* ── Scrollbar ── */
    ::-webkit-scrollbar {{ width: 6px; }}
    ::-webkit-scrollbar-track {{ background: {BG_DARK}; }}
    ::-webkit-scrollbar-thumb {{ background: {BORDER_COLOR}; border-radius: 3px; }}
    </style>
    """, unsafe_allow_html=True)

    # ── Carga de recursos ─────────────────────────────────────
    with st.spinner("Iniciando GlucoVibes..."):
        model_peak, calibrator, feat_peak, model_ttp, feat_ttp = load_models()
        medians  = load_medians()
        food_db  = load_food_db()

    food_names = food_db["nombre"].tolist()

    # ── Cabecera ──────────────────────────────────────────────
    st.markdown(
        f'<h1 style="color:{BRAND_COLOR};margin-bottom:2px;font-size:2.4em;letter-spacing:-0.5px">GlucoVibes</h1>'
        f'<p style="color:{TEXT_MUTED};margin-top:0;font-size:1em">'
        'Predictor de respuesta glucémica postprandial &nbsp;·&nbsp; '
        'LightGBM &nbsp;·&nbsp; 15.000+ comidas reales &nbsp;·&nbsp; MAE ≈ 13 mg/dL'
        '</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    # ── SIDEBAR: contexto del usuario ─────────────────────────
    with st.sidebar:
        st.markdown(f'<h3 style="color:{BRAND_COLOR};margin-bottom:12px">Tu estado ahora</h3>', unsafe_allow_html=True)

        glucose_preprandial = st.slider(
            "Glucosa en sangre (mg/dL)", 60, 220, 100, step=5,
            help="Tu nivel de glucosa justo antes de comer",
        )

        # Semáforo de glucosa en ayunas
        if glucose_preprandial < 70:
            st.warning("Hipoglucemia — nivel bajo")
        elif glucose_preprandial <= 100:
            st.success("Glucosa en rango óptimo")
        elif glucose_preprandial <= 140:
            st.warning("Glucosa ligeramente elevada")
        else:
            st.error("Glucosa alta — ten cuidado")

        st.markdown("---")
        trend_label = st.selectbox(
            "Tendencia de glucosa (últimos 30 min)",
            list(CGM_SLOPE_MAP.keys()), index=2,
            help="Como está evolucionando tu glucosa en este momento",
        )
        meal_period = st.selectbox(
            "Momento del día",
            list(MEAL_PERIOD_HOURS.keys()), index=2,
        )

        st.markdown("---")
        st.markdown(f'<h3 style="color:{BRAND_COLOR};margin-bottom:8px">Ejercicio hoy</h3>', unsafe_allow_html=True)
        has_sport = st.toggle("He hecho ejercicio hoy")

        sport_2h_dur = 0.0
        sport_6h_dur = 0.0
        sport_intensity_val = 2.0

        if has_sport:
            sport_intensity_label = st.selectbox("Intensidad", list(SPORT_INTENSITY.keys()), index=1)
            sport_intensity_val   = SPORT_INTENSITY[sport_intensity_label]
            sport_duration        = st.slider("Duración (min)", 10, 180, 45, step=5)
            sport_when            = st.radio(
                "¿Cuándo?",
                ["Hace menos de 2h", "Entre 2h y 6h", "Hace más de 6h"],
                horizontal=True,
            )
            if sport_when == "Hace menos de 2h":
                sport_2h_dur = float(sport_duration)
            elif sport_when == "Entre 2h y 6h":
                sport_6h_dur = float(sport_duration)
            # > 6h no afecta significativamente

    # ── ZONA PRINCIPAL ────────────────────────────────────────
    col_food, col_result = st.columns([1.1, 1], gap="large")

    with col_food:
        st.markdown("### ¿Qué has comido?")

        # — Comidas rápidas de demo —
        st.markdown("**Elige una comida de ejemplo o busca tus alimentos:**")
        preset_cols = st.columns(len(PRESET_MEALS))
        for idx, (preset_name, _) in enumerate(PRESET_MEALS.items()):
            if preset_cols[idx].button(preset_name, use_container_width=True, key=f"preset_{idx}"):
                # Cargar preset en session_state
                st.session_state["selected_foods"] = [p[0] for p in PRESET_MEALS[preset_name]]
                for food_name, grams in PRESET_MEALS[preset_name]:
                    st.session_state[f"qty_{food_name}"] = grams
                st.session_state.pop("last_result", None)

        st.markdown("")

        # — Selector de alimentos —
        default_selection = st.session_state.get("selected_foods", [])
        # Filtrar defaults que no existan en la DB
        default_selection = [f for f in default_selection if f in food_names]

        selected_foods = st.multiselect(
            "Busca y añade alimentos",
            options=food_names,
            default=default_selection,
            placeholder="Escribe para buscar (ej: pollo, arroz, manzana...)",
            key="food_multiselect",
        )
        st.session_state["selected_foods"] = selected_foods

        # — Cantidades por alimento —
        selections = {}   # {nombre: gramos}
        if selected_foods:
            st.markdown("**Cantidad de cada alimento:**")
            food_indexed = food_db.set_index("nombre")

            for food_name in selected_foods:
                portion_default = int(food_indexed.loc[food_name, "porcion_default"]) if food_name in food_indexed.index else 100
                qty_key = f"qty_{food_name}"
                col_n, col_g = st.columns([3, 1])
                with col_n:
                    st.markdown(f"<div class='food-row'>🍴 {food_name}</div>", unsafe_allow_html=True)
                with col_g:
                    grams = st.number_input(
                        "g", min_value=5, max_value=1000,
                        value=st.session_state.get(qty_key, portion_default),
                        step=5, key=qty_key, label_visibility="collapsed",
                    )
                selections[food_name] = grams

            # — Resumen nutricional —
            macros = compute_meal_macros(food_db, selections)
            st.markdown("")
            st.markdown("**Resumen nutricional de la comida:**")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Carbohidratos", f"{macros['carbs']:.0f} g")
            m2.metric("Grasas",        f"{macros['fat']:.0f} g")
            m3.metric("Proteínas",     f"{macros['protein']:.0f} g")
            m4.metric("Fibra",         f"{macros['fibre']:.0f} g")

            cal_total = macros["carbs"]*4 + macros["fat"]*9 + macros["protein"]*4
            tags = []
            if macros["n_high_gi"] > 0:
                tags.append(f"⚡ {macros['n_high_gi']} alimento(s) de alto IG")
            if macros["n_ultraprocessed"] > 0:
                tags.append(f"🏭 {macros['n_ultraprocessed']} ultraprocesado(s)")
            tags.append(f"🔥 {cal_total:.0f} kcal totales")
            st.markdown("  ·  ".join(tags))

        else:
            macros = None
            st.info("Selecciona alimentos o elige una comida de ejemplo arriba.")

        st.markdown("")

        # — Botón de predicción —
        predict_btn = st.button(
            "🔮  Predecir respuesta glucémica",
            type="primary",
            use_container_width=True,
            disabled=(macros is None or macros["n_items"] == 0),
        )

    # ── COLUMNA DE RESULTADOS ─────────────────────────────────
    with col_result:
        if predict_btn and macros and macros["n_items"] > 0:
            cgm_slope = CGM_SLOPE_MAP[trend_label]
            X_peak, X_ttp = build_feature_vector(
                glucose_preprandial,
                macros["carbs"], macros["fat"], macros["protein"],
                macros["fibre"], macros["n_high_gi"], macros["n_ultraprocessed"],
                macros["avg_sat_fat"], cgm_slope,
                sport_2h_dur, sport_6h_dur, sport_intensity_val,
                meal_period, macros["n_items"], medians, feat_peak, feat_ttp,
            )
            result = predict_glucose_peak(X_peak, X_ttp, model_peak, calibrator, feat_peak, model_ttp, feat_ttp)
            st.session_state["last_result"]             = result
            st.session_state["last_glucose_preprandial"] = glucose_preprandial

        if "last_result" in st.session_state:
            result              = st.session_state["last_result"]
            glucose_preprandial_saved = st.session_state.get("last_glucose_preprandial", glucose_preprandial)

            # — Métricas principales —
            st.markdown("### Predicción")
            c1, c2 = st.columns(2)
            sign = "+" if result["amplitude"] >= 0 else ""
            c1.metric("🎯 Pico glucémico predicho",
                      f"{result['peak_value']:.0f} mg/dL",
                      delta=f"{sign}{result['amplitude']:.0f} mg/dL sobre basal",
                      delta_color="inverse")
            c2.metric("⏱️ Tiempo hasta el pico", result["ttp_label"])

            # — Badge de clasificación —
            bc = result["class_color"]
            st.markdown(
                f'<div style="background:linear-gradient(135deg,{bc}22,{bc}0a);'
                f'border:1px solid {bc}55;border-radius:14px;'
                f'padding:16px;text-align:center;margin:12px 0;'
                f'box-shadow:0 0 20px {bc}22;">'
                f'<span style="color:{bc};font-size:1.6em;font-weight:700;letter-spacing:-0.3px">'
                f'{result["class_emoji"]} Respuesta {result["classification"]}'
                f'</span></div>',
                unsafe_allow_html=True,
            )

            # — Curva glucémica —
            time_axis, glucose_curve = build_glucose_curve(
                glucose_preprandial_saved, result["peak_value"], result["ttp_minutes"]
            )
            st.plotly_chart(
                build_plotly_figure(time_axis, glucose_curve,
                                    result["peak_value"], result["ttp_minutes"],
                                    glucose_preprandial_saved),
                use_container_width=True,
            )

            # — Probabilidad TTP —
            p_t = result["ttp_proba"]
            st.progress(p_t, text=f"Probabilidad pico tardío (>75 min): {p_t:.0%}  ·  Pico rápido (<45 min): {1-p_t:.0%}")

            # — Explicación —
            with st.expander("ℹ️ Cómo interpretar la predicción"):
                st.markdown(f"""
**Zonas glucémicas:**
| | Pico (mg/dL) | Significado |
|---|---|---|
| 🟢 Normal | < 140 | Respuesta controlada |
| 🟡 Elevado | 140–180 | Monitorizar |
| 🔴 Muy elevado | > 180 | Ajustar dieta |

**El modelo tiene en cuenta:**  carbohidratos, grasas, fibra, índice glucémico, tu glucosa actual,
tendencia del CGM, ejercicio reciente y el momento del día.

**Precisión:** error medio ≈ **13 mg/dL** · entrenado con **>15.000 comidas reales** de 92 usuarios con CGM continuo.
                """)

        else:
            # Pantalla de espera
            st.markdown(f"### Tu predicción aparecerá aquí")
            st.markdown("")
            st.markdown(
                f'<div style="background:{BG_CARD};border:1px solid {BORDER_COLOR};'
                f'border-radius:12px;padding:16px 20px;color:{TEXT_MUTED};font-size:0.95em;">'
                f'1. Elige o busca los alimentos que has comido<br>'
                f'2. Ajusta tu glucosa actual y contexto en el panel lateral<br>'
                f'3. Pulsa <strong style="color:{BRAND_COLOR}">Predecir respuesta glucémica</strong>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown("")
            st.markdown(
                f'<div style="background:{BG_CARD};border:1px dashed {BORDER_COLOR};'
                f'border-radius:16px;padding:60px 40px;text-align:center;">'
                f'<div style="font-size:3.5em;margin-bottom:12px">📈</div>'
                f'<div style="font-size:1.1em;color:{TEXT_MUTED};font-weight:500">'
                f'La curva glucémica aparecerá aquí</div>'
                f'<div style="font-size:0.85em;color:{BORDER_COLOR};margin-top:6px">'
                f'Rango normal · Elevado · Muy elevado</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
