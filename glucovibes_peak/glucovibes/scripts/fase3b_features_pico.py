"""
FASE 3B — FEATURE ENGINEERING PARA PREDICCIÓN DE PICO GLUCÉMICO
Proyecto Glucovibes Challenge 2026

OBJETIVO:
  Ampliar el dataset de modelado con features nuevas específicas para predecir:
    - peak_value     : valor máximo de glucosa postprandial (mg/dL)
    - time_to_peak_min : minutos desde la comida hasta el pico

NUEVAS FEATURES RESPECTO A FASE 3:
  1. CGM preprandial (ventana -60 a 0 min):
       - cgm_slope_30m         : pendiente de glucosa en los últimos 30 min (mg/dL/min)
       - cgm_slope_60m         : pendiente en los últimos 60 min
       - cgm_std_30m           : variabilidad en los últimos 30 min
       - cgm_delta_30m         : diferencia última - primera lectura en 30 min
       - cgm_delta_60m         : diferencia última - primera lectura en 60 min
       - cgm_tir_pre           : % tiempo en rango (70-140) en los 60 min previos
       - cgm_pct_above_target  : % lecturas >140 mg/dL en los 60 min previos
       - n_readings_pre_60m    : número de lecturas en la ventana previa

  2. Features nutricionales para predicción de tiempo al pico:
       - net_carbs             : carbs - fibra (ya en fase4, aquí lo añadimos antes)
       - pct_high_gi_items     : proporción de ítems con alto IG sobre total de ítems
       - fat_delay_score       : total_fat * total_carbs (captura el retardo por grasa)
       - effective_gi_score    : n_high_gi * (1 - pct_cal_fat/100)
       - carbs_per_item        : densidad de carbs por ítem

  3. Actividad física con ventanas múltiples (2h, 6h, 48h) + leakage fix:
       - sport_2h_duration     : duración total de ejercicio en las últimas 2h
       - sport_6h_duration     : duración total en las últimas 6h
       - sport_48h_duration    : duración total en las últimas 48h
       - sport_48h_sessions    : sesiones en las últimas 48h
       - sport_intensity_score : score numérico ponderado de intensidad (low=1, mod=2, high=3)

  4. Features rolling del usuario (SIN leakage — solo datos previos a la comida):
       - user_peak_mean_roll   : media de peak_value del usuario en los 30 días previos
       - user_peak_std_roll    : std de peak_value del usuario en los 30 días previos
       - user_ttp_mean_roll    : media de time_to_peak del usuario en los 30 días previos
       - user_glucose_mean_roll: media de glucosa CGM del usuario en los 14 días previos
       - user_glucose_std_roll : std de glucosa CGM del usuario en los 14 días previos

  5. Estado metabólico del día (comidas previas al mismo día):
       - prev_meals_today      : número de comidas registradas antes en el mismo día
       - prev_peak_today_max   : pico máximo de glucosa de comidas previas del día
       - hours_since_last_meal : horas desde la última comida registrada
       - carbs_load_today      : carbs acumulados de comidas previas del día

PREREQUISITOS:
  - ./clean_data/modeling_dataset.csv  (salida de fase3)
  - ./clean_data/glucose_clean.csv
  - ./clean_data/sport_clean.csv

EJECUCIÓN:
  python fase3b_features_pico.py
  → Genera ./clean_data/modeling_dataset_pico.csv
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("FASE 3B — FEATURES PARA PREDICCIÓN DE PICO GLUCÉMICO")
print("=" * 70)

CLEAN = './clean_data'

# ============================================================
# PASO 0: CARGAR DATOS
# ============================================================
print("\n📦 Cargando datos...")

dataset = pd.read_csv(f'{CLEAN}/modeling_dataset.csv')
dataset['meal_timestamp'] = pd.to_datetime(dataset['meal_timestamp'], format='mixed', utc=True)

glucose = pd.read_csv(f'{CLEAN}/glucose_clean.csv')
glucose['timestamp'] = pd.to_datetime(glucose['timestamp'], format='mixed', utc=True)
glucose = glucose.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

sport = pd.read_csv(f'{CLEAN}/sport_clean.csv')
sport['timestamp'] = pd.to_datetime(sport['timestamp'], format='mixed', utc=True)
sport = sport.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

print(f"  Dataset base: {len(dataset):,} filas × {dataset.shape[1]} columnas")
print(f"  Usuarios: {dataset['user_id'].nunique()}")
print(f"  Lecturas CGM: {len(glucose):,}")

# Pre-indexar por usuario para eficiencia
glucose_by_user = {uid: grp.reset_index(drop=True) for uid, grp in glucose.groupby('user_id')}
sport_by_user   = {uid: grp.reset_index(drop=True) for uid, grp in sport.groupby('user_id')}

# ============================================================
# PASO 1: FEATURES CGM PREPRANDIALES (ventana -60 a 0 min)
# ============================================================
print("\n📈 PASO 1: Extrayendo features CGM preprandiales (ventana -60min)...")
print("  (Tarda ~2-3 minutos...)")

PRE_60 = 60   # minutos
PRE_30 = 30

cgm_pre_features = []

for idx, row in dataset.iterrows():
    uid        = row['user_id']
    meal_time  = row['meal_timestamp']
    meal_id    = row['meal_id']

    feat = {
        'meal_id': meal_id,
        'cgm_slope_30m': np.nan,
        'cgm_slope_60m': np.nan,
        'cgm_std_30m': np.nan,
        'cgm_delta_30m': np.nan,
        'cgm_delta_60m': np.nan,
        'cgm_tir_pre': np.nan,
        'cgm_pct_above_target': np.nan,
        'n_readings_pre_60m': 0,
    }

    if uid not in glucose_by_user:
        cgm_pre_features.append(feat)
        continue

    g = glucose_by_user[uid]
    t60_start = meal_time - timedelta(minutes=PRE_60)
    t30_start = meal_time - timedelta(minutes=PRE_30)

    mask_60 = (g['timestamp'] >= t60_start) & (g['timestamp'] <= meal_time)
    mask_30 = (g['timestamp'] >= t30_start) & (g['timestamp'] <= meal_time)

    g60 = g.loc[mask_60].copy()
    g30 = g.loc[mask_30].copy()

    n60 = len(g60)
    feat['n_readings_pre_60m'] = n60

    if n60 >= 2:
        vals60 = g60['value_decimal'].values
        mins60 = (g60['timestamp'] - meal_time).dt.total_seconds().values / 60  # negativos

        # Pendiente 60m (regresión lineal sobre el tiempo)
        slope60, _, _, _, _ = stats.linregress(mins60, vals60)
        feat['cgm_slope_60m'] = slope60

        feat['cgm_delta_60m'] = vals60[-1] - vals60[0]
        feat['cgm_tir_pre']   = ((vals60 >= 70) & (vals60 <= 140)).mean() * 100
        feat['cgm_pct_above_target'] = (vals60 > 140).mean() * 100

    if len(g30) >= 2:
        vals30 = g30['value_decimal'].values
        mins30 = (g30['timestamp'] - meal_time).dt.total_seconds().values / 60

        slope30, _, _, _, _ = stats.linregress(mins30, vals30)
        feat['cgm_slope_30m'] = slope30
        feat['cgm_std_30m']   = vals30.std()
        feat['cgm_delta_30m'] = vals30[-1] - vals30[0]

    cgm_pre_features.append(feat)

    if (idx + 1) % 3000 == 0:
        print(f"    {idx+1:,}/{len(dataset):,} comidas procesadas...")

cgm_pre_df = pd.DataFrame(cgm_pre_features)
dataset = dataset.merge(cgm_pre_df, on='meal_id', how='left')
print(f"  Features CGM preprandiales añadidas: {cgm_pre_df.shape[1]-1} columnas")
print(f"  Comidas con slope_30m válido: {dataset['cgm_slope_30m'].notna().sum():,}")
print(f"  Comidas con slope_60m válido: {dataset['cgm_slope_60m'].notna().sum():,}")

# ============================================================
# PASO 2: FEATURES NUTRICIONALES ADICIONALES PARA TIEMPO AL PICO
# ============================================================
print("\n🥗 PASO 2: Features nutricionales para tiempo al pico...")

# net_carbs (carbs absorbibles)
dataset['net_carbs'] = (dataset['total_carbs'] - dataset['total_fibre'].fillna(0)).clip(lower=0)

# Proporción de ítems con alto IG sobre total ítems
dataset['pct_high_gi_items'] = dataset['n_high_gi'] / dataset['n_items'].replace(0, np.nan)

# Score de retardo por grasa: más grasa + más carbs = pico más tardío
dataset['fat_delay_score'] = dataset['total_fat'] * dataset['total_carbs']

# IG efectivo corregido por el efecto buffer de la grasa
dataset['effective_gi_score'] = (
    dataset['n_high_gi'] * (1 - (dataset['pct_cal_fat'].fillna(0) / 100).clip(0, 1))
)

# Densidad de carbs por ítem (comidas con pocos ítems muy cargados en carbs = pico más rápido)
dataset['carbs_per_item'] = dataset['total_carbs'] / dataset['n_items'].replace(0, np.nan)

print(f"  net_carbs, pct_high_gi_items, fat_delay_score, effective_gi_score, carbs_per_item → añadidos")

# ============================================================
# PASO 3: ACTIVIDAD FÍSICA CON VENTANAS MÚLTIPLES
# ============================================================
print("\n🏃 PASO 3: Actividad física con ventanas múltiples (2h, 6h, 48h)...")

intensity_score_map = {'none': 0, 'low': 1, 'moderate': 2, 'high': 3}

sport_multi_features = []

for _, row in dataset.iterrows():
    uid       = row['user_id']
    meal_time = row['meal_timestamp']
    meal_id   = row['meal_id']

    feat = {
        'meal_id': meal_id,
        'sport_2h_duration': 0.0,
        'sport_6h_duration': 0.0,
        'sport_48h_duration': 0.0,
        'sport_48h_sessions': 0,
        'sport_intensity_score': 0.0,
    }

    if uid not in sport_by_user:
        sport_multi_features.append(feat)
        continue

    s = sport_by_user[uid]

    for window_h, col_dur, col_sess in [
        (2,  'sport_2h_duration',  None),
        (6,  'sport_6h_duration',  None),
        (48, 'sport_48h_duration', 'sport_48h_sessions'),
    ]:
        t_start = meal_time - timedelta(hours=window_h)
        mask    = (s['timestamp'] >= t_start) & (s['timestamp'] < meal_time)
        prior   = s[mask]
        dur     = prior['duration'].sum() if len(prior) > 0 else 0.0
        feat[col_dur] = 0.0 if pd.isna(dur) else dur
        if col_sess:
            feat[col_sess] = len(prior)

    # Score ponderado de intensidad en 48h (intensidad × duración)
    t48_start = meal_time - timedelta(hours=48)
    mask48    = (s['timestamp'] >= t48_start) & (s['timestamp'] < meal_time)
    prior48   = s[mask48]
    if len(prior48) > 0:
        scores = prior48['intensity_category'].map(intensity_score_map).fillna(0)
        durs   = prior48['duration'].fillna(0)
        total_dur = durs.sum()
        if total_dur > 0:
            feat['sport_intensity_score'] = (scores * durs).sum() / total_dur
        else:
            feat['sport_intensity_score'] = scores.mean()

    sport_multi_features.append(feat)

sport_multi_df = pd.DataFrame(sport_multi_features)

# Eliminar columnas de sport originales que reemplazamos con versiones mejoradas
# (las originales se mantienen como están, añadimos las nuevas)
dataset = dataset.merge(sport_multi_df, on='meal_id', how='left')
print(f"  Features de actividad añadidas: sport_2h/6h/48h_duration, sport_48h_sessions, sport_intensity_score")

# ============================================================
# PASO 4: FEATURES ROLLING DEL USUARIO (SIN LEAKAGE)
# ============================================================
print("\n👤 PASO 4: Features rolling del usuario (sin leakage)...")
print("  (Tarda ~3-4 minutos...)")

# Ordenar el dataset cronológicamente por usuario
dataset = dataset.sort_values(['user_id', 'meal_timestamp']).reset_index(drop=True)

rolling_features = []

# Pre-calcular índices por usuario para eficiencia
user_meal_groups = {uid: grp for uid, grp in dataset.groupby('user_id')}

for uid, user_meals in user_meal_groups.items():
    g_user = glucose_by_user.get(uid, None)

    for idx, row in user_meals.iterrows():
        meal_time = row['meal_timestamp']
        meal_id   = row['meal_id']
        feat      = {'meal_id': meal_id}

        # ---- Rolling de picos y tiempo al pico (últimos 30 días de comidas previas) ----
        cutoff_30d = meal_time - timedelta(days=30)
        prev_meals = user_meals[
            (user_meals['meal_timestamp'] >= cutoff_30d) &
            (user_meals['meal_timestamp'] <  meal_time)
        ]

        if len(prev_meals) >= 3:
            feat['user_peak_mean_roll'] = prev_meals['peak_value'].mean()
            feat['user_peak_std_roll']  = prev_meals['peak_value'].std()
            feat['user_ttp_mean_roll']  = prev_meals['time_to_peak_min'].mean()
        else:
            feat['user_peak_mean_roll'] = np.nan
            feat['user_peak_std_roll']  = np.nan
            feat['user_ttp_mean_roll']  = np.nan

        # ---- Rolling glucosa CGM (últimos 14 días de lecturas PREVIAS a la comida) ----
        if g_user is not None:
            cutoff_14d = meal_time - timedelta(days=14)
            mask_roll  = (g_user['timestamp'] >= cutoff_14d) & (g_user['timestamp'] < meal_time)
            g_roll     = g_user.loc[mask_roll, 'value_decimal']

            if len(g_roll) >= 10:
                feat['user_glucose_mean_roll'] = g_roll.mean()
                feat['user_glucose_std_roll']  = g_roll.std()
            else:
                feat['user_glucose_mean_roll'] = np.nan
                feat['user_glucose_std_roll']  = np.nan
        else:
            feat['user_glucose_mean_roll'] = np.nan
            feat['user_glucose_std_roll']  = np.nan

        rolling_features.append(feat)

rolling_df = pd.DataFrame(rolling_features)
dataset = dataset.merge(rolling_df, on='meal_id', how='left')

valid_roll = dataset['user_peak_mean_roll'].notna().sum()
print(f"  Features rolling añadidas para {valid_roll:,} comidas ({valid_roll/len(dataset)*100:.1f}%)")
print(f"  (Las primeras comidas de cada usuario tendrán NaN — se imputarán en modelado)")

# ============================================================
# PASO 5: ESTADO METABÓLICO DEL DÍA (comidas previas)
# ============================================================
print("\n📅 PASO 5: Estado metabólico acumulado del día...")

day_context_features = []

for uid, user_meals in user_meal_groups.items():
    user_meals_sorted = user_meals.sort_values('meal_timestamp')

    for idx, row in user_meals_sorted.iterrows():
        meal_time = row['meal_timestamp']
        meal_id   = row['meal_id']
        meal_date = meal_time.date()

        # Comidas del mismo día ANTERIORES a esta
        same_day_prev = user_meals_sorted[
            (user_meals_sorted['meal_timestamp'].dt.date == meal_date) &
            (user_meals_sorted['meal_timestamp'] < meal_time)
        ]

        if len(same_day_prev) == 0:
            day_context_features.append({
                'meal_id': meal_id,
                'prev_meals_today': 0,
                'prev_peak_today_max': np.nan,
                'hours_since_last_meal': np.nan,
                'carbs_load_today': 0.0,
            })
        else:
            last_meal_time = same_day_prev['meal_timestamp'].max()
            hours_since    = (meal_time - last_meal_time).total_seconds() / 3600

            day_context_features.append({
                'meal_id': meal_id,
                'prev_meals_today': len(same_day_prev),
                'prev_peak_today_max': same_day_prev['peak_value'].max(),
                'hours_since_last_meal': hours_since,
                'carbs_load_today': same_day_prev['total_carbs'].sum(),
            })

day_ctx_df = pd.DataFrame(day_context_features)
dataset = dataset.merge(day_ctx_df, on='meal_id', how='left')
print(f"  Features de contexto diario añadidas: prev_meals_today, prev_peak_today_max, hours_since_last_meal, carbs_load_today")

# ============================================================
# PASO 6: VERIFICACIÓN Y GUARDAR
# ============================================================
print("\n✅ PASO 6: Verificación y guardado...")

new_features = [
    # CGM preprandial
    'cgm_slope_30m', 'cgm_slope_60m', 'cgm_std_30m',
    'cgm_delta_30m', 'cgm_delta_60m', 'cgm_tir_pre',
    'cgm_pct_above_target', 'n_readings_pre_60m',
    # Nutricionales adicionales
    'net_carbs', 'pct_high_gi_items', 'fat_delay_score',
    'effective_gi_score', 'carbs_per_item',
    # Actividad multi-ventana
    'sport_2h_duration', 'sport_6h_duration', 'sport_48h_duration',
    'sport_48h_sessions', 'sport_intensity_score',
    # Rolling usuario
    'user_peak_mean_roll', 'user_peak_std_roll', 'user_ttp_mean_roll',
    'user_glucose_mean_roll', 'user_glucose_std_roll',
    # Contexto diario
    'prev_meals_today', 'prev_peak_today_max',
    'hours_since_last_meal', 'carbs_load_today',
]

print(f"\n  Dataset final: {len(dataset):,} filas × {dataset.shape[1]} columnas")
print(f"  Columnas nuevas añadidas: {len(new_features)}")
print(f"\n  Cobertura de features nuevas:")
for col in new_features:
    if col in dataset.columns:
        pct = dataset[col].notna().mean() * 100
        print(f"    {col:<30}: {pct:>5.1f}% no-nulo")
    else:
        print(f"    {col:<30}: ⚠️  NO ENCONTRADA")

# Guardar
out_path = f'{CLEAN}/modeling_dataset_pico.csv'
dataset.to_csv(out_path, index=False)
print(f"\n  💾 Guardado: {out_path}")
print("\n✅ FASE 3B COMPLETADA")
print("   Siguiente paso: ejecutar fase5_modelado_pico.py")
