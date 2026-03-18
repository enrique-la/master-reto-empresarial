"""
FASE 3 — FEATURE ENGINEERING
Proyecto Glucovibes Challenge 2026

EJECUCIÓN:
  1. Asegúrate de que ./clean_data/ existe (salida de fase2)
  2. python fase3_feature_engineering_local.py
  3. Genera ./clean_data/modeling_dataset.csv

Construye dataset de modelado a nivel COMIDA con features nutricionales,
glucémicas, contextuales, de actividad, sueño y perfil de usuario.
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("FASE 3 — FEATURE ENGINEERING")
print("=" * 70)

CLEAN = './clean_data'

# ============================================================
# PASO 0: CARGAR DATOS LIMPIOS
# ============================================================
print("\n📦 Cargando datos limpios...")
glucose = pd.read_csv(f'{CLEAN}/glucose_clean.csv')
meal = pd.read_csv(f'{CLEAN}/meal_clean.csv')
meal_item = pd.read_csv(f'{CLEAN}/meal_item_clean.csv')
sport = pd.read_csv(f'{CLEAN}/sport_clean.csv')
quest_morning = pd.read_csv(f'{CLEAN}/quest_morning_clean.csv')
quest_night = pd.read_csv(f'{CLEAN}/quest_night_clean.csv')
food_comp = pd.read_csv(f'{CLEAN}/food_composition.csv')

for name, df in [('glucose', glucose), ('meal', meal), ('sport', sport),
                  ('quest_morning', quest_morning), ('quest_night', quest_night)]:
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)

print("  Datos cargados correctamente")

# ============================================================
# PASO 1: FEATURES NUTRICIONALES POR COMIDA
# ============================================================
print("\n🥗 PASO 1: Agregando macronutrientes por comida...")

meal_nutrition = meal_item.groupby('meal_id').agg(
    total_calories=('calories', 'sum'),
    total_protein=('protein', 'sum'),
    total_fat=('fat', 'sum'),
    total_carbs=('carbs', 'sum'),
    n_items=('id', 'count'),
    has_nan_calories=('calories', lambda x: x.isna().any()),
    has_nan_macros=('protein', lambda x: x.isna().any()),
).reset_index()

# Fibra desde food_composition
fibre_lookup = food_comp[['food_id', 'fibre_gram_value']].drop_duplicates('food_id')
fibre_lookup = fibre_lookup.set_index('food_id')['fibre_gram_value']

meal_item_fibre = meal_item.copy()
meal_item_fibre['fibre_per_100g'] = meal_item_fibre['food_id'].map(fibre_lookup)
meal_item_fibre['fibre_grams'] = meal_item_fibre['fibre_per_100g'] * meal_item_fibre['amount'] / 100

fibre_by_meal = meal_item_fibre.groupby('meal_id').agg(
    total_fibre=('fibre_grams', 'sum'),
    fibre_available=('fibre_per_100g', lambda x: x.notna().any()),
    pct_items_with_fibre=('fibre_per_100g', lambda x: x.notna().mean()),
).reset_index()

meal_nutrition = meal_nutrition.merge(fibre_by_meal, on='meal_id', how='left')

# Atributos food_composition
attr_cols = ['food_id', 'food_group_id', 'atri_ultraprocessed', 'atri_high_glycemic_index',
             'atri_sugar_add', 'atri_processed', 'atri_natural',
             'saturated_fatty_acid', 'monounsaturated_fatty_acid', 'polyunsaturated_fatty_acid']
food_attrs = food_comp[[c for c in attr_cols if c in food_comp.columns]].drop_duplicates('food_id')
meal_item_attrs = meal_item.merge(food_attrs, on='food_id', how='left')

ultra_by_meal = meal_item_attrs.groupby('meal_id').agg(
    n_ultraprocessed=('atri_ultraprocessed', lambda x: (x == 'Si').sum()),
    n_high_gi=('atri_high_glycemic_index', lambda x: (x == 'Si').sum()),
    n_sugar_added=('atri_sugar_add', lambda x: x.sum() if x.dtype == bool else 0),
    avg_saturated_fat=('saturated_fatty_acid', 'mean'),
    n_food_groups=('food_group_id', 'nunique'),
).reset_index()
meal_nutrition = meal_nutrition.merge(ultra_by_meal, on='meal_id', how='left')

# Ratios
meal_nutrition['pct_cal_carbs'] = (meal_nutrition['total_carbs'] * 4) / meal_nutrition['total_calories'].replace(0, np.nan) * 100
meal_nutrition['pct_cal_protein'] = (meal_nutrition['total_protein'] * 4) / meal_nutrition['total_calories'].replace(0, np.nan) * 100
meal_nutrition['pct_cal_fat'] = (meal_nutrition['total_fat'] * 9) / meal_nutrition['total_calories'].replace(0, np.nan) * 100
meal_nutrition['fibre_carb_ratio'] = meal_nutrition['total_fibre'] / meal_nutrition['total_carbs'].replace(0, np.nan)

print(f"  Comidas con nutrición: {len(meal_nutrition):,}")

# ============================================================
# PASO 2: MERGE MEAL + NUTRICIÓN
# ============================================================
print("\n🔗 PASO 2: Merge meal + nutrición...")
meals_valid = meal[meal['has_items'] == True].copy()
meals_valid = meals_valid.merge(meal_nutrition, left_on='id', right_on='meal_id', how='inner')
print(f"  Comidas válidas: {len(meals_valid):,}")

# ============================================================
# PASO 3: CURVAS GLUCÉMICAS POSTPRANDIALES
# ============================================================
print("\n📈 PASO 3: Extrayendo curvas glucémicas postprandiales...")
print("  (Esto tarda ~2-3 minutos...)")

glucose_by_user = {uid: grp.sort_values('timestamp').reset_index(drop=True)
                   for uid, grp in glucose.groupby('user_id')}

WINDOW_PRE = 15
WINDOW_POST = 180
MIN_READINGS_POST = 5

results = []
n_processed = 0
n_skipped_no_glucose = 0
n_skipped_few_readings = 0

for _, meal_row in meals_valid.iterrows():
    uid = meal_row['user_id']
    meal_time = meal_row['timestamp']

    if uid not in glucose_by_user:
        n_skipped_no_glucose += 1
        continue

    g = glucose_by_user[uid]
    t_pre_start = meal_time - timedelta(minutes=WINDOW_PRE)
    t_post_end = meal_time + timedelta(minutes=WINDOW_POST)

    mask_pre = (g['timestamp'] >= t_pre_start) & (g['timestamp'] <= meal_time)
    mask_post = (g['timestamp'] > meal_time) & (g['timestamp'] <= t_post_end)

    readings_pre = g.loc[mask_pre, 'value_decimal']
    readings_post = g.loc[mask_post]

    if len(readings_post) < MIN_READINGS_POST:
        n_skipped_few_readings += 1
        continue

    glucose_preprandial = readings_pre.iloc[-1] if len(readings_pre) > 0 else np.nan
    glucose_preprandial_mean = readings_pre.mean() if len(readings_pre) > 0 else np.nan

    post_values = readings_post['value_decimal'].values
    post_times = readings_post['timestamp']
    minutes_post = (post_times - meal_time).dt.total_seconds().values / 60
    baseline = glucose_preprandial if not np.isnan(glucose_preprandial) else post_values[0]

    peak_value = post_values.max()
    peak_idx = post_values.argmax()
    time_to_peak = minutes_post[peak_idx]
    amplitude = peak_value - baseline

    incremental = np.maximum(post_values - baseline, 0)
    iauc = np.trapezoid(incremental, minutes_post) if len(minutes_post) >= 2 else np.nan

    recovery_time = np.nan
    for i in range(peak_idx + 1, len(post_values)):
        if post_values[i] <= baseline + 5:
            recovery_time = minutes_post[i] - time_to_peak
            break

    cv_post = (post_values.std() / post_values.mean() * 100) if post_values.mean() > 0 else np.nan
    mask_2h = (minutes_post >= 110) & (minutes_post <= 130)
    glucose_2h = post_values[mask_2h].mean() if mask_2h.any() else np.nan

    results.append({
        'meal_id': meal_row['id'], 'user_id': uid, 'meal_timestamp': meal_time,
        'glucose_preprandial': glucose_preprandial,
        'glucose_preprandial_mean': glucose_preprandial_mean,
        'peak_value': peak_value, 'time_to_peak_min': time_to_peak,
        'amplitude': amplitude, 'iauc': iauc, 'recovery_time_min': recovery_time,
        'cv_postprandial': cv_post, 'glucose_2h': glucose_2h,
        'n_readings_post': len(post_values),
    })

    n_processed += 1
    if n_processed % 2000 == 0:
        print(f"    Procesadas {n_processed:,} comidas...")

glycemic_response = pd.DataFrame(results)
print(f"\n  Comidas procesadas: {n_processed:,}")
print(f"  Sin datos glucosa: {n_skipped_no_glucose}")
print(f"  Pocas lecturas: {n_skipped_few_readings:,}")

# ============================================================
# PASO 4: FEATURES CONTEXTUALES
# ============================================================
print("\n🌐 PASO 4: Features contextuales...")

gr = glycemic_response.copy()
gr['hour_of_day'] = gr['meal_timestamp'].dt.hour
gr['day_of_week'] = gr['meal_timestamp'].dt.dayofweek
gr['is_weekend'] = gr['day_of_week'].isin([5, 6]).astype(int)

def meal_period(hour):
    if 5 <= hour < 10: return 'breakfast'
    elif 10 <= hour < 13: return 'mid_morning'
    elif 13 <= hour < 16: return 'lunch'
    elif 16 <= hour < 19: return 'snack'
    elif 19 <= hour < 23: return 'dinner'
    else: return 'night'

gr['meal_period'] = gr['hour_of_day'].apply(meal_period)

# 4b. Actividad física previa
print("  Calculando actividad previa...")
sport_sorted = sport.sort_values(['user_id', 'timestamp'])
sport_by_user = {uid: grp for uid, grp in sport_sorted.groupby('user_id')}

def get_prior_activity(user_id, meal_time, hours_window=24):
    if user_id not in sport_by_user:
        return {'sport_prior_duration': 0, 'sport_prior_sessions': 0,
                'sport_prior_intensity': 'none', 'hours_since_last_sport': np.nan}
    s = sport_by_user[user_id]
    t_start = meal_time - timedelta(hours=hours_window)
    mask = (s['timestamp'] >= t_start) & (s['timestamp'] < meal_time)
    prior = s[mask]
    if len(prior) == 0:
        return {'sport_prior_duration': 0, 'sport_prior_sessions': 0,
                'sport_prior_intensity': 'none', 'hours_since_last_sport': np.nan}
    total_dur = prior['duration'].sum()
    n_sessions = len(prior)
    max_intensity = 'low'
    if 'high' in prior['intensity_category'].values: max_intensity = 'high'
    elif 'moderate' in prior['intensity_category'].values: max_intensity = 'moderate'
    hours_since = (meal_time - prior['timestamp'].max()).total_seconds() / 3600
    return {'sport_prior_duration': total_dur if not pd.isna(total_dur) else 0,
            'sport_prior_sessions': n_sessions,
            'sport_prior_intensity': max_intensity,
            'hours_since_last_sport': hours_since}

activity_features = []
for _, row in gr.iterrows():
    act = get_prior_activity(row['user_id'], row['meal_timestamp'])
    act['meal_id'] = row['meal_id']
    activity_features.append(act)
gr = gr.merge(pd.DataFrame(activity_features), on='meal_id', how='left')
print(f"  Actividad calculada para {len(gr):,} comidas")

# 4c. Sueño previo
print("  Vinculando sueño...")
quest_morning['date'] = quest_morning['timestamp'].dt.date
sleep_by_user_date = quest_morning.set_index(['user_id', 'date'])[
    ['sleep_time', 'sleep_quality', 'tiredness', 'fasting_hunger',
     'resting_hr', 'hr_variability', 'glucose_basal', 'sickness']
]
gr['meal_date'] = gr['meal_timestamp'].dt.date

sleep_features = []
for _, row in gr.iterrows():
    key = (row['user_id'], row['meal_date'])
    if key in sleep_by_user_date.index:
        vals = sleep_by_user_date.loc[key]
        if isinstance(vals, pd.DataFrame): vals = vals.iloc[0]
        sleep_features.append({
            'meal_id': row['meal_id'],
            'sleep_time_prev': vals.get('sleep_time', np.nan),
            'sleep_quality_prev': vals.get('sleep_quality', np.nan),
            'tiredness': vals.get('tiredness', np.nan),
            'fasting_hunger': vals.get('fasting_hunger', np.nan),
            'resting_hr_morning': vals.get('resting_hr', np.nan),
            'hrv_morning': vals.get('hr_variability', np.nan),
            'glucose_basal_quest': vals.get('glucose_basal', np.nan),
            'is_sick': vals.get('sickness', False),
        })
    else:
        sleep_features.append({'meal_id': row['meal_id'], 'sleep_time_prev': np.nan,
            'sleep_quality_prev': np.nan, 'tiredness': np.nan, 'fasting_hunger': np.nan,
            'resting_hr_morning': np.nan, 'hrv_morning': np.nan,
            'glucose_basal_quest': np.nan, 'is_sick': False})
gr = gr.merge(pd.DataFrame(sleep_features), on='meal_id', how='left')

# 4d. Cuestionario nocturno
print("  Vinculando cuestionario nocturno...")
quest_night['date'] = quest_night['timestamp'].dt.date
night_by_user_date = quest_night.set_index(['user_id', 'date'])[
    ['trainning_effort', 'anxiety_level', 'nutrition_plan', 'day_evaluation', 'out_of_routine']
]
night_features = []
for _, row in gr.iterrows():
    prev_date = row['meal_date'] - timedelta(days=1)
    key = (row['user_id'], prev_date)
    if key in night_by_user_date.index:
        vals = night_by_user_date.loc[key]
        if isinstance(vals, pd.DataFrame): vals = vals.iloc[0]
        night_features.append({
            'meal_id': row['meal_id'],
            'training_effort_prev': vals.get('trainning_effort', np.nan),
            'anxiety_prev': vals.get('anxiety_level', np.nan),
            'nutrition_plan_prev': vals.get('nutrition_plan', np.nan),
            'day_eval_prev': vals.get('day_evaluation', np.nan),
            'out_of_routine_prev': vals.get('out_of_routine', False),
        })
    else:
        night_features.append({'meal_id': row['meal_id'], 'training_effort_prev': np.nan,
            'anxiety_prev': np.nan, 'nutrition_plan_prev': np.nan,
            'day_eval_prev': np.nan, 'out_of_routine_prev': False})
gr = gr.merge(pd.DataFrame(night_features), on='meal_id', how='left')

# ============================================================
# PASO 5: FEATURES DEL USUARIO
# ============================================================
print("\n👤 PASO 5: Features del usuario...")
user_glucose_stats = glucose.groupby('user_id')['value_decimal'].agg(
    user_glucose_mean='mean', user_glucose_std='std', user_glucose_median='median',
    user_glucose_q25=lambda x: x.quantile(0.25), user_glucose_q75=lambda x: x.quantile(0.75),
).reset_index()

def time_in_range(values):
    total = len(values)
    return ((values >= 70) & (values <= 140)).sum() / total * 100 if total > 0 else np.nan

user_tir = glucose.groupby('user_id')['value_decimal'].apply(time_in_range).reset_index()
user_tir.columns = ['user_id', 'user_time_in_range']
user_stats = user_glucose_stats.merge(user_tir, on='user_id')
gr = gr.merge(user_stats, on='user_id', how='left')

# ============================================================
# PASO 6: MERGE FINAL
# ============================================================
print("\n🔗 PASO 6: Merge final...")
nutrition_cols = ['meal_id', 'total_calories', 'total_protein', 'total_fat', 'total_carbs',
                  'n_items', 'total_fibre', 'fibre_available', 'pct_items_with_fibre',
                  'n_ultraprocessed', 'n_high_gi', 'avg_saturated_fat', 'n_food_groups',
                  'pct_cal_carbs', 'pct_cal_protein', 'pct_cal_fat', 'fibre_carb_ratio']
nutrition_subset = meal_nutrition[[c for c in nutrition_cols if c in meal_nutrition.columns]]
meal_type = meals_valid[['id', 'type']].rename(columns={'id': 'meal_id', 'type': 'meal_type'})

dataset = gr.merge(nutrition_subset, on='meal_id', how='left')
dataset = dataset.merge(meal_type, on='meal_id', how='left')

# Limpieza final
dataset = dataset.drop(columns=['meal_date'], errors='ignore')

# Ordenar columnas
id_cols = ['meal_id', 'user_id', 'meal_timestamp']
target_cols = ['glucose_preprandial', 'glucose_preprandial_mean', 'peak_value',
               'time_to_peak_min', 'amplitude', 'iauc', 'recovery_time_min',
               'cv_postprandial', 'glucose_2h', 'n_readings_post']
all_ordered = id_cols + target_cols
existing = [c for c in all_ordered if c in dataset.columns]
remaining = [c for c in dataset.columns if c not in existing]
dataset = dataset[existing + remaining]

# ============================================================
# GUARDAR
# ============================================================
print(f"\n  Dataset final: {len(dataset):,} filas × {len(dataset.columns)} columnas")
dataset.to_csv(f'{CLEAN}/modeling_dataset.csv', index=False)
print(f"  Guardado: {CLEAN}/modeling_dataset.csv")
print("\n✅ FASE 3 COMPLETADA")
