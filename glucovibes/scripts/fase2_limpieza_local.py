"""
FASE 2 — LIMPIEZA CIENTÍFICA Y JUSTIFICADA
Proyecto Glucovibes Challenge 2026

EJECUCIÓN:
  1. Coloca este script en la misma carpeta que los CSV crudos (meal.csv, glucose.csv, etc.)
  2. python fase2_limpieza_local.py
  3. Se crea carpeta ./clean_data/ con los CSV limpios

Cada decisión de limpieza está documentada con justificación fisiológica.
"""

import pandas as pd
import numpy as np
from dateutil import parser as dateutil_parser
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# RUTAS — Ajustar si es necesario
# ============================================================
DATA = './data'          # Carpeta con los CSV crudos
OUT = './clean_data'    # Carpeta de salida
os.makedirs(OUT, exist_ok=True)

# ============================================================
# REGISTRO DE LIMPIEZA
# ============================================================
cleaning_log = []

def log_action(table, action, n_affected, n_total, justification):
    pct = (n_affected / n_total * 100) if n_total > 0 else 0
    cleaning_log.append({
        'table': table, 'action': action,
        'n_affected': n_affected, 'n_total': n_total,
        'pct_affected': round(pct, 3), 'justification': justification
    })
    print(f"  [{table}] {action}: {n_affected}/{n_total} ({pct:.3f}%) — {justification}")

print("=" * 70)
print("FASE 2 — LIMPIEZA CIENTÍFICA")
print("=" * 70)

# ============================================================
# PASO 0: CARGA
# ============================================================
print("\n📦 Cargando datos crudos...")
glucose_raw = pd.read_csv(f'{DATA}/glucose.csv')
meal_raw = pd.read_csv(f'{DATA}/meal.csv')
meal_item_raw = pd.read_csv(f'{DATA}/meal_item.csv')
food_comp = pd.read_csv(f'{DATA}/food_composition.csv')
sport_raw = pd.read_csv(f'{DATA}/sport.csv')
event_raw = pd.read_csv(f'{DATA}/event.csv')
quest_morning_raw = pd.read_csv(f'{DATA}/questionnaire_morning.csv')
quest_night_raw = pd.read_csv(f'{DATA}/questionnaire_night.csv')

for name, df in [('glucose', glucose_raw), ('meal', meal_raw), ('meal_item', meal_item_raw),
                  ('sport', sport_raw), ('event', event_raw),
                  ('quest_morning', quest_morning_raw), ('quest_night', quest_night_raw)]:
    print(f"  {name}: {len(df):,} filas")

# ============================================================
# PASO 1: PARSEO ROBUSTO DE TIMESTAMPS
# ============================================================
print("\n⏱️ PASO 1: Parseo de timestamps con dateutil...")

def parse_timestamps_robust(df, col='timestamp', table_name=''):
    original_count = len(df)
    parsed = []
    failed = 0
    for val in df[col]:
        try:
            parsed.append(dateutil_parser.parse(str(val)))
        except:
            parsed.append(pd.NaT)
            failed += 1
    df[col] = pd.to_datetime(parsed, utc=True)
    log_action(table_name, 'Timestamp parseado con dateutil',
               original_count - failed, original_count,
               'Formato +00 requiere dateutil')
    if failed > 0:
        log_action(table_name, 'Timestamps no parseables → NaT',
                   failed, original_count, 'Formato irreconocible')
    return df

glucose = parse_timestamps_robust(glucose_raw.copy(), 'timestamp', 'glucose')
meal = parse_timestamps_robust(meal_raw.copy(), 'timestamp', 'meal')
sport = parse_timestamps_robust(sport_raw.copy(), 'timestamp', 'sport')
event = parse_timestamps_robust(event_raw.copy(), 'timestamp', 'event')
quest_morning = parse_timestamps_robust(quest_morning_raw.copy(), 'timestamp', 'quest_morning')
quest_night = parse_timestamps_robust(quest_night_raw.copy(), 'timestamp', 'quest_night')

# ============================================================
# PASO 2: RENOMBRAR glucose.id → user_id
# ============================================================
print("\n🏷️ PASO 2: Renombrar glucose.id → user_id...")
glucose = glucose.rename(columns={'id': 'user_id'})
log_action('glucose', 'Rename id→user_id', 1, 1,
           'id contiene 92 valores únicos = user_id')

# ============================================================
# PASO 3: LIMPIEZA DE GLUCOSE
# ============================================================
print("\n🩸 PASO 3: Limpieza de glucose...")

n_below_40 = (glucose['value_decimal'] < 40).sum()
n_above_400 = (glucose['value_decimal'] > 400).sum()
log_action('glucose', 'Lecturas <40 mg/dL → excluidas', n_below_40, len(glucose),
           '<40 mg/dL = hipoglucemia severa/error sensor')
log_action('glucose', 'Lecturas >400 mg/dL → excluidas', n_above_400, len(glucose),
           '>400 = crisis hiperglucémica/error sensor')
glucose = glucose[(glucose['value_decimal'] >= 40) & (glucose['value_decimal'] <= 400)].copy()

n_pre_2020 = (glucose['timestamp'] < '2020-01-01').sum()
if n_pre_2020 > 0:
    glucose = glucose[glucose['timestamp'] >= '2020-01-01'].copy()
    log_action('glucose', 'Pre-2020 → excluidas', n_pre_2020, len(glucose_raw), 'Fuera del estudio')

glucose = glucose.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
print(f"  Glucose final: {len(glucose):,} filas, {glucose['user_id'].nunique()} usuarios")

# ============================================================
# PASO 4: LIMPIEZA DE MEAL_ITEM
# ============================================================
print("\n🍽️ PASO 4: Limpieza de meal_item...")
meal_item = meal_item_raw.copy()

for col in ['protein', 'fat', 'carbs', 'calories']:
    n_neg = (meal_item[col] < 0).sum()
    if n_neg > 0:
        meal_item.loc[meal_item[col] < 0, col] = np.nan
        log_action('meal_item', f'{col} negativos → NaN', n_neg, len(meal_item),
                   'Código error plataforma')

thresholds = {'calories': 3000, 'protein': 150, 'fat': 200, 'carbs': 500, 'amount': 2000}
for col, thresh in thresholds.items():
    n_extreme = (meal_item[col] > thresh).sum()
    if n_extreme > 0:
        meal_item.loc[meal_item[col] > thresh, col] = np.nan
        log_action('meal_item', f'{col} > {thresh} → NaN', n_extreme, len(meal_item),
                   f'Imposible para 1 item (umbral: {thresh})')

# ============================================================
# PASO 5: MARCAR COMIDAS SIN ITEMS
# ============================================================
print("\n🥗 PASO 5: Marcar comidas sin items...")
meals_with_items = set(meal_item['meal_id'].unique())
meal['has_items'] = meal['id'].isin(meals_with_items)
n_empty = (~meal['has_items']).sum()
log_action('meal', 'Comidas sin items → flag', n_empty, len(meal), 'Sin alimentos registrados')

# ============================================================
# PASO 6: LIMPIEZA DE SPORT
# ============================================================
print("\n🏃 PASO 6: Limpieza de sport...")

n_zero = (sport['duration'] == 0).sum()
sport.loc[sport['duration'] == 0, 'duration'] = np.nan
log_action('sport', 'duration=0 → NaN', n_zero, len(sport), 'Sesión sin duración')

n_extreme = (sport['duration'] > 300).sum()
sport.loc[sport['duration'] > 300, 'duration'] = np.nan
log_action('sport', 'duration >300 min → NaN', n_extreme, len(sport), '>5h = error registro')

n_old = (sport['timestamp'] < '2020-01-01').sum()
if n_old > 0:
    sport = sport[sport['timestamp'] >= '2020-01-01'].copy()
    log_action('sport', 'Pre-2020 → excluidas', n_old, len(sport_raw), 'Timestamps erróneos')

# Normalización de tipos
sport_type_map = {
    'Running': 'Running', 'Run': 'Running', 'Correr': 'Running',
    'Correr al aire libre': 'Running', 'Trail running': 'Trail Running',
    'Trail Running': 'Trail Running',
    'Ride': 'Cycling', 'Cycling': 'Cycling', 'Ciclismo': 'Cycling',
    'Ciclismo al aire libre': 'Cycling', 'VirtualRide': 'Cycling_Indoor',
    'Bicicleta estática': 'Cycling_Indoor',
    'Walking': 'Walking', 'Walk': 'Walking', 'Caminar': 'Walking',
    'Caminar al aire libre': 'Walking', 'Hike': 'Hiking', 'Hiking': 'Hiking',
    'Senderismo': 'Hiking',
    'Swim': 'Swimming', 'Swimming': 'Swimming', 'Natación': 'Swimming',
    'WeightTraining': 'Strength', 'Strength_training': 'Strength',
    'Entrenamiento de fuerza': 'Strength', 'Workout': 'Workout',
    'Entrenamiento funcional': 'Workout', 'CrossFit': 'CrossFit', 'Crossfit': 'CrossFit',
    'Yoga': 'Yoga', 'Pilates': 'Pilates', 'Stretching': 'Stretching',
    'Soccer': 'Soccer', 'Fútbol': 'Soccer', 'Padel': 'Padel',
    'Tennis': 'Tennis', 'Tenis': 'Tennis',
    'Surf': 'Surf', 'Rowing': 'Rowing', 'Kayaking': 'Kayaking',
    'EBikeRide': 'EBike', 'Skiing': 'Skiing', 'Esquí': 'Skiing',
    'AlpineSki': 'Skiing', 'NordicSki': 'Nordic_Ski',
    'Elliptical': 'Elliptical', 'Elíptica': 'Elliptical',
    'StairStepper': 'StairStepper', 'RockClimbing': 'Climbing', 'Escalada': 'Climbing',
}
sport['type_original'] = sport['type']
sport['type_normalized'] = sport['type'].map(sport_type_map)
sport.loc[sport['type_normalized'].isna(), 'type_normalized'] = sport.loc[sport['type_normalized'].isna(), 'type_original']
n_mapped = sport['type_normalized'].notna().sum()
log_action('sport', 'Tipos normalizados', n_mapped, len(sport), f'{len(sport_type_map)} mapeos ES↔EN')

intensity_map = {
    'Running': 'high', 'Trail Running': 'high', 'CrossFit': 'high',
    'Swimming': 'high', 'Soccer': 'high', 'Rowing': 'high',
    'Cycling': 'moderate', 'Cycling_Indoor': 'moderate', 'Hiking': 'moderate',
    'Strength': 'moderate', 'Workout': 'moderate', 'Tennis': 'moderate',
    'Padel': 'moderate', 'Climbing': 'moderate', 'Elliptical': 'moderate',
    'Surf': 'moderate', 'Skiing': 'moderate', 'Nordic_Ski': 'moderate',
    'Walking': 'low', 'Yoga': 'low', 'Pilates': 'low',
    'Stretching': 'low', 'EBike': 'low',
}
sport['intensity_category'] = sport['type_normalized'].map(intensity_map).fillna('moderate')

# ============================================================
# PASO 7: LIMPIEZA QUESTIONNAIRE_MORNING
# ============================================================
print("\n🌅 PASO 7: Limpieza questionnaire_morning...")

n_hr_zero = (quest_morning['resting_hr'] == 0).sum()
quest_morning.loc[quest_morning['resting_hr'] == 0, 'resting_hr'] = np.nan
log_action('quest_morning', 'resting_hr=0 → NaN', n_hr_zero, len(quest_morning), 'FC=0 imposible')

n_hr_ext = ((quest_morning['resting_hr'] > 0) & (quest_morning['resting_hr'] < 30)).sum() + \
           (quest_morning['resting_hr'] > 120).sum()
quest_morning.loc[(quest_morning['resting_hr'] > 0) & (quest_morning['resting_hr'] < 30), 'resting_hr'] = np.nan
quest_morning.loc[quest_morning['resting_hr'] > 120, 'resting_hr'] = np.nan
log_action('quest_morning', 'resting_hr extremos → NaN', n_hr_ext, len(quest_morning), '<30 o >120')

n_gb_zero = (quest_morning['glucose_basal'] == 0).sum()
quest_morning.loc[quest_morning['glucose_basal'] == 0, 'glucose_basal'] = np.nan
log_action('quest_morning', 'glucose_basal=0 → NaN', n_gb_zero, len(quest_morning), 'Glucosa=0 imposible')

n_gb_ext = ((quest_morning['glucose_basal'] > 0) & (quest_morning['glucose_basal'] < 40)).sum() + \
           (quest_morning['glucose_basal'] > 300).sum()
quest_morning.loc[(quest_morning['glucose_basal'] > 0) & (quest_morning['glucose_basal'] < 40), 'glucose_basal'] = np.nan
quest_morning.loc[quest_morning['glucose_basal'] > 300, 'glucose_basal'] = np.nan
log_action('quest_morning', 'glucose_basal extremos → NaN', n_gb_ext, len(quest_morning), '<40 o >300')

n_sleep = (quest_morning['sleep_time'] > 840).sum()
quest_morning.loc[quest_morning['sleep_time'] > 840, 'sleep_time'] = np.nan
log_action('quest_morning', 'sleep_time >840min → NaN', n_sleep, len(quest_morning), '>14h anómalo')

n_hrv = (quest_morning['hr_variability'] > 300).sum()
quest_morning.loc[quest_morning['hr_variability'] > 300, 'hr_variability'] = np.nan
log_action('quest_morning', 'hr_variability >300 → NaN', n_hrv, len(quest_morning), 'Extremo')

# ============================================================
# PASO 8: FLAGS DE CALIDAD
# ============================================================
print("\n🚩 PASO 8: Flags de calidad por usuario...")
user_ids = sorted(glucose['user_id'].unique())
quality = pd.DataFrame({'user_id': user_ids})
quality['n_glucose_readings'] = quality['user_id'].map(glucose.groupby('user_id').size())
quality['n_meals'] = quality['user_id'].map(meal[meal['has_items']].groupby('user_id').size()).fillna(0).astype(int)
quality['n_sport_sessions'] = quality['user_id'].map(sport.groupby('user_id').size()).fillna(0).astype(int)
quality['n_quest_morning'] = quality['user_id'].map(quest_morning.groupby('user_id').size()).fillna(0).astype(int)
quality['n_quest_night'] = quality['user_id'].map(quest_night.groupby('user_id').size()).fillna(0).astype(int)
quality['has_glucose'] = quality['n_glucose_readings'] >= 500
quality['has_meals'] = quality['n_meals'] >= 20
quality['has_sport'] = quality['n_sport_sessions'] >= 5
quality['has_questionnaires'] = (quality['n_quest_morning'] >= 5) | (quality['n_quest_night'] >= 5)
quality['completeness_score'] = (quality['has_glucose'].astype(int) + quality['has_meals'].astype(int) +
                                  quality['has_sport'].astype(int) + quality['has_questionnaires'].astype(int))
quality['viable_glycemic_model'] = quality['has_glucose'] & quality['has_meals']
print(f"  Viables para modelado: {quality['viable_glycemic_model'].sum()}/92")

# ============================================================
# GUARDAR
# ============================================================
print("\n💾 Guardando...")
glucose.to_csv(f'{OUT}/glucose_clean.csv', index=False)
meal.to_csv(f'{OUT}/meal_clean.csv', index=False)
meal_item.to_csv(f'{OUT}/meal_item_clean.csv', index=False)
sport.to_csv(f'{OUT}/sport_clean.csv', index=False)
quest_morning.to_csv(f'{OUT}/quest_morning_clean.csv', index=False)
quest_night.to_csv(f'{OUT}/quest_night_clean.csv', index=False)
quality.to_csv(f'{OUT}/user_quality.csv', index=False)
food_comp.to_csv(f'{OUT}/food_composition.csv', index=False)

log_df = pd.DataFrame(cleaning_log)
log_df.to_csv(f'{OUT}/cleaning_log.csv', index=False)

print(f"\n  Archivos guardados en {OUT}/")
print(f"  Total acciones: {len(log_df)}")

# Comparación
print("\nCOMPARACIÓN ANTES/DESPUÉS:")
print(f"{'Tabla':<20} {'Antes':>10} {'Después':>10} {'Δ':>8}")
print("-" * 50)
for name, bef, aft in [('glucose', len(glucose_raw), len(glucose)),
                        ('meal', len(meal_raw), len(meal)),
                        ('meal_item', len(meal_item_raw), len(meal_item)),
                        ('sport', len(sport_raw), len(sport)),
                        ('quest_morning', len(quest_morning_raw), len(quest_morning)),
                        ('quest_night', len(quest_night_raw), len(quest_night))]:
    print(f"{name:<20} {bef:>10,} {aft:>10,} {aft-bef:>8,}")

print("\n✅ FASE 2 COMPLETADA")
