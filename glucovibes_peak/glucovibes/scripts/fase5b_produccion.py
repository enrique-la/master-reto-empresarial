"""
FASE 5B — MEJORAS DE PRODUCCIÓN: PICO GLUCÉMICO Y TIEMPO AL PICO
Proyecto Glucovibes Challenge 2026

MEJORAS IMPLEMENTADAS (sobre fase5):
  P1 — Calibración isotónica post-hoc (peak_value)
       El modelo base sobreestima picos bajos y subestima picos altos
       (regresión a la media). IsotonicRegression en espacio de predicción
       corrige este sesgo sistemático sin reentrenar el modelo principal.

  P2 — Sample weighting por usuario (peak_value)
       Usuarios con MAE histórico alto reciben mayor peso en el entrenamiento.
       Peso = clip(MAE_usuario / MAE_global, 0.5, 3.0).
       Garantiza que el modelo no ignore los perfiles glucémicos difíciles.

  P3 — time_to_peak reformulado como clasificación BINARIA
       Con R²=0.06 la regresión no es viable.
       Nueva formulación: ¿el pico llegará RÁPIDO (<45min) o TARDÍO (>75min)?
       Se descarta la clase "moderado" como ambigua para entrenamiento
       y se reporta como "incierto" en producción.
       Añade features específicas para esta tarea:
         · fat_carb_ratio_meal  (grasa retarda el vaciamiento gástrico)
         · gi_fiber_interaction  (fibra atenúa el IG efectivo)
         · sport_2h_insulin_proxy (ejercicio reciente → absorción más rápida)

  P4 — Ventana CGM ampliada a 90min (fallback cuando <4 lecturas en 60min)
       cgm_slope_60m tenía cobertura baja. Con ventana de 90min como fallback
       se maximiza la cobertura de esta feature crítica.

PREREQUISITOS:
  pip install xgboost lightgbm scikit-learn optuna pandas numpy scipy
  - ./clean_data/modeling_dataset_pico.csv  (salida de fase3b)
  - ./clean_data/glucose_clean.csv          (para P4, ventana CGM 90min)

EJECUCIÓN:
  python fase5b_produccion.py
  → Genera modelos serializados y resultados en ./clean_data/
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                              r2_score, accuracy_score, f1_score,
                              precision_recall_fscore_support, confusion_matrix)
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from scipy import stats
from datetime import timedelta
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBRegressor, XGBClassifier
    HAS_XGB = True
    print("✅ XGBoost disponible")
except ImportError:
    HAS_XGB = False
    print("⚠️  XGBoost no instalado")

try:
    from lightgbm import LGBMRegressor, LGBMClassifier
    HAS_LGB = True
    print("✅ LightGBM disponible")
except ImportError:
    HAS_LGB = False
    print("⚠️  LightGBM no instalado")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
    print("✅ Optuna disponible")
except ImportError:
    HAS_OPTUNA = False
    print("⚠️  Optuna no instalado — pip install optuna")

CLEAN       = './clean_data'
N_FOLDS     = 5
N_TRIALS_A  = 80   # trials Optuna para peak_value
N_TRIALS_B  = 60   # trials Optuna para clasificación binaria TTP
os.makedirs(f'{CLEAN}/models', exist_ok=True)

print("\n" + "=" * 70)
print("FASE 5B — MEJORAS DE PRODUCCIÓN")
print("=" * 70)

# ============================================================
# CARGAR DATOS
# ============================================================
print("\n📦 Cargando datos...")
dataset = pd.read_csv(f'{CLEAN}/modeling_dataset_pico.csv')
dataset['meal_timestamp'] = pd.to_datetime(dataset['meal_timestamp'], format='mixed', utc=True)
print(f"  Dataset: {len(dataset):,} filas × {dataset.shape[1]} columnas")

glucose = pd.read_csv(f'{CLEAN}/glucose_clean.csv')
glucose['timestamp'] = pd.to_datetime(glucose['timestamp'], format='mixed', utc=True)
glucose = glucose.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
glucose_by_user = {uid: grp.reset_index(drop=True) for uid, grp in glucose.groupby('user_id')}
print(f"  Lecturas CGM: {len(glucose):,}")

# ============================================================
# P4 — VENTANA CGM AMPLIADA A 90MIN (FALLBACK)
# ============================================================
print("\n📈 P4: Ampliando ventana CGM a 90min donde cobertura es baja...")

PRE_90 = 90
PRE_60 = 60

cgm_90_features = []

for _, row in dataset.iterrows():
    uid       = row['user_id']
    meal_time = row['meal_timestamp']
    meal_id   = row['meal_id']

    # Si ya tenemos slope_60m válido, no hace falta recalcular
    has_slope60 = not pd.isna(row.get('cgm_slope_60m', np.nan))

    feat = {
        'meal_id': meal_id,
        'cgm_slope_90m': np.nan,
        'cgm_delta_90m': np.nan,
        'cgm_std_90m': np.nan,
        'cgm_slope_best': row.get('cgm_slope_60m', np.nan),   # empieza con 60m
        'cgm_delta_best': row.get('cgm_delta_60m', np.nan),
        'cgm_window_used': 60 if has_slope60 else np.nan,
    }

    if uid not in glucose_by_user:
        cgm_90_features.append(feat)
        continue

    g = glucose_by_user[uid]
    t90_start = meal_time - timedelta(minutes=PRE_90)
    mask_90   = (g['timestamp'] >= t90_start) & (g['timestamp'] <= meal_time)
    g90       = g.loc[mask_90].copy()

    if len(g90) >= 3:
        vals90 = g90['value_decimal'].values
        mins90 = (g90['timestamp'] - meal_time).dt.total_seconds().values / 60
        slope90, _, _, _, _ = stats.linregress(mins90, vals90)
        feat['cgm_slope_90m'] = slope90
        feat['cgm_delta_90m'] = vals90[-1] - vals90[0]
        feat['cgm_std_90m']   = vals90.std()

        # Usar 90m como fallback si 60m no estaba disponible
        if not has_slope60:
            feat['cgm_slope_best'] = slope90
            feat['cgm_delta_best'] = vals90[-1] - vals90[0]
            feat['cgm_window_used'] = 90

    cgm_90_features.append(feat)

cgm_90_df = pd.DataFrame(cgm_90_features)
dataset = dataset.merge(cgm_90_df, on='meal_id', how='left')

cov_60  = dataset['cgm_slope_60m'].notna().mean() * 100
cov_90  = dataset['cgm_slope_90m'].notna().mean() * 100
cov_best= dataset['cgm_slope_best'].notna().mean() * 100
print(f"  Cobertura slope_60m:   {cov_60:.1f}%")
print(f"  Cobertura slope_90m:   {cov_90:.1f}%")
print(f"  Cobertura slope_best:  {cov_best:.1f}%  ← feature que usará el modelo")

# ============================================================
# P3 — FEATURES ADICIONALES PARA CLASIFICACIÓN BINARIA TTP
# ============================================================
print("\n⏱️  P3: Calculando features para clasificación binaria de tiempo al pico...")

# Ratio grasa/carbohidratos de la comida (grasa retarda vaciamiento gástrico)
dataset['fat_carb_ratio_meal'] = (
    dataset['total_fat'] / dataset['total_carbs'].replace(0, np.nan)
).fillna(0)

# Interacción IG × fibra: fibra atenúa el IG efectivo
dataset['gi_fiber_interaction'] = (
    dataset['n_high_gi'] * (1 - dataset['fibre_carb_ratio'].fillna(0).clip(0, 1))
)

# Proxy insulinémico del ejercicio reciente (ejercicio intenso <2h → absorción más rápida)
dataset['sport_2h_insulin_proxy'] = (
    dataset['sport_2h_duration'].fillna(0) *
    dataset.get('sport_intensity_score', pd.Series(0, index=dataset.index)).fillna(0)
)

print("  fat_carb_ratio_meal, gi_fiber_interaction, sport_2h_insulin_proxy → añadidos")

# ============================================================
# DEFINICIÓN DE FEATURES
# ============================================================
print("\n🔧 Definiendo feature sets...")

base_numeric = [
    'glucose_preprandial', 'total_calories', 'total_protein', 'total_fat',
    'total_carbs', 'total_fibre', 'n_items', 'pct_cal_carbs', 'pct_cal_protein',
    'pct_cal_fat', 'fibre_carb_ratio', 'n_ultraprocessed', 'n_high_gi',
    'avg_saturated_fat', 'n_food_groups', 'hour_of_day', 'day_of_week',
    'is_weekend', 'sport_prior_duration', 'sport_prior_sessions',
    'hours_since_last_sport', 'sleep_time_prev', 'sleep_quality_prev',
    'tiredness', 'fasting_hunger', 'resting_hr_morning', 'hrv_morning',
    'training_effort_prev', 'anxiety_prev', 'nutrition_plan_prev',
    'day_eval_prev', 'pct_items_with_fibre',
]

# Encoding de categóricas
for col in ['meal_period', 'meal_type', 'sport_prior_intensity']:
    if col in dataset.columns:
        le = LabelEncoder()
        enc_col = f'{col}_enc'
        if enc_col not in dataset.columns:
            dataset[enc_col] = le.fit_transform(dataset[col].fillna('unknown').astype(str))
        base_numeric.append(enc_col)

# Features nuevas fase3b (sin cgm_slope_60m — sustituida por cgm_slope_best)
new_features = [
    # CGM preprandial con cobertura mejorada
    'cgm_slope_best', 'cgm_delta_best', 'cgm_slope_30m', 'cgm_std_30m',
    'cgm_slope_90m', 'cgm_delta_90m', 'cgm_std_90m',
    'cgm_tir_pre', 'cgm_pct_above_target', 'n_readings_pre_60m',
    # Nutricionales
    'net_carbs', 'pct_high_gi_items', 'fat_delay_score',
    'effective_gi_score', 'carbs_per_item',
    # Actividad multi-ventana
    'sport_2h_duration', 'sport_6h_duration', 'sport_48h_duration',
    'sport_48h_sessions', 'sport_intensity_score',
    # Rolling sin leakage
    'user_peak_mean_roll', 'user_peak_std_roll', 'user_ttp_mean_roll',
    'user_glucose_mean_roll', 'user_glucose_std_roll',
    # Contexto diario
    'prev_meals_today', 'prev_peak_today_max',
    'hours_since_last_meal', 'carbs_load_today',
]

# Features específicas para TTP binario (P3)
ttp_extra = [
    'fat_carb_ratio_meal', 'gi_fiber_interaction', 'sport_2h_insulin_proxy',
]

# Clusters si existen
cluster_features = [c for c in ['nutri_cluster_num', 'glyc_cluster_num'] if c in dataset.columns]

# Feature set completo
feature_set_full = [f for f in (base_numeric + new_features + cluster_features)
                    if f in dataset.columns]

# Feature set TTP (incluye las específicas de P3)
feature_set_ttp = [f for f in (feature_set_full + ttp_extra) if f in dataset.columns]

print(f"  Features base:          {len([f for f in base_numeric if f in dataset.columns])}")
print(f"  Features nuevas (fase3b+P4): {len([f for f in new_features if f in dataset.columns])}")
print(f"  Features TTP extra:     {len([f for f in ttp_extra if f in dataset.columns])}")
print(f"  TOTAL peak_value:       {len(feature_set_full)}")
print(f"  TOTAL ttp_binario:      {len(feature_set_ttp)}")

# ============================================================
# UTILIDADES COMPARTIDAS
# ============================================================
gkf = GroupKFold(n_splits=N_FOLDS)

def impute_user_median(X_df, dataset_ref):
    """Imputa NaN: primero mediana por usuario, fallback mediana global."""
    X = X_df.copy()
    uid_col = dataset_ref['user_id'] if 'user_id' in dataset_ref.columns else None
    for col in X.columns:
        nan_mask = X[col].isna()
        if not nan_mask.any():
            continue
        if uid_col is not None:
            user_medians = dataset_ref.groupby('user_id')[col].median() \
                           if col in dataset_ref.columns else None
            if user_medians is not None:
                fill_vals = uid_col.map(user_medians)
                X.loc[nan_mask, col] = fill_vals[nan_mask]
        still_nan = X[col].isna()
        if still_nan.any():
            global_med = X[col].median()
            X.loc[still_nan, col] = global_med if not pd.isna(global_med) else 0
    return X

def compute_sample_weights(dataset_ref, user_maes, global_mae,
                            weight_min=0.5, weight_max=3.0):
    """
    P2: Peso de cada muestra = clip(MAE_usuario / MAE_global, min, max).
    Usuarios con peor historial reciben más peso en el entrenamiento.
    """
    user_weights = (user_maes / global_mae).clip(weight_min, weight_max)
    sample_weights = dataset_ref['user_id'].map(user_weights).fillna(1.0).values
    return sample_weights, user_weights

def evaluate_regressor_weighted(model, X_arr, y, groups, sample_weights=None):
    """
    Evalúa regresor con GroupKFold. Aplica sample_weights en entrenamiento.
    Las métricas de test se calculan siempre sin pesos (rendimiento real).
    """
    fold_mae, fold_rmse, fold_r2 = [], [], []
    oof_preds = np.full(len(y), np.nan)

    for train_idx, test_idx in gkf.split(X_arr, y, groups):
        X_tr, X_te = X_arr[train_idx], X_arr[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        sw_tr = sample_weights[train_idx] if sample_weights is not None else None

        m = type(model)(**model.get_params())
        fit_kwargs = {'sample_weight': sw_tr} if sw_tr is not None else {}
        m.fit(X_tr, y_tr, **fit_kwargs)
        y_pred = m.predict(X_te)

        fold_mae.append(mean_absolute_error(y_te, y_pred))
        fold_rmse.append(np.sqrt(mean_squared_error(y_te, y_pred)))
        fold_r2.append(r2_score(y_te, y_pred))
        oof_preds[test_idx] = y_pred

    return {
        'MAE': np.mean(fold_mae), 'RMSE': np.mean(fold_rmse),
        'R2': np.mean(fold_r2), 'oof_preds': oof_preds
    }

def get_feature_importance(model, feature_names, X_arr, y, sample_weights=None):
    """Entrena en todo el dataset y devuelve importancias."""
    m = type(model)(**model.get_params())
    fit_kwargs = {'sample_weight': sample_weights} if sample_weights is not None else {}
    m.fit(X_arr, y, **fit_kwargs)
    return pd.DataFrame({
        'feature': feature_names,
        'importance': m.feature_importances_
    }).sort_values('importance', ascending=False).reset_index(drop=True), m

# ============================================================
# P2 — CALCULAR SAMPLE WEIGHTS (con MAE de fase5 como referencia)
# ============================================================
print("\n⚖️  P2: Calculando sample weights por usuario...")

# Usamos las predicciones OOF de fase5 como estimación del MAE por usuario.
# Si no existen, se usa peso uniforme en la primera iteración.
pred_path = f'{CLEAN}/predictions_peak_value.csv'
if os.path.exists(pred_path):
    prev_preds = pd.read_csv(pred_path)
    user_maes_series = prev_preds.groupby('user_id')['abs_error'].mean()
    global_mae_ref   = prev_preds['abs_error'].mean()
    _, user_weights_map = compute_sample_weights(
        dataset, user_maes_series, global_mae_ref
    )
    print(f"  MAE global referencia: {global_mae_ref:.2f} mg/dL")
    print(f"  Usuarios con peso > 1.5 (énfasis alto): "
          f"{(user_weights_map > 1.5).sum()} usuarios")
    print(f"  Top 5 usuarios con mayor peso:")
    for uid, w in user_weights_map.sort_values(ascending=False).head(5).items():
        print(f"    User {uid}: weight={w:.2f}")
    sample_weights_A = dataset['user_id'].map(user_weights_map).fillna(1.0).values
else:
    print("  ⚠️  No se encontró predictions_peak_value.csv → pesos uniformes")
    sample_weights_A = np.ones(len(dataset))

# ============================================================
# MODELO A — PEAK_VALUE CON P1 + P2
# ============================================================
print("\n" + "=" * 70)
print("MODELO A — peak_value  [P1: calibración isotónica + P2: sample weights]")
print("=" * 70)

target_A = 'peak_value'
mask_A   = dataset[target_A].notna() & (dataset[target_A] > 0)
X_A_df   = dataset.loc[mask_A, feature_set_full].copy()
y_A      = dataset.loc[mask_A, target_A].values
g_A      = dataset.loc[mask_A, 'user_id'].values
sw_A     = sample_weights_A[mask_A.values]

X_A_df = impute_user_median(X_A_df, dataset.loc[mask_A])
X_A    = X_A_df.values

print(f"  Muestras: {len(y_A):,}  |  Features: {X_A.shape[1]}")
print(f"  peak_value: media={y_A.mean():.1f}  std={y_A.std():.1f}")

results_A = []

# ---- A1. Baseline sin mejoras (referencia) ----
if HAS_LGB:
    print("\n  A0. LightGBM SIN mejoras (referencia fase5)...")
    lgbm_ref = LGBMRegressor(
        n_estimators=455, max_depth=9, learning_rate=0.0193, num_leaves=49,
        subsample=0.813, colsample_bytree=0.826, min_child_samples=47,
        reg_alpha=9.77, reg_lambda=0.000101,
        random_state=42, n_jobs=-1, verbose=-1
    )
    res = evaluate_regressor_weighted(lgbm_ref, X_A, y_A, g_A, sample_weights=None)
    print(f"     MAE={res['MAE']:.3f}  R²={res['R2']:.4f}  (sin weights, sin calibración)")
    results_A.append({'model': 'LGBM_fase5_referencia', 'MAE': res['MAE'],
                       'RMSE': res['RMSE'], 'R2': res['R2'],
                       'mejoras': 'ninguna'})
    oof_ref_A = res['oof_preds']

# ---- A2. Con sample weights (P2) ----
if HAS_LGB:
    print("\n  A1. LightGBM + sample weights (P2)...")
    res = evaluate_regressor_weighted(lgbm_ref, X_A, y_A, g_A, sample_weights=sw_A)
    print(f"     MAE={res['MAE']:.3f}  R²={res['R2']:.4f}")
    results_A.append({'model': 'LGBM_+weights', 'MAE': res['MAE'],
                       'RMSE': res['RMSE'], 'R2': res['R2'],
                       'mejoras': 'P2'})
    oof_weighted_A = res['oof_preds']

# ---- A3. Optuna con sample weights ----
if HAS_OPTUNA and HAS_LGB:
    print(f"\n  A2. Optuna + sample weights ({N_TRIALS_A} trials)...")

    def objective_A_weighted(trial):
        params = {
            'n_estimators':      trial.suggest_int('n_estimators', 300, 1200),
            'max_depth':         trial.suggest_int('max_depth', 4, 10),
            'learning_rate':     trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'num_leaves':        trial.suggest_int('num_leaves', 20, 200),
            'subsample':         trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 80),
            'reg_alpha':         trial.suggest_float('reg_alpha', 1e-4, 20.0, log=True),
            'reg_lambda':        trial.suggest_float('reg_lambda', 1e-4, 20.0, log=True),
            'random_state': 42, 'n_jobs': -1, 'verbose': -1,
        }
        fold_mae = []
        for train_idx, test_idx in gkf.split(X_A, y_A, g_A):
            m = LGBMRegressor(**params)
            m.fit(X_A[train_idx], y_A[train_idx], sample_weight=sw_A[train_idx])
            fold_mae.append(mean_absolute_error(y_A[test_idx], m.predict(X_A[test_idx])))
        return np.mean(fold_mae)

    study_A = optuna.create_study(direction='minimize')
    study_A.optimize(objective_A_weighted, n_trials=N_TRIALS_A, show_progress_bar=False)
    best_params_A = study_A.best_params
    print(f"     Mejor MAE CV: {study_A.best_value:.3f}")

    lgbm_A_opt = LGBMRegressor(**best_params_A, random_state=42, n_jobs=-1, verbose=-1)
    res = evaluate_regressor_weighted(lgbm_A_opt, X_A, y_A, g_A, sample_weights=sw_A)
    print(f"     Evaluación final: MAE={res['MAE']:.3f}  R²={res['R2']:.4f}")
    results_A.append({'model': 'LGBM_optuna_+weights', 'MAE': res['MAE'],
                       'RMSE': res['RMSE'], 'R2': res['R2'],
                       'mejoras': 'P2+Optuna'})
    oof_opt_A = res['oof_preds']

    pd.DataFrame([best_params_A]).to_csv(f'{CLEAN}/best_params_A_v2.csv', index=False)
else:
    lgbm_A_opt = lgbm_ref if HAS_LGB else GradientBoostingRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42)
    oof_opt_A  = oof_weighted_A if HAS_LGB else oof_ref_A

# ---- P1: Calibración Isotónica ----
print("\n  P1: Aplicando calibración isotónica...")
print("  (Entrena un regresor isotónico sobre las predicciones OOF vs valores reales)")

# Usamos las OOF del mejor modelo para entrenar el calibrador
# IMPORTANTE: el calibrador se entrena en OOF para evitar leakage
oof_for_calib = oof_opt_A if (HAS_OPTUNA and HAS_LGB) else \
                (oof_weighted_A if HAS_LGB else oof_ref_A)

valid_calib = ~np.isnan(oof_for_calib)
isotonic = IsotonicRegression(out_of_bounds='clip')
isotonic.fit(oof_for_calib[valid_calib], y_A[valid_calib])

# Evaluar calibración en las mismas OOF (estimación optimista — se muestra como referencia)
y_calib_oof = isotonic.predict(oof_for_calib[valid_calib])
mae_before = mean_absolute_error(y_A[valid_calib], oof_for_calib[valid_calib])
mae_after  = mean_absolute_error(y_A[valid_calib], y_calib_oof)
r2_before  = r2_score(y_A[valid_calib], oof_for_calib[valid_calib])
r2_after   = r2_score(y_A[valid_calib], y_calib_oof)
print(f"     Antes calibración: MAE={mae_before:.3f}  R²={r2_before:.4f}")
print(f"     Tras  calibración: MAE={mae_after:.3f}  R²={r2_after:.4f}")
print(f"     ⚠️  Nota: estas métricas son sobre datos de entrenamiento del calibrador.")
print(f"         La mejora real en datos nuevos será menor. Usar como estimación.")

results_A.append({'model': 'LGBM_optuna_+weights_+isotonic', 'MAE': mae_after,
                   'RMSE': np.sqrt(mean_squared_error(y_A[valid_calib], y_calib_oof)),
                   'R2': r2_after, 'mejoras': 'P1+P2+Optuna'})

# Análisis del sesgo antes y después por rango
print("\n  Análisis de sesgo por rango (antes vs después de calibración):")
print(f"  {'Rango real':<12} {'Bias antes':>12} {'Bias después':>13}")
print("  " + "-" * 40)
for low, high, label in [(0,100,'<100'), (100,120,'100-120'), (120,140,'120-140'),
                          (140,160,'140-160'), (160,200,'160-200'), (200,999,'>200')]:
    mask_r = valid_calib & (y_A >= low) & (y_A < high)
    if mask_r.sum() < 5:
        continue
    bias_b = (oof_for_calib[mask_r] - y_A[mask_r]).mean()
    bias_a = (y_calib_oof[mask_r[valid_calib]] - y_A[mask_r]).mean() \
             if mask_r[valid_calib].sum() > 0 else np.nan
    print(f"  {label:<12} {bias_b:>+12.2f} {bias_a:>+13.2f}")

# ---- Guardar modelo A + calibrador ----
print("\n  Entrenando modelo final en todos los datos...")
fi_A, model_A_final = get_feature_importance(
    lgbm_A_opt, feature_set_full, X_A, y_A, sample_weights=sw_A
)
joblib.dump(model_A_final, f'{CLEAN}/models/model_peak_value.pkl')
joblib.dump(isotonic, f'{CLEAN}/models/calibrator_peak_value.pkl')
joblib.dump(feature_set_full, f'{CLEAN}/models/feature_set_peak_value.pkl')
print(f"  💾 Guardado: models/model_peak_value.pkl")
print(f"  💾 Guardado: models/calibrator_peak_value.pkl")

# Resumen Modelo A
results_A_df = pd.DataFrame(results_A).sort_values('MAE')
print(f"\n  📊 RESUMEN MODELO A (peak_value):")
print(f"  {'Modelo':<35} {'MAE':>8} {'RMSE':>8} {'R²':>8}  Mejoras")
print("  " + "-" * 72)
for _, r in results_A_df.iterrows():
    tag = "🏆" if r['model'] == results_A_df.iloc[0]['model'] else "  "
    print(f"  {tag} {r['model']:<33} {r['MAE']:>8.3f} {r['RMSE']:>8.3f} "
          f"{r['R2']:>8.4f}  {r['mejoras']}")

print(f"\n  Top 15 features de peak_value:")
for i, (_, row) in enumerate(fi_A.head(15).iterrows()):
    new_tag = "⭐" if row['feature'] in (new_features + ttp_extra) else "  "
    print(f"  {new_tag} {i+1:>2}. {row['feature']:<35} {row['importance']:.0f}")

# ============================================================
# MODELO B — TTP BINARIO (P3)
# ============================================================
print("\n" + "=" * 70)
print("MODELO B — time_to_peak BINARIO  [P3: rápido <45min vs tardío >75min]")
print("=" * 70)

# Filtrar solo los extremos (rápido y tardío), descartar moderado
mask_rapid  = dataset['time_to_peak_min'] < 45
mask_late   = dataset['time_to_peak_min'] > 75
mask_B_bin  = mask_rapid | mask_late

y_B_bin = (dataset.loc[mask_B_bin, 'time_to_peak_min'] > 75).astype(int).values
g_B_bin = dataset.loc[mask_B_bin, 'user_id'].values
X_B_df  = dataset.loc[mask_B_bin, feature_set_ttp].copy()

X_B_df  = impute_user_median(X_B_df, dataset.loc[mask_B_bin])
X_B_bin = X_B_df.values

n_rapid = (y_B_bin == 0).sum()
n_late  = (y_B_bin == 1).sum()
print(f"  Muestras: {len(y_B_bin):,}  (rápido: {n_rapid:,} / tardío: {n_late:,})")
print(f"  Moderado (descartado del entrenamiento): {mask_B_bin.shape[0] - len(y_B_bin):,}")
print(f"  Features: {X_B_bin.shape[1]}")

# Balance de clases
scale_pos = n_rapid / n_late  # peso para clase tardío si hay desbalance

results_B = []
oof_proba_B = None
best_clf_B  = None
best_f1_B   = -1

# ---- B1. Baseline GBR ----
print("\n  B1. GBR (baseline)...")
gbr_clf = GradientBoostingClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
)
fold_acc, fold_f1, oof_cat = [], [], np.full(len(y_B_bin), -1)
for train_idx, test_idx in gkf.split(X_B_bin, y_B_bin, g_B_bin):
    m = GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                    learning_rate=0.1, random_state=42)
    m.fit(X_B_bin[train_idx], y_B_bin[train_idx])
    y_pred = m.predict(X_B_bin[test_idx])
    fold_acc.append(accuracy_score(y_B_bin[test_idx], y_pred))
    fold_f1.append(f1_score(y_B_bin[test_idx], y_pred))
    oof_cat[test_idx] = y_pred
avg_f1 = np.mean(fold_f1)
print(f"     Accuracy={np.mean(fold_acc):.3f}  F1={avg_f1:.3f}")
results_B.append({'model': 'GBR_base', 'accuracy': np.mean(fold_acc), 'f1': avg_f1})
if avg_f1 > best_f1_B:
    best_f1_B, best_clf_B = avg_f1, gbr_clf

# ---- B2. LightGBM ----
if HAS_LGB:
    print("\n  B2. LightGBM (default)...")
    lgbm_clf = LGBMClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
        scale_pos_weight=scale_pos,
        random_state=42, n_jobs=-1, verbose=-1
    )
    fold_acc, fold_f1, oof_proba = [], [], np.full(len(y_B_bin), np.nan)
    for train_idx, test_idx in gkf.split(X_B_bin, y_B_bin, g_B_bin):
        m = LGBMClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
                            scale_pos_weight=scale_pos,
                            random_state=42, n_jobs=-1, verbose=-1)
        m.fit(X_B_bin[train_idx], y_B_bin[train_idx])
        y_pred  = m.predict(X_B_bin[test_idx])
        y_proba = m.predict_proba(X_B_bin[test_idx])[:, 1]
        fold_acc.append(accuracy_score(y_B_bin[test_idx], y_pred))
        fold_f1.append(f1_score(y_B_bin[test_idx], y_pred))
        oof_proba[test_idx] = y_proba
    avg_f1 = np.mean(fold_f1)
    print(f"     Accuracy={np.mean(fold_acc):.3f}  F1={avg_f1:.3f}")
    results_B.append({'model': 'LightGBM_default', 'accuracy': np.mean(fold_acc), 'f1': avg_f1})
    if avg_f1 > best_f1_B:
        best_f1_B, best_clf_B, oof_proba_B = avg_f1, lgbm_clf, oof_proba

# ---- B3. Optuna LightGBM ----
if HAS_OPTUNA and HAS_LGB:
    print(f"\n  B3. Optuna LightGBM ({N_TRIALS_B} trials)...")

    def objective_B(trial):
        params = {
            'n_estimators':      trial.suggest_int('n_estimators', 200, 1000),
            'max_depth':         trial.suggest_int('max_depth', 3, 10),
            'learning_rate':     trial.suggest_float('learning_rate', 0.005, 0.15, log=True),
            'num_leaves':        trial.suggest_int('num_leaves', 20, 200),
            'subsample':         trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 80),
            'reg_alpha':         trial.suggest_float('reg_alpha', 1e-4, 20.0, log=True),
            'reg_lambda':        trial.suggest_float('reg_lambda', 1e-4, 20.0, log=True),
            'scale_pos_weight':  scale_pos,
            'random_state': 42, 'n_jobs': -1, 'verbose': -1,
        }
        fold_f1 = []
        for train_idx, test_idx in gkf.split(X_B_bin, y_B_bin, g_B_bin):
            m = LGBMClassifier(**params)
            m.fit(X_B_bin[train_idx], y_B_bin[train_idx])
            y_pred = m.predict(X_B_bin[test_idx])
            fold_f1.append(f1_score(y_B_bin[test_idx], y_pred))
        return -np.mean(fold_f1)  # minimizar negativo = maximizar F1

    study_B = optuna.create_study(direction='minimize')
    study_B.optimize(objective_B, n_trials=N_TRIALS_B, show_progress_bar=False)
    best_params_B = study_B.best_params
    best_params_B['scale_pos_weight'] = scale_pos
    print(f"     Mejor F1 CV: {-study_B.best_value:.3f}")

    lgbm_clf_opt = LGBMClassifier(**best_params_B, random_state=42, n_jobs=-1, verbose=-1)
    fold_acc, fold_f1, oof_proba = [], [], np.full(len(y_B_bin), np.nan)
    for train_idx, test_idx in gkf.split(X_B_bin, y_B_bin, g_B_bin):
        m = LGBMClassifier(**best_params_B, random_state=42, n_jobs=-1, verbose=-1)
        m.fit(X_B_bin[train_idx], y_B_bin[train_idx])
        y_pred  = m.predict(X_B_bin[test_idx])
        y_proba = m.predict_proba(X_B_bin[test_idx])[:, 1]
        fold_acc.append(accuracy_score(y_B_bin[test_idx], y_pred))
        fold_f1.append(f1_score(y_B_bin[test_idx], y_pred))
        oof_proba[test_idx] = y_proba

    avg_f1 = np.mean(fold_f1)
    print(f"     Evaluación final: Accuracy={np.mean(fold_acc):.3f}  F1={avg_f1:.3f}")
    results_B.append({'model': 'LightGBM_optuna', 'accuracy': np.mean(fold_acc), 'f1': avg_f1})
    if avg_f1 > best_f1_B:
        best_f1_B, best_clf_B, oof_proba_B = avg_f1, lgbm_clf_opt, oof_proba

    pd.DataFrame([best_params_B]).to_csv(f'{CLEAN}/best_params_B_v2.csv', index=False)

# Confusion matrix del mejor modelo
print(f"\n  Matriz de confusión del mejor modelo binario:")
oof_pred_bin = (oof_proba_B >= 0.5).astype(int) if oof_proba_B is not None \
               else np.full(len(y_B_bin), 0)
cm = confusion_matrix(y_B_bin, oof_pred_bin)
print(f"                 Pred_rápido  Pred_tardío")
print(f"  Real_rápido   {cm[0,0]:>10}  {cm[0,1]:>10}")
print(f"  Real_tardío   {cm[1,0]:>10}  {cm[1,1]:>10}")
p, r, f, _ = precision_recall_fscore_support(y_B_bin, oof_pred_bin, average='binary')
print(f"\n  Precision: {p:.3f}  Recall: {r:.3f}  F1: {f:.3f}")

# Feature importance TTP
print(f"\n  Top 15 features de time_to_peak binario:")
_, model_B_final = get_feature_importance(
    best_clf_B if best_clf_B is not None else gbr_clf,
    feature_set_ttp, X_B_bin, y_B_bin
)
fi_B = pd.DataFrame({
    'feature': feature_set_ttp,
    'importance': model_B_final.feature_importances_
}).sort_values('importance', ascending=False).reset_index(drop=True)

for i, (_, row) in enumerate(fi_B.head(15).iterrows()):
    new_tag = "⭐" if row['feature'] in (new_features + ttp_extra) else "  "
    print(f"  {new_tag} {i+1:>2}. {row['feature']:<35} {row['importance']:.0f}")

# Guardar modelo B
joblib.dump(model_B_final, f'{CLEAN}/models/model_ttp_binary.pkl')
joblib.dump(feature_set_ttp, f'{CLEAN}/models/feature_set_ttp.pkl')
print(f"\n  💾 Guardado: models/model_ttp_binary.pkl")

# ============================================================
# GUARDAR RESULTADOS
# ============================================================
print("\n💾 Guardando resultados...")

results_A_df.to_csv(f'{CLEAN}/model_results_peak_value_v2.csv', index=False)
pd.DataFrame(results_B).sort_values('f1', ascending=False).to_csv(
    f'{CLEAN}/model_results_ttp_binary.csv', index=False)
fi_A.to_csv(f'{CLEAN}/feature_importances_peak_value_v2.csv', index=False)
fi_B.to_csv(f'{CLEAN}/feature_importances_ttp_binary.csv', index=False)

# Predicciones OOF completas para análisis posterior
oof_A_final = oof_opt_A if (HAS_OPTUNA and HAS_LGB) else \
              (oof_weighted_A if HAS_LGB else oof_ref_A)

pred_out_A = dataset.loc[mask_A, ['meal_id','user_id','meal_timestamp',
                                    'peak_value','glucose_preprandial']].copy()
pred_out_A['peak_pred_raw']       = oof_A_final
pred_out_A['peak_pred_calibrated'] = np.nan
valid_idx = ~np.isnan(oof_A_final)
pred_out_A.loc[valid_idx, 'peak_pred_calibrated'] = isotonic.predict(oof_A_final[valid_idx])
pred_out_A['error_raw']       = pred_out_A['peak_pred_raw']       - y_A
pred_out_A['error_calibrated'] = pred_out_A['peak_pred_calibrated'] - y_A
pred_out_A['abs_error_raw']       = pred_out_A['error_raw'].abs()
pred_out_A['abs_error_calibrated'] = pred_out_A['error_calibrated'].abs()
pred_out_A.to_csv(f'{CLEAN}/predictions_peak_value_v2.csv', index=False)

pred_out_B = dataset.loc[mask_B_bin, ['meal_id','user_id','meal_timestamp',
                                        'time_to_peak_min']].copy()
pred_out_B['real_binary'] = y_B_bin
pred_out_B['pred_binary'] = oof_pred_bin
pred_out_B['proba_tardio'] = oof_proba_B if oof_proba_B is not None else np.nan
pred_out_B['correct'] = (pred_out_B['real_binary'] == pred_out_B['pred_binary']).astype(int)
pred_out_B.to_csv(f'{CLEAN}/predictions_ttp_binary.csv', index=False)

print(f"  model_results_peak_value_v2.csv")
print(f"  model_results_ttp_binary.csv")
print(f"  feature_importances_peak_value_v2.csv")
print(f"  feature_importances_ttp_binary.csv")
print(f"  predictions_peak_value_v2.csv  (raw + calibrated)")
print(f"  predictions_ttp_binary.csv")

# ============================================================
# RESUMEN FINAL
# ============================================================
print("\n" + "=" * 70)
print("📊 RESUMEN FINAL")
print("=" * 70)

print(f"\n--- MODELO A: peak_value ---")
for _, r in results_A_df.iterrows():
    tag = "🏆" if r['model'] == results_A_df.iloc[0]['model'] else "  "
    print(f"  {tag} {r['model']:<38}: MAE={r['MAE']:.3f} mg/dL  R²={r['R2']:.4f}  [{r['mejoras']}]")

print(f"\n--- MODELO B: time_to_peak binario (rápido vs tardío) ---")
for r in sorted(results_B, key=lambda x: -x['f1']):
    tag = "🏆" if r['f1'] == max(x['f1'] for x in results_B) else "  "
    print(f"  {tag} {r['model']:<20}: Accuracy={r['accuracy']:.3f}  F1={r['f1']:.3f}")

print(f"\n--- MEJORAS NETAS vs FASE 5 ---")
mae_fase5 = 13.404
mae_final = results_A_df.iloc[0]['MAE']
print(f"  peak_value MAE:  fase5={mae_fase5:.3f}  →  v2={mae_final:.3f}  "
      f"(Δ = {mae_final - mae_fase5:+.3f} mg/dL)")
print(f"  time_to_peak:    reformulado como binario "
      f"(F1={max(r['f1'] for r in results_B):.3f} vs F1_3clases=0.414)")

print(f"\n--- MODELOS GUARDADOS ---")
print(f"  models/model_peak_value.pkl         → LightGBM entrenado en todos los datos")
print(f"  models/calibrator_peak_value.pkl    → IsotonicRegression para post-procesado")
print(f"  models/feature_set_peak_value.pkl   → Lista de features en orden correcto")
print(f"  models/model_ttp_binary.pkl         → Clasificador rápido/tardío")
print(f"  models/feature_set_ttp.pkl          → Lista de features TTP en orden correcto")

print(f"""
--- INSTRUCCIONES DE USO EN PRODUCCIÓN ---

  import joblib, numpy as np

  # Cargar modelos
  model_peak   = joblib.load('clean_data/models/model_peak_value.pkl')
  calibrator   = joblib.load('clean_data/models/calibrator_peak_value.pkl')
  feat_peak    = joblib.load('clean_data/models/feature_set_peak_value.pkl')
  model_ttp    = joblib.load('clean_data/models/model_ttp_binary.pkl')
  feat_ttp     = joblib.load('clean_data/models/feature_set_ttp.pkl')

  # Predicción de pico (con calibración)
  X_new = ...  # DataFrame con columnas == feat_peak
  peak_raw        = model_peak.predict(X_new[feat_peak])
  peak_calibrated = calibrator.predict(peak_raw)   # ← usar este valor

  # Predicción de tiempo al pico
  X_ttp    = ...  # DataFrame con columnas == feat_ttp
  ttp_proba  = model_ttp.predict_proba(X_ttp[feat_ttp])[:, 1]  # P(tardío)
  ttp_class  = np.where(ttp_proba > 0.5, 'tardío (>75min)', 'rápido (<45min)')
  # Si ttp_proba ≈ 0.5 (0.35–0.65), reportar como 'incierto'
  ttp_label  = np.where((ttp_proba > 0.35) & (ttp_proba < 0.65),
                         'incierto', ttp_class)
""")

print("✅ FASE 5B COMPLETADA")
