"""
FASE 5 — MODELADO PREDICTIVO: PICO GLUCÉMICO Y TIEMPO AL PICO
Proyecto Glucovibes Challenge 2026

OBJETIVOS:
  - Predecir peak_value     (mg/dL) con el menor MAE posible
  - Predecir time_to_peak_min (min) — también como clasificación (rápido/moderado/tardío)

ARQUITECTURA:
  Modelo A — peak_value
    · LightGBM / XGBoost con features completas (incluidas las nuevas de fase3b)
    · Búsqueda de hiperparámetros con Optuna (GroupKFold 5-fold, anti-leakage)
    · Fallback a GBR si Optuna no está disponible

  Modelo B — time_to_peak_min
    · Regresión: LightGBM / XGBoost directo
    · Clasificación (3 clases): rápido <45min / moderado 45-75min / tardío >75min
    · Se reportan ambos y se recomienda el clasificador para producción

  Modelo C — peak_value usando predicción de tiempo al pico (stacking en cascada)
    · Añade la predicción de time_to_peak_min como feature al modelo de peak_value
    · Valida si el stacking mejora el MAE de peak_value

PREREQUISITOS:
  pip install xgboost lightgbm scikit-learn optuna pandas numpy scipy
  - ./clean_data/modeling_dataset_pico.csv  (salida de fase3b)

EJECUCIÓN:
  python fase5_modelado_pico.py
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                              r2_score, accuracy_score, f1_score)
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
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
    print("✅ Optuna disponible — se realizará búsqueda de hiperparámetros")
except ImportError:
    HAS_OPTUNA = False
    print("⚠️  Optuna no instalado — se usarán hiperparámetros por defecto")
    print("    Para instalarlo: pip install optuna")

CLEAN = './clean_data'
N_FOLDS  = 5
N_OPTUNA_TRIALS = 50   # Reducir a 20 si tarda demasiado

print("\n" + "=" * 70)
print("FASE 5 — MODELADO: PEAK VALUE Y TIME TO PEAK")
print("=" * 70)

# ============================================================
# CARGAR DATOS
# ============================================================
print("\n📦 Cargando dataset...")
dataset = pd.read_csv(f'{CLEAN}/modeling_dataset_pico.csv')
dataset['meal_timestamp'] = pd.to_datetime(dataset['meal_timestamp'], format='mixed', utc=True)
print(f"  Dataset: {len(dataset):,} filas × {dataset.shape[1]} columnas")
print(f"  Usuarios: {dataset['user_id'].nunique()}")

# ============================================================
# DEFINICIÓN DE FEATURES
# ============================================================
print("\n🔧 Definiendo features...")

# Features base (las mismas que en fase4b)
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

# Features rolling del usuario (reemplazan a las stats globales con leakage)
rolling_features = [
    'user_peak_mean_roll', 'user_peak_std_roll', 'user_ttp_mean_roll',
    'user_glucose_mean_roll', 'user_glucose_std_roll',
]

# Features CGM preprandiales (nuevas en fase3b)
cgm_pre_features = [
    'cgm_slope_30m', 'cgm_slope_60m', 'cgm_std_30m',
    'cgm_delta_30m', 'cgm_delta_60m', 'cgm_tir_pre',
    'cgm_pct_above_target', 'n_readings_pre_60m',
]

# Features nutricionales adicionales (nuevas en fase3b)
nutrition_new = [
    'net_carbs', 'pct_high_gi_items', 'fat_delay_score',
    'effective_gi_score', 'carbs_per_item',
]

# Actividad multi-ventana (nuevas en fase3b)
sport_new = [
    'sport_2h_duration', 'sport_6h_duration', 'sport_48h_duration',
    'sport_48h_sessions', 'sport_intensity_score',
]

# Contexto diario (nuevas en fase3b)
day_context = [
    'prev_meals_today', 'prev_peak_today_max',
    'hours_since_last_meal', 'carbs_load_today',
]

# Clusters (si existen del pipeline anterior)
cluster_features = []
for col in ['nutri_cluster_num', 'glyc_cluster_num']:
    if col in dataset.columns:
        cluster_features.append(col)

# Encoding de categóricas
cat_features = ['meal_period', 'meal_type', 'sport_prior_intensity']
for col in cat_features:
    if col in dataset.columns:
        le = LabelEncoder()
        enc_col = f'{col}_enc'
        dataset[enc_col] = le.fit_transform(dataset[col].fillna('unknown').astype(str))
        base_numeric.append(enc_col)

# Feature set completo
all_new = cgm_pre_features + nutrition_new + sport_new + day_context + rolling_features

feature_set_full = [f for f in (base_numeric + all_new + cluster_features) if f in dataset.columns]

# Feature set sin rolling (para comparar el aporte del rolling)
feature_set_no_roll = [f for f in feature_set_full if f not in rolling_features]

print(f"  Features base:    {len([f for f in base_numeric if f in dataset.columns])}")
print(f"  Features nuevas:  {len([f for f in all_new if f in dataset.columns])}")
print(f"  Features clusters:{len(cluster_features)}")
print(f"  TOTAL:            {len(feature_set_full)}")

# ============================================================
# UTILIDADES
# ============================================================
gkf = GroupKFold(n_splits=N_FOLDS)

def impute_with_user_median(X, dataset_ref, feature_cols):
    """Imputa NaN con mediana por usuario. Fallback a mediana global."""
    X = X.copy()
    for col in feature_cols:
        if col not in X.columns:
            continue
        nan_mask = X[col].isna()
        if not nan_mask.any():
            continue
        # Mediana por usuario
        user_medians = dataset_ref.groupby('user_id')[col].median()
        user_ids = dataset_ref.loc[X.index, 'user_id'] if 'user_id' in dataset_ref.columns else None
        if user_ids is not None:
            fill_vals = user_ids.map(user_medians)
            X.loc[nan_mask, col] = fill_vals[nan_mask]
        # Fallback a mediana global
        still_nan = X[col].isna()
        if still_nan.any():
            X.loc[still_nan, col] = X[col].median()
    return X

def evaluate_regressor(model, X_arr, y, groups, use_log=False):
    """Evalúa un regresor con GroupKFold. Devuelve métricas y predicciones OOF."""
    fold_mae, fold_rmse, fold_r2 = [], [], []
    oof_preds = np.full(len(y), np.nan)

    for train_idx, test_idx in gkf.split(X_arr, y, groups):
        X_tr, X_te = X_arr[train_idx], X_arr[test_idx]
        y_tr = np.log1p(y[train_idx]) if use_log else y[train_idx]

        m = type(model)(**model.get_params())
        m.fit(X_tr, y_tr)
        y_pred = m.predict(X_te)

        if use_log:
            y_pred = np.expm1(y_pred)
            y_pred = np.maximum(y_pred, 0)

        fold_mae.append(mean_absolute_error(y[test_idx], y_pred))
        fold_rmse.append(np.sqrt(mean_squared_error(y[test_idx], y_pred)))
        fold_r2.append(r2_score(y[test_idx], y_pred))
        oof_preds[test_idx] = y_pred

    return {
        'MAE': np.mean(fold_mae), 'RMSE': np.mean(fold_rmse),
        'R2': np.mean(fold_r2), 'oof_preds': oof_preds
    }

def get_feature_importance(model, feature_names, X_arr, y):
    """Entrena modelo en todos los datos y devuelve importancias."""
    m = type(model)(**model.get_params())
    m.fit(X_arr, y)
    return pd.DataFrame({
        'feature': feature_names,
        'importance': m.feature_importances_
    }).sort_values('importance', ascending=False).reset_index(drop=True)

# ============================================================
# MODELO A — PEAK_VALUE
# ============================================================
print("\n" + "=" * 70)
print("MODELO A — PREDICCIÓN DE peak_value (mg/dL)")
print("=" * 70)

target_A = 'peak_value'
mask_A   = dataset[target_A].notna() & (dataset[target_A] > 0)
X_A_df   = dataset.loc[mask_A, feature_set_full].copy()
y_A      = dataset.loc[mask_A, target_A].values
g_A      = dataset.loc[mask_A, 'user_id'].values

X_A_df = impute_with_user_median(X_A_df, dataset.loc[mask_A], feature_set_full)
X_A    = X_A_df.values

print(f"  Muestras: {len(y_A):,}  |  Features: {X_A.shape[1]}")
print(f"  peak_value — media: {y_A.mean():.1f}, std: {y_A.std():.1f}, "
      f"min: {y_A.min():.0f}, max: {y_A.max():.0f}")

results_A = []

# ---- A1. Baseline: GBR ----
print("\n  A1. GBR (baseline)...")
gbr_A = GradientBoostingRegressor(
    n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
)
res = evaluate_regressor(gbr_A, X_A, y_A, g_A)
print(f"     MAE={res['MAE']:.2f}  RMSE={res['RMSE']:.2f}  R²={res['R2']:.4f}")
results_A.append({'model': 'GBR_base', 'MAE': res['MAE'], 'RMSE': res['RMSE'], 'R2': res['R2']})

# ---- A2. LightGBM ----
if HAS_LGB:
    print("\n  A2. LightGBM (default hyperparams)...")
    lgbm_A = LGBMRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbose=-1
    )
    res = evaluate_regressor(lgbm_A, X_A, y_A, g_A)
    print(f"     MAE={res['MAE']:.2f}  RMSE={res['RMSE']:.2f}  R²={res['R2']:.4f}")
    results_A.append({'model': 'LightGBM_default', 'MAE': res['MAE'], 'RMSE': res['RMSE'], 'R2': res['R2']})
    oof_lgbm_A = res['oof_preds']

# ---- A3. XGBoost ----
if HAS_XGB:
    print("\n  A3. XGBoost (default hyperparams)...")
    xgb_A = XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbosity=0
    )
    res = evaluate_regressor(xgb_A, X_A, y_A, g_A)
    print(f"     MAE={res['MAE']:.2f}  RMSE={res['RMSE']:.2f}  R²={res['R2']:.4f}")
    results_A.append({'model': 'XGB_default', 'MAE': res['MAE'], 'RMSE': res['RMSE'], 'R2': res['R2']})

# ---- A4. Optuna: búsqueda de hiperparámetros para LightGBM ----
if HAS_OPTUNA and HAS_LGB:
    print(f"\n  A4. Optuna — LightGBM ({N_OPTUNA_TRIALS} trials, ~5-10 min)...")

    def objective_A(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 60),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
            'random_state': 42, 'n_jobs': -1, 'verbose': -1,
        }
        model = LGBMRegressor(**params)
        fold_mae = []
        for train_idx, test_idx in gkf.split(X_A, y_A, g_A):
            m = LGBMRegressor(**params)
            m.fit(X_A[train_idx], y_A[train_idx])
            fold_mae.append(mean_absolute_error(y_A[test_idx], m.predict(X_A[test_idx])))
        return np.mean(fold_mae)

    study_A = optuna.create_study(direction='minimize')
    study_A.optimize(objective_A, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)

    best_params_A = study_A.best_params
    print(f"     Mejor MAE en CV: {study_A.best_value:.2f}")
    print(f"     Mejores params: {best_params_A}")

    lgbm_A_opt = LGBMRegressor(**best_params_A, random_state=42, n_jobs=-1, verbose=-1)
    res = evaluate_regressor(lgbm_A_opt, X_A, y_A, g_A)
    print(f"     Evaluación final — MAE={res['MAE']:.2f}  R²={res['R2']:.4f}")
    results_A.append({'model': 'LightGBM_optuna', 'MAE': res['MAE'], 'RMSE': res['RMSE'], 'R2': res['R2']})
    oof_lgbm_A = res['oof_preds']

    # Guardar best params
    pd.DataFrame([best_params_A]).to_csv(f'{CLEAN}/best_params_peak_value.csv', index=False)
    print(f"     Params guardados: best_params_peak_value.csv")

# ---- A5. Resumen Modelo A ----
results_A_df = pd.DataFrame(results_A).sort_values('MAE')
print(f"\n  📊 RESUMEN MODELO A (peak_value):")
print(f"  {'Modelo':<25} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
print("  " + "-" * 52)
for _, r in results_A_df.iterrows():
    tag = "🏆" if r['model'] == results_A_df.iloc[0]['model'] else "  "
    print(f"  {tag} {r['model']:<23} {r['MAE']:>8.2f} {r['RMSE']:>8.2f} {r['R2']:>8.4f}")

# Feature importance del mejor modelo
best_model_A_name = results_A_df.iloc[0]['model']
print(f"\n  Calculando feature importance para {best_model_A_name}...")
if HAS_OPTUNA and HAS_LGB and 'optuna' in best_model_A_name:
    best_model_A = LGBMRegressor(**best_params_A, random_state=42, n_jobs=-1, verbose=-1)
elif HAS_LGB:
    best_model_A = LGBMRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
                                  random_state=42, n_jobs=-1, verbose=-1)
else:
    best_model_A = GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                              learning_rate=0.1, random_state=42)

fi_A = get_feature_importance(best_model_A, feature_set_full, X_A, y_A)
print(f"\n  Top 15 predictores de peak_value:")
for i, (_, row) in enumerate(fi_A.head(15).iterrows()):
    new_tag = "⭐" if row['feature'] in all_new else "  "
    print(f"  {new_tag} {i+1:>2}. {row['feature']:<35} {row['importance']:.4f}")

# ============================================================
# MODELO B — TIME_TO_PEAK_MIN
# ============================================================
print("\n" + "=" * 70)
print("MODELO B — PREDICCIÓN DE time_to_peak_min (min)")
print("=" * 70)

target_B = 'time_to_peak_min'
mask_B   = dataset[target_B].notna() & (dataset[target_B] > 0) & (dataset[target_B] <= 180)
X_B_df   = dataset.loc[mask_B, feature_set_full].copy()
y_B      = dataset.loc[mask_B, target_B].values
g_B      = dataset.loc[mask_B, 'user_id'].values

X_B_df = impute_with_user_median(X_B_df, dataset.loc[mask_B], feature_set_full)
X_B    = X_B_df.values

print(f"  Muestras: {len(y_B):,}  |  Features: {X_B.shape[1]}")
print(f"  time_to_peak — media: {y_B.mean():.1f} min, std: {y_B.std():.1f}, "
      f"min: {y_B.min():.0f}, max: {y_B.max():.0f}")

# Distribución por categoría
q_rapido  = (y_B < 45).sum()
q_moderad = ((y_B >= 45) & (y_B <= 75)).sum()
q_tardio  = (y_B > 75).sum()
print(f"  Distribución: rápido (<45min)={q_rapido:,} ({q_rapido/len(y_B)*100:.1f}%)  "
      f"moderado={q_moderad:,} ({q_moderad/len(y_B)*100:.1f}%)  "
      f"tardío (>75min)={q_tardio:,} ({q_tardio/len(y_B)*100:.1f}%)")

results_B = []

# ---- B1. Regresión directa ----
print("\n  B1. Regresión directa — GBR (baseline)...")
gbr_B = GradientBoostingRegressor(
    n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
)
res = evaluate_regressor(gbr_B, X_B, y_B, g_B)
print(f"     MAE={res['MAE']:.2f}min  RMSE={res['RMSE']:.2f}  R²={res['R2']:.4f}")
results_B.append({'task': 'regression', 'model': 'GBR_base',
                   'MAE': res['MAE'], 'RMSE': res['RMSE'], 'R2': res['R2']})

if HAS_LGB:
    print("\n  B2. Regresión directa — LightGBM...")
    lgbm_B = LGBMRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbose=-1
    )
    res = evaluate_regressor(lgbm_B, X_B, y_B, g_B)
    print(f"     MAE={res['MAE']:.2f}min  RMSE={res['RMSE']:.2f}  R²={res['R2']:.4f}")
    results_B.append({'task': 'regression', 'model': 'LightGBM_default',
                       'MAE': res['MAE'], 'RMSE': res['RMSE'], 'R2': res['R2']})
    oof_lgbm_B = res['oof_preds']

# ---- B3. Optuna para regresión de tiempo al pico ----
if HAS_OPTUNA and HAS_LGB:
    print(f"\n  B3. Optuna — LightGBM regresión ({N_OPTUNA_TRIALS} trials)...")

    def objective_B(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 60),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
            'random_state': 42, 'n_jobs': -1, 'verbose': -1,
        }
        fold_mae = []
        for train_idx, test_idx in gkf.split(X_B, y_B, g_B):
            m = LGBMRegressor(**params)
            m.fit(X_B[train_idx], y_B[train_idx])
            fold_mae.append(mean_absolute_error(y_B[test_idx], m.predict(X_B[test_idx])))
        return np.mean(fold_mae)

    study_B = optuna.create_study(direction='minimize')
    study_B.optimize(objective_B, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)

    best_params_B = study_B.best_params
    print(f"     Mejor MAE en CV: {study_B.best_value:.2f} min")

    lgbm_B_opt = LGBMRegressor(**best_params_B, random_state=42, n_jobs=-1, verbose=-1)
    res = evaluate_regressor(lgbm_B_opt, X_B, y_B, g_B)
    print(f"     Evaluación final — MAE={res['MAE']:.2f}min  R²={res['R2']:.4f}")
    results_B.append({'task': 'regression', 'model': 'LightGBM_optuna',
                       'MAE': res['MAE'], 'RMSE': res['RMSE'], 'R2': res['R2']})
    oof_lgbm_B = res['oof_preds']

    pd.DataFrame([best_params_B]).to_csv(f'{CLEAN}/best_params_time_to_peak.csv', index=False)

# ---- B4. Clasificación (rápido / moderado / tardío) ----
print("\n  B4. Clasificación 3 clases: rápido / moderado / tardío...")

def categorize_ttp(val):
    if val < 45:  return 0  # rápido
    elif val <= 75: return 1  # moderado
    else:           return 2  # tardío

y_B_cat = np.array([categorize_ttp(v) for v in y_B])
class_names = {0: 'rápido (<45min)', 1: 'moderado (45-75min)', 2: 'tardío (>75min)'}

clf_results = []
classifiers = {}

if HAS_LGB:
    classifiers['LightGBM'] = LGBMClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
        random_state=42, n_jobs=-1, verbose=-1
    )
if HAS_XGB:
    classifiers['XGBoost'] = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbosity=0,
        eval_metric='mlogloss', use_label_encoder=False
    )
classifiers['GBR'] = GradientBoostingClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
)

oof_clf_B = None
best_clf_name = None
best_clf_f1   = -1

for clf_name, clf in classifiers.items():
    fold_acc, fold_f1 = [], []
    oof_clf = np.full(len(y_B_cat), -1)

    for train_idx, test_idx in gkf.split(X_B, y_B_cat, g_B):
        m = type(clf)(**clf.get_params())
        m.fit(X_B[train_idx], y_B_cat[train_idx])
        y_pred_cat = m.predict(X_B[test_idx])
        fold_acc.append(accuracy_score(y_B_cat[test_idx], y_pred_cat))
        fold_f1.append(f1_score(y_B_cat[test_idx], y_pred_cat, average='macro'))
        oof_clf[test_idx] = y_pred_cat

    avg_acc = np.mean(fold_acc)
    avg_f1  = np.mean(fold_f1)
    tag = "🏆" if avg_f1 > best_clf_f1 else "  "
    print(f"     {tag} {clf_name:<12}: Accuracy={avg_acc:.3f}  F1-macro={avg_f1:.3f}")
    clf_results.append({'model': clf_name, 'accuracy': avg_acc, 'f1_macro': avg_f1})

    if avg_f1 > best_clf_f1:
        best_clf_f1   = avg_f1
        best_clf_name = clf_name
        oof_clf_B     = oof_clf

results_B_df = pd.DataFrame(results_B).sort_values('MAE')

print(f"\n  📊 RESUMEN MODELO B — Regresión (time_to_peak_min):")
print(f"  {'Modelo':<25} {'MAE(min)':>10} {'RMSE':>8} {'R²':>8}")
print("  " + "-" * 55)
for _, r in results_B_df.iterrows():
    tag = "🏆" if r['model'] == results_B_df.iloc[0]['model'] else "  "
    print(f"  {tag} {r['model']:<23} {r['MAE']:>10.2f} {r['RMSE']:>8.2f} {r['R2']:>8.4f}")

# Feature importance time_to_peak
best_reg_B = lgbm_B_opt if (HAS_OPTUNA and HAS_LGB) else (lgbm_B if HAS_LGB else gbr_B)
fi_B = get_feature_importance(best_reg_B, feature_set_full, X_B, y_B)
print(f"\n  Top 15 predictores de time_to_peak_min:")
for i, (_, row) in enumerate(fi_B.head(15).iterrows()):
    new_tag = "⭐" if row['feature'] in all_new else "  "
    print(f"  {new_tag} {i+1:>2}. {row['feature']:<35} {row['importance']:.4f}")

# ============================================================
# MODELO C — STACKING EN CASCADA (peak_value + pred_ttp como feature)
# ============================================================
print("\n" + "=" * 70)
print("MODELO C — STACKING: peak_value con pred_time_to_peak como feature")
print("=" * 70)

# Alinear índices de A y B (pueden diferir por distintos filtros de NaN)
idx_A = dataset.loc[mask_A].index
idx_B = dataset.loc[mask_B].index
common_idx = idx_A.intersection(idx_B)

print(f"  Muestras comunes A∩B: {len(common_idx):,} / A={len(idx_A):,} / B={len(idx_B):,}")

if len(common_idx) > 500:
    # Recuperar las predicciones OOF alineadas
    oof_A_series = pd.Series(oof_lgbm_A if HAS_LGB else np.full(len(idx_A), np.nan), index=idx_A)
    oof_B_series = pd.Series(oof_lgbm_B if HAS_LGB else np.full(len(idx_B), np.nan), index=idx_B)

    # Dataset de stacking
    stacking_df = dataset.loc[common_idx, feature_set_full + [target_A, 'user_id']].copy()
    stacking_df['pred_time_to_peak'] = oof_B_series.loc[common_idx].values
    stacking_df['pred_ttp_category'] = np.array([categorize_ttp(v) if not np.isnan(v) else 1
                                                   for v in stacking_df['pred_time_to_peak']])

    feature_set_C = feature_set_full + ['pred_time_to_peak', 'pred_ttp_category']
    X_C_df = stacking_df[feature_set_C].copy()
    y_C    = stacking_df[target_A].values
    g_C    = stacking_df['user_id'].values

    X_C_df = impute_with_user_median(X_C_df, stacking_df, feature_set_C)
    X_C    = X_C_df.values

    if HAS_LGB:
        stacking_model = LGBMRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
            random_state=42, n_jobs=-1, verbose=-1
        )
        res_C = evaluate_regressor(stacking_model, X_C, y_C, g_C)
        print(f"  LightGBM + pred_ttp: MAE={res_C['MAE']:.2f}  R²={res_C['R2']:.4f}")

        best_A_mae = results_A_df.iloc[0]['MAE']
        delta_mae  = res_C['MAE'] - best_A_mae
        if delta_mae < 0:
            print(f"  ✅ Stacking MEJORA: Δ MAE = {delta_mae:.2f} (mejor que modelo A solo)")
        else:
            print(f"  ℹ️  Stacking no mejora: Δ MAE = +{delta_mae:.2f} (modelo A sin stacking es mejor)")
else:
    print("  ⚠️  Insuficientes muestras comunes para stacking, saltando Modelo C.")

# ============================================================
# GUARDAR RESULTADOS
# ============================================================
print("\n💾 Guardando resultados...")

# Resultados Modelo A
results_A_df.to_csv(f'{CLEAN}/model_results_peak_value.csv', index=False)
print(f"  model_results_peak_value.csv")

# Resultados Modelo B (regresión + clasificación)
results_B_df.to_csv(f'{CLEAN}/model_results_time_to_peak.csv', index=False)
pd.DataFrame(clf_results).to_csv(f'{CLEAN}/model_results_ttp_classification.csv', index=False)
print(f"  model_results_time_to_peak.csv")
print(f"  model_results_ttp_classification.csv")

# Feature importances
fi_A.to_csv(f'{CLEAN}/feature_importances_peak_value.csv', index=False)
fi_B.to_csv(f'{CLEAN}/feature_importances_time_to_peak.csv', index=False)
print(f"  feature_importances_peak_value.csv")
print(f"  feature_importances_time_to_peak.csv")

# Predicciones OOF para análisis de errores
if HAS_LGB:
    pred_df_A = dataset.loc[mask_A, ['meal_id', 'user_id', 'meal_timestamp',
                                      'peak_value', 'glucose_preprandial']].copy()
    pred_df_A['peak_pred']  = oof_lgbm_A
    pred_df_A['error']      = oof_lgbm_A - y_A
    pred_df_A['abs_error']  = np.abs(pred_df_A['error'])
    pred_df_A.to_csv(f'{CLEAN}/predictions_peak_value.csv', index=False)

    pred_df_B = dataset.loc[mask_B, ['meal_id', 'user_id', 'meal_timestamp',
                                      'time_to_peak_min', 'total_fat', 'total_carbs']].copy()
    pred_df_B['ttp_pred']   = oof_lgbm_B
    pred_df_B['error']      = oof_lgbm_B - y_B
    pred_df_B['abs_error']  = np.abs(pred_df_B['error'])
    pred_df_B['real_cat']   = y_B_cat
    pred_df_B['pred_cat']   = oof_clf_B if oof_clf_B is not None else np.nan
    pred_df_B.to_csv(f'{CLEAN}/predictions_time_to_peak.csv', index=False)

    print(f"  predictions_peak_value.csv")
    print(f"  predictions_time_to_peak.csv")

# ============================================================
# RESUMEN FINAL
# ============================================================
print("\n" + "=" * 70)
print("📊 RESUMEN FINAL")
print("=" * 70)

print(f"\n--- MODELO A: peak_value ---")
for _, r in results_A_df.iterrows():
    tag = "🏆" if r['model'] == results_A_df.iloc[0]['model'] else "  "
    print(f"  {tag} {r['model']:<25}: MAE={r['MAE']:.2f} mg/dL  R²={r['R2']:.4f}")

print(f"\n--- MODELO B: time_to_peak_min (regresión) ---")
for _, r in results_B_df.iterrows():
    tag = "🏆" if r['model'] == results_B_df.iloc[0]['model'] else "  "
    print(f"  {tag} {r['model']:<25}: MAE={r['MAE']:.2f} min  R²={r['R2']:.4f}")

print(f"\n--- MODELO B: time_to_peak_min (clasificación) ---")
for r in clf_results:
    tag = "🏆" if r['model'] == best_clf_name else "  "
    print(f"  {tag} {r['model']:<12}: Accuracy={r['accuracy']:.3f}  F1-macro={r['f1_macro']:.3f}")
    print(f"       Clases: rápido (<45min) / moderado (45-75min) / tardío (>75min)")

print(f"\n--- TOP 5 FEATURES CLAVE ---")
print(f"  peak_value:      {', '.join(fi_A.head(5)['feature'].tolist())}")
print(f"  time_to_peak:    {', '.join(fi_B.head(5)['feature'].tolist())}")

print("\n✅ FASE 5 COMPLETADA")
print("   Archivos clave para análisis:")
print("   · model_results_peak_value.csv")
print("   · model_results_time_to_peak.csv")
print("   · model_results_ttp_classification.csv")
print("   · feature_importances_peak_value.csv")
print("   · feature_importances_time_to_peak.csv")
print("   · predictions_peak_value.csv")
print("   · predictions_time_to_peak.csv")
