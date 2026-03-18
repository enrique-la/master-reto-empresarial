"""
FASE 4B — PARCHE: Modelos mejorados SIN log-transform
Ejecutar después de fase4.

Problema detectado: log-transform mejora R² en log-space (0.315) pero
la back-transformación con expm1() pierde calidad (R²_orig=0.098).

Solución: Ejecutar XGBoost/LightGBM en iAUC directo con features de
interacción para aislar la mejora real de algoritmo+features.

EJECUCIÓN:
  cd glucovibes
  python scripts/fase4b_parche_modelos.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

CLEAN = './clean_data'

print("=" * 70)
print("FASE 4B — PARCHE: Modelos directos (sin log-transform)")
print("=" * 70)

# Cargar dataset con features de interacción
dataset = pd.read_csv(f'{CLEAN}/modeling_dataset_final.csv')
print(f"Dataset: {len(dataset):,} filas × {dataset.shape[1]} columnas")

# ============================================================
# FEATURES (las mismas mejoradas de fase4)
# ============================================================
from sklearn.preprocessing import LabelEncoder

numeric_features = [
    'glucose_preprandial', 'total_calories', 'total_protein', 'total_fat',
    'total_carbs', 'total_fibre', 'n_items', 'pct_cal_carbs', 'pct_cal_protein',
    'pct_cal_fat', 'fibre_carb_ratio', 'n_ultraprocessed', 'n_high_gi',
    'avg_saturated_fat', 'n_food_groups', 'hour_of_day', 'day_of_week',
    'is_weekend', 'sport_prior_duration', 'sport_prior_sessions',
    'hours_since_last_sport', 'sleep_time_prev', 'sleep_quality_prev',
    'tiredness', 'fasting_hunger', 'resting_hr_morning', 'hrv_morning',
    'training_effort_prev', 'anxiety_prev', 'nutrition_plan_prev',
    'day_eval_prev', 'user_glucose_mean', 'user_glucose_std',
    'user_glucose_median', 'user_glucose_q25', 'user_glucose_q75',
    'user_time_in_range', 'pct_items_with_fibre',
]

# Encoding categóricas
for col in ['meal_period', 'meal_type', 'sport_prior_intensity']:
    if col in dataset.columns:
        le = LabelEncoder()
        dataset[f'{col}_enc'] = le.fit_transform(dataset[col].fillna('unknown').astype(str))
        numeric_features.append(f'{col}_enc')

# Features de interacción (ya existen en el dataset)
interaction_features = ['net_carbs', 'carbs_x_hour', 'carbs_x_preprandial',
                        'fat_protein_ratio', 'nutri_cluster_num', 'glyc_cluster_num']

# Separar features base vs mejoradas
base_features = [f for f in numeric_features if f in dataset.columns]
improved_features = base_features + [f for f in interaction_features if f in dataset.columns]

print(f"Features base: {len(base_features)}")
print(f"Features mejoradas: {len(improved_features)}")

# ============================================================
# MODELOS: Predicción DIRECTA de iAUC (sin log-transform)
# ============================================================
gkf = GroupKFold(n_splits=5)

models_to_test = {
    # Originales (para confirmar baseline)
    'GBR_base': (GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
    ), base_features),
    
    # Mejorados: mismo algoritmo, features mejoradas
    'GBR_improved_features': (GradientBoostingRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=5, random_state=42
    ), improved_features),
}

if HAS_XGB:
    # XGBoost con features base (para ver efecto del algoritmo solo)
    models_to_test['XGB_base'] = (XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbosity=0
    ), base_features)
    
    # XGBoost con features mejoradas (algoritmo + features)
    models_to_test['XGB_improved'] = (XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbosity=0
    ), improved_features)

if HAS_LGB:
    models_to_test['LGBM_base'] = (LGBMRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
        random_state=42, n_jobs=-1, verbose=-1
    ), base_features)
    
    models_to_test['LGBM_improved'] = (LGBMRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
        random_state=42, n_jobs=-1, verbose=-1
    ), improved_features)

# También probar log-transform + corrección de sesgo
if HAS_XGB:
    models_to_test['XGB_log_corrected'] = (XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbosity=0
    ), improved_features)

# ============================================================
# EVALUACIÓN
# ============================================================
print("\n📊 Evaluando modelos en predicción DIRECTA de iAUC...")
print("-" * 80)

results = []
best_predictions = None
best_r2 = -np.inf
best_name = ""

for model_name, (model, features) in models_to_test.items():
    
    use_log = 'log_corrected' in model_name
    target_col = 'iauc'
    
    mask = dataset[target_col].notna()
    X = dataset.loc[mask, features].copy()
    y = dataset.loc[mask, target_col].values
    groups = dataset.loc[mask, 'user_id'].values
    
    # Imputar NaN
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())
    
    X_arr = X.values
    
    if use_log:
        y_train_target = np.log1p(np.clip(y, 0, None))
    else:
        y_train_target = y
    
    fold_mae, fold_rmse, fold_r2 = [], [], []
    all_preds = np.full(len(y), np.nan)
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_arr, y_train_target, groups)):
        X_train, X_test = X_arr[train_idx], X_arr[test_idx]
        yt_train = y_train_target[train_idx]
        y_test_orig = y[test_idx]
        
        m = type(model)(**model.get_params())
        m.fit(X_train, yt_train)
        y_pred = m.predict(X_test)
        
        if use_log:
            # Corrección de sesgo: añadir varianza residual / 2
            y_pred_train_log = m.predict(X_train)
            residuals = yt_train - y_pred_train_log
            bias_correction = np.var(residuals) / 2
            y_pred_orig = np.expm1(y_pred + bias_correction)
            y_pred_orig = np.maximum(y_pred_orig, 0)
        else:
            y_pred_orig = y_pred
        
        fold_mae.append(mean_absolute_error(y_test_orig, y_pred_orig))
        fold_rmse.append(np.sqrt(mean_squared_error(y_test_orig, y_pred_orig)))
        fold_r2.append(r2_score(y_test_orig, y_pred_orig))
        all_preds[test_idx] = y_pred_orig
    
    avg_mae = np.mean(fold_mae)
    avg_rmse = np.mean(fold_rmse)
    avg_r2 = np.mean(fold_r2)
    
    feature_type = 'improved' if 'improved' in model_name or 'log' in model_name else 'base'
    log_flag = 'log+bias_corr' if use_log else 'direct'
    
    results.append({
        'model': model_name,
        'MAE': round(avg_mae, 1),
        'RMSE': round(avg_rmse, 1),
        'R²': round(avg_r2, 4),
        'features': feature_type,
        'target_transform': log_flag,
        'n_features': len(features),
    })
    
    tag = "🏆" if avg_r2 > best_r2 else "  "
    print(f"  {tag} {model_name:<25} MAE={avg_mae:>8.1f}  R²={avg_r2:>7.4f}  [{log_flag}, {len(features)}f]")
    
    if avg_r2 > best_r2:
        best_r2 = avg_r2
        best_name = model_name
        best_predictions = pd.DataFrame({
            'meal_id': dataset.loc[mask, 'meal_id'].values,
            'user_id': dataset.loc[mask, 'user_id'].values,
            'iauc_real': y,
            'iauc_predicted': all_preds,
            'error': all_preds - y,
            'abs_error': np.abs(all_preds - y),
        })
        
        # Feature importance del mejor modelo
        m_full = type(model)(**model.get_params())
        if use_log:
            m_full.fit(X_arr, y_train_target)
        else:
            m_full.fit(X_arr, y)
        best_fi = pd.DataFrame({
            'feature': features,
            'importance': m_full.feature_importances_
        }).sort_values('importance', ascending=False)

print(f"\n{'='*80}")
print(f"🏆 MEJOR MODELO: {best_name} con R²={best_r2:.4f}")
print(f"{'='*80}")

# ============================================================
# GUARDAR RESULTADOS
# ============================================================
print("\n💾 Guardando...")

results_df = pd.DataFrame(results)
results_df.to_csv(f'{CLEAN}/model_results_v2.csv', index=False)
print(f"  model_results_v2.csv — Comparativa completa")

if best_predictions is not None:
    best_predictions.to_csv(f'{CLEAN}/predictions_iauc_best.csv', index=False)
    print(f"  predictions_iauc_best.csv — Predicciones del mejor modelo")

if best_fi is not None:
    best_fi.to_csv(f'{CLEAN}/feature_importances_best.csv', index=False)
    print(f"  feature_importances_best.csv — Importancias del mejor modelo")

# ============================================================
# TABLA RESUMEN BONITA
# ============================================================
print("\n" + "=" * 80)
print("📊 TABLA RESUMEN — Desglose de mejoras")
print("=" * 80)

print(f"\n{'Modelo':<28} {'MAE':>8} {'R²':>8} {'Features':>10} {'Transform':>14}")
print("-" * 72)
for _, row in results_df.sort_values('R²', ascending=False).iterrows():
    print(f"  {row['model']:<26} {row['MAE']:>8.1f} {row['R²']:>8.4f} {row['features']:>10} {row['target_transform']:>14}")

# Desglosar fuentes de mejora
print(f"\n--- DESGLOSE DE MEJORAS ---")
gbr_base = results_df[results_df['model'] == 'GBR_base']['R²'].values[0]
print(f"  Baseline (GBR, features base):                R² = {gbr_base:.4f}")

if 'XGB_base' in results_df['model'].values:
    xgb_base = results_df[results_df['model'] == 'XGB_base']['R²'].values[0]
    print(f"  + Mejor algoritmo (XGB, features base):        R² = {xgb_base:.4f}  (Δ = {xgb_base - gbr_base:+.4f})")

if 'XGB_improved' in results_df['model'].values:
    xgb_imp = results_df[results_df['model'] == 'XGB_improved']['R²'].values[0]
    if 'XGB_base' in results_df['model'].values:
        print(f"  + Features interacción (XGB, improved):        R² = {xgb_imp:.4f}  (Δ = {xgb_imp - xgb_base:+.4f})")
    else:
        print(f"  + Features interacción (XGB, improved):        R² = {xgb_imp:.4f}")

if 'XGB_log_corrected' in results_df['model'].values:
    xgb_log = results_df[results_df['model'] == 'XGB_log_corrected']['R²'].values[0]
    print(f"  + Log-transform + bias corr:                   R² = {xgb_log:.4f}  (Δ = {xgb_log - xgb_imp:+.4f})")

print(f"\n  Total mejora: R² {gbr_base:.4f} → {best_r2:.4f} ({(best_r2-gbr_base)/abs(gbr_base)*100:+.1f}%)")

# Top 15 features
print(f"\n--- TOP 15 FEATURES ({best_name}) ---")
if best_fi is not None:
    for i, (_, row) in enumerate(best_fi.head(15).iterrows()):
        is_new = "⭐" if row['feature'] in interaction_features else "  "
        print(f"  {is_new} {i+1:>2}. {row['feature']:<30} {row['importance']:.4f}")

print(f"\n✅ FASE 4B COMPLETADA")
print(f"   Archivos para el notebook: model_results_v2.csv, predictions_iauc_best.csv, feature_importances_best.csv")
