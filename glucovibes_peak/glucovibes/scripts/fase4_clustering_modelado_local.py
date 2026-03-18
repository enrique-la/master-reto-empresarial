"""
FASE 4 — CLUSTERING + MODELADO PREDICTIVO (CON MEJORAS)
Proyecto Glucovibes Challenge 2026

EJECUCIÓN:
  1. Asegúrate de que ./clean_data/modeling_dataset.csv existe (salida de fase3)
  2. pip install xgboost lightgbm scikit-learn pandas numpy
  3. python fase4_clustering_modelado_local.py
  4. Genera múltiples CSV en ./clean_data/

CONTENIDO:
  - Clustering nutricional (KMeans K=3)
  - Clustering glucémico (KMeans K=2)
  - Modelado original (Ridge, RF, GBR) con GroupKFold
  - Modelado mejorado (XGBoost, LightGBM, log-transform, features interacción)
  - Feature importances
  - Predicciones
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

# Intentar importar modelos avanzados
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
    print("✅ XGBoost disponible")
except ImportError:
    HAS_XGB = False
    print("⚠️ XGBoost no instalado (pip install xgboost)")

try:
    from lightgbm import LGBMRegressor
    HAS_LGB = True
    print("✅ LightGBM disponible")
except ImportError:
    HAS_LGB = False
    print("⚠️ LightGBM no instalado (pip install lightgbm)")

CLEAN = './clean_data'

print("=" * 70)
print("FASE 4 — CLUSTERING + MODELADO PREDICTIVO")
print("=" * 70)

# ============================================================
# CARGAR DATOS
# ============================================================
print("\n📦 Cargando modeling_dataset...")
dataset = pd.read_csv(f'{CLEAN}/modeling_dataset.csv')
dataset['meal_timestamp'] = pd.to_datetime(dataset['meal_timestamp'], format='mixed', utc=True)
print(f"  Dataset: {len(dataset):,} filas × {dataset.shape[1]} columnas")
print(f"  Usuarios: {dataset['user_id'].nunique()}")

# ============================================================
# PARTE A: CLUSTERING
# ============================================================
print("\n" + "=" * 70)
print("PARTE A — CLUSTERING DE USUARIOS")
print("=" * 70)

# -------------------------------------------------------
# A1. Perfiles nutricionales por usuario
# -------------------------------------------------------
print("\n🥗 A1: Construyendo perfiles nutricionales...")

user_nutri = dataset.groupby('user_id').agg(
    avg_calories=('total_calories', 'mean'),
    avg_protein=('total_protein', 'mean'),
    avg_fat=('total_fat', 'mean'),
    avg_carbs=('total_carbs', 'mean'),
    avg_fibre=('total_fibre', 'mean'),
    avg_pct_carbs=('pct_cal_carbs', 'mean'),
    avg_pct_protein=('pct_cal_protein', 'mean'),
    avg_pct_fat=('pct_cal_fat', 'mean'),
    avg_fibre_carb_ratio=('fibre_carb_ratio', 'mean'),
    avg_ultraprocessed=('n_ultraprocessed', 'mean'),
    avg_food_groups=('n_food_groups', 'mean'),
    n_meals=('meal_id', 'count'),
    avg_items_per_meal=('n_items', 'mean'),
).reset_index()

# Calorías diarias estimadas
user_dates = dataset.groupby('user_id').agg(
    total_calories_all=('total_calories', 'sum'),
    n_days=('meal_timestamp', lambda x: (x.max() - x.min()).days + 1),
    total_fibre_all=('total_fibre', 'sum'),
).reset_index()
user_dates['est_daily_calories'] = user_dates['total_calories_all'] / user_dates['n_days'].replace(0, np.nan)
user_dates['est_daily_fibre'] = user_dates['total_fibre_all'] / user_dates['n_days'].replace(0, np.nan)

user_nutri = user_nutri.merge(user_dates, on='user_id')
print(f"  Perfiles nutricionales: {len(user_nutri)} usuarios")

# -------------------------------------------------------
# A2. Clustering nutricional
# -------------------------------------------------------
print("\n📊 A2: Clustering nutricional...")

nutri_features = ['avg_pct_carbs', 'avg_pct_protein', 'avg_pct_fat',
                   'avg_fibre_carb_ratio', 'avg_ultraprocessed', 'avg_food_groups',
                   'est_daily_calories', 'avg_items_per_meal']

X_nutri = user_nutri[nutri_features].fillna(0)
scaler_nutri = StandardScaler()
X_nutri_scaled = scaler_nutri.fit_transform(X_nutri)

# Silhouette scores para K=2..7
silhouette_results = []
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_nutri_scaled)
    sil = silhouette_score(X_nutri_scaled, labels)
    silhouette_results.append({'k': k, 'silhouette_nutri': sil})
    print(f"  K={k}: silhouette={sil:.3f}")

# K=3 para nutricional
km_nutri = KMeans(n_clusters=3, random_state=42, n_init=10)
user_nutri['nutri_cluster'] = km_nutri.fit_predict(X_nutri_scaled)

# Nombrar clusters según calorías medias
cluster_cal_means = user_nutri.groupby('nutri_cluster')['est_daily_calories'].mean().sort_values()
nutri_names = {}
for i, (cluster_id, _) in enumerate(cluster_cal_means.items()):
    if i == 0:
        nutri_names[cluster_id] = 'Bajo registro'
    elif i == 1:
        nutri_names[cluster_id] = 'Moderado'
    else:
        nutri_names[cluster_id] = 'Alto registro'

user_nutri['nutri_cluster_name'] = user_nutri['nutri_cluster'].map(nutri_names)
print(f"\n  Clusters nutricionales:")
for name, count in user_nutri['nutri_cluster_name'].value_counts().items():
    print(f"    {name}: {count} usuarios")

# -------------------------------------------------------
# A3. Perfiles glucémicos por usuario
# -------------------------------------------------------
print("\n📈 A3: Construyendo perfiles glucémicos...")

user_glyc = dataset.groupby('user_id').agg(
    avg_iauc=('iauc', 'mean'),
    avg_amplitude=('amplitude', 'mean'),
    avg_peak=('peak_value', 'mean'),
    avg_time_to_peak=('time_to_peak_min', 'mean'),
    avg_cv_post=('cv_postprandial', 'mean'),
).reset_index()

# Stats desde glucose_clean
glucose_clean = pd.read_csv(f'{CLEAN}/glucose_clean.csv')
user_gluc_stats = glucose_clean.groupby('user_id')['value_decimal'].agg(
    time_in_range=lambda x: ((x >= 70) & (x <= 140)).mean() * 100,
    glucose_mean='mean',
    glucose_std='std',
).reset_index()

# Picos altos (>160 mg/dL)
high_spikes = dataset.groupby('user_id').apply(
    lambda x: (x['peak_value'] > 160).mean() * 100
).reset_index()
high_spikes.columns = ['user_id', 'pct_high_spikes']

# Consistencia de respuesta
response_consistency = dataset.groupby('user_id')['iauc'].std().reset_index()
response_consistency.columns = ['user_id', 'response_consistency']

# Glucosa preprandial media
avg_preprandial = dataset.groupby('user_id')['glucose_preprandial'].mean().reset_index()
avg_preprandial.columns = ['user_id', 'avg_preprandial']

user_glyc = user_glyc.merge(user_gluc_stats, on='user_id')
user_glyc = user_glyc.merge(high_spikes, on='user_id')
user_glyc = user_glyc.merge(response_consistency, on='user_id')
user_glyc = user_glyc.merge(avg_preprandial, on='user_id')

# -------------------------------------------------------
# A4. Clustering glucémico
# -------------------------------------------------------
print("\n📊 A4: Clustering glucémico...")

glyc_features = ['avg_iauc', 'avg_amplitude', 'avg_peak', 'time_in_range',
                  'glucose_std', 'pct_high_spikes', 'avg_preprandial']

X_glyc = user_glyc[glyc_features].fillna(0)
scaler_glyc = StandardScaler()
X_glyc_scaled = scaler_glyc.fit_transform(X_glyc)

for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_glyc_scaled)
    sil = silhouette_score(X_glyc_scaled, labels)
    # Añadir al mismo silhouette_results
    for r in silhouette_results:
        if r['k'] == k:
            r['silhouette_glyc'] = sil
    print(f"  K={k}: silhouette={sil:.3f}")

# K=2 para glucémico (mejor silhouette)
km_glyc = KMeans(n_clusters=2, random_state=42, n_init=10)
user_glyc['glyc_cluster'] = km_glyc.fit_predict(X_glyc_scaled)

# Nombrar clusters
cluster_tir_means = user_glyc.groupby('glyc_cluster')['time_in_range'].mean().sort_values()
glyc_names = {}
for i, (cluster_id, tir_val) in enumerate(cluster_tir_means.items()):
    if i == 0:
        glyc_names[cluster_id] = 'Reactivo-Alto'
    else:
        glyc_names[cluster_id] = 'Estable'

user_glyc['glyc_cluster_name'] = user_glyc['glyc_cluster'].map(glyc_names)
print(f"\n  Clusters glucémicos:")
for name, count in user_glyc['glyc_cluster_name'].value_counts().items():
    avg_tir = user_glyc[user_glyc['glyc_cluster_name'] == name]['time_in_range'].mean()
    print(f"    {name}: {count} usuarios (TIR medio: {avg_tir:.1f}%)")

# -------------------------------------------------------
# A5. Guardar silhouette scores
# -------------------------------------------------------
sil_df = pd.DataFrame(silhouette_results)
sil_df.to_csv(f'{CLEAN}/silhouette_scores.csv', index=False)
print(f"\n  Silhouette scores guardados")

# -------------------------------------------------------
# A6. Merge perfiles y guardar
# -------------------------------------------------------
print("\n💾 Guardando perfiles de usuario...")
user_profiles = user_nutri.merge(
    user_glyc[['user_id', 'glyc_cluster', 'glyc_cluster_name',
               'avg_iauc', 'avg_amplitude', 'avg_peak', 'avg_time_to_peak',
               'avg_cv_post', 'time_in_range', 'glucose_mean', 'glucose_std',
               'pct_high_spikes', 'response_consistency', 'avg_preprandial']],
    on='user_id', how='left'
)
user_profiles.to_csv(f'{CLEAN}/user_profiles_clustered.csv', index=False)

# Añadir clusters al dataset de modelado
cluster_map_nutri = user_nutri.set_index('user_id')[['nutri_cluster', 'nutri_cluster_name']]
cluster_map_glyc = user_glyc.set_index('user_id')[['glyc_cluster', 'glyc_cluster_name']]

dataset = dataset.merge(cluster_map_nutri, on='user_id', how='left')
dataset = dataset.merge(cluster_map_glyc, on='user_id', how='left')
dataset.to_csv(f'{CLEAN}/modeling_dataset_clustered.csv', index=False)
print(f"  modeling_dataset_clustered.csv guardado ({len(dataset):,} × {dataset.shape[1]})")

# ============================================================
# PARTE B: MODELADO PREDICTIVO ORIGINAL
# ============================================================
print("\n" + "=" * 70)
print("PARTE B — MODELADO PREDICTIVO ORIGINAL")
print("=" * 70)

# -------------------------------------------------------
# B1. Preparar features
# -------------------------------------------------------
print("\n🔧 B1: Preparando features para modelado...")

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

categorical_features = ['meal_period', 'meal_type', 'sport_prior_intensity']

# Encoding categóricas
label_encoders = {}
for col in categorical_features:
    if col in dataset.columns:
        le = LabelEncoder()
        dataset[f'{col}_enc'] = le.fit_transform(dataset[col].fillna('unknown').astype(str))
        label_encoders[col] = le
        numeric_features.append(f'{col}_enc')

# Features disponibles
available_features = [f for f in numeric_features if f in dataset.columns]
print(f"  Features disponibles: {len(available_features)}")

# -------------------------------------------------------
# B2. Modelado con GroupKFold
# -------------------------------------------------------
targets = ['iauc', 'amplitude', 'peak_value']
models_config = {
    'Ridge': Ridge(alpha=1.0),
    'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
}

all_results = []
all_importances = {}
all_predictions = {}

gkf = GroupKFold(n_splits=5)
groups = dataset['user_id'].values

for target in targets:
    print(f"\n📊 Modelando: {target}")
    
    # Preparar X, y
    mask = dataset[target].notna()
    X = dataset.loc[mask, available_features].copy()
    y = dataset.loc[mask, target].values
    g = groups[mask]
    
    # Imputar NaN en features con mediana
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())
    
    X_arr = X.values
    
    for model_name, model in models_config.items():
        fold_mae, fold_rmse, fold_r2 = [], [], []
        fold_predictions = np.full(len(y), np.nan)
        
        for fold, (train_idx, test_idx) in enumerate(gkf.split(X_arr, y, g)):
            X_train, X_test = X_arr[train_idx], X_arr[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model_clone = type(model)(**model.get_params())
            model_clone.fit(X_train, y_train)
            y_pred = model_clone.predict(X_test)
            
            fold_mae.append(mean_absolute_error(y_test, y_pred))
            fold_rmse.append(np.sqrt(mean_squared_error(y_test, y_pred)))
            fold_r2.append(r2_score(y_test, y_pred))
            fold_predictions[test_idx] = y_pred
        
        avg_mae = np.mean(fold_mae)
        avg_rmse = np.mean(fold_rmse)
        avg_r2 = np.mean(fold_r2)
        
        # MAPE (evitar div/0)
        valid_mask = y != 0
        mape = np.mean(np.abs((y[valid_mask] - fold_predictions[valid_mask]) / y[valid_mask])) * 100 if valid_mask.any() else np.nan
        
        all_results.append({
            'target': target, 'model': model_name,
            'MAE': round(avg_mae, 2), 'RMSE': round(avg_rmse, 2),
            'R²': round(avg_r2, 4), 'MAPE': round(mape, 2)
        })
        
        print(f"  {model_name}: MAE={avg_mae:.1f}, R²={avg_r2:.4f}")
        
        # Feature importance (para el mejor modelo por target)
        if model_name == 'GradientBoosting' and target == 'iauc':
            # Entrenar en todos los datos para importances
            model_full = type(model)(**model.get_params())
            model_full.fit(X_arr, y)
            fi = pd.DataFrame({
                'feature': available_features,
                'importance': model_full.feature_importances_
            }).sort_values('importance', ascending=False)
            all_importances['original'] = fi
        
        # Guardar predicciones para iauc
        if target == 'iauc' and model_name == 'GradientBoosting':
            pred_df = dataset.loc[mask, ['meal_id', 'user_id']].copy()
            pred_df['iauc_real'] = y
            pred_df['iauc_predicted'] = fold_predictions
            pred_df['error'] = fold_predictions - y
            pred_df['abs_error'] = np.abs(pred_df['error'])
            all_predictions['original'] = pred_df

# Guardar resultados originales
results_df = pd.DataFrame(all_results)
results_df.to_csv(f'{CLEAN}/model_results.csv', index=False)
print(f"\n  Resultados guardados: model_results.csv")

if 'original' in all_importances:
    all_importances['original'].to_csv(f'{CLEAN}/feature_importances.csv', index=False)
    print(f"  Feature importances guardadas")

if 'original' in all_predictions:
    all_predictions['original'].to_csv(f'{CLEAN}/predictions_iauc.csv', index=False)
    print(f"  Predicciones iAUC guardadas")

# ============================================================
# PARTE C: MODELADO MEJORADO
# ============================================================
print("\n" + "=" * 70)
print("PARTE C — MODELADO MEJORADO (XGBoost, LightGBM, Log-transform)")
print("=" * 70)

# -------------------------------------------------------
# C1. Features de interacción
# -------------------------------------------------------
print("\n🔬 C1: Creando features de interacción...")

dataset['net_carbs'] = dataset['total_carbs'] - dataset['total_fibre'].fillna(0)
dataset['carbs_x_hour'] = dataset['total_carbs'] * dataset['hour_of_day']
dataset['carbs_x_preprandial'] = dataset['total_carbs'] * dataset['glucose_preprandial'].fillna(dataset['glucose_preprandial'].median())
dataset['fat_protein_ratio'] = dataset['total_fat'] / dataset['total_protein'].replace(0, np.nan)
dataset['fat_protein_ratio'] = dataset['fat_protein_ratio'].fillna(dataset['fat_protein_ratio'].median())

# Clusters como features numéricas
dataset['nutri_cluster_num'] = dataset['nutri_cluster'].fillna(0).astype(int)
dataset['glyc_cluster_num'] = dataset['glyc_cluster'].fillna(0).astype(int)

new_features = ['net_carbs', 'carbs_x_hour', 'carbs_x_preprandial',
                'fat_protein_ratio', 'nutri_cluster_num', 'glyc_cluster_num']
improved_features = available_features + new_features

print(f"  Features totales (mejoradas): {len(improved_features)}")

# -------------------------------------------------------
# C2. Log-transform del target
# -------------------------------------------------------
print("\n📐 C2: Log-transformación de iAUC...")
dataset['log_iauc'] = np.log1p(dataset['iauc'].clip(lower=0))
print(f"  iAUC original: media={dataset['iauc'].mean():.1f}, mediana={dataset['iauc'].median():.1f}")
print(f"  log_iauc:       media={dataset['log_iauc'].mean():.2f}, mediana={dataset['log_iauc'].median():.2f}")

# -------------------------------------------------------
# C3. Modelos mejorados
# -------------------------------------------------------
print("\n🚀 C3: Entrenando modelos mejorados...")

improved_models = {
    'GBR_improved': GradientBoostingRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=5, random_state=42
    ),
}

if HAS_XGB:
    improved_models['XGBoost'] = XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbosity=0
    )

if HAS_LGB:
    improved_models['LightGBM'] = LGBMRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
        random_state=42, n_jobs=-1, verbose=-1
    )

improved_results = []

# Target: log_iauc (evaluamos en escala original)
target = 'log_iauc'
mask = dataset[target].notna() & np.isfinite(dataset[target])
X_imp = dataset.loc[mask, improved_features].copy()
y_log = dataset.loc[mask, target].values
y_orig = dataset.loc[mask, 'iauc'].values
g_imp = dataset.loc[mask, 'user_id'].values

for col in X_imp.columns:
    if X_imp[col].isna().any():
        X_imp[col] = X_imp[col].fillna(X_imp[col].median())

X_imp_arr = X_imp.values

for model_name, model in improved_models.items():
    fold_mae, fold_rmse, fold_r2, fold_r2_log = [], [], [], []
    fold_predictions_orig = np.full(len(y_orig), np.nan)
    fold_predictions_log = np.full(len(y_log), np.nan)
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_imp_arr, y_log, g_imp)):
        X_train, X_test = X_imp_arr[train_idx], X_imp_arr[test_idx]
        y_train_log, y_test_log = y_log[train_idx], y_log[test_idx]
        y_test_orig_fold = y_orig[test_idx]
        
        model_clone = type(model)(**model.get_params())
        model_clone.fit(X_train, y_train_log)
        
        y_pred_log = model_clone.predict(X_test)
        y_pred_orig = np.expm1(y_pred_log)  # Inversa de log1p
        y_pred_orig = np.maximum(y_pred_orig, 0)  # No negativos
        
        fold_mae.append(mean_absolute_error(y_test_orig_fold, y_pred_orig))
        fold_rmse.append(np.sqrt(mean_squared_error(y_test_orig_fold, y_pred_orig)))
        fold_r2.append(r2_score(y_test_orig_fold, y_pred_orig))
        fold_r2_log.append(r2_score(y_test_log, y_pred_log))
        
        fold_predictions_orig[test_idx] = y_pred_orig
        fold_predictions_log[test_idx] = y_pred_log
    
    avg_mae = np.mean(fold_mae)
    avg_rmse = np.mean(fold_rmse)
    avg_r2 = np.mean(fold_r2)
    avg_r2_log = np.mean(fold_r2_log)
    
    valid_mask = y_orig != 0
    mape = np.mean(np.abs((y_orig[valid_mask] - fold_predictions_orig[valid_mask]) / y_orig[valid_mask])) * 100
    
    improved_results.append({
        'target': 'iauc', 'model': model_name,
        'MAE': round(avg_mae, 2), 'RMSE': round(avg_rmse, 2),
        'R²': round(avg_r2, 4), 'R²_log': round(avg_r2_log, 4),
        'MAPE': round(mape, 2), 'type': 'improved'
    })
    
    print(f"  {model_name}: MAE={avg_mae:.1f}, R²={avg_r2:.4f}, R²_log={avg_r2_log:.4f}")
    
    # Feature importance para el mejor modelo
    model_full = type(model)(**model.get_params())
    model_full.fit(X_imp_arr, y_log)
    fi_imp = pd.DataFrame({
        'feature': improved_features,
        'importance': model_full.feature_importances_
    }).sort_values('importance', ascending=False)
    all_importances[model_name] = fi_imp
    
    # Predicciones
    pred_imp_df = dataset.loc[mask, ['meal_id', 'user_id']].copy()
    pred_imp_df['iauc_real'] = y_orig
    pred_imp_df['iauc_predicted'] = fold_predictions_orig
    pred_imp_df['error'] = fold_predictions_orig - y_orig
    pred_imp_df['abs_error'] = np.abs(pred_imp_df['error'])
    all_predictions[model_name] = pred_imp_df

# También entrenar modelos mejorados para amplitude y peak_value
print("\n  Modelando amplitude y peak_value con modelos mejorados...")

for target in ['amplitude', 'peak_value']:
    mask_t = dataset[target].notna()
    X_t = dataset.loc[mask_t, improved_features].copy()
    y_t = dataset.loc[mask_t, target].values
    g_t = dataset.loc[mask_t, 'user_id'].values
    
    for col in X_t.columns:
        if X_t[col].isna().any():
            X_t[col] = X_t[col].fillna(X_t[col].median())
    
    X_t_arr = X_t.values
    
    # Solo el mejor modelo mejorado
    best_model_name = 'XGBoost' if HAS_XGB else ('LightGBM' if HAS_LGB else 'GBR_improved')
    best_model = improved_models[best_model_name]
    
    fold_mae, fold_rmse, fold_r2 = [], [], []
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_t_arr, y_t, g_t)):
        model_clone = type(best_model)(**best_model.get_params())
        model_clone.fit(X_t_arr[train_idx], y_t[train_idx])
        y_pred = model_clone.predict(X_t_arr[test_idx])
        fold_mae.append(mean_absolute_error(y_t[test_idx], y_pred))
        fold_rmse.append(np.sqrt(mean_squared_error(y_t[test_idx], y_pred)))
        fold_r2.append(r2_score(y_t[test_idx], y_pred))
    
    valid_mask = y_t != 0
    improved_results.append({
        'target': target, 'model': best_model_name,
        'MAE': round(np.mean(fold_mae), 2), 'RMSE': round(np.mean(fold_rmse), 2),
        'R²': round(np.mean(fold_r2), 4), 'R²_log': np.nan,
        'MAPE': np.nan, 'type': 'improved'
    })
    print(f"  {target} ({best_model_name}): MAE={np.mean(fold_mae):.1f}, R²={np.mean(fold_r2):.4f}")

# -------------------------------------------------------
# C4. Guardar resultados mejorados
# -------------------------------------------------------
print("\n💾 Guardando resultados mejorados...")

improved_results_df = pd.DataFrame(improved_results)
improved_results_df.to_csv(f'{CLEAN}/model_results_improved.csv', index=False)

# Guardar feature importances mejoradas
for name, fi in all_importances.items():
    if name != 'original':
        fi.to_csv(f'{CLEAN}/feature_importances_{name}.csv', index=False)
        print(f"  feature_importances_{name}.csv")

# Guardar predicciones mejoradas
for name, pred in all_predictions.items():
    if name != 'original':
        pred.to_csv(f'{CLEAN}/predictions_iauc_{name}.csv', index=False)
        print(f"  predictions_iauc_{name}.csv")

# Guardar dataset con features de interacción
dataset.to_csv(f'{CLEAN}/modeling_dataset_final.csv', index=False)
print(f"  modeling_dataset_final.csv ({len(dataset):,} × {dataset.shape[1]})")

# ============================================================
# RESUMEN FINAL
# ============================================================
print("\n" + "=" * 70)
print("📊 RESUMEN FINAL")
print("=" * 70)

print("\n--- MODELADO ORIGINAL (iAUC) ---")
orig_iauc = results_df[results_df['target'] == 'iauc']
for _, row in orig_iauc.iterrows():
    print(f"  {row['model']}: MAE={row['MAE']}, R²={row['R²']}")

print("\n--- MODELADO MEJORADO (iAUC) ---")
imp_iauc = improved_results_df[improved_results_df['target'] == 'iauc']
for _, row in imp_iauc.iterrows():
    r2_log_str = f", R²_log={row['R²_log']}" if pd.notna(row.get('R²_log')) else ""
    print(f"  {row['model']}: MAE={row['MAE']}, R²={row['R²']}{r2_log_str}")

# Mejor original vs mejor mejorado
best_orig_r2 = orig_iauc['R²'].max()
best_imp_r2 = imp_iauc['R²'].max()
improvement = ((best_imp_r2 - best_orig_r2) / abs(best_orig_r2)) * 100 if best_orig_r2 != 0 else 0

print(f"\n  🎯 Mejor original: R²={best_orig_r2:.4f}")
print(f"  🚀 Mejor mejorado: R²={best_imp_r2:.4f}")
print(f"  📈 Mejora: {improvement:+.1f}%")

print("\n--- CLUSTERS ---")
print(f"  Nutricionales: {user_nutri['nutri_cluster_name'].value_counts().to_dict()}")
print(f"  Glucémicos: {user_glyc['glyc_cluster_name'].value_counts().to_dict()}")

print("\n--- ARCHIVOS GENERADOS ---")
for f in ['silhouette_scores.csv', 'user_profiles_clustered.csv',
          'modeling_dataset_clustered.csv', 'model_results.csv',
          'feature_importances.csv', 'predictions_iauc.csv',
          'model_results_improved.csv', 'modeling_dataset_final.csv']:
    print(f"  ✅ {CLEAN}/{f}")

print("\n✅ FASE 4 COMPLETADA — Listo para generar notebook")
