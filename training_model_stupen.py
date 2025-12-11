import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor 
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

pd.set_option('display.max_rows', None)
df = pd.read_csv('data_kapasitor_lengkap.csv')
print(f"Total Data ditemukan: {len(df)} baris")
print("-" * 30)
print("Isi Data Lengkap:") 
print(df) 

from sklearn.preprocessing import MinMaxScaler
import pandas as pd 
import joblib 

# cleaning data, handling missing value, konversi, dan normalisasi data

df = pd.read_csv('data_kapasitor_lengkap.csv')

cek_kosong = df.isnull().sum().sum()
print(f"1. Missing value: {cek_kosong}")
df = df.dropna()
df = df.drop_duplicates()
print(f"2. Sisa data bersih: {len(df)} baris")

df['Mode_Charge'] = df['Mode'].apply(lambda x: 1 if x == 'Charge' else 0)
df['Mode_Discharge'] = df['Mode'].apply(lambda x: 1 if x == 'Discharge' else 0)
print(df[['Mode', 'Mode_Charge', 'Mode_Discharge']].to_string())

scaler = MinMaxScaler()
kolom_angka = ['Kapasitansi', 'Waktu']
df_norm = df.copy()
df_norm[kolom_angka] = scaler.fit_transform(df[kolom_angka])
print(df_norm[kolom_angka].head()) 

joblib.dump(scaler, "scaler.pkl")
print("✔ Scaler berhasil disimpan sebagai 'scaler.pkl'")

import matplotlib.pyplot as plt
import seaborn as sns

# eksplorasi dan visualisasi awal
# grafik waktu, tegangan, dan kuat arus, scatter tegangan dan kuat arus

fig, (ax_kiri, ax_tengah, ax_kanan) = plt.subplots(1, 3, figsize=(18, 5))

sns.lineplot(data=df, x='Waktu', y='Tegangan', hue='Mode', style='Kapasitansi', markers=True, ax=ax_kiri)
ax_kiri.set_title('Grafik Waktu vs Tegangan')
ax_kiri.set_xlabel('Waktu (ms)')
ax_kiri.set_ylabel('Tegangan (V)')

sns.lineplot(data=df, x='Waktu', y='Arus', hue='Mode', style='Kapasitansi', markers=True, ax=ax_tengah)
ax_tengah.set_title('Grafik Waktu vs Kuat Arus')
ax_tengah.set_xlabel('Waktu (ms)')
ax_tengah.set_ylabel('Arus (mA)')

sns.scatterplot(data=df, x='Tegangan', y='Arus', hue='Mode', ax=ax_kanan)
ax_kanan.set_title('Scatter Tegangan vs Arus')
ax_kanan.set_xlabel('Tegangan (V)')
ax_kanan.set_ylabel('Arus (mA)')

plt.tight_layout()
plt.show()

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# train split

X = df[['Kapasitansi', 'Waktu', 'Mode_Charge', 'Mode_Discharge']]
y = df['Tegangan']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("-" * 40)
print("STATISTIK PEMBAGIAN DATA:")
print("-" * 40)
print(f"Total Data Asli : {len(df)} baris (100%)")
print(f"Data Latih      : {len(X_train)} baris (80%)")
print(f"Data Uji        : {len(X_test)}  baris (20%)")
print("-" * 40)

train_data = X_train.copy()
train_data['Status'] = 'Latih (Training)'
test_data = X_test.copy()
test_data['Status'] = 'Uji (Testing)'
combined = pd.concat([train_data, test_data])

fig, (ax_kiri, ax_kanan) = plt.subplots(1, 2, figsize=(14, 5))

sns.countplot(data=combined, x='Status', palette=['blue', 'orange'], ax=ax_kiri)
ax_kiri.set_title('Perbandingan Jumlah Data Latih vs Uji')
ax_kiri.set_ylabel('Jumlah Baris Data')

sns.scatterplot(data=df, x='Waktu', y='Tegangan', hue=combined['Status'], style=combined['Status'], ax=ax_kanan)
ax_kanan.set_title('Visualisasi Sebaran Data (Random Split)')
ax_kanan.set_xlabel('Waktu (ms)')
ax_kanan.set_ylabel('Tegangan (V)')

plt.tight_layout()
plt.show() 

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import pprint

# random forest regressor
print("-" * 40)
print("random forest regressor.")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
print("pemodelan selesai")
print(f"   Bukti Model: {rf_model}")
pprint.pprint(rf_model.get_params())

# XGBoost regressor
print("-" * 40)
print("XGBoost regressor.")
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)
print("pemodelan selesai")
print(f"   Bukti Model: {xgb_model}")
pprint.pprint(xgb_model.get_params())
print("-" * 40) 

# Decision Tree Regressor
print("-" * 40)
print("Decision Tree Regressor.")
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
print("pemodelan selesai")
print(f"   Bukti Model: {dt_model}")
pprint.pprint(dt_model.get_params())
print("-" * 40)

from sklearn.model_selection import GridSearchCV

#tuning random forest
print("-" * 40)
print("1. mencari parameter terbaik random forest")

param_rf = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'random_state': [42]
}

rf_grid = GridSearchCV(RandomForestRegressor(), param_rf, cv=3, n_jobs=-1)
rf_grid.fit(X_train, y_train)

rf_best = rf_grid.best_estimator_

print("  tuning random forest selesai!")
print("   parameter terbaik :", rf_grid.best_params_)
print(f"   skor akurasi (R²)    : {rf_grid.best_score_:.4f}")
print("-" * 40)

#tuning XGBoost
print("-" * 40)
print("2. mencari parameter XGBoost terbaik")

param_xgb = {
    'learning_rate': [0.01, 0.1],
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'subsample': [0.8, 1.0]
}

xgb_grid = GridSearchCV(XGBRegressor(random_state=42), param_xgb, cv=3, n_jobs=-1)
xgb_grid.fit(X_train, y_train)

xgb_best = xgb_grid.best_estimator_

print("  tuning XGBoost selesai!")
print("   parameter terbaik XGBoost :", xgb_grid.best_params_)
print(f"   skor akurasi (R²)     : {xgb_grid.best_score_:.4f}")
print("-" * 40)

#tuning Decision Tree
print("-" * 40)
print("3. mencari parameter Decision Tree terbaik")

param_dt = {
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'random_state': [42]
}

dt_grid = GridSearchCV(DecisionTreeRegressor(), param_dt, cv=3, n_jobs=-1)
dt_grid.fit(X_train, y_train)

dt_best = dt_grid.best_estimator_

print("   tuning Decision Tree selesai!")
print("   parameter terbaik Decision Tree :", dt_grid.best_params_)
print(f"   skor akurasi (R²)     : {dt_grid.best_score_:.4f}")
print("-" * 40)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# evaluasi MSE, RMSE, MAE, dan R²

# 1. Hitung Prediksi untuk SEMUA model
pred_rf = rf_best.predict(X_test)
pred_xgb = xgb_best.predict(X_test)
pred_dt = dt_best.predict(X_test) 

def hitung_skor(nama, y_asli, y_prediksi):
    mse = mean_squared_error(y_asli, y_prediksi)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_asli, y_prediksi)
    r2 = r2_score(y_asli, y_prediksi)
    return mse, rmse, mae, r2

# 2. Hitung Skor masing-masing
mse_rf, rmse_rf, mae_rf, r2_rf = hitung_skor("Random Forest", y_test, pred_rf)
mse_xgb, rmse_xgb, mae_xgb, r2_xgb = hitung_skor("XGBoost", y_test, pred_xgb)
mse_dt, rmse_dt, mae_dt, r2_dt = hitung_skor("Decision Tree", y_test, pred_dt)

# 3. Print Tabel Perbandingan 
print("\n" + "="*95)
print(f"{'METRIK':<10} | {'RANDOM FOREST':<20} | {'XGBOOST':<20} | {'DECISION TREE':<20}")
print("="*95)
print(f"{'MSE':<10} | {mse_rf:<20.6f} | {mse_xgb:<20.6f} | {mse_dt:<20.6f}")
print(f"{'RMSE':<10} | {rmse_rf:<20.6f} | {rmse_xgb:<20.6f} | {rmse_dt:<20.6f}")
print(f"{'MAE':<10} | {mae_rf:<20.6f} | {mae_xgb:<20.6f} | {mae_dt:<20.6f}")
print(f"{'R²':<10} | {r2_rf:<20.6f} | {r2_xgb:<20.6f} | {r2_dt:<20.6f}")
print("="*95)
print("*Semakin kecil Error (MSE/RMSE/MAE) semakin baik.")
print("*Semakin besar R² (mendekati 1.0) semakin akurat.")

# 4. Visualisasi 
fig, ax = plt.subplots(2, 3, figsize=(18, 10))

# --- BARIS 1: Scatter Plot (Prediksi vs Aktual) ---

# Random Forest
ax[0, 0].scatter(y_test, pred_rf, color='green', alpha=0.5)
ax[0, 0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
ax[0, 0].set_title(f'Random Forest (R²={r2_rf:.3f})')
ax[0, 0].set_ylabel('Nilai Prediksi')
ax[0, 0].set_xlabel('Nilai Aktual')

# XGBoost
ax[0, 1].scatter(y_test, pred_xgb, color='purple', alpha=0.5)
ax[0, 1].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
ax[0, 1].set_title(f'XGBoost (R²={r2_xgb:.3f})')
ax[0, 1].set_xlabel('Nilai Aktual')

# Decision Tree 
ax[0, 2].scatter(y_test, pred_dt, color='blue', alpha=0.5)
ax[0, 2].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
ax[0, 2].set_title(f'Decision Tree (R²={r2_dt:.3f})')
ax[0, 2].set_xlabel('Nilai Aktual')


# --- BARIS 2: Distribusi Error ---

# Error RF
sns.histplot(y_test - pred_rf, kde=True, color='green', ax=ax[1, 0])
ax[1, 0].set_title('Error: Random Forest')
ax[1, 0].set_xlabel('Selisih Error')

# Error XGB
sns.histplot(y_test - pred_xgb, kde=True, color='purple', ax=ax[1, 1])
ax[1, 1].set_title('Error: XGBoost')
ax[1, 1].set_xlabel('Selisih Error')

# Error DT 
sns.histplot(y_test - pred_dt, kde=True, color='blue', ax=ax[1, 2])
ax[1, 2].set_title('Error: Decision Tree')
ax[1, 2].set_xlabel('Selisih Error')

plt.tight_layout()
plt.show()

import joblib
import os

# Simpan ketiga model
joblib.dump(rf_best, 'rf_model.pkl')
joblib.dump(xgb_best, 'xgb_model.pkl')
joblib.dump(dt_best, 'dt_model.pkl')  # <--- Tambahan

print("BUKTI FILE BERHASIL DISIMPAN:")
print("-" * 50)

# Cek File Random Forest
if os.path.exists('rf_model.pkl'):
    size_rf = os.path.getsize('rf_model.pkl') / 1024
    print(f"   DITEMUKAN: 'rf_model.pkl'")
    print(f"   Ukuran File : {size_rf:.2f} KB")
else:
    print("GAGAL: File rf_model.pkl tidak ditemukan!")

print("." * 30)

# Cek File XGBoost
if os.path.exists('xgb_model.pkl'):
    size_xgb = os.path.getsize('xgb_model.pkl') / 1024
    print(f"   DITEMUKAN: 'xgb_model.pkl'")
    print(f"   Ukuran File : {size_xgb:.2f} KB")
else:
    print("GAGAL: File xgb_model.pkl tidak ditemukan!")

print("." * 30)

# Cek File Decision Tree (Tambahan)
if os.path.exists('dt_model.pkl'):
    size_dt = os.path.getsize('dt_model.pkl') / 1024
    print(f"   DITEMUKAN: 'dt_model.pkl'")
    print(f"   Ukuran File : {size_dt:.2f} KB")
    print(f"   Lokasi      : {os.getcwd()}")
else:
    print("GAGAL: File dt_model.pkl tidak ditemukan!")

print("-" * 50)
print("SIAP LANJUT KE STREAMLIT!")