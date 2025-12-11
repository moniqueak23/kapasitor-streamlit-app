import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
import os

# --- 1. CONFIG & TITLE ---
st.set_page_config(page_title="Prediksi Tegangan Kapasitor", layout="wide")
st.title("Aplikasi Prediksi Tegangan Kapasitor")
st.caption("Model: Random Forest, XGBoost & Decision Tree • Grafik Interaktif Altair")

# --- 2. LOAD ARTIFACTS (MODEL + SCALER) ---
def load_artifact(path: str):
    """Load model or scaler safely."""
    if not os.path.exists(path):
        st.error(f"File '{path}' tidak ditemukan. Pastikan file ada di folder yang sama.")
        return None
    try:
        return joblib.load(path)
    except Exception as err:
        st.error(f"Gagal memuat '{path}'. Detail: {err}")
        return None

# Load ketiga model dan scaler
rf_model = load_artifact("rf_model.pkl")
xgb_model = load_artifact("xgb_model.pkl")
dt_model  = load_artifact("dt_model.pkl")  
scaler    = load_artifact("scaler.pkl")

# Load dataset untuk visualisasi
if os.path.exists("data_kapasitor_lengkap.csv"):
    df = pd.read_csv("data_kapasitor_lengkap.csv")
    df["Mode_Charge"] = df["Mode"].apply(lambda x: 1 if x == "Charge" else 0)
    df["Mode_Discharge"] = df["Mode"].apply(lambda x: 1 if x == "Discharge" else 0)
else:
    st.warning("File 'data_kapasitor_lengkap.csv' tidak ditemukan. Tab Visualisasi mungkin kosong.")
    df = pd.DataFrame() 

# --- 3. TABS LAYOUT ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Visualisasi Data", "🔮 Prediksi Tegangan", "📈 Evaluasi Model", "ℹ️ Tentang"])

# === TAB 1 — VISUALISASI DATA (ALTAIR) ===
with tab1:
    st.header("Visualisasi Interaktif Data Eksperimen")

    if not df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("1. Waktu vs Tegangan")
            chart1 = (
                alt.Chart(df)
                .mark_line(point=True)
                .encode(
                    x="Waktu",
                    y="Tegangan",
                    color="Mode",
                    tooltip=["Waktu", "Tegangan", "Mode", "Kapasitansi"],
                )
                .interactive()
            )
            st.altair_chart(chart1, use_container_width=True)

        with col2:
            st.subheader("2. Waktu vs Arus")
            chart2 = (
                alt.Chart(df)
                .mark_line(point=True)
                .encode(
                    x="Waktu",
                    y="Arus",
                    color="Mode",
                    tooltip=["Waktu", "Arus", "Kapasitansi"],
                )
                .interactive()
            )
            st.altair_chart(chart2, use_container_width=True)

        st.subheader("3. Scatter Tegangan vs Arus")
        chart3 = (
            alt.Chart(df)
            .mark_circle(size=80)
            .encode(
                x="Tegangan",
                y="Arus",
                color="Mode",
                tooltip=["Tegangan", "Arus", "Mode"],
            )
            .interactive()
        )
        st.altair_chart(chart3, use_container_width=True)
    else:
        st.info("Data CSV tidak ditemukan, visualisasi tidak dapat ditampilkan.")

# === TAB 2 — PREDIKSI TEGANGAN ===
with tab2:
    st.header("Simulasi Prediksi Tegangan")

    col_input, col_result = st.columns([1, 1])

    with col_input:
        st.subheader("Parameter Input")
        kapasitansi = st.number_input("Kapasitansi (µF)", min_value=0.0, step=100.0, value=0.0)
        waktu = st.number_input("Waktu (ms)", min_value=0.0, step=10.0, value=0.0)
        mode = st.selectbox("Mode Operasi", ["Charge", "Discharge"])
        
        # Update Pilihan Model 
        selected_model_name = st.selectbox("Pilih Model", ["Random Forest", "XGBoost", "Decision Tree"]) 

        tombol = st.button("Hitung Prediksi 🚀")

    with col_result:
        if tombol:
            if scaler is None:
                st.error("Scaler tidak tersedia. Cek file scaler.pkl")
                st.stop()
            
            # Logika Pemilihan Model
            model = None
            if selected_model_name == "Random Forest":
                model = rf_model
            elif selected_model_name == "XGBoost":
                model = xgb_model
            elif selected_model_name == "Decision Tree": 
                model = dt_model

            if model is None:
                st.error(f"Model {selected_model_name} gagal dimuat.")
            else:
                # Normalisasi input
                try:
                    scaled = scaler.transform([[kapasitansi, waktu]])
                    cap_scaled, time_scaled = scaled[0]

                    mode_c = 1 if mode == "Charge" else 0
                    mode_d = 1 - mode_c

                    input_data = pd.DataFrame([{
                        "Kapasitansi": cap_scaled,
                        "Waktu": time_scaled,
                        "Mode_Charge": mode_c,
                        "Mode_Discharge": mode_d
                    }])

                    pred = model.predict(input_data)[0]

                    st.subheader("Hasil Prediksi")
                    st.success(f"Tegangan diprediksi:\n# **{pred:.4f} Volt**")
                    st.info(f"Menggunakan algoritma: {selected_model_name}")
                except Exception as e:
                    st.error(f"Terjadi error saat prediksi: {e}")

# === TAB 3 — EVALUASI MODEL ===
with tab3:
    st.header("Perbandingan Performa Model")
    st.write("Nilai metrik di bawah ini berdasarkan hasil training di Notebook.")

    # UPDATE DATA EVALUASI
    data_eval = {
        "Metrik": ["MSE", "RMSE", "MAE", "R²"],
        "Random Forest": [0.097662, 0.312509, 0.172165, 0.989113], 
        "XGBoost": [0.061856, 0.248709, 0.140304, 0.993104],       
        "Decision Tree": [0.195915, 0.442622, 0.200086, 0.978159]  
    }

    df_eval = pd.DataFrame(data_eval)
    
    # Highlight nilai terbaik 
    st.dataframe(df_eval, use_container_width=True)

    st.subheader("Kesimpulan Sementara")
    st.markdown("""
    - **Decision Tree** biasanya memiliki akurasi yang baik namun cenderung *overfitting* dibanding Random Forest.
    - **Random Forest** & **XGBoost** umumnya lebih stabil untuk data yang belum pernah dilihat sebelumnya.
    """)

# === TAB 4 — TENTANG ===
with tab4:
    st.header("Tentang Aplikasi")
    st.write("""
    Aplikasi ini adalah hasil revisi untuk memenuhi kebutuhan laporan magang.
    
    **Fitur Utama:**
    1.  **Multi-Model:** Mendukung Random Forest, XGBoost, dan Decision Tree.
    2.  **Visualisasi:** Grafik interaktif menggunakan Altair.
    3.  **Skalabilitas:** Input data dinormalisasi secara otomatis menggunakan MinMaxScaler.
    
    Dikembangkan oleh **Monique**.
    """)