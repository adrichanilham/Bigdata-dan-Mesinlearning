import streamlit as st
import pickle
import pandas as pd
import numpy as np

# --- 1. Memuat Model dan Tools ---
try:
    # Muat Model
    with open('decision_tree_model.pkl', 'rb') as file:
        model = pickle.load(file)

    # Muat Label Encoder
    with open('label_encoder.pkl', 'rb') as file:
        le = pickle.load(file)
        # Pastikan kelas target sesuai
        target_names = le.classes_

    # Muat Daftar Fitur
    with open('feature_names.pkl', 'rb') as file:
        feature_names = pickle.load(file)

except FileNotFoundError:
    st.error("File model atau encoder tidak ditemukan. Pastikan Anda sudah menjalankan Langkah 1.")
    st.stop()


# --- 2. Konfigurasi Mapping Data Input ---
# Mapping yang sama seperti pada tahap preprocessing
NILAI_RAPOR_MAP = {'Rendah': 1, 'Sedang': 2, 'Tinggi': 3}
WAWANCARA_MAP = {'Kurang': 1, 'Cukup': 2, 'Baik': 3}
PRESTASI_MAP = {'Ada': 1, 'Tidak Ada': 0}

# Ambil pilihan unik untuk dropdown dari nama fitur yang di One-Hot
def get_one_hot_options(prefix):
    options = [f.replace(prefix + '_', '') for f in feature_names if f.startswith(prefix)]
    return sorted(options)

J_RUMAH_OPTIONS = get_one_hot_options('Jarak_Rumah')
S_EKONOMI_OPTIONS = get_one_hot_options('Status_Ekonomi')
J_SAUDARA_OPTIONS = get_one_hot_options('Jumlah_Saudara')


# --- 3. Fungsi Prediksi ---
def predict_keputusan(data_input):
    # Buat dataframe dari input
    input_df = pd.DataFrame(data_input, index=[0])

    # 1. Encoding Ordinal/Biner
    input_df['Nilai_Rapor_Calistung_Encoded'] = input_df['Nilai_Rapor_Calistung'].map(NILAI_RAPOR_MAP)
    input_df['Hasil_Wawancara_Encoded'] = input_df['Hasil_Wawancara'].map(WAWANCARA_MAP)
    input_df['Prestasi_Non_Akademik_Encoded'] = input_df['Prestasi_Non_Akademik'].map(PRESTASI_MAP)

    # 2. One-Hot Encoding (Manual untuk memastikan urutan kolom)
    final_input = {}

    # Ambil nilai encoded
    final_input['Nilai_Rapor_Calistung_Encoded'] = input_df['Nilai_Rapor_Calistung_Encoded'].iloc[0]
    final_input['Hasil_Wawancara_Encoded'] = input_df['Hasil_Wawancara_Encoded'].iloc[0]
    final_input['Prestasi_Non_Akademik_Encoded'] = input_df['Prestasi_Non_Akademik_Encoded'].iloc[0]

    # One-Hot Encoding untuk Jarak_Rumah
    for opt in J_RUMAH_OPTIONS:
        col_name = f'Jarak_Rumah_{opt}'
        final_input[col_name] = 1 if input_df['Jarak_Rumah'].iloc[0] == opt else 0

    # One-Hot Encoding untuk Status_Ekonomi
    for opt in S_EKONOMI_OPTIONS:
        col_name = f'Status_Ekonomi_{opt}'
        final_input[col_name] = 1 if input_df['Status_Ekonomi'].iloc[0] == opt else 0

    # One-Hot Encoding untuk Jumlah_Saudara
    for opt in J_SAUDARA_OPTIONS:
        col_name = f'Jumlah_Saudara_{opt}'
        final_input[col_name] = 1 if input_df['Jumlah_Saudara'].iloc[0] == opt else 0

    # Ubah ke DataFrame dengan urutan kolom yang benar (sesuai feature_names)
    final_df = pd.DataFrame([final_input], columns=feature_names)

    # Lakukan prediksi
    prediction_encoded = model.predict(final_df)[0]
    prediction_proba = model.predict_proba(final_df)

    # Decode hasil
    result = target_names[prediction_encoded]

    return result, prediction_proba


# --- 4. Tampilan Streamlit ---
st.set_page_config(page_title="Prediksi Penerimaan Siswa MIN 4 Padang", layout="wide")
st.title("🌳 Decision Tree: Prediksi Penerimaan Siswa MIN 4 Padang")
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.header("Data Calon Siswa")

    nilai_rapor = st.selectbox(
        "Nilai Rapor Calistung (Akurasi):",
        options=list(NILAI_RAPOR_MAP.keys()),
        index=1 # Sedang
    )

    wawancara = st.selectbox(
        "Hasil Wawancara (Interaksi):",
        options=list(WAWANCARA_MAP.keys()),
        index=1 # Cukup
    )

    prestasi = st.selectbox(
        "Prestasi Non-Akademik:",
        options=list(PRESTASI_MAP.keys()),
        index=1 # Tidak Ada
    )

with col2:
    st.header("Data Pendukung")

    jarak_rumah = st.selectbox(
        "Jarak Rumah:",
        options=J_RUMAH_OPTIONS,
        index=0 # Dekat
    )

    status_ekonomi = st.selectbox(
        "Status Ekonomi Keluarga:",
        options=S_EKONOMI_OPTIONS,
        index=0 # Mampu
    )

    jumlah_saudara = st.selectbox(
        "Jumlah Saudara:",
        options=J_SAUDARA_OPTIONS,
        index=1 # Sedikit
    )

# Tombol Prediksi
st.markdown("---")
if st.button("📣 Prediksi Keputusan Penerimaan"):
    # Siapkan data input
    input_data = {
        'Nilai_Rapor_Calistung': nilai_rapor,
        'Hasil_Wawancara': wawancara,
        'Prestasi_Non_Akademik': prestasi,
        'Jarak_Rumah': jarak_rumah,
        'Status_Ekonomi': status_ekonomi,
        'Jumlah_Saudara': jumlah_saudara,
    }

    # Lakukan prediksi
    result, proba = predict_keputusan(input_data)

    # Tampilkan Hasil
    st.header(f"Hasil Prediksi Keputusan: **{result}**")

    if result == "Diterima":
        st.success("✅ Calon siswa ini **DIPREDIKSI DITERIMA** di MIN 4 Padang.")
    else:
        st.warning("❌ Calon siswa ini **DIPREDIKSI TIDAK DITERIMA** di MIN 4 Padang.")

    # Tampilkan Probabilitas
    col_prob1, col_prob2 = st.columns(2)
    with col_prob1:
        st.info(f"Probabilitas Diterima: **{proba[0][0]*100:.2f}%**")
    with col_prob2:
        st.error(f"Probabilitas Tidak Diterima: **{proba[0][1]*100:.2f}%**")

    st.markdown("---")
    st.subheader("Data Input yang Diberikan:")
    st.write(pd.Series(input_data))
