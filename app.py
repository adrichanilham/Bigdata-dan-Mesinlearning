import streamlit as st
import pickle
import pandas as pd
import numpy as np
import sys

# --- 1. Memuat Model dan Tools ---
# Fungsi untuk memuat file .pkl dengan penanganan error
def load_pickle_file(filepath):
    try:
        with open(filepath, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error(f"Error: File model tidak ditemukan ({filepath}). Pastikan Anda sudah mengunggah semua file .pkl ke GitHub.")
        st.stop()
    except Exception as e:
        st.error(f"Error saat memuat file {filepath}: {e}")
        st.stop()
        
model = load_pickle_file('decision_tree_model.pkl')
le = load_pickle_file('label_encoder.pkl')
feature_names = load_pickle_file('feature_names.pkl')

if le is None or feature_names is None:
    st.stop()

# Ambil nama kelas target (misalnya: ['Diterima', 'Tidak Diterima'])
try:
    target_names = le.classes_
except AttributeError:
    # Fallback jika le tidak memiliki atribut classes_
    st.error("Error: Label Encoder (.pkl) tidak valid.")
    target_names = ['Diterima', 'Tidak Diterima']


# --- 2. Konfigurasi Mapping Data Input ---
NILAI_RAPOR_MAP = {'Rendah': 1, 'Sedang': 2, 'Tinggi': 3}
WAWANCARA_MAP = {'Kurang': 1, 'Cukup': 2, 'Baik': 3}
PRESTASI_MAP = {'Ada': 1, 'Tidak Ada': 0}

# FUNGSI PERBAIKAN BUG OPSI (Baseline yang Hilang)
def get_one_hot_options(prefix):
    # Dapatkan opsi yang menjadi kolom di feature_names (yang TIDAK di-drop)
    options = [f.replace(prefix + '_', '') for f in feature_names if f.startswith(prefix)]
    
    # PERBAIKAN: MENAMBAHKAN KEMBALI OPSI BASELINE YANG HILANG
    
    # 1. Perbaiki Status Ekonomi (Baseline: Mampu, Opsional: Kurang Mampu)
    if prefix == 'Status_Ekonomi':
        if 'Mampu' not in options:
            options.append('Mampu')
            
    # 2. Perbaiki Jumlah Saudara (Baseline: Sedikit, Opsional: Banyak)
    if prefix == 'Jumlah_Saudara':
        # Asumsi: Jika 'Banyak' adalah yang tersisa, maka 'Sedikit' yang hilang (atau sebaliknya)
        # Berdasarkan umumnya data, 'Sedikit' atau 'Banyak' adalah baseline. Kita pastikan keduanya ada
        if 'Sedikit' not in options:
             options.append('Sedikit')
        if 'Banyak' not in options:
             options.append('Banyak')

    # 3. Perbaiki Jarak Rumah (Baseline: Dekat, Opsional: Jauh)
    if prefix == 'Jarak_Rumah':
        if 'Dekat' not in options:
            options.append('Dekat')

    # Pastikan opsi unik dan diurutkan
    return sorted(list(set(options)))

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

    final_input = {}
    
    # Ambil nilai encoded Ordinal/Biner
    final_input['Nilai_Rapor_Calistung_Encoded'] = input_df['Nilai_Rapor_Calistung_Encoded'].iloc[0]
    final_input['Hasil_Wawancara_Encoded'] = input_df['Hasil_Wawancara_Encoded'].iloc[0]
    final_input['Prestasi_Non_Akademik_Encoded'] = input_df['Prestasi_Non_Akademik_Encoded'].iloc[0]

    # 2. One-Hot Encoding Manual
    
    # Jarak_Rumah
    for opt in J_RUMAH_OPTIONS:
        col_name = f'Jarak_Rumah_{opt}'
        if col_name in feature_names: 
             final_input[col_name] = 1 if input_df['Jarak_Rumah'].iloc[0] == opt else 0

    # Status_Ekonomi
    for opt in S_EKONOMI_OPTIONS:
        col_name = f'Status_Ekonomi_{opt}'
        if col_name in feature_names:
            final_input[col_name] = 1 if input_df['Status_Ekonomi'].iloc[0] == opt else 0

    # Jumlah_Saudara
    for opt in J_SAUDARA_OPTIONS:
        col_name = f'Jumlah_Saudara_{opt}'
        if col_name in feature_names:
            final_input[col_name] = 1 if input_df['Jumlah_Saudara'].iloc[0] == opt else 0
            
    # Pastikan semua kolom fitur yang dibutuhkan model ada di final_input
    for col in feature_names:
        if col not in final_input:
            final_input[col] = 0


    # Ubah ke DataFrame dengan urutan kolom yang benar (sesuai feature_names.pkl)
    final_df = pd.DataFrame([final_input], columns=feature_names)
    
    # Lakukan prediksi (Penerapan Rumus Decision Tree)
    prediction_encoded = model.predict(final_df)[0]
    prediction_proba = model.predict_proba(final_df)

    # Decode hasil
    result = target_names[prediction_encoded]
    
    return result, prediction_proba


# --- 4. Tampilan Streamlit ---
st.set_page_config(page_title="Prediksi Penerimaan Siswa MIN 4 Padang", layout="wide")
st.title("🌳 Decision Tree: Prediksi Penerimaan Siswa MIN 4 Padang")
st.markdown("---")

# Input Identitas Siswa
st.subheader("Data Identitas Siswa")
col_identity_1, col_identity_2 = st.columns(2)

with col_identity_1:
    input_nama = st.text_input("Nama Lengkap Calon Siswa:")
    
with col_identity_2:
    input_nisn = st.text_input("NISN (Nomor Induk Siswa Nasional):")

st.markdown("---")

# Input Kriteria
col1, col2 = st.columns(2)

with col1:
    st.header("Kriteria Akademik & Personal")
    
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
    st.header("Kriteria Penunjang")
    
    jarak_rumah = st.selectbox(
        "Jarak Rumah:",
        options=J_RUMAH_OPTIONS,
        index=0 # Index aman
    )
    
    # SUDAH DIPERBAIKI: Pilihan Mampu dan Kurang Mampu
    status_ekonomi = st.selectbox(
        "Status Ekonomi Keluarga:",
        options=S_EKONOMI_OPTIONS,
        index=0 # Index aman
    )

    # SUDAH DIPERBAIKI: Pilihan Banyak dan Sedikit
    jumlah_saudara = st.selectbox(
        "Jumlah Saudara:",
        options=J_SAUDARA_OPTIONS,
        index=0 # Index aman
    )

# Tombol Prediksi
st.markdown("---")
if st.button("📣 Prediksi Keputusan Penerimaan", type="primary"):
    if not input_nama or not input_nisn:
        st.error("⚠️ Mohon lengkapi Nama Lengkap dan NISN sebelum melakukan prediksi.")
        st.stop()
        
    # Siapkan data input (hanya fitur yang digunakan model)
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
    
    # Tampilkan Hasil Utama
    st.header(f"Hasil Prediksi untuk {input_nama}")
    st.subheader(f"NISN: **{input_nisn}**")
    st.markdown("---")
    
    # Probabilitas
    # Menemukan index probabilitas untuk 'Diterima' dan 'Tidak Diterima'
    try:
        idx_diterima = np.where(target_names == 'Diterima')[0][0]
        idx_tidak_diterima = np.where(target_names == 'Tidak Diterima')[0][0]
        prob_diterima = proba[0][idx_diterima]
        prob_tidak_diterima = proba[0][idx_tidak_diterima]
    except IndexError:
        prob_diterima = 0.5
        prob_tidak_diterima = 0.5
        st.warning("Gagal menghitung probabilitas spesifik, menggunakan 50%.")


    if result == "Diterima":
        st.success(f"✅ Status Keputusan: **DIPREDIKSI DITERIMA** di MIN 4 Padang.")
    else:
        st.error(f"❌ Status Keputusan: **DIPREDIKSI TIDAK DITERIMA** di MIN 4 Padang.")

    # Tampilkan Probabilitas
    col_prob1, col_prob2 = st.columns(2)
    with col_prob1:
        st.info(f"Probabilitas Diterima: **{prob_diterima*100:.2f}%**")
    with col_prob2:
        st.error(f"Probabilitas Tidak Diterima: **{prob_tidak_diterima*100:.2f}%**")

    st.markdown("---")
    st.subheader("Rincian Kriteria yang Digunakan:")
    st.write(pd.Series(input_data))
