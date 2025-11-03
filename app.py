import streamlit as st
import pickle
import pandas as pd
import numpy as np

# --- 1. Memuat Model dan Tools ---
try:
    with open('decision_tree_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('label_encoder.pkl', 'rb') as file:
        le = pickle.load(file)
        target_names = le.classes_
    with open('feature_names.pkl', 'rb') as file:
        feature_names = pickle.load(file)

except FileNotFoundError:
    st.error("Error: File model (.pkl) tidak ditemukan. Pastikan Anda sudah mengunggah semua file .pkl ke GitHub.")
    st.stop()


# --- 2. Konfigurasi Mapping Data Input ---
NILAI_RAPOR_MAP = {'Rendah': 1, 'Sedang': 2, 'Tinggi': 3}
WAWANCARA_MAP = {'Kurang': 1, 'Cukup': 2, 'Baik': 3}
PRESTASI_MAP = {'Ada': 1, 'Tidak Ada': 0}

def get_one_hot_options(prefix):
    options = [f.replace(prefix + '_', '') for f in feature_names if f.startswith(prefix)]
    # Tambahkan opsi baseline yang hilang karena drop_first=True
    if 'Dekat' not in options and prefix == 'Jarak_Rumah':
        options.append('Dekat')
    if 'Mampu' not in options and prefix == 'Status_Ekonomi':
        options.append('Mampu')
    if 'Sedikit' not in options and prefix == 'Jumlah_Saudara':
        options.append('Sedikit')
        
    return sorted(options)

J_RUMAH_OPTIONS = get_one_hot_options('Jarak_Rumah')
S_EKONOMI_OPTIONS = get_one_hot_options('Status_Ekonomi')
J_SAUDARA_OPTIONS = get_one_hot_options('Jumlah_Saudara')


# --- 3. Fungsi Prediksi (Tidak Berubah) ---
def predict_keputusan(data_input):
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
    for opt in J_RUMAH_OPTIONS:
        col_name = f'Jarak_Rumah_{opt}'
        if col_name in feature_names: final_input[col_name] = 1 if input_df['Jarak_Rumah'].iloc[0] == opt else 0

    for opt in S_EKONOMI_OPTIONS:
        col_name = f'Status_Ekonomi_{opt}'
        if col_name in feature_names: final_input[col_name] = 1 if input_df['Status_Ekonomi'].iloc[0] == opt else 0

    for opt in J_SAUDARA_OPTIONS:
        col_name = f'Jumlah_Saudara_{opt}'
        if col_name in feature_names: final_input[col_name] = 1 if input_df['Jumlah_Saudara'].iloc[0] == opt else 0
            
    # Pastikan semua kolom fitur yang dibutuhkan model ada (default 0 jika tidak ada di input)
    for col in feature_names:
        if col not in final_input: final_input[col] = 0

    final_df = pd.DataFrame([final_input], columns=feature_names)
    
    prediction_encoded = model.predict(final_df)[0]
    prediction_proba = model.predict_proba(final_df)

    result = target_names[prediction_encoded]
    
    return result, prediction_proba


# --- 4. Tampilan Streamlit ---
st.set_page_config(page_title="Prediksi Penerimaan Siswa MIN 4 Padang", layout="wide")
st.title("🌳 Decision Tree: Prediksi Penerimaan Siswa MIN 4 Padang")
st.markdown("---")

# **********************************************
# PERUBAHAN BARU: INPUT NAMA DAN NISN
# **********************************************
col_identity_1, col_identity_2 = st.columns(2)

with col_identity_1:
    input_nama = st.text_input("Nama Lengkap Calon Siswa:")
    
with col_identity_2:
    input_nisn = st.text_input("NISN (Nomor Induk Siswa Nasional):")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.header("Data Kriteria Akademik & Personal")
    
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
    st.header("Data Kriteria Penunjang")
    
    jarak_rumah = st.selectbox(
        "Jarak Rumah:",
        options=J_RUMAH_OPTIONS,
        index=0 # Indeks aman
    )
    
    status_ekonomi = st.selectbox(
        "Status Ekonomi Keluarga:",
        options=S_EKONOMI_OPTIONS,
        index=0 # Indeks aman
    )

    jumlah_saudara = st.selectbox(
        "Jumlah Saudara:",
        options=J_SAUDARA_OPTIONS,
        index=0 # Indeks aman
    )

# Tombol Prediksi
st.markdown("---")
if st.button("📣 Prediksi Keputusan Penerimaan"):
    if not input_nama or not input_nisn:
        st.error("⚠️ Mohon lengkapi Nama Lengkap dan NISN sebelum melakukan prediksi.")
        st.stop() # Hentikan eksekusi jika identitas kosong
        
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
    prob_diterima = proba[0][np.where(target_names == 'Diterima')[0][0]]
    prob_tidak_diterima = proba[0][np.where(target_names == 'Tidak Diterima')[0][0]]


    if result == "Diterima":
        st.success(f"✅ Status Keputusan: **DIPREDIKSI DITERIMA** di MIN 4 Padang.")
    else:
        st.warning(f"❌ Status Keputusan: **DIPREDIKSI TIDAK DITERIMA** di MIN 4 Padang.")

    # Tampilkan Probabilitas
    col_prob1, col_prob2 = st.columns(2)
    with col_prob1:
        st.info(f"Probabilitas Diterima: **{prob_diterima*100:.2f}%**")
    with col_prob2:
        st.error(f"Probabilitas Tidak Diterima: **{prob_tidak_diterima*100:.2f}%**")

    st.markdown("---")
    st.subheader("Rincian Kriteria yang Digunakan untuk Prediksi:")
    st.write(pd.Series(input_data))
