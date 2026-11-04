"""
Aplikasi Web Edukasi, Skrining Risiko, dan Visualisasi Teknis Diabetes Tipe 2
Menggunakan Top Navigation (st.navigation position='top'), LightGBM & SMOTE-Tomek Pipeline
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Compatibility shim untuk unpickling model scikit-learn 1.6 di versi berbeda
try:
    import sklearn.compose._column_transformer as _ct
    if not hasattr(_ct, "_RemainderColsList"):
        class _RemainderColsList(list):
            pass
        _ct._RemainderColsList = _RemainderColsList
except Exception:
    pass

# Konfigurasi Tampilan Halaman (Sidebar disembunyikan/collapsed karena memakai Top Navigation)
st.set_page_config(
    page_title="DiabetesCare - Skrining & Edukasi Diabetes",
    page_icon="https://cdn-icons-png.flaticon.com/512/2966/2966327.png",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_FILENAME = "diabetes_baseline_model.joblib"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

EXPORTED_DIR = os.path.join(os.path.dirname(__file__), "exported")
PLOTS_DIR = os.path.join(EXPORTED_DIR, "plots")
OUTPUTS_DIR = os.path.join(EXPORTED_DIR, "outputs")

# Preset data pasien
PRESETS = {
    "healthy": {
        "gender": "Female",
        "smoking_status": "Never",
        "age": 32,
        "alcohol_consumption": 0,
        "physical_activity": 250,
        "diet_score": 8.5,
        "sleep_hours": 7.5,
        "screen_time": 4.0,
        "family_history": 0,
        "hypertension_history": 0,
        "cardio_history": 0,
        "bmi": 22.4,
        "diastolic_bp": 70,
        "heart_rate": 68,
        "cholesterol_total": 170.0,
        "hdl_cholesterol": 60.0,
        "triglycerides": 95.0,
        "glucose_fasting": 88.0,
        "insulin_level": 6.5,
        "name": "Pasien Sehat (Bebas Risiko)",
    },
    "high_risk": {
        "gender": "Male",
        "smoking_status": "Current",
        "age": 58,
        "alcohol_consumption": 4,
        "physical_activity": 40,
        "diet_score": 3.5,
        "sleep_hours": 5.5,
        "screen_time": 9.0,
        "family_history": 1,
        "hypertension_history": 1,
        "cardio_history": 1,
        "bmi": 33.8,
        "diastolic_bp": 92,
        "heart_rate": 84,
        "cholesterol_total": 260.0,
        "hdl_cholesterol": 36.0,
        "triglycerides": 240.0,
        "glucose_fasting": 145.0,
        "insulin_level": 18.5,
        "name": "Pasien Berisiko Tinggi Diabetes",
    },
    "median": {
        "gender": "Male",
        "smoking_status": "Never",
        "age": 45,
        "alcohol_consumption": 1,
        "physical_activity": 120,
        "diet_score": 6.0,
        "sleep_hours": 7.0,
        "screen_time": 6.0,
        "family_history": 0,
        "hypertension_history": 0,
        "cardio_history": 0,
        "bmi": 25.5,
        "diastolic_bp": 75,
        "heart_rate": 70,
        "cholesterol_total": 185.0,
        "hdl_cholesterol": 52.0,
        "triglycerides": 120.0,
        "glucose_fasting": 100.0,
        "insulin_level": 8.9,
        "name": "Nilai Rata-rata (Median Populasi)",
    },
}


def apply_preset(preset_key):
    """Mengisi nilai session_state berdasarkan preset."""
    preset_data = PRESETS[preset_key]
    for k, v in preset_data.items():
        if k != "name":
            st.session_state[f"input_{k}"] = v
    st.session_state["active_preset"] = preset_data["name"]


def init_form_state():
    """Inisialisasi nilai form jika belum tersedia."""
    if "input_age" not in st.session_state:
        apply_preset("median")


@st.cache_resource
def load_trained_model():
    """Memuat pipeline model dari folder model/."""
    candidates = [
        os.path.join(MODEL_DIR, "diabetes_baseline_model.joblib"),
        os.path.join(MODEL_DIR, "diabetes_model.joblib"),
        os.path.join(MODEL_DIR, "model.joblib"),
        os.path.join(MODEL_DIR, "diabetes_model.pkl"),
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                return model, os.path.basename(path)
            except Exception as e:
                st.error(f"Gagal memuat model dari {path}: {e}")
                return None, None
    return None, None


@st.cache_data
def load_experiment_export():
    """Memuat data JSON hasil eksperimen dari folder exported/outputs/."""
    if not os.path.exists(OUTPUTS_DIR):
        return None, None
    for f in os.listdir(OUTPUTS_DIR):
        if f.endswith(".json"):
            json_path = os.path.join(OUTPUTS_DIR, f)
            try:
                with open(json_path, "r", encoding="utf-8") as file:
                    return json.load(file), f
            except Exception:
                continue
    return None, None


# ==========================================
# HALAMAN 1: OVERVIEW & EDUKASI
# ==========================================
def render_overview_page():
    """Halaman edukasi dan tinjauan umum tentang Diabetes Melitus."""
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 14px; margin-bottom: 10px;">
            <img src="https://cdn-icons-png.flaticon.com/512/3004/3004458.png" width="38" height="38" alt="Blood test icon"/>
            <h1 style="margin: 0; font-size: 1.85rem; font-weight: 700;">Overview Diabetes Melitus</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "Panduan komprehensif mengenai definisi klinis, klasifikasi tipe, faktor risiko etiologis, "
        "serta langkah pencegahan berbasis rekomendasi PERKENI dan Kemenkes RI."
    )
    st.write("---")

    # 1. Definisi
    st.markdown("### 1. Definisi Klinis")
    c_def1, c_def2 = st.columns([1.6, 1], gap="large")
    with c_def1:
        st.markdown(
            """
            **Diabetes Melitus (DM)** adalah gangguan metabolik menahun yang ditandai dengan peningkatan kadar 
            glukosa darah melebihi batas normal (**Hiperglikemia**). 
            
            Kondisi ini terjadi akibat:
            * Gangguan sekresi insulin oleh sel beta pankreas,
            * Penurunan responsivitas jaringan perifer terhadap kerja insulin (*Resistensi Insulin*), atau
            * Kombinasi keduanya.
            
            **Peran Insulin dalam Tubuh:**  
            Insulin berfungsi memfasilitasi masuknya glukosa darah ke dalam sel otot dan adiposa untuk diubah menjadi energi. 
            Ketika aksi insulin terganggu, glukosa terakumulasi di dalam plasma darah dan memicu kerusakan endotel vaskular.
            """
        )
    with c_def2:
        st.info(
            """
            **Konteks Epidemiologi & Komplikasi**  
            * Di Indonesia, estimasi penderita diabetes dewasa melampaui **19,5 juta jiwa** (peringkat ke-5 global).
            * Lebih dari separuh penderita diabetes belum terdiagnosis secara dini karena minimnya gejala pada fase awal.
            * Komplikasi jangka panjang meliputi nefropati diabetik, retinopati, neuropati perifer, dan penyakit kardiovaskular.
            """
        )

    st.write("---")

    # 2. Tipe-Tipe Diabetes
    st.markdown("### 2. Klasifikasi Tipe Diabetes")
    st.write("Klasifikasi etiologis diabetes menurut standar WHO dan American Diabetes Association (ADA):")

    t1, t2, t3, t4 = st.columns(4)
    with t1:
        st.markdown(
            """
            <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-top: 4px solid #0284c7; padding: 16px; border-radius: 8px; min-height: 250px;">
                <h4 style="margin-top: 0; color: #0284c7; font-size: 1.05rem;">Diabetes Tipe 1</h4>
                <p style="font-size: 0.88rem; line-height: 1.5; color: #334155;">
                    <b>Etiologi:</b> Destruksi autoimun terhadap sel beta pankreas yang menyebabkan defisiensi insulin absolut.<br><br>
                    <b>Onset:</b> Mayoritas anak-anak, remaja, dan dewasa muda.<br><br>
                    <b>Penanganan:</b> Terapi insulin eksogen mutlak seumur hidup.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with t2:
        st.markdown(
            """
            <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-top: 4px solid #16a34a; padding: 16px; border-radius: 8px; min-height: 250px;">
                <h4 style="margin-top: 0; color: #16a34a; font-size: 1.05rem;">Diabetes Tipe 2 (Fokus)</h4>
                <p style="font-size: 0.88rem; line-height: 1.5; color: #334155;">
                    <b>Etiologi:</b> Resistensi insulin perifer disertai defisiensi insulin relatif.<br><br>
                    <b>Proporsi:</b> Mencakup <b>90%–95%</b> total kasus diabetes global.<br><br>
                    <b>Karakteristik:</b> Berkaitan erat dengan obesitas dan inaktivitas fisik. Sangat dapat dicegah.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with t3:
        st.markdown(
            """
            <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-top: 4px solid #ea580c; padding: 16px; border-radius: 8px; min-height: 250px;">
                <h4 style="margin-top: 0; color: #ea580c; font-size: 1.05rem;">Diabetes Gestasional</h4>
                <p style="font-size: 0.88rem; line-height: 1.5; color: #334155;">
                    <b>Etiologi:</b> Intoleransi glukosa yang pertama kali dikenali selama masa kehamilan akibat hormon plasenta.<br><br>
                    <b>Prognosis:</b> Kadar gula darah umumnya kembali normal pasca persalinan, namun risiko DM Tipe 2 meningkat di masa depan.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with t4:
        st.markdown(
            """
            <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-top: 4px solid #dc2626; padding: 16px; border-radius: 8px; min-height: 250px;">
                <h4 style="margin-top: 0; color: #dc2626; font-size: 1.05rem;">Prediabetes</h4>
                <p style="font-size: 0.88rem; line-height: 1.5; color: #334155;">
                    <b>Etiologi:</b> Kadar glukosa plasma di atas batas normal (100–125 mg/dL puasa), tetapi belum memenuhi kriteria diagnostik DM.<br><br>
                    <b>Urgensi:</b> Merupakan fase reversibel untuk intervensi intensif gaya hidup sebelum terjadi kerusakan permanen.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("---")

    # 3. Faktor Penyebab & Risiko
    st.markdown("### 3. Etiologi & Faktor Risiko")
    st.write("Faktor penentu risiko timbulnya Diabetes Melitus Tipe 2:")

    col_risk1, col_risk2 = st.columns(2, gap="large")
    with col_risk1:
        st.markdown("#### Faktor Non-Modifikasi (Bawaan)")
        st.markdown(
            """
            * **Predisposisi Genetik:** Riwayat diabetes pada keluarga tingkat pertama (orang tua atau saudara kandung).
            * **Usia:** Risiko meningkat progresif seiring penuaan biologis, terutama pada individu berusia >= 35 tahun.
            * **Riwayat Obstetrik:** Riwayat melahirkan bayi dengan berat lahir > 4.000 gram atau pernah mengalami diabetes gestasional.
            """
        )

    with col_risk2:
        st.markdown("#### Faktor Modifikasi (Gaya Hidup & Lingkungan)")
        st.markdown(
            """
            * **Kelebihan Berat Badan & Obesitas:** Indeks Massa Tubuh (BMI) >= 25 kg/m² atau adipositas sentral (lingkar pinggang > 90 cm pria, > 80 cm wanita).
            * **Aktivitas Fisik Rendah:** Kurang dari 150 menit latihan fisik aerobik per minggu serta sedentary lifestyle.
            * **Pola Nutrisi Tidak Seimbang:** Konsumsi berlebih karbohidrat sederhana, makanan olahan, serta rendah serat nabati.
            * **Konsumsi Tembakau & Alkohol:** Merokok memicu disfungsi endotel vaskular dan memperparah resistensi insulin.
            * **Durasi & Kualitas Tidur:** Gangguan tidur (< 6 jam/hari) memicu peningkatan hormon stres kortisol.
            * **Komorbiditas Kardiometabolik:** Hipertensi (>= 140/90 mmHg) dan dislipidemia (trigliserida >= 150 mg/dL, HDL < 40 mg/dL).
            """
        )

    st.write("---")

    # 4. Pencegahan
    st.markdown("### 4. Pencegahan & Pengendalian")
    st.markdown(
        "Penelitian klinis membuktikan bahwa modifikasi gaya hidup intensif dapat menurunkan insiden diabetes tipe 2 hingga **58%**:"
    )

    prev1, prev2, prev3 = st.columns(3)
    with prev1:
        st.markdown(
            """
            <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; padding: 14px; border-radius: 6px; min-height: 180px;">
                <h5 style="margin-top: 0; color: #0f172a;">Nutrisi Terencana</h5>
                <ul style="font-size: 0.88rem; color: #475569; padding-left: 20px; margin-bottom: 0;">
                    <li>Prinsip gizi seimbang: perbanyak sayuran berserat dan protein tanpa lemak.</li>
                    <li>Batasi asupan gula bebas maksimal 4 sendok makan (50 gram) per hari.</li>
                    <li>Hindari minuman manis kemasan, boba, dan sirup tinggi fruktosa.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with prev2:
        st.markdown(
            """
            <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; padding: 14px; border-radius: 6px; min-height: 180px;">
                <h5 style="margin-top: 0; color: #0f172a;">Aktivitas Fisik Aerobik</h5>
                <ul style="font-size: 0.88rem; color: #475569; padding-left: 20px; margin-bottom: 0;">
                    <li>Latihan intensitas sedang minimal 150 menit per minggu (30 menit/hari, 5 hari/minggu).</li>
                    <li>Kombinasikan dengan latihan beban 2 kali seminggu untuk meningkatkan massa otot penyerap glukosa.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with prev3:
        st.markdown(
            """
            <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; padding: 14px; border-radius: 6px; min-height: 180px;">
                <h5 style="margin-top: 0; color: #0f172a;">Manajemen Berat Badan</h5>
                <ul style="font-size: 0.88rem; color: #475569; padding-left: 20px; margin-bottom: 0;">
                    <li>Pertahankan indeks massa tubuh (BMI) dalam batas ideal Asia-Pasifik (18.5 – 22.9 kg/m²).</li>
                    <li>Penurunan 5%–7% dari berat badan berlebih signifikan menormalkan sensitivitas insulin.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    prev4, prev5, prev6 = st.columns(3)
    with prev4:
        st.markdown(
            """
            <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; padding: 14px; border-radius: 6px; min-height: 180px;">
                <h5 style="margin-top: 0; color: #0f172a;">Penghentian Tembakau</h5>
                <ul style="font-size: 0.88rem; color: #475569; padding-left: 20px; margin-bottom: 0;">
                    <li>Berhenti merokok aktif dan hindari paparan asap rokok lingkungan.</li>
                    <li>Sensitivitas reseptor insulin berangsur membaik dalam 8 minggu pasca henti rokok.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with prev5:
        st.markdown(
            """
            <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; padding: 14px; border-radius: 6px; min-height: 180px;">
                <h5 style="margin-top: 0; color: #0f172a;">Higienitas Tidur & Stres</h5>
                <ul style="font-size: 0.88rem; color: #475569; padding-left: 20px; margin-bottom: 0;">
                    <li>Penuhi kebutuhan istirahat teratur 7–8 jam per malam.</li>
                    <li>Manajemen stres untuk menekan pelepasan hormon kortisol dan katekolamin.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with prev6:
        st.markdown(
            """
            <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; padding: 14px; border-radius: 6px; min-height: 180px;">
                <h5 style="margin-top: 0; color: #0f172a;">Pemeriksaan Teratur</h5>
                <ul style="font-size: 0.88rem; color: #475569; padding-left: 20px; margin-bottom: 0;">
                    <li>Skrining glukosa puasa tahunan untuk individu berisiko atau usia >= 35 tahun.</li>
                    <li>Gunakan menu skrining pada aplikasi ini untuk kalkulasi estimasi risiko dini.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("---")

    # 5. Tabel Acuan Medis
    st.markdown("### 5. Kriteria Diagnostik Laboratorium (PERKENI & ADA)")
    criteria_df = pd.DataFrame(
        {
            "Parameter Pemeriksaan": [
                "Glukosa Darah Puasa (GDP)",
                "Glukosa 2 Jam Pasca Beban (GD2PP)",
                "Hemoglobin Terglikasi (HbA1c)",
                "Glukosa Darah Sewaktu (GDS)",
            ],
            "Normal": [
                "< 100 mg/dL",
                "< 140 mg/dL",
                "< 5.7 %",
                "< 140 mg/dL",
            ],
            "Prediabetes": [
                "100 – 125 mg/dL",
                "140 – 199 mg/dL",
                "5.7 – 6.4 %",
                "140 – 199 mg/dL",
            ],
            "Diabetes": [
                ">= 126 mg/dL",
                ">= 200 mg/dL",
                ">= 6.5 %",
                ">= 200 mg/dL (dengan gejala klasik)",
            ],
        }
    )
    st.table(criteria_df)


# ==========================================
# HALAMAN 2: SKRINING & PREDIKSI RISIKO
# ==========================================
def render_prediction_page():
    """Halaman skrining mandiri dan prediksi risiko Diabetes Tipe 2."""
    init_form_state()

    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 14px; margin-bottom: 10px;">
            <img src="https://cdn-icons-png.flaticon.com/512/3063/3063176.png" width="38" height="38" alt="Analysis icon"/>
            <h1 style="margin: 0; font-size: 1.85rem; font-weight: 700;">Skrining Risiko Diabetes Tipe 2</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "Kalkulasi probabilitas risiko Diabetes Melitus Tipe 2 berbasis **LightGBM Pipeline** "
        "menggunakan 19 indikator klinis, demografi, dan kebiasaan hidup pasien."
    )
    st.write("---")

    model, loaded_model_name = load_trained_model()
    if model is None:
        st.warning(
            """
            Model machine learning belum dapat dimuat dari folder `model/`.
            Pastikan file model `diabetes_baseline_model.joblib` tersedia di direktori kerja.
            """
        )
        return

    # PRESET INPUT CEPAT (Di halaman utama skrining)
    st.markdown("##### Preset Input Cepat")
    st.caption("Pilih salah satu tombol profil di bawah ini untuk mengisi seluruh isian form secara otomatis:")

    col_p1, col_p2, col_p3 = st.columns([1, 1, 1])
    with col_p1:
        if st.button("Pasien Sehat (Bebas Risiko)", use_container_width=True):
            apply_preset("healthy")
            st.rerun()
    with col_p2:
        if st.button("Pasien Berisiko Tinggi", use_container_width=True):
            apply_preset("high_risk")
            st.rerun()
    with col_p3:
        if st.button("Reset Nilai Standar (Median)", use_container_width=True):
            apply_preset("median")
            st.rerun()

    active_name = st.session_state.get("active_preset", "Nilai Rata-rata (Median Populasi)")
    st.info(f"Profil data saat ini: **{active_name}** | Model aktif: **{loaded_model_name}**")

    # Form Input 19 Fitur dibagi dalam 3 Tab
    st.markdown("#### Input Parameter Kesehatan")
    tab1, tab2, tab3 = st.tabs([
        "1. Demografi & Gaya Hidup",
        "2. Riwayat Medis & Fisik",
        "3. Profil Laboratorium"
    ])

    with tab1:
        st.markdown("##### Parameter Demografi dan Kebiasaan Harian")
        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox(
                "Jenis Kelamin",
                options=["Male", "Female", "Other"],
                key="input_gender",
            )
            age = st.slider(
                "Usia (Tahun)",
                min_value=18,
                max_value=90,
                key="input_age",
            )
            smoking_status = st.selectbox(
                "Status Merokok",
                options=["Never", "Former", "Current"],
                key="input_smoking_status",
                help="Never = Tidak pernah, Former = Mantan perokok, Current = Perokok aktif",
            )
        with c2:
            alcohol_consumption = st.slider(
                "Konsumsi Alkohol (porsi/minggu)",
                min_value=0,
                max_value=10,
                key="input_alcohol_consumption",
            )
            physical_activity = st.slider(
                "Aktivitas Fisik (menit/minggu)",
                min_value=0,
                max_value=800,
                step=10,
                key="input_physical_activity",
            )
            diet_score = st.slider(
                "Skor Kualitas Diet (0 - 10)",
                min_value=0.0,
                max_value=10.0,
                step=0.1,
                key="input_diet_score",
                help="0 = Sangat rendah serat / tinggi gula, 10 = Gizi seimbang",
            )
        with c3:
            sleep_hours = st.slider(
                "Durasi Tidur (jam/hari)",
                min_value=3.0,
                max_value=10.0,
                step=0.5,
                key="input_sleep_hours",
            )
            screen_time = st.slider(
                "Waktu di Depan Layar (jam/hari)",
                min_value=0.5,
                max_value=16.0,
                step=0.5,
                key="input_screen_time",
            )

    with tab2:
        st.markdown("##### Riwayat Penyakit & Pengukuran Fisik")
        c4, c5, c6 = st.columns(3)
        with c4:
            family_history = st.radio(
                "Riwayat Diabetes Keluarga",
                options=[0, 1],
                format_func=lambda x: "Ada Riwayat" if x == 1 else "Tidak Ada",
                key="input_family_history",
                horizontal=True,
            )
            hypertension_history = st.radio(
                "Riwayat Hipertensi",
                options=[0, 1],
                format_func=lambda x: "Hipertensi" if x == 1 else "Tidak Ada",
                key="input_hypertension_history",
                horizontal=True,
            )
            cardio_history = st.radio(
                "Riwayat Kardiovaskular",
                options=[0, 1],
                format_func=lambda x: "Ada Riwayat" if x == 1 else "Tidak Ada",
                key="input_cardio_history",
                horizontal=True,
            )
        with c5:
            bmi = st.number_input(
                "Indeks Massa Tubuh / BMI (kg/m²)",
                min_value=15.0,
                max_value=40.0,
                step=0.1,
                format="%.1f",
                key="input_bmi",
            )
            diastolic_bp = st.slider(
                "Tekanan Darah Diastolik (mmHg)",
                min_value=50,
                max_value=115,
                key="input_diastolic_bp",
            )
        with c6:
            heart_rate = st.slider(
                "Denyut Nadi Istirahat (bpm)",
                min_value=40,
                max_value=110,
                key="input_heart_rate",
            )

    with tab3:
        st.markdown("##### Hasil Pemeriksaan Darah & Profil Lipid")
        c7, c8 = st.columns(2)
        with c7:
            glucose_fasting = st.number_input(
                "Glukosa Darah Puasa (mg/dL)",
                min_value=60.0,
                max_value=200.0,
                step=1.0,
                format="%.1f",
                key="input_glucose_fasting",
                help="Batas normal: < 100 mg/dL",
            )
            insulin_level = st.number_input(
                "Insulin Puasa (mu U/ml)",
                min_value=2.0,
                max_value=30.0,
                step=0.1,
                format="%.2f",
                key="input_insulin_level",
            )
        with c8:
            cholesterol_total = st.number_input(
                "Kolesterol Total (mg/dL)",
                min_value=100.0,
                max_value=320.0,
                step=1.0,
                format="%.1f",
                key="input_cholesterol_total",
            )
            hdl_cholesterol = st.number_input(
                "Kolesterol HDL (mg/dL)",
                min_value=20.0,
                max_value=100.0,
                step=1.0,
                format="%.1f",
                key="input_hdl_cholesterol",
            )
            triglycerides = st.number_input(
                "Kadar Trigliserida (mg/dL)",
                min_value=30.0,
                max_value=320.0,
                step=1.0,
                format="%.1f",
                key="input_triglycerides",
            )

    st.write("---")

    # Posisi tombol diperkecil dan diletakkan di sebelah paling kanan
    _, col_btn = st.columns([3.8, 1.2])
    with col_btn:
        predict_btn = st.button("Analisis Risiko", type="primary", use_container_width=True)

    if predict_btn:
        patient_dict = {
            "age": [age],
            "gender": [gender],
            "smoking_status": [smoking_status],
            "alcohol_consumption_per_week": [alcohol_consumption],
            "physical_activity_minutes_per_week": [physical_activity],
            "diet_score": [diet_score],
            "sleep_hours_per_day": [sleep_hours],
            "screen_time_hours_per_day": [screen_time],
            "family_history_diabetes": [family_history],
            "hypertension_history": [hypertension_history],
            "cardiovascular_history": [cardio_history],
            "bmi": [bmi],
            "diastolic_bp": [diastolic_bp],
            "heart_rate": [heart_rate],
            "cholesterol_total": [cholesterol_total],
            "hdl_cholesterol": [hdl_cholesterol],
            "triglycerides": [triglycerides],
            "glucose_fasting": [glucose_fasting],
            "insulin_level": [insulin_level],
        }
        input_df = pd.DataFrame(patient_dict)

        try:
            prediction = model.predict(input_df)[0]
            probabilities = model.predict_proba(input_df)[0]

            prob_healthy = probabilities[0] * 100
            prob_diabetes = probabilities[1] * 100

            st.markdown("### Hasil Analisis Risiko")
            res_col1, res_col2 = st.columns([1.2, 1], gap="medium")

            with res_col1:
                if prediction == 1:
                    st.error("Terindikasi Berisiko Tinggi: Diabetes Melitus Tipe 2")
                    st.markdown(
                        f"Kalkulasi model menunjukkan estimasi probabilitas **{prob_diabetes:.1f}%** "
                        "berada dalam kategori risiko diabetes melitus tipe 2."
                    )
                else:
                    st.success("Terindikasi Berisiko Rendah (Normal)")
                    st.markdown(
                        f"Kalkulasi model menunjukkan estimasi probabilitas **{prob_healthy:.1f}%** "
                        "berada dalam kategori sehat / bebas risiko diabetes melitus tipe 2."
                    )

                st.progress(int(prob_diabetes))
                m1, m2 = st.columns(2)
                m1.metric("Probabilitas Bebas Risiko", f"{prob_healthy:.1f}%")
                m2.metric("Probabilitas Risiko DM Tipe 2", f"{prob_diabetes:.1f}%")

            with res_col2:
                st.markdown("##### Catatan Indikator Klinis")
                warnings_list = []
                if glucose_fasting >= 126:
                    warnings_list.append("Glukosa Puasa tinggi (>= 126 mg/dL - ambang diagnostik diabetes)")
                elif glucose_fasting >= 100:
                    warnings_list.append("Glukosa Puasa batas prediabetes (100 - 125 mg/dL)")

                if bmi >= 30:
                    warnings_list.append("Status Obesitas (BMI >= 30 kg/m²)")
                elif bmi >= 25:
                    warnings_list.append("Status Kelebihan Berat Badan / Overweight (BMI 25 - 29.9)")

                if triglycerides >= 150:
                    warnings_list.append("Kadar Trigliserida tinggi (>= 150 mg/dL)")

                if family_history == 1:
                    warnings_list.append("Terdapat riwayat genetik diabetes pada keluarga")

                if physical_activity < 150:
                    warnings_list.append("Aktivitas fisik di bawah anjuran (minimal 150 menit/minggu)")

                if warnings_list:
                    for w in warnings_list:
                        st.write(f"- {w}")
                else:
                    st.write("Semua parameter berada dalam rentang fisiologis normal.")

            st.write("---")
            st.markdown("#### Rekomendasi Klinis")
            if prediction == 1:
                st.warning(
                    """
                    1. **Pemeriksaan Laboratorium Lanjutan:** Konsultasikan ke dokter untuk pemeriksaan diagnostik lanjutan (HbA1c dan tes toleransi glukosa oral).
                    2. **Intervensi Pola Makan:** Batasi ketat gula bebas, karbohidrat sederhana, dan tingkatkan asupan serat pangan.
                    3. **Aktivitas Fisik Terjadwal:** Rutin berolahraga aerobik minimal 30 menit sehari.
                    """
                )
            else:
                st.info(
                    """
                    1. **Pertahankan Pola Hidup Sehat:** Jaga asupan gizi seimbang dan pertahankan berat badan ideal.
                    2. **Olahraga Berkelanjutan:** Pertahankan rutinitas latihan fisik minimal 150 menit per minggu.
                    3. **Skrining Berkala:** Lakukan pemeriksaan berkala minimal satu kali per tahun.
                    """
                )

        except Exception as err:
            st.error(f"Gagal memproses prediksi: {err}")

    st.write("---")
    st.caption(
        "Disclaimer: Sistem skrining ini merupakan alat bantu prediktif berbasis machine learning "
        "dan tidak menggantikan diagnosis resmi oleh tenaga medis berwenang."
    )


# ==========================================
# HALAMAN 3: TEKNIS MODEL & HASIL VISUALISASI
# ==========================================
def render_technical_page():
    """Halaman spesifikasi teknis file joblib dan galeri visualisasi dari exported/plots & exported/outputs/."""
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 14px; margin-bottom: 10px;">
            <img src="https://cdn-icons-png.flaticon.com/512/1055/1055687.png" width="38" height="38" alt="Analytics icon"/>
            <h1 style="margin: 0; font-size: 1.85rem; font-weight: 700;">Spesifikasi Model & Analisis Output</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "Dokumentasi teknis pipeline machine learning terintegrasi dari file `.joblib`, "
        "disertai analisis mendalam hasil eksperimen menggunakan data riil dari folder `exported/outputs/` "
        "dan visualisasi dari `exported/plots/`."
    )
    st.write("---")

    exp_data, json_filename = load_experiment_export()

    # 1. Ringkasan Teknis Pipeline Joblib
    st.markdown("### 1. Spesifikasi Teknis File Model (`.joblib`)")
    
    col_t1, col_t2 = st.columns([1.2, 1], gap="large")
    with col_t1:
        st.markdown(
            """
            * **Nama File Model:** `model/diabetes_baseline_model.joblib`
            * **Ukuran File:** ~2.08 MB
            * **Tipe Objek:** `imblearn.pipeline.Pipeline`
            * **Komponen Pipeline:**
              1. **`preprocessor` (`ColumnTransformer`):**
                 - `OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)` untuk 2 fitur kategorikal (`gender`, `smoking_status`).
                 - `passthrough` untuk 17 fitur numerikal.
              2. **`resampler` (`SMOTETomek`):**
                 - Menangani ketimpangan kelas (*class imbalance*) pada subset data latih.
                 - *Catatan:* Otomatis dilewati (*bypassed*) saat mode inferensi/prediksi.
              3. **`classifier` (`LGBMClassifier` / LightGBM):**
                 - Algoritma ensemble berbasis *Gradient Boosted Decision Trees (GBDT)*.
            """
        )
        if exp_data and "data_info" in exp_data:
            resamp = exp_data["data_info"].get("resampling_report", [])
            if resamp:
                st.markdown("##### Laporan Penyeimbangan Data (SMOTE-Tomek)")
                resamp_df = pd.DataFrame(resamp)
                st.dataframe(resamp_df, hide_index=True, use_container_width=True)

    with col_t2:
        st.markdown("##### Metrik Evaluasi Model (Data Riil `exported/outputs/`)")
        if exp_data and "experiment_results" in exp_data:
            metrics_summary = exp_data["experiment_results"].get("metrics_summary", [])
            if metrics_summary:
                metrics_df = pd.DataFrame(metrics_summary)
                # Format persentase
                cols_to_pct = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "PR-AUC"]
                display_df = metrics_df[["Set"] + [c for c in cols_to_pct if c in metrics_df.columns]].copy()
                for col in cols_to_pct:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"{x * 100:.2f}%" if pd.notnull(x) else "-")
                st.table(display_df)
                if json_filename:
                    st.caption(f"Sumber data metrik: `{json_filename}`")
        else:
            eval_metrics = pd.DataFrame(
                {
                    "Metrik Evaluasi": ["Akurasi", "Presisi", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"],
                    "Testing Set": ["93.29%", "98.08%", "94.24%", "96.12%", "97.15%", "99.64%"],
                }
            )
            st.table(eval_metrics)

    st.write("---")

    # 2. Galeri Visualisasi Hasil Eksperimen (/exported/plots/)
    st.markdown("### 2. Galeri Visualisasi Hasil Eksperimen (`exported/plots/`)")
    st.markdown(
        "Setiap gambar visualisasi di bawah ini disertai narasi analitis berbasis output data kuantitatif "
        "yang tersimpan di dalam folder `exported/outputs/`:"
    )

    tab_eda, tab_eval, tab_shap = st.tabs([
        "Eksplorasi Data (EDA)",
        "Evaluasi Model LightGBM",
        "Explainable AI (SHAP)",
    ])

    # TAB 1: EDA
    with tab_eda:
        st.markdown("#### Analisis Sebaran Data & Korelasi Fitur")
        
        # Plot 01: Boxplots
        plot_box = os.path.join(PLOTS_DIR, "01_numerical_features_boxplots.png")
        if os.path.exists(plot_box):
            _, c_box, _ = st.columns([15, 70, 15])
            with c_box:
                st.image(
                    plot_box,
                    caption="01. Boxplot Sebaran & Deteksi Outlier Fitur Numerik",
                    use_container_width=True,
                )
            st.markdown(
                """
                **Analisis Output Berdasarkan Data Eksperimen:**
                * Grafik boxplot di atas memetakan rentang interkuartil (IQR), nilai median, serta sebaran pencilan (*outliers*) pada 17 fitur numerik.
                * Parameter biomarker darah—khususnya **Glukosa Puasa**, **Insulin**, dan **Trigliserida**—menunjukkan variasi rentang yang lebar antara kelompok sehat dan diabetes.
                * Outlier pada variabel klinis seperti insulin tetap dipertahankan karena mencerminkan variabilitas fisiologis ekstrem pada kondisi resistensi insulin berat.
                """
            )
            st.write("---")

        # Plot 02: Distributions
        plot_dist = os.path.join(PLOTS_DIR, "02_numerical_features_distributions.png")
        if os.path.exists(plot_dist):
            _, c_dist, _ = st.columns([15, 70, 15])
            with c_dist:
                st.image(
                    plot_dist,
                    caption="02. Histogram Distribusi Seluruh Fitur Numerik Pasien",
                    use_container_width=True,
                )
            st.markdown(
                """
                **Analisis Output Berdasarkan Data Eksperimen:**
                * Menampilkan fungsi kerapatan probabilitas (KDE) dan frekuensi histogram populasi data setelah filtering tahap *Type 2* dan *No Diabetes* (total 67.755 baris).
                * Fitur usia terdistribusi merata pada rentang dewasa (18–90 tahun, rata-rata 51 tahun).
                * Variabel biomarker laboratorium menunjukkan pola distribusi multimodal yang mencerminkan pemisahan alami antara subpopulasi normoglikemia dan hiperglikemia.
                """
            )
            st.write("---")

        # Plot 03: Correlation Heatmap
        plot_corr = os.path.join(PLOTS_DIR, "03_correlation_heatmap.png")
        if os.path.exists(plot_corr):
            _, c_corr, _ = st.columns([15, 70, 15])
            with c_corr:
                st.image(
                    plot_corr,
                    caption="03. Heatmap Korelasi Pearson Antar Fitur Numerik",
                    use_container_width=True,
                )
            st.markdown(
                """
                **Analisis Output Berdasarkan Data Eksperimen:**
                * Matriks korelasi Pearson ini menjadi dasar ilmiah proses **Feature Selection** di notebook:
                  Variabel yang memiliki koefisien multikolinearitas tinggi (|r| >= 0.50) dieliminasi untuk mencegah redundansi informasi.
                * Sebanyak **6 fitur tereliminasi** dari 31 kolom awal: `waist_to_hip_ratio`, `systolic_bp`, `ldl_cholesterol`, `glucose_postprandial`, `hba1c`, dan `diabetes_risk_score`.
                * Dengan seleksi ini, model menjadi lebih efisien (*parsimonious*), stabil, dan bebas dari distorsi bobot multikolinearitas.
                """
            )

    # TAB 2: Evaluasi Model
    with tab_eval:
        st.markdown("#### Performa Klasifikasi & Kepentingan Fitur")
        
        # Plot 04 & 05
        c_ev1, c_ev2 = st.columns(2, gap="large")

        with c_ev1:
            plot_cm = os.path.join(PLOTS_DIR, "05_confusion_matrices.png")
            if os.path.exists(plot_cm):
                st.image(
                    plot_cm,
                    caption="05. Confusion Matrix: Evaluasi Training vs Testing Set",
                    use_container_width=True,
                )
                if exp_data and "experiment_results" in exp_data:
                    cm_test = exp_data["experiment_results"].get("confusion_matrices", {}).get("testing", [])
                    if len(cm_test) == 2:
                        tn, fp = cm_test[0]
                        fn, tp = cm_test[1]
                        total_test = tn + fp + fn + tp
                        st.markdown(
                            f"""
                            **Bedah Angka Confusion Matrix (Testing Set = {total_test:,} sampel):**
                            * **True Negative (TN):** **{tn:,}** pasien sehat teridentifikasi normal dengan benar.
                            * **True Positive (TP):** **{tp:,}** pasien diabetes teridentifikasi positif dengan tepat.
                            * **False Positive (FP):** Hanya **{fp:,}** ({fp/total_test*100:.1f}%) false alarm.
                            * **False Negative (FN):** **{fn:,}** ({fn/total_test*100:.1f}%) kasus terlewat.
                            * **Tingkat Presisi:** **98.08%**, meminimalkan rujukan salah bagi pasien sehat.
                            """
                        )

        with c_ev2:
            plot_fi = os.path.join(PLOTS_DIR, "04_lgbm_feature_importance.png")
            if os.path.exists(plot_fi):
                st.image(
                    plot_fi,
                    caption="04. LightGBM Feature Importance (Gain & Split)",
                    use_container_width=True,
                )
                st.markdown(
                    """
                    **Analisis Kontribusi Pohon Keputusan (LightGBM):**
                    * Feature Importance mengukur seberapa sering suatu fitur digunakan untuk membagi node (*split count*) serta seberapa besar penurunan loss (*gain*).
                    * **Glukosa Puasa (`glucose_fasting`)** menjadi fitur paling dominan dengan total gain terbesar.
                    * Variabel gaya hidup (konsumsi alkohol, status merokok) serta faktor biologis (gender, riwayat hipertensi) konsisten menduduki peringkat atas dalam pembagian cabang pohon.
                    """
                )

    # TAB 3: SHAP
    with tab_shap:
        st.markdown("#### Interpretasi Model Global & Studi Kasus Pasien (SHAP / TreeExplainer)")
        st.caption(
            "SHAP (SHapley Additive exPlanations) menghitung kontribusi marjinal tiap fitur "
            "menggunakan konsep teori permainan (Game Theory) untuk transparansi keputusan klinis."
        )

        c_sh1, c_sh2 = st.columns(2, gap="large")
        with c_sh1:
            plot_beeswarm = os.path.join(PLOTS_DIR, "06_shap_beeswarm.png")
            if os.path.exists(plot_beeswarm):
                st.image(
                    plot_beeswarm,
                    caption="06. SHAP Summary Beeswarm Plot",
                    use_container_width=True,
                )
                st.markdown(
                    """
                    **Cara Membaca Beeswarm Plot:**
                    * Sumbu X merepresentasikan nilai SHAP. Titik di sebelah kanan garis 0 mendorong prediksi ke arah **Positif Diabetes**.
                    * Warna titik menunjukkan nilai fitur (Merah = Nilai tinggi, Biru = Nilai rendah).
                    * Titik merah pada `glucose_fasting` berkumpul jauh di sebelah kanan: **Makin tinggi glukosa puasa, makin masif lonjakan risiko diabetes**.
                    """
                )
        with c_sh2:
            plot_shap_bar = os.path.join(PLOTS_DIR, "07_shap_feature_importance_bar.png")
            if os.path.exists(plot_shap_bar):
                st.image(
                    plot_shap_bar,
                    caption="07. SHAP Global Feature Importance Bar Chart",
                    use_container_width=True,
                )
                if exp_data and "interpretability_shap" in exp_data:
                    base_shap = exp_data["interpretability_shap"].get("Baseline Model", {})
                    gfi = base_shap.get("global_feature_importance", {})
                    if gfi:
                        st.markdown("**Top 5 Nilai Rata-rata |SHAP Value| Global:**")
                        top_5_items = list(gfi.items())[:5]
                        top_df = pd.DataFrame(
                            [{"Fitur": k, "Rata-rata |SHAP|": f"{v:.4f}"} for k, v in top_5_items]
                        )
                        st.dataframe(top_df, hide_index=True, use_container_width=True)

        st.write("---")
        st.markdown("##### Studi Kasus Lokal Pasien (Local Waterfall Plots)")
        st.markdown(
            "Waterfall plot memperlihatkan perjalanan kalkulasi log-odds dari nilai dasar (*base value*) "
            "hingga menghasilkan keputusan akhir untuk pasien individual:"
        )

        c_wf1, c_wf2 = st.columns(2, gap="large")
        with c_wf1:
            plot_wf_pos = os.path.join(PLOTS_DIR, "08_shap_waterfall_true_positive.png")
            if os.path.exists(plot_wf_pos):
                st.image(
                    plot_wf_pos,
                    caption="08. Kasus True Positive (Pasien Terdiagnosa Diabetes)",
                    use_container_width=True,
                )
                if exp_data and "interpretability_shap" in exp_data:
                    tp_data = exp_data["interpretability_shap"].get("Baseline Model", {}).get("local_explanation_true_positive", {})
                    if tp_data:
                        bv = tp_data.get("base_value", 0)
                        pv = tp_data.get("prediction_value", 0)
                        st.markdown(
                            f"""
                            **Bedah Kasus Pasien Positif (Indeks #{tp_data.get('sample_index', 0)}):**
                            * **Base Value ($E[f(x)]$):** `{bv:.4f}`
                            * **Prediction Value ($f(x)$):** `+{pv:.4f}` (Probabilitas > 99.9%)
                            * Glukosa puasa tinggi memberikan dorongan terbesar (+6.80), disusul status merokok (+0.25) dan konsumsi alkohol (+0.20).
                            """
                        )

        with c_wf2:
            plot_wf_neg = os.path.join(PLOTS_DIR, "09_shap_waterfall_true_negative.png")
            if os.path.exists(plot_wf_neg):
                st.image(
                    plot_wf_neg,
                    caption="09. Kasus True Negative (Pasien Normal / Sehat)",
                    use_container_width=True,
                )
                if exp_data and "interpretability_shap" in exp_data:
                    tn_data = exp_data["interpretability_shap"].get("Baseline Model", {}).get("local_explanation_true_negative", {})
                    if tn_data:
                        bv = tn_data.get("base_value", 0)
                        pv = tn_data.get("prediction_value", 0)
                        st.markdown(
                            f"""
                            **Bedah Kasus Pasien Bebas Risiko (Indeks #{tn_data.get('sample_index', 39)}):**
                            * **Base Value ($E[f(x)]$):** `{bv:.4f}`
                            * **Prediction Value ($f(x)$):** `{pv:.4f}` (Probabilitas < 0.1%)
                            * Glukosa puasa dalam batas normal menarik nilai prediksi turun sebesar -6.80, mengonfirmasi status bebas diabetes.
                            """
                        )


# ==========================================
# KONFIGURASI TOP NAVIGATION STREAMLIT (st.navigation)
# ==========================================
page_overview = st.Page(
    render_overview_page,
    title="Overview & Edukasi",
    icon=":material/menu_book:",
    default=True,
)

page_prediction = st.Page(
    render_prediction_page,
    title="Skrining Risiko",
    icon=":material/monitor_heart:",
)

page_technical = st.Page(
    render_technical_page,
    title="Model & Visualisasi",
    icon=":material/analytics:",
)

# Navigasi di bagian ATAS (position="top") tanpa sidebar
pg = st.navigation(
    [
        page_overview,
        page_prediction,
        page_technical,
    ],
    position="top",
)

pg.run()
