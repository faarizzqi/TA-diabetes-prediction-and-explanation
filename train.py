"""
Script Pelatihan Model Diabetes Tipe 2 (Baseline LGBM Pipeline)
Sesuai dengan eksperimen pada notebooks/TA2025-v3.ipynb
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek
from lightgbm import LGBMClassifier

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "diabetes_dataset.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "diabetes_baseline_model.joblib")

# Fitur-fitur yang digunakan sesuai seleksi fitur di TA2025-v3.ipynb
CATEGORICAL_FEATURES = ["gender", "smoking_status"]
NUMERICAL_FEATURES = [
    "age",
    "alcohol_consumption_per_week",
    "physical_activity_minutes_per_week",
    "diet_score",
    "sleep_hours_per_day",
    "screen_time_hours_per_day",
    "family_history_diabetes",
    "hypertension_history",
    "cardiovascular_history",
    "bmi",
    "diastolic_bp",
    "heart_rate",
    "cholesterol_total",
    "hdl_cholesterol",
    "triglycerides",
    "glucose_fasting",
    "insulin_level",
]
ALL_FEATURES = [
    "age",
    "gender",
    "smoking_status",
    "alcohol_consumption_per_week",
    "physical_activity_minutes_per_week",
    "diet_score",
    "sleep_hours_per_day",
    "screen_time_hours_per_day",
    "family_history_diabetes",
    "hypertension_history",
    "cardiovascular_history",
    "bmi",
    "diastolic_bp",
    "heart_rate",
    "cholesterol_total",
    "hdl_cholesterol",
    "triglycerides",
    "glucose_fasting",
    "insulin_level",
]
TARGET_COLUMN = "diagnosed_diabetes"


def train_baseline_model():
    """Memuat data, memfilter Type 2 & No Diabetes, melatih Pipeline LGBM, dan menyimpannya."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset tidak ditemukan di {DATA_PATH}!")

    print(f"Membaca dataset dari {DATA_PATH}...")
    df_raw = pd.read_csv(DATA_PATH)
    print(f"Dataset awal: {df_raw.shape}")

    # 1. Filtering: Hanya sertakan 'Type 2' dan 'No Diabetes'
    df_filtered = df_raw[df_raw["diabetes_stage"].isin(["Type 2", "No Diabetes"])].copy()
    print(f"Dataset setelah filter stage: {df_filtered.shape}")

    # 2. Pisahkan Fitur (X) dan Target (y)
    X = df_filtered[ALL_FEATURES]
    y = df_filtered[TARGET_COLUMN]

    # 3. Train-Test Split (80:20, Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Jumlah sampel train: {len(X_train)}, test: {len(X_test)}")

    # 4. Bangun Komponen Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                CATEGORICAL_FEATURES,
            ),
            ("num", "passthrough", NUMERICAL_FEATURES),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    resampler = SMOTETomek(random_state=42, n_jobs=-1)
    classifier = LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)

    baseline_pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("resampler", resampler),
            ("classifier", classifier),
        ]
    )

    # 5. Latih Pipeline
    print("Melatih baseline pipeline (Preprocess -> SMOTE-Tomek -> LightGBM)...")
    baseline_pipeline.fit(X_train, y_train)

    # 6. Evaluasi
    y_pred = baseline_pipeline.predict(X_test)
    y_prob = baseline_pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\n--- Hasil Evaluasi Test Set ---")
    print(f"Akurasi: {acc * 100:.2f}%")
    print(f"ROC-AUC: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 7. Simpan Model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(baseline_pipeline, MODEL_PATH)
    print(f"Model berhasil disimpan di: {MODEL_PATH}")


if __name__ == "__main__":
    train_baseline_model()
