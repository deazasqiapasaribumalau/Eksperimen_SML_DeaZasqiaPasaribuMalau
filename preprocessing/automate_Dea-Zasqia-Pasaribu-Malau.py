"""
automate_Dea-Zasqia-Pasaribu-Malau.py
=====================
Script otomatisasi preprocessing dataset Credit Scoring.
Mengkonversi langkah eksperimen pada notebook menjadi pipeline otomatis.

Usage:
    python automate_Dea-Zasqia-Pasaribu-Malau.py --input credit_scoring_raw.csv --output credit_scoring_preprocessing
"""

import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer


# ==============================================================
# 1. LOAD DATA
# ==============================================================
def load_data(filepath: str) -> pd.DataFrame:
    """Membaca dataset dari file CSV."""
    print(f"[1/6] Loading data dari: {filepath}")
    df = pd.read_csv(filepath)
    print(f"      Shape awal: {df.shape}")
    print(f"      Kolom: {list(df.columns)}")
    return df


# ==============================================================
# 2. HANDLE MISSING VALUES
# ==============================================================
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Mengisi missing values dengan strategi yang sesuai per tipe kolom."""
    print("[2/6] Handling missing values...")
    df = df.copy()

    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    num_missing = df[num_cols].columns[df[num_cols].isnull().any()]
    cat_missing = df[cat_cols].columns[df[cat_cols].isnull().any()]

    if len(num_missing) > 0:
        imputer_num = SimpleImputer(strategy="median")
        df[num_missing] = imputer_num.fit_transform(df[num_missing])
        print(f"      Numerik diimputasi (median): {list(num_missing)}")

    if len(cat_missing) > 0:
        imputer_cat = SimpleImputer(strategy="most_frequent")
        df[cat_missing] = imputer_cat.fit_transform(df[cat_missing])
        print(f"      Kategorikal diimputasi (modus): {list(cat_missing)}")

    total_missing = df.isnull().sum().sum()
    print(f"      Total missing values setelah imputasi: {total_missing}")
    assert total_missing == 0, "Masih ada missing values!"
    return df


# ==============================================================
# 3. HANDLE DUPLICATES
# ==============================================================
def handle_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Menghapus baris duplikat."""
    print("[3/6] Handling duplicates...")
    n_before = len(df)
    df = df.drop_duplicates()
    n_after = len(df)
    print(f"      Baris duplikat dihapus: {n_before - n_after}")
    return df


# ==============================================================
# 4. HANDLE OUTLIER (IQR Capping / Winsorization)
# ==============================================================
def handle_outliers(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Mengatasi outlier menggunakan IQR capping (Winsorization)."""
    print("[4/6] Handling outliers dengan IQR capping...")
    df = df.copy()

    for col in columns:
        if col not in df.columns:
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower=lower, upper=upper)
        print(f"      {col}: {n_outliers} outlier → cap [{lower:.2f}, {upper:.2f}]")

    return df


# ==============================================================
# 5. ENCODING
# ==============================================================
def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Melakukan encoding pada fitur kategorikal."""
    print("[5/6] Encoding fitur kategorikal...")
    df = df.copy()

    # Ordinal Encoding: loan_grade (A=0 paling baik, G=6 paling buruk)
    if "loan_grade" in df.columns:
        grade_order = [["A", "B", "C", "D", "E", "F", "G"]]
        ord_enc = OrdinalEncoder(categories=grade_order)
        df["loan_grade"] = ord_enc.fit_transform(df[["loan_grade"]]).ravel()
        print("      loan_grade → OrdinalEncoder")

    # Label Encoding: binary column
    if "cb_person_default_on_file" in df.columns:
        le = LabelEncoder()
        df["cb_person_default_on_file"] = le.fit_transform(df["cb_person_default_on_file"])
        print("      cb_person_default_on_file → LabelEncoder")

    # One-Hot Encoding: nominal columns
    ohe_cols = [c for c in ["person_home_ownership", "loan_intent"] if c in df.columns]
    if ohe_cols:
        df = pd.get_dummies(df, columns=ohe_cols, drop_first=False, dtype=int)
        print(f"      {ohe_cols} → One-Hot Encoding")

    print(f"      Shape setelah encoding: {df.shape}")
    return df


# ==============================================================
# 6. SCALING & SPLIT
# ==============================================================
def scale_and_split(
    df: pd.DataFrame,
    target_col: str = "loan_status",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Scaling fitur numerik dan split data train-test."""
    print("[6/6] Feature scaling dan train-test split...")

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    scale_cols = [
        "person_age",
        "person_income",
        "person_emp_length",
        "loan_amnt",
        "loan_int_rate",
        "loan_percent_income",
        "cb_person_cred_hist_length",
        "loan_grade",
    ]
    scale_cols = [c for c in scale_cols if c in X.columns]

    scaler = StandardScaler()
    X = X.copy()
    X[scale_cols] = scaler.fit_transform(X[scale_cols])
    print(f"      StandardScaler diterapkan pada: {scale_cols}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"      Train: {X_train.shape} | Test: {X_test.shape}")

    return X_train, X_test, y_train, y_test


# ==============================================================
# MAIN PIPELINE
# ==============================================================
def run_preprocessing_pipeline(input_path: str, output_dir: str):
    """Menjalankan seluruh pipeline preprocessing dari awal hingga akhir."""
    print("=" * 60)
    print("  PIPELINE PREPROCESSING - CREDIT SCORING DATASET")
    print("=" * 60)

    OUTLIER_COLS = [
        "person_age",
        "person_income",
        "person_emp_length",
        "loan_amnt",
        "cb_person_cred_hist_length",
    ]

    df = load_data(input_path)
    df = handle_missing_values(df)
    df = handle_duplicates(df)
    df = handle_outliers(df, OUTLIER_COLS)
    df = encode_features(df)
    X_train, X_test, y_train, y_test = scale_and_split(df)

    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

    print()
    print("=" * 60)
    print("✅  PREPROCESSING SELESAI!")
    print(f"    Output disimpan di: {output_dir}/")
    print(f"    - X_train.csv  : {X_train.shape}")
    print(f"    - X_test.csv   : {X_test.shape}")
    print(f"    - y_train.csv  : {y_train.shape}")
    print(f"    - y_test.csv   : {y_test.shape}")
    print("=" * 60)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automate preprocessing pipeline")
    parser.add_argument(
        "--input",
        type=str,
        default="credit_scoring_raw.csv",
        help="Path ke file raw CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="credit_scoring_preprocessing",
        help="Direktori output hasil preprocessing",
    )
    args = parser.parse_args()
    run_preprocessing_pipeline(args.input, args.output)
