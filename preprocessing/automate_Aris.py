import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def auto_preprocessing(file_path, target_column=None, save_path=None):

    # 1. Load data
    df = pd.read_csv(file_path)

    # 2. Hapus duplikat & missing value
    df = df.drop_duplicates()
    df = df.dropna()

    # 3. Pisahkan target (jika ada)
    if target_column:
        y = df[target_column]
        df = df.drop(columns=[target_column])
    else:
        y = None

    # 4. Identifikasi kolom
    kolom_numerik = df.select_dtypes(include=np.number).columns
    kolom_kategori = df.select_dtypes(include='object').columns

    # 5. Encoding kategorikal
    for col in kolom_kategori:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # 6. Scaling numerik
    scaler = StandardScaler()
    df[kolom_numerik] = scaler.fit_transform(df[kolom_numerik])

    # 7. Handling outlier (IQR)
    for col in kolom_numerik:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower, upper)

    # 8. Gabungkan kembali target
    if target_column:
        df[target_column] = y.values

    # 9. Simpan hasil (opsional)
    if save_path:
        df.to_csv(save_path, index=False)

    return df
