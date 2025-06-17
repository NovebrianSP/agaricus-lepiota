import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, 'rf_mushroom.pkl'), 'rb') as f:
    model = pickle.load(f)
with open(os.path.join(BASE_DIR, 'le_dict.pkl'), 'rb') as f:
    le_dict = pickle.load(f)

# Load dataset
df = pd.read_csv('agaricus-lepiota-mapped.csv')
column_mapping = {
    'class': 'kelas',
    'odor': 'bau',
    'spore-print-color': 'warna_spora',
    'gill-color': 'warna_insang',
    'gill-size': 'ukuran_insang',
    'bruises': 'memar',
    'population': 'populasi',
    'habitat': 'habitat'
}
df.rename(columns=column_mapping, inplace=True)
selected_cols = ['bau', 'warna_spora', 'warna_insang', 'ukuran_insang', 'memar', 'populasi', 'habitat', 'kelas']
df = df[selected_cols]

# Encode data sesuai encoder yang digunakan saat training
for col in df.columns:
    le = le_dict[col]
    df[col] = le.transform(df[col])

X = df.drop('kelas', axis=1)
y = df['kelas']

# Prediksi dan hitung akurasi
y_pred = model.predict(X)
acc = accuracy_score(y, y_pred)
print(f"Akurasi model pada seluruh data: {acc:.4f}")