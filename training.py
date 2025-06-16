import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Mapping nama kolom ke bahasa Indonesia
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

# 1. Baca dataset mapped
df = pd.read_csv('agaricus-lepiota-mapped.csv')
df.rename(columns=column_mapping, inplace=True)

# 2. Pilih hanya fitur penting + target
selected_cols = ['bau', 'warna_spora', 'warna_insang', 'ukuran_insang', 'memar', 'populasi', 'habitat', 'kelas']
df = df[selected_cols]

# 3. Encode semua kolom kategorikal
le_dict = {}
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# 4. Split data
X = df.drop('kelas', axis=1)
y = df['kelas']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 6. Simpan model dan encoder ke folder src
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
os.makedirs(src_dir, exist_ok=True)

with open(os.path.join(src_dir, 'rf_mushroom.pkl'), 'wb') as f:
    pickle.dump(rf, f)
with open(os.path.join(src_dir, 'le_dict.pkl'), 'wb') as f:
    pickle.dump(le_dict, f)

print("Training selesai. Model dan encoder telah disimpan di folder src.")