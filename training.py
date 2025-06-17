import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset dan mapping kolom
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

# Encode
le_dict = {}
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# Split & train
X = df.drop('kelas', axis=1)
y = df['kelas']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=8000, random_state=42)
model.fit(X_train, y_train)

# Simpan model dan encoder
with open('rf_mushroom.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('le_dict.pkl', 'wb') as f:
    pickle.dump(le_dict, f)

print("Model dan encoder berhasil disimpan!")