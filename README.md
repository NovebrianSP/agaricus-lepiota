# Mushroom Classification Project 🍄

Aplikasi ini merupakan sistem klasifikasi jamur berbasis **Streamlit** yang dapat memprediksi apakah jamur dapat dimakan atau beracun, serta menampilkan dashboard visualisasi data.  
Model yang digunakan adalah **Random Forest** dengan akurasi tinggi, dan seluruh proses training serta prediksi sudah dipisahkan agar aplikasi berjalan cepat dan efisien.

---

## 📁 Struktur Project

```
mushroom-streamlit-app/
│
├── agaricus-lepiota.data              # Dataset asli dari UCI
├── agaricus-lepiota-mapped.csv        # Dataset hasil mapping kategori
├── processing.py                      # Script mapping kode kategori ke label string
├── training.py                        # Script training model & encoding label
├── rf_mushroom.pkl                    # File model Random Forest hasil training
├── le_dict.pkl                        # File LabelEncoder dictionary
├── src/
│   ├── app.py                         # Aplikasi utama Streamlit (dashboard & prediksi)
│   └── accuracy.py                    # Script untuk menguji akurasi model
├── explanation.txt                    # Penjelasan detail setiap script
└── README.md                          # Dokumentasi project
```

---

## 🚀 Alur Kerja Project

1. **Preprocessing Data**  
   Mapping kode kategori pada dataset asli menjadi label string yang mudah dibaca menggunakan `processing.py`.

2. **Training Model**  
   Melatih model Random Forest pada data hasil mapping, melakukan encoding label, dan menyimpan model serta encoder ke file `.pkl` menggunakan `training.py`.

3. **Aplikasi Streamlit**  
   Menampilkan dashboard visualisasi data, akurasi model, dan fitur klasifikasi jamur pada `app.py`.

4. **Evaluasi Akurasi**  
   Menguji akurasi model pada seluruh data menggunakan `accuracy.py`.

---

## 📝 Penjelasan Script

### 1. `processing.py`  
**Fungsi:**  
Melakukan mapping kode kategori pada dataset jamur (`agaricus-lepiota.data`) menjadi label string yang mudah dibaca, lalu menyimpan hasilnya ke file CSV (`agaricus-lepiota-mapped.csv`).

**Line-by-line:**
```python
import pandas as pd  # Mengimpor library pandas untuk manipulasi data.

input_file = 'agaricus-lepiota.data'
output_file = 'agaricus-lepiota-mapped.csv'
# Mendefinisikan nama file input (data asli) dan output (hasil mapping).

columns = [ ... ]
# Mendefinisikan nama kolom sesuai dokumentasi UCI agar data mudah diakses.

mapping_dict = { ... }
# Dictionary untuk mapping kode kategori ke label string.

df = pd.read_csv(input_file, header=None, names=columns)
# Membaca file data asli tanpa header, lalu memberi nama kolom.

for col in columns:
    if col in mapping_dict:
        df[col] = df[col].map(mapping_dict[col])
# Mapping setiap kolom dari kode ke label string.

df.to_csv(output_file, index=False)
# Menyimpan hasil mapping ke file CSV tanpa index.
```

---

### 2. `training.py`  
**Fungsi:**  
Melakukan training model Random Forest pada data hasil mapping, melakukan encoding label, dan menyimpan model serta encoder ke file `.pkl`.

**Line-by-line:**
```python
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# Mengimpor library yang dibutuhkan.

df = pd.read_csv('agaricus-lepiota-mapped.csv')
# Membaca data hasil mapping dari file CSV.

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
# Mapping nama kolom dan memilih kolom yang relevan.

le_dict = {}
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le
# Melakukan encoding label pada setiap kolom dan menyimpan encoder ke dictionary.

X = df.drop('kelas', axis=1)
y = df['kelas']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=8000, random_state=42)
model.fit(X_train, y_train)
# Membagi data dan melatih model Random Forest.

with open('rf_mushroom.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('le_dict.pkl', 'wb') as f:
    pickle.dump(le_dict, f)
# Menyimpan model dan dictionary encoder ke file .pkl.
```

---

### 3. `src/app.py`  
**Fungsi:**  
Aplikasi utama Streamlit untuk dashboard visualisasi data dan klasifikasi jamur.

**Highlight kode penting:**
```python
import pandas as pd
import streamlit as st
import pickle
import os

# CSS responsif untuk tampilan tabel di Streamlit

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, 'rf_mushroom.pkl'), 'rb') as f:
    model = pickle.load(f)
with open(os.path.join(BASE_DIR, 'le_dict.pkl'), 'rb') as f:
    le_dict = pickle.load(f)
# Memuat model dan encoder dari file .pkl.

df = pd.read_csv(os.path.join(BASE_DIR, 'agaricus-lepiota-mapped.csv'))
# Membaca data hasil mapping.

# Mapping nama kolom dan memilih kolom yang relevan
columns = [col for col in df.columns if col != 'kelas']

# Sidebar navigasi Streamlit
st.sidebar.title("Menu")
page = st.sidebar.radio("Pilih Halaman", ("Dashboard", "Klasifikasi"))

if page == "Dashboard":
    # Visualisasi distribusi fitur, contoh data, dan akurasi model
    # Akurasi model:
    benar = (y == y_pred).sum()
    total = len(y)
    akurasi = benar / total * 100
    st.metric("Akurasi Model", f"{akurasi:.2f}%")

elif page == "Klasifikasi":
    # Form input fitur, prediksi, dan tampilan hasil sesuai skor edible
    if edible_score >= 8.0:
        st.success("Jamur ini **dapat dimakan**.")
    elif 6.0 <= edible_score < 8.0:
        st.warning("Jamur ini **tidak disarankan dimakan**.")
    else:
        st.error("Jamur ini **tidak bisa dimakan**.")
```

---

### 4. `src/accuracy.py`  
**Fungsi:**  
Script untuk menguji akurasi model pada seluruh data.

**Line-by-line:**
```python
import pandas as pd
import pickle
import os
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, 'rf_mushroom.pkl'), 'rb') as f:
    model = pickle.load(f)
with open(os.path.join(BASE_DIR, 'le_dict.pkl'), 'rb') as f:
    le_dict = pickle.load(f)
# Memuat model dan encoder.

df = pd.read_csv(os.path.join(BASE_DIR, 'agaricus-lepiota-mapped.csv'))
# Membaca data hasil mapping.

# Mapping nama kolom dan memilih kolom yang relevan

for col in df.columns:
    le = le_dict[col]
    df[col] = le.transform(df[col])
# Melakukan encoding pada data.

X = df.drop('kelas', axis=1)
y = df['kelas']

y_pred = model.predict(X)
benar = (y == y_pred).sum()
total = len(y)
akurasi = benar / total * 100
print(f"Akurasi model pada seluruh data: {akurasi:.2f}%")
# Menghitung dan menampilkan akurasi model secara manual dalam persen.
```

---

## 📊 Cara Menjalankan Project

1. **Preprocessing Data**
   ```bash
   python processing.py
   ```
2. **Training Model**
   ```bash
   python training.py
   ```
3. **Menjalankan Aplikasi Streamlit**
   ```bash
   streamlit run src/app.py
   ```
4. **Cek Akurasi Model**
   ```bash
   python src/accuracy.py
   ```

---

## 💡 Catatan

- Semua file `.pkl` dan `.csv` harus berada di folder yang sama dengan script yang menggunakannya, atau gunakan path absolut seperti pada contoh.
- Untuk deployment di Streamlit Cloud, pastikan semua file sudah di-push ke repository.
- Model sudah dipisahkan dari aplikasi agar runtime prediksi sangat cepat.

---

## 📚 Referensi

- [UCI Mushroom Dataset](https://archive.ics.uci.edu/ml/datasets/Mushroom)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [scikit-learn Documentation](https://scikit-learn.org/)

---

**Selamat mencoba dan semoga