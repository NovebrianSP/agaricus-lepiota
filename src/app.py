import pandas as pd
import streamlit as st
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
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

columns = [col for col in df.columns if col != 'kelas']

# Sidebar untuk navigasi halaman
st.sidebar.title("Menu")
page = st.sidebar.radio("Pilih Halaman", ("Dashboard", "Rekomendasi"))

if page == "Dashboard":
    st.title("Dashboard Jamur")
    # Tampilkan distribusi fitur penting dengan nama kolom yang lebih rapi
    for col in ['bau', 'warna_spora']:
        label = col.replace('_', ' ').title()
        # Ambil value_counts dan mapping ke label asli
        value_counts = df[col].value_counts().sort_index()
        labels = le_dict[col].inverse_transform(value_counts.index)
        value_counts.index = labels
        st.subheader(f"Distribusi Jumlah Jamur Berdasarkan {label}")
        st.bar_chart(value_counts)
    st.markdown("---")
    st.write("Contoh data:")
    # Ubah nama kolom pada tabel contoh data
    df_display = df.copy()
    df_display.columns = [c.replace('_', ' ').title() for c in df_display.columns]
    st.dataframe(df_display.head())
    
elif page == "Rekomendasi":
    st.title("Mushroom Classification App")
    st.subheader("Masukkan Fitur Jamur untuk Prediksi")

    # User input
    input_features = {}
    for col in columns:
        label = col.replace('_', ' ').title()
        options = le_dict[col].classes_
        input_features[col] = st.selectbox(f"Pilih {label}", options)

    # Prediction
    if st.button('Predict'):
        input_data = []
        for col in columns:
            le = le_dict[col]
            val_enc = le.transform([input_features[col]])[0]
            input_data.append(val_enc)
        
        input_df = pd.DataFrame([input_data], columns=columns)
        proba = model.predict_proba(input_df)[0]
        class_le = le_dict['kelas']
        class_names = class_le.inverse_transform([0, 1])

        # Skor 0-10
        edible_score = proba[class_names.tolist().index('bisa dimakan')] * 10
        poisonous_score = proba[class_names.tolist().index('beracun')] * 10

        # Warna dinamis
        edible_color = f"rgba(0, 200, 0, {edible_score/10:.2f})"
        poisonous_color = f"rgba(200, 0, 0, {poisonous_score/10:.2f})"

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"""
                <div style="padding:16px;border-radius:8px;background:{edible_color};text-align:center;">
                    <b>Bisa Dimakan</b><br>
                    <span style="font-size:1.5em;">{edible_score:.2f} / 10</span>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                f"""
                <div style="padding:16px;border-radius:8px;background:{poisonous_color};text-align:center;">
                    <b>Beracun</b><br>
                    <span style="font-size:1.5em;">{poisonous_score:.2f} / 10</span>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Hasil utama
        pred = model.predict(input_df)[0]
        prediction = class_le.inverse_transform([pred])[0]
        if prediction == 'beracun':
            st.error(f'Prediksi kelas jamur: {prediction}')
        else:
            st.success(f'Prediksi kelas jamur: {prediction}')