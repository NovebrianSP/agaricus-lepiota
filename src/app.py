import pandas as pd
import pickle
import streamlit as st

# Load the model and label encoders
with open('rf_mushroom.pkl', 'rb') as f:
    model = pickle.load(f)

with open('le_dict.pkl', 'rb') as f:
    le_dict = pickle.load(f)

# Load the dataset and prepare columns
df_map = pd.read_csv('agaricus-lepiota-mapped.csv')
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
df_map.rename(columns=column_mapping, inplace=True)

# Pilih hanya fitur penting
selected_cols = ['bau', 'warna_spora', 'warna_insang', 'ukuran_insang', 'memar', 'populasi', 'habitat', 'kelas']
df_map = df_map[selected_cols]
columns = [col for col in df_map.columns if col != 'kelas']

# Sidebar untuk navigasi halaman
st.sidebar.title("Menu")
page = st.sidebar.radio("Pilih Halaman", ("Dashboard", "Rekomendasi"))

if page == "Dashboard":
    st.title("Dashboard Jamur")
    # Tampilkan distribusi fitur penting dengan nama kolom yang lebih rapi
    for col in ['bau', 'warna_spora']:
        label = col.replace('_', ' ').title()
        st.subheader(f"Distribusi Jumlah Jamur Berdasarkan {label}")
        st.bar_chart(df_map[col].value_counts())
    st.markdown("---")
    st.write("Contoh data:")
    # Ubah nama kolom pada tabel contoh data
    df_display = df_map.copy()
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