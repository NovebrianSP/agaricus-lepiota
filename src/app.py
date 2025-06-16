import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# CSS untuk responsif dan tabel compact
st.markdown("""
    <style>
    /* Tabel responsif dan font kecil */
    .stDataFrame, .stTable {font-size: 0.85em;}
    .stDataFrame th, .stDataFrame td {padding: 0.2em 0.5em;}
    @media (max-width: 600px) {
        .stDataFrame, .stTable {font-size: 0.7em;}
        .stDataFrame th, .stDataFrame td {padding: 0.1em 0.2em;}
        .block-container {padding-left: 0.5rem; padding-right: 0.5rem;}
    }
    </style>
""", unsafe_allow_html=True)

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
    for col in ['bau', 'warna_spora']:
        label = col.replace('_', ' ').title()
        value_counts = df[col].value_counts().sort_index()
        labels = le_dict[col].inverse_transform(value_counts.index)
        value_counts.index = labels
        st.subheader(f"Distribusi Jamur: {label}")
        st.bar_chart(value_counts)
    st.markdown("---")
    st.write("Contoh Data:")
    # Tampilkan kata asli pada tabel contoh data
    df_display = df.copy()
    for col in df_display.columns:
        if col in le_dict:
            df_display[col] = le_dict[col].inverse_transform(df_display[col])
    df_display.columns = [c.replace('_', ' ').title() for c in df_display.columns]
    st.dataframe(df_display.head(), use_container_width=True)

elif page == "Rekomendasi":
    st.title("Klasifikasi Jamur")
    st.subheader("Masukkan Fitur Jamur")

    # User input responsif: 2 kolom per baris di desktop, 1 kolom di mobile
    input_features = {}
    n_cols = 2 if st.columns(2)[0].width > 300 else 1
    input_cols = st.columns(n_cols)
    for idx, col in enumerate(columns):
        label = col.replace('_', ' ').title()
        options = le_dict[col].classes_
        with input_cols[idx % n_cols]:
            input_features[col] = st.selectbox(f"{label}", options, key=col)

    # Prediction
    if st.button('Prediksi'):
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
                <div style="padding:12px;border-radius:8px;background:{edible_color};text-align:center;">
                    <b>Bisa Dimakan</b><br>
                    <span style="font-size:1.2em;">{edible_score:.2f} / 10</span>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                f"""
                <div style="padding:12px;border-radius:8px;background:{poisonous_color};text-align:center;">
                    <b>Beracun</b><br>
                    <span style="font-size:1.2em;">{poisonous_score:.2f} / 10</span>
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