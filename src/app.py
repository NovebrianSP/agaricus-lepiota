import pandas as pd
import streamlit as st
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Set page config
st.set_page_config(
    page_title="🍄 Mushroom Classification System",
    page_icon="🍄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling with theme support
st.markdown("""
    <style>
    /* Base theme detection and variables */
    :root {
        --text-primary: #1f2937;
        --text-secondary: #6b7280;
        --bg-primary: #ffffff;
        --bg-secondary: #f9fafb;
        --bg-card: #ffffff;
        --border-color: #e5e7eb;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --error-color: #ef4444;
        --accent-color: #3b82f6;
    }

    /* Dark theme variables */
    @media (prefers-color-scheme: dark) {
        :root {
            --text-primary: #f9fafb;
            --text-secondary: #d1d5db;
            --bg-primary: #111827;
            --bg-secondary: #1f2937;
            --bg-card: #374151;
            --border-color: #4b5563;
            --success-color: #34d399;
            --warning-color: #fbbf24;
            --error-color: #f87171;
            --accent-color: #60a5fa;
        }
    }

    /* Streamlit dark theme detection */
    .stApp[data-theme="dark"] {
        --text-primary: #f9fafb;
        --text-secondary: #d1d5db;
        --bg-primary: #111827;
        --bg-secondary: #1f2937;
        --bg-card: #374151;
        --border-color: #4b5563;
        --success-color: #34d399;
        --warning-color: #fbbf24;
        --error-color: #f87171;
        --accent-color: #60a5fa;
    }

    .stApp[data-theme="light"] {
        --text-primary: #1f2937;
        --text-secondary: #6b7280;
        --bg-primary: #ffffff;
        --bg-secondary: #f9fafb;
        --bg-card: #ffffff;
        --border-color: #e5e7eb;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --error-color: #ef4444;
        --accent-color: #3b82f6;
    }

    /* Adaptive styling using CSS variables */
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: var(--success-color);
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }

    .sub-header {
        font-size: 1.5rem;
        color: var(--success-color);
        margin-bottom: 1rem;
        border-left: 4px solid var(--success-color);
        padding-left: 1rem;
    }

    .metric-card {
        background: linear-gradient(135deg, var(--accent-color) 0%, var(--success-color) 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        border: 1px solid var(--border-color);
    }

    .info-box {
        background-color: var(--bg-card);
        color: var(--text-primary);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid var(--success-color);
        border: 1px solid var(--border-color);
        margin: 1rem 0;
    }

    .info-box h4 {
        color: var(--text-primary);
        margin-top: 0;
    }

    .info-box p {
        color: var(--text-secondary);
        margin-bottom: 0.5rem;
    }

    .spore-gallery {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 1rem;
        margin: 1rem 0;
    }

    .spore-card {
        background: var(--bg-card);
        color: var(--text-primary);
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid var(--border-color);
        text-align: center;
        width: 200px;
    }

    .prediction-result {
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.2rem;
        border: 2px solid var(--border-color);
    }

    /* Custom styling for image placeholders */
    .image-placeholder {
        height: 200px;
        background: var(--bg-secondary);
        color: var(--text-secondary);
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 10px;
        border: 2px dashed var(--border-color);
        margin-bottom: 1rem;
    }

    .image-placeholder h3 {
        color: var(--text-secondary);
    }

    .image-placeholder p {
        color: var(--text-secondary);
    }

    .image-placeholder small {
        color: var(--text-secondary);
        opacity: 0.7;
    }

    /* Responsive design */
    @media (max-width: 600px) {
        .stDataFrame, .stTable {font-size: 0.7em;}
        .stDataFrame th, .stDataFrame td {padding: 0.1em 0.2em;}
        .block-container {padding-left: 0.5rem; padding-right: 0.5rem;}
        .spore-card {width: 150px;}
        .main-header {font-size: 2rem;}
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: var(--bg-secondary);
    }

    /* Ensure text contrast in all elements */
    .stMarkdown, .stText {
        color: var(--text-primary);
    }

    /* Fix for prediction result backgrounds with better contrast */
    .prediction-result.safe {
        background: linear-gradient(135deg, var(--success-color), #059669);
        color: white;
    }

    .prediction-result.warning {
        background: linear-gradient(135deg, var(--warning-color), #d97706);
        color: white;
    }

    .prediction-result.danger {
        background: linear-gradient(135deg, var(--error-color), #dc2626);
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Load model & encoder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
IMAGES_DIR = os.path.join(PARENT_DIR, 'images')

# Debug: Print paths
print(f"BASE_DIR: {BASE_DIR}")
print(f"PARENT_DIR: {PARENT_DIR}")
print(f"IMAGES_DIR: {IMAGES_DIR}")
print(f"Images directory exists: {os.path.exists(IMAGES_DIR)}")

if os.path.exists(IMAGES_DIR):
    print(f"Files in images directory: {os.listdir(IMAGES_DIR)}")

with open(os.path.join(BASE_DIR, 'rf_mushroom.pkl'), 'rb') as f:
    model = pickle.load(f)
with open(os.path.join(BASE_DIR, 'le_dict.pkl'), 'rb') as f:
    le_dict = pickle.load(f)

# Load dataset untuk tampilan dashboard
try:
    df = pd.read_csv(os.path.join(BASE_DIR, 'agaricus-lepiota-mapped.csv'))
except FileNotFoundError:
    # Fallback to parent directory
    df = pd.read_csv(os.path.join(PARENT_DIR, 'agaricus-lepiota-mapped.csv'))

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
columns = [col for col in df.columns if col != 'kelas']

# Enhanced spore image mapping with fallback
spore_images = {
    'hitam': 'spora-hitam-jamur.jpg',
    'coklat': 'spora-coklat-jamur.jpg',
    'krem': 'spora-krem-jamur.jpg',
    'coklat tua': 'spora-coklattua-jamur.jpg',
    'hijau': 'spora-hijau-jamur.jpg',
    'oranye': 'spora-oranye-jamur.jpg',
    'ungu': 'spora-ungu-jamur.jpg',
    'putih': 'spora-putih-jamur.jpg',
    'kuning': 'spora-kuning-jamur.png'
}

# Function to display image with fallback
def display_spore_image(spore_color, caption="", width=None):
    """Display spore image with fallback to placeholder if image not found"""
    image_file = spore_images.get(spore_color, 'spora-putih-jamur.jpg')
    image_path = os.path.join(IMAGES_DIR, image_file)
    
    if os.path.exists(image_path):
        try:
            if width:
                st.image(image_path, caption=caption, width=width)
            else:
                st.image(image_path, caption=caption, use_container_width=True)
            return True
        except Exception as e:
            st.error(f"Error loading image: {e}")
            return False
    else:
        # Enhanced placeholder with theme-aware styling
        st.markdown(f"""
            <div class="image-placeholder">
                <div style="text-align: center;">
                    <h3 style="color: var(--text-secondary); margin: 0; font-size: 2rem;">🍄</h3>
                    <p style="color: var(--text-primary); margin: 10px 0 5px 0; font-size: 16px; font-weight: bold;">{spore_color.title()}</p>
                    <small style="color: var(--text-secondary); opacity: 0.8;">Gambar tidak tersedia</small>
                </div>
            </div>
        """, unsafe_allow_html=True)
        return False

# Enhanced Sidebar
st.sidebar.markdown("""
    <div style='text-align: center; padding: 1rem; background: linear-gradient(135deg, #2E7D32, #4CAF50); border-radius: 10px; margin-bottom: 1rem;'>
        <h2 style='color: white; margin: 0;'>🍄 Menu Navigasi</h2>
    </div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Pilih Halaman", 
    ("🏠 Dashboard", "🔍 Klasifikasi", "📊 Galeri Spora"),
    index=0
)

# Add sidebar info with debug information
st.sidebar.markdown(f"""
    <div class='info-box'>
        <h4>ℹ️ Informasi Sistem</h4>
        <p><strong>Model:</strong> Random Forest</p>
        <p><strong>Fitur:</strong> 7 karakteristik jamur</p>
        <p><strong>Dataset:</strong> UCI Mushroom</p>
        <p><strong>Akurasi:</strong> >99%</p>
        <hr>
        <small><strong>Debug Info:</strong></small><br>
        <small>Images Dir: {os.path.exists(IMAGES_DIR)}</small><br>
        <small>Data Loaded: {len(df)} rows</small>
    </div>
""", unsafe_allow_html=True)

if page == "🏠 Dashboard":
    # Main Header
    st.markdown('<h1 class="main-header">🍄 Dashboard Klasifikasi Jamur</h1>', unsafe_allow_html=True)
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    total_samples = len(df)
    edible_count = len(df[df['kelas'] == 'bisa dimakan'])
    poisonous_count = len(df[df['kelas'] == 'beracun'])
    
    # Calculate accuracy
    df_encoded = df.copy()
    for col in df_encoded.columns:
        le = le_dict[col]
        df_encoded[col] = le.transform(df_encoded[col])
    X = df_encoded.drop('kelas', axis=1)
    y = df_encoded['kelas']
    y_pred = model.predict(X)
    accuracy = (y == y_pred).sum() / len(y) * 100
    
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <h3>📊 Total Sampel</h3>
                <h2>{total_samples:,}</h2>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <h3>✅ Dapat Dimakan</h3>
                <h2>{edible_count:,}</h2>
                <p>{edible_count/total_samples*100:.1f}%</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <h3>⚠️ Beracun</h3>
                <h2>{poisonous_count:,}</h2>
                <p>{poisonous_count/total_samples*100:.1f}%</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
            <div class="metric-card">
                <h3>🎯 Akurasi Model</h3>
                <h2>{accuracy:.2f}%</h2>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Interactive Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h3 class="sub-header">📈 Distribusi Berdasarkan Bau</h3>', unsafe_allow_html=True)
        bau_counts = df['bau'].value_counts()
        fig_bau = px.pie(
            values=bau_counts.values, 
            names=bau_counts.index,
            title="Distribusi Jamur Berdasarkan Bau",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_bau.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_bau, use_container_width=True)
    
    with col2:
        st.markdown('<h3 class="sub-header">🎨 Distribusi Berdasarkan Warna Spora</h3>', unsafe_allow_html=True)
        spora_counts = df['warna_spora'].value_counts()
        fig_spora = px.bar(
            x=spora_counts.index, 
            y=spora_counts.values,
            title="Distribusi Jamur Berdasarkan Warna Spora",
            color=spora_counts.values,
            color_continuous_scale="Viridis"
        )
        fig_spora.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_spora, use_container_width=True)
    
    # Correlation Matrix
    st.markdown('<h3 class="sub-header">🔗 Matriks Korelasi Fitur</h3>', unsafe_allow_html=True)
    df_corr = df_encoded[columns].corr()
    fig_corr = px.imshow(
        df_corr,
        title="Korelasi Antar Fitur Jamur",
        color_continuous_scale="RdBu"
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Sample Data with Enhanced Display
    st.markdown('<h3 class="sub-header">🔍 Contoh Data Jamur</h3>', unsafe_allow_html=True)
    df_display = df.copy()
    df_display.columns = [c.replace('_', ' ').title() for c in df_display.columns]
    
    # Add color coding for the class
    def color_class(val):
        if val == 'bisa dimakan':
            return 'background-color: #10b981'
        elif val == 'beracun':
            return 'background-color: #ef4444'
        return ''
    
    st.dataframe(
        df_display.head(10).style.applymap(color_class, subset=['Kelas']),
        use_container_width=True
    )

elif page == "📊 Galeri Spora":
    st.markdown('<h1 class="main-header">🎨 Galeri Warna Spora Jamur</h1>', unsafe_allow_html=True)
    
    st.markdown("""
        <div class='info-box'>
            <h4>💡 Tentang Warna Spora</h4>
            <p>Warna spora adalah salah satu karakteristik penting dalam identifikasi jamur. 
            Setiap warna memiliki karakteristik unik yang membantu dalam klasifikasi keamanan jamur.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Debug information
    if not os.path.exists(IMAGES_DIR):
        st.warning(f"⚠️ Folder gambar tidak ditemukan di: {IMAGES_DIR}")
        st.info("💡 Pastikan folder 'images' berada di direktori yang benar dengan file gambar spora.")
    
    # Create gallery layout
    spore_colors = df['warna_spora'].unique()
    spore_counts = df['warna_spora'].value_counts()
    
    cols = st.columns(3)
    col_idx = 0
    
    for spore_color in sorted(spore_colors):
        count = spore_counts[spore_color]
        percentage = count / len(df) * 100
        
        with cols[col_idx % 3]:
            # Use the enhanced display function
            display_spore_image(spore_color, f"Spora {spore_color.title()}")
            
            st.markdown(f"""
                <div class="metric-card">
                    <h4>{spore_color.title()}</h4>
                    <p><strong>{count}</strong> sampel ({percentage:.1f}%)</p>
                </div>
            """, unsafe_allow_html=True)
            
        col_idx += 1
    
    # Distribution chart for spore colors
    st.markdown('<h3 class="sub-header">📊 Distribusi Warna Spora</h3>', unsafe_allow_html=True)
    fig_spore_dist = px.sunburst(
        df,
        path=['kelas', 'warna_spora'],
        title="Distribusi Warna Spora berdasarkan Kelas Jamur"
    )
    st.plotly_chart(fig_spore_dist, use_container_width=True)

elif page == "🔍 Klasifikasi":
    st.markdown('<h1 class="main-header">🔍 Sistem Klasifikasi Jamur</h1>', unsafe_allow_html=True)
    
    st.markdown("""
        <div class='info-box'>
            <h4>🎯 Cara Menggunakan</h4>
            <p>Pilih karakteristik jamur yang ingin Anda klasifikasi menggunakan dropdown di bawah ini. 
            Sistem akan memberikan prediksi keamanan jamur berdasarkan model Random Forest yang telah dilatih.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Enhanced input form
    st.markdown('<h3 class="sub-header">📝 Masukkan Karakteristik Jamur</h3>', unsafe_allow_html=True)
    
    input_features = {}
    cols_input = st.columns(2)
    
    feature_descriptions = {
        'bau': '👃 Aroma jamur yang tercium',
        'warna_spora': '🎨 Warna serbuk spora',
        'warna_insang': '🌈 Warna bagian insang',
        'ukuran_insang': '📏 Ukuran insang jamur',
        'memar': '🔵 Apakah jamur memar saat ditekan',
        'populasi': '👥 Pola pertumbuhan jamur',
        'habitat': '🌲 Lingkungan tempat tumbuh'
    }
    
    for idx, col in enumerate(columns):
        label = col.replace('_', ' ').title()
        description = feature_descriptions.get(col, '')
        options = le_dict[col].classes_
        
        with cols_input[idx % 2]:
            st.markdown(f"**{description}**")
            input_features[col] = st.selectbox(
                f"{label}", 
                options, 
                key=col,
                help=f"Pilih {label.lower()} yang sesuai dengan jamur yang diamati"
            )
    
    # Enhanced prediction button
    if st.button('🔮 Prediksi Keamanan Jamur', type="primary"):
        with st.spinner('🔄 Menganalisis karakteristik jamur...'):
            input_data = []
            for col in columns:
                le = le_dict[col]
                val_enc = le.transform([input_features[col]])[0]
                input_data.append(val_enc)
            
            input_df = pd.DataFrame([input_data], columns=columns)
            proba = model.predict_proba(input_df)[0]
            class_le = le_dict['kelas']
            class_names = class_le.inverse_transform([0, 1])
            edible_score = proba[class_names.tolist().index('bisa dimakan')] * 10
            poisonous_score = proba[class_names.tolist().index('beracun')] * 10
            
            # Enhanced result display
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Skor Prediksi")
                st.progress(edible_score / 10)
                st.markdown(f"**Skor Dapat Dimakan:** {edible_score:.2f}/10")
                st.progress(poisonous_score / 10)
                st.markdown(f"**Skor Beracun:** {poisonous_score:.2f}/10")
            
            with col2:
                # Show spore image if available
                if 'warna_spora' in input_features:
                    spore_color = input_features['warna_spora']
                    st.markdown("### 🎨 Warna Spora")
                    display_spore_image(spore_color, f"Spora {spore_color.title()}", width=200)
            
            # Enhanced result display with recommendations
            if edible_score >= 8.0:
                st.markdown(f"""
                    <div class="prediction-result" style="background: linear-gradient(135deg, #c8e6c9, #a5d6a7);">
                        <h2>✅ DAPAT DIMAKAN</h2>
                        <p><strong>Skor Keamanan: {edible_score:.2f}/10</strong></p>
                        <p>Jamur ini memiliki karakteristik yang menunjukkan aman untuk dikonsumsi.</p>
                        <p><em>💡 Rekomendasi: Tetap berhati-hati dan pastikan identifikasi yang akurat sebelum mengonsumsi jamur liar.</em></p>
                    </div>
                """, unsafe_allow_html=True)
            elif 6.0 <= edible_score < 8.0:
                st.markdown(f"""
                    <div class="prediction-result" style="background: linear-gradient(135deg, #fff3cd, #ffeaa7);">
                        <h2>⚠️ TIDAK DISARANKAN</h2>
                        <p><strong>Skor Keamanan: {edible_score:.2f}/10</strong></p>
                        <p>Jamur ini memiliki karakteristik yang ambigu dan tidak disarankan untuk dikonsumsi.</p>
                        <p><em>💡 Rekomendasi: Hindari mengonsumsi jamur ini karena tingkat kepastian yang rendah.</em></p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="prediction-result" style="background: linear-gradient(135deg, #ffcdd2, #ef9a9a);">
                        <h2>❌ TIDAK BISA DIMAKAN</h2>
                        <p><strong>Skor Keamanan: {edible_score:.2f}/10</strong></p>
                        <p>Jamur ini memiliki karakteristik yang menunjukkan kemungkinan besar beracun.</p>
                        <p><em>💡 Rekomendasi: JANGAN mengonsumsi jamur ini dalam keadaan apapun!</em></p>
                    </div>
                """, unsafe_allow_html=True)
            
            # Feature importance for this prediction
            st.markdown("### 📋 Ringkasan Input")
            input_summary = pd.DataFrame({
                'Karakteristik': [feature_descriptions[col] for col in columns],
                'Nilai': [input_features[col] for col in columns]
            })
            st.dataframe(input_summary, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: var(--text-secondary); padding: 2rem; background: var(--bg-secondary); border-radius: 10px; margin-top: 2rem;'>
        <p style='margin: 0; font-size: 1.1rem;'>
            🍄 <strong style='color: var(--success-color);'>Mushroom Classification System</strong> | 
            <span style='color: var(--accent-color);'>Powered by Random Forest & Streamlit</span>
        </p>
        <p style='margin: 10px 0 0 0; font-style: italic; color: var(--text-secondary);'>
            <em>⚠️ Selalu konsultasi dengan ahli mikologi untuk identifikasi jamur liar!</em>
        </p>
    </div>
""", unsafe_allow_html=True)