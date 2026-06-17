import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, MinMaxScaler, label_binarize
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Dashboard Sosial-Ekonomi Indonesia 2021",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Premium untuk tampilan Glassmorphism & Gradient modern
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    /* Judul Utama */
    .title-gradient {
        background: linear-gradient(135deg, #FF4B4B, #FF8F00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.6rem;
        font-weight: 800;
        margin-bottom: 5px;
    }
    
    .subtitle {
        color: #A0A0A0;
        font-size: 1.1rem;
        margin-bottom: 25px;
    }
    
    /* Card Container */
    .dashboard-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .dashboard-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 20px rgba(0, 0, 0, 0.15);
    }

    /* Premium Metric Card */
    .metric-card {
        background: var(--secondary-background-color) !important;
        border: 1px solid rgba(128, 128, 128, 0.2) !important;
        border-radius: 12px !important;
        padding: 16px 14px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05) !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 155px;
    }
    .metric-card:hover {
        transform: translateY(-3px) !important;
        border-color: rgba(128, 128, 128, 0.4) !important;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1) !important;
    }
    .metric-label {
        font-size: 0.78rem !important;
        color: var(--text-color) !important;
        opacity: 0.7 !important;
        font-weight: 600 !important;
        margin-bottom: 8px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.8px !important;
        line-height: 1.2 !important;
    }
    .metric-value {
        font-size: 1.4rem !important;
        font-weight: 700 !important;
        color: var(--text-color) !important;
        margin-bottom: 10px !important;
        font-family: 'Outfit', sans-serif !important;
        white-space: normal !important;
        word-wrap: break-word !important;
        line-height: 1.2 !important;
    }
    .metric-comparison {
        display: flex !important;
        flex-direction: column !important;
        gap: 6px !important;
    }
    .comp-badge {
        font-size: 0.72rem !important;
        padding: 3px 8px !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        display: inline-flex !important;
        align-items: center !important;
        width: fit-content !important;
        letter-spacing: 0.2px !important;
    }
    .comp-badge.better {
        background-color: rgba(46, 204, 113, 0.12) !important;
        color: #2ECC71 !important;
        border: 1px solid rgba(46, 204, 113, 0.2) !important;
    }
    .comp-badge.worse {
        background-color: rgba(231, 76, 60, 0.12) !important;
        color: #E74C3C !important;
        border: 1px solid rgba(231, 76, 60, 0.2) !important;
    }
    .comp-badge.warning {
        background-color: rgba(243, 156, 18, 0.12) !important;
        color: #E67E22 !important;
        border: 1px solid rgba(243, 156, 18, 0.2) !important;
    }
</style>
""", unsafe_allow_html=True)

# Load data dengan caching
@st.cache_data
def load_data():
    return pd.read_csv("socio_economic_final.csv")

try:
    df = load_data()
except Exception as e:
    st.error(f"Gagal memuat dataset: {e}")
    st.info("Pastikan file 'socio_economic_final.csv' berada di direktori yang sama dengan dashboard.py.")
    st.stop()

# Definisi fitur
features_cluster = ["poorpeople_percentage", "reg_gdp", "life_exp", "avg_schooltime", "exp_percap"]
features_model = ["poorpeople_percentage", "avg_schooltime", "life_exp", "exp_percap"]

# Cache modeling K-Means untuk performa cepat
@st.cache_resource
def run_kmeans_model(dataframe):
    X_cluster = dataframe[features_cluster]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_result = dataframe.copy()
    df_result["cluster"] = kmeans.fit_predict(X_scaled)
    
    # Pelabelan dinamis berdasarkan rata-rata exp_percap
    cluster_mean_exp = df_result.groupby("cluster")["exp_percap"].mean().sort_values()
    label_order = ["Pembangunan Rendah", "Pembangunan Sedang", "Pembangunan Tinggi"]
    cluster_label_map = {cluster_id: label for cluster_id, label in zip(cluster_mean_exp.index, label_order)}
    df_result["label_cluster"] = df_result["cluster"].map(cluster_label_map)
    df_result["label_cluster"] = pd.Categorical(
        df_result["label_cluster"],
        categories=["Pembangunan Rendah", "Pembangunan Sedang", "Pembangunan Tinggi"],
        ordered=True
    )
    
    # Hitung centroid
    centroids_scaled = kmeans.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids_scaled)
    centroids_df = pd.DataFrame(centroids_original, columns=features_cluster)
    centroids_df["cluster"] = range(3)
    centroids_df["label_cluster"] = centroids_df["cluster"].map(cluster_label_map)
    
    return df_result, centroids_df

df_clustered, centroids_df = run_kmeans_model(df)

# Cache modeling Random Forest
@st.cache_resource
def train_rf_model(dataframe):
    X = dataframe[features_model]
    y = dataframe["target_kategori_pembangunan"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    return rf, X_train, X_test, y_train, y_test

rf_model, X_train, X_test, y_train, y_test = train_rf_model(df)

def create_custom_metric(label, value, diff_nat, diff_prov, is_poverty=False):
    # Untuk kemiskinan, nilai lebih rendah = lebih baik (is_poverty=True)
    # Untuk yang lainnya, nilai lebih tinggi = lebih baik (is_poverty=False)
    
    # Check Nas
    if diff_nat < 0:
        nat_class = "better" if is_poverty else "worse"
        nat_symbol = "↓"
    else:
        nat_class = "worse" if is_poverty else "better"
        nat_symbol = "↑"
        
    # Check Prov
    if diff_prov < 0:
        prov_class = "better" if is_poverty else "worse"
        prov_symbol = "↓"
    else:
        prov_class = "worse" if is_poverty else "better"
        prov_symbol = "↑"
        
    # Format diff strings
    if label == "Pengeluaran per Kapita":
        nat_diff_str = f"Rp {abs(diff_nat):,.0f}"
        prov_diff_str = f"Rp {abs(diff_prov):,.0f}"
    elif label == "Persentase Kemiskinan":
        nat_diff_str = f"{abs(diff_nat):.2f}%"
        prov_diff_str = f"{abs(diff_prov):.2f}%"
    elif label == "PDRB (Regional GDP)":
        nat_diff_str = f"{abs(diff_nat):.2f}"
        prov_diff_str = f"{abs(diff_prov):.2f}"
    else:
        nat_diff_str = f"{abs(diff_nat):.2f} Thn"
        prov_diff_str = f"{abs(diff_prov):.2f} Thn"

    html = f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-comparison">
            <span class="comp-badge {nat_class}">{nat_symbol} {nat_diff_str} vs Nas</span>
            <span class="comp-badge {prov_class}">{prov_symbol} {prov_diff_str} vs Prov</span>
        </div>
    </div>
    """
    return html

# SIDEBAR NAVIGASI
st.sidebar.image("https://img.icons8.com/clouds/200/dashboard.png", width=100)
st.sidebar.markdown("### **Navigasi Menu**")
page = st.sidebar.radio(
    "Pilih Halaman Analisis:",
    [
        "📊 Overview & Filter Wilayah",
        "🔮 Predictor"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Kelompok Data Mining**  
    *Tugas Besar: Analisis Sosial-Ekonomi Indonesia 2021*
    """
)

# ----------------- HALAMAN 1: OVERVIEW & FILTER WILAYAH -----------------
if page == "📊 Overview & Filter Wilayah":
    st.markdown("<div class='title-gradient'>Dashboard Profil Sosial-Ekonomi Indonesia</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Analisis Profil Wilayah & Parameter Pembangunan Utama (Tahun 2021)</div>", unsafe_allow_html=True)
    
    # 1. Ringkasan Statistik Nasional (Collapsible)
    with st.expander("📊 Lihat Ringkasan Statistik Nasional (Tahun 2021)"):
        col_kpi1, col_kpi2, col_kpi3, col_kpi4, col_kpi5 = st.columns(5)
        
        with col_kpi1:
            st.metric("Total Wilayah", f"{len(df)} Daerah", help="Kabupaten dan Kota di Indonesia")
        with col_kpi2:
            st.metric("Rerata Kemiskinan", f"{df['poorpeople_percentage'].mean():.2f}%", help="Persentase Penduduk Miskin")
        with col_kpi3:
            st.metric("Rerata Pengeluaran", f"Rp {df['exp_percap'].mean():,.0f}", help="Pengeluaran per Kapita disesuaikan (ribu rupiah)")
        with col_kpi4:
            st.metric("Harapan Hidup", f"{df['life_exp'].mean():.2f} Thn", help="Rata-rata Angka Harapan Hidup")
        with col_kpi5:
            st.metric("Lama Sekolah", f"{df['avg_schooltime'].mean():.2f} Thn", help="Rata-rata Lama Sekolah")
            
    st.markdown("---")
    
    # 2. Tabs: Nasional vs Detail Wilayah
    tab1, tab2 = st.tabs(["🗺️ Distribusi Pembangunan Nasional", "📌 Detail Profil per Wilayah"])
    
    with tab1:
        df_tree = df_clustered.copy()
        df_tree["const"] = 1
        
        # Convert Categorical columns to string to prevent Plotly parent aggregation crashes
        for col in df_tree.columns:
            if isinstance(df_tree[col].dtype, pd.CategoricalDtype):
                df_tree[col] = df_tree[col].astype(str)
        
        fig_tree = px.treemap(
            df_tree,
            path=["province", "cities_reg"],
            values="exp_percap",
            color="skor_pembangunan",
            color_continuous_scale="RdYlGn",
            title="Treemap Distribusi Pembangunan Indonesia (Ukuran: Pengeluaran, Warna: Skor Pembangunan)",
            hover_data={
                "skor_pembangunan": ":.4f",
                "label_cluster": True,
                "poorpeople_percentage": ":.2f%",
                "exp_percap": ":,d",
                "life_exp": ":.2f",
                "avg_schooltime": ":.2f"
            }
        )
        fig_tree.update_layout(margin=dict(t=50, l=10, r=10, b=10), height=550)
        st.plotly_chart(fig_tree, use_container_width=True)

        
    with tab2:
        # 2. Pemilihan Wilayah (Cascading Filter)
        st.markdown("### 🗺️ Pilih Wilayah Analisis")
        col_sel1, col_sel2 = st.columns(2)

        with col_sel1:
            provinces = sorted(df["province"].unique())
            # Tentukan default index ke Jawa Barat jika ada untuk visual awal yang lebih familiar, jika tidak index 0
            default_prov_idx = provinces.index("Jawa Barat") if "Jawa Barat" in provinces else 0
            selected_province = st.selectbox("Pilih Provinsi:", provinces, index=default_prov_idx)

        with col_sel2:
            # Filter kabupaten/kota berdasarkan provinsi terpilih
            available_cities = sorted(df[df["province"] == selected_province]["cities_reg"].unique())
            selected_city = st.selectbox("Pilih Kabupaten/Kota:", available_cities)

        # Ambil data wilayah terpilih
        city_data = df[df["cities_reg"] == selected_city].iloc[0]
        city_province = city_data["province"]

        # Hitung rata-rata provinsi & nasional
        df_prov = df[df["province"] == city_province]
        prov_avg = df_prov.mean(numeric_only=True)
        national_avg = df.mean(numeric_only=True)

        st.markdown("---")

        # 3. Overview Wilayah & Parameter Utama Pembangunan
        st.markdown(f"## 📌 Overview Wilayah: {selected_city} (Provinsi {city_province})")

        st.markdown("#### 🏆 Parameter Utama Pembangunan")
        col_p1, col_p2, col_p3 = st.columns(3)

        kat_pembangunan = city_data["target_kategori_pembangunan"]
        kluster_pembangunan = city_data["label_cluster"]
        skor_pem = city_data["skor_pembangunan"]
        skor_nat_diff = skor_pem - national_avg["skor_pembangunan"]

        # 1) Skor Pembangunan Card HTML
        skor_badge_class = "better" if skor_nat_diff >= 0 else "worse"
        skor_badge_symbol = "↑" if skor_nat_diff >= 0 else "↓"
        html_skor = f"""
        <div class="metric-card">
            <div class="metric-label">Skor Pembangunan</div>
            <div class="metric-value">{skor_pem:.4f}</div>
            <div class="metric-comparison">
                <span class="comp-badge {skor_badge_class}">{skor_badge_symbol} {abs(skor_nat_diff):.4f} vs Nas</span>
            </div>
        </div>
        """

        # 2) Kategori Pembangunan Card HTML
        kat_badge_class = "better" if kat_pembangunan == "Tinggi" else ("warning" if kat_pembangunan == "Sedang" else "worse")
        html_kat = f"""
        <div class="metric-card">
            <div class="metric-label">Kategori Pembangunan (RF)</div>
            <div class="metric-value">{kat_pembangunan}</div>
            <div class="metric-comparison">
                <span class="comp-badge {kat_badge_class}">Random Forest</span>
            </div>
        </div>
        """

        # 3) Kluster K-Means Card HTML
        klust_badge_class = "better" if "Tinggi" in kluster_pembangunan else ("warning" if "Sedang" in kluster_pembangunan else "worse")
        html_klust = f"""
        <div class="metric-card">
            <div class="metric-label">Kluster K-Means</div>
            <div class="metric-value">{kluster_pembangunan}</div>
            <div class="metric-comparison">
                <span class="comp-badge {klust_badge_class}">K-Means Segmentasi</span>
            </div>
        </div>
        """

        with col_p1:
            st.markdown(html_skor, unsafe_allow_html=True)
        with col_p2:
            st.markdown(html_kat, unsafe_allow_html=True)
        with col_p3:
            st.markdown(html_klust, unsafe_allow_html=True)

        st.markdown("---")

        # 4. Detail Indikator Sosial-Ekonomi
        st.markdown("#### 📊 Detail Indikator Pembangunan")
        col_d1, col_d2, col_d3, col_d4, col_d5 = st.columns(5)

        with col_d1:
            val_pov = city_data["poorpeople_percentage"]
            diff_pov_nat = val_pov - national_avg["poorpeople_percentage"]
            diff_pov_prov = val_pov - prov_avg["poorpeople_percentage"]
            html_pov = create_custom_metric("Persentase Kemiskinan", f"{val_pov:.2f}%", diff_pov_nat, diff_pov_prov, is_poverty=True)
            st.markdown(html_pov, unsafe_allow_html=True)

        with col_d2:
            val_exp = city_data["exp_percap"]
            diff_exp_nat = val_exp - national_avg["exp_percap"]
            diff_exp_prov = val_exp - prov_avg["exp_percap"]
            html_exp = create_custom_metric("Pengeluaran per Kapita", f"Rp {val_exp:,.0f}", diff_exp_nat, diff_exp_prov, is_poverty=False)
            st.markdown(html_exp, unsafe_allow_html=True)

        with col_d3:
            val_life = city_data["life_exp"]
            diff_life_nat = val_life - national_avg["life_exp"]
            diff_life_prov = val_life - prov_avg["life_exp"]
            html_life = create_custom_metric("Angka Harapan Hidup", f"{val_life:.2f} Tahun", diff_life_nat, diff_life_prov, is_poverty=False)
            st.markdown(html_life, unsafe_allow_html=True)

        with col_d4:
            val_school = city_data["avg_schooltime"]
            diff_school_nat = val_school - national_avg["avg_schooltime"]
            diff_school_prov = val_school - prov_avg["avg_schooltime"]
            html_school = create_custom_metric("Rata-rata Lama Sekolah", f"{val_school:.2f} Tahun", diff_school_nat, diff_school_prov, is_poverty=False)
            st.markdown(html_school, unsafe_allow_html=True)

        with col_d5:
            val_gdp = city_data["reg_gdp"]
            diff_gdp_nat = val_gdp - national_avg["reg_gdp"]
            diff_gdp_prov = val_gdp - prov_avg["reg_gdp"]
            html_gdp = create_custom_metric("PDRB (Regional GDP)", f"{val_gdp:.2f}", diff_gdp_nat, diff_gdp_prov, is_poverty=False)
            st.markdown(html_gdp, unsafe_allow_html=True)

        st.markdown("---")

        # 5. Visualisasi Analisis Komparatif
        st.markdown(f"#### 📊 Perbandingan Indikator Se-Provinsi {city_province}")

        # Pilihan Indikator untuk Bar Chart
        selected_ind = st.selectbox(
            "Pilih Indikator untuk Grafik Perbandingan:",
            [
                ("poorpeople_percentage", "Persentase Kemiskinan (%)"),
                ("exp_percap", "Pengeluaran Per Kapita (Rp)"),
                ("life_exp", "Angka Harapan Hidup (Tahun)"),
                ("avg_schooltime", "Rata-rata Lama Sekolah (Tahun)"),
                ("reg_gdp", "PDRB (Regional GDP)")
            ],
            format_func=lambda x: x[1],
            key="overview_bar_indicator"
        )

        # Plot Perbandingan Bar Chart
        df_prov_sorted = df_prov.sort_values(by=selected_ind[0], ascending=True)

        # Berikan warna pembeda untuk kota terpilih
        colors_bar = []
        for city in df_prov_sorted["cities_reg"]:
            if city == selected_city:
                colors_bar.append("#FF4B4B")  # Warna merah untuk yang terpilih
            else:
                colors_bar.append("#5A9EED")  # Warna biru default

        fig_bar = px.bar(
            df_prov_sorted,
            x=selected_ind[0],
            y="cities_reg",
            orientation='h',
            title=f"Perbandingan {selected_ind[1]} di Provinsi {city_province}",
            labels={selected_ind[0]: selected_ind[1], "cities_reg": "Kabupaten/Kota"}
        )
        # Update warna bar
        fig_bar.update_traces(marker_color=colors_bar)
        fig_bar.update_layout(height=450, margin=dict(l=150))
        st.plotly_chart(fig_bar, use_container_width=True)

        # 6. Data Tabel Kabupaten/Kota di Provinsi Terpilih
        st.markdown(f"#### 📋 Peringkat Kinerja Pembangunan se-Provinsi {city_province}")
        st.markdown("Tabel di bawah ini diurutkan berdasarkan **Skor Pembangunan** dari yang tertinggi (Top 1) hingga terendah:")

        # Urutkan berdasarkan skor pembangunan secara descending
        df_prov_ranked = df_prov.sort_values(by="skor_pembangunan", ascending=False).copy()
        # Tambahkan kolom Peringkat (Top 1, 2, 3...)
        df_prov_ranked.insert(0, "Peringkat", range(1, len(df_prov_ranked) + 1))

        # Fungsi styling untuk highlight kota terpilih
        def highlight_selected(row):
            if row["cities_reg"] == selected_city:
                return ["background-color: rgba(255, 75, 75, 0.15); font-weight: bold;"] * len(row)
            return [""] * len(row)

        styled_df = df_prov_ranked[[
            "Peringkat", "cities_reg", "skor_pembangunan", "poorpeople_percentage", 
            "exp_percap", "life_exp", "avg_schooltime", "reg_gdp", "label_cluster", "target_kategori_pembangunan"
        ]].style.apply(highlight_selected, axis=1)

        st.dataframe(
            styled_df,
            column_config={
                "Peringkat": st.column_config.NumberColumn("🏆 Rank", format="%d", help="Peringkat daerah berdasarkan Skor Pembangunan"),
                "cities_reg": st.column_config.TextColumn("Kabupaten/Kota"),
                "skor_pembangunan": st.column_config.NumberColumn("Skor Pembangunan", format="%.4f", help="Skor gabungan indeks pembangunan (0-1)"),
                "poorpeople_percentage": st.column_config.NumberColumn("Kemiskinan (%)", format="%.2f%%"),
                "exp_percap": st.column_config.NumberColumn("Pengeluaran per Kapita", format="Rp %d"),
                "life_exp": st.column_config.NumberColumn("Harapan Hidup (Thn)", format="%.2f"),
                "avg_schooltime": st.column_config.NumberColumn("Lama Sekolah (Thn)", format="%.2f"),
                "reg_gdp": st.column_config.NumberColumn("PDRB (GDP)", format="%.2f"),
                "label_cluster": st.column_config.TextColumn("Kluster K-Means"),
                "target_kategori_pembangunan": st.column_config.TextColumn("Kategori (RF)")
            },
            use_container_width=True,
            hide_index=True
        )
elif page == "🔮 Predictor":
    st.markdown("<div class='title-gradient'>Predictor Kategori Pembangunan</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Simulasi Prediksi Kategori Pembangunan Daerah Baru</div>", unsafe_allow_html=True)
    
    st.markdown("Masukkan nilai indikator di bawah ini untuk melihat hasil klasifikasi prediksi kategori pembangunan secara langsung:")
    
    # Dapatkan nilai rata-rata, min, maks dari dataset asli untuk batasan slider
    desc = df[features_model].describe()
    
    col_inp1, col_inp2 = st.columns(2)
    
    with col_inp1:
        st.markdown("#### 📝 Input Indikator Sosial-Ekonomi")
        inp_poverty = st.slider(
            "Persentase Penduduk Miskin (%)",
            min_value=0.0,
            max_value=40.0,
            value=float(desc.loc["mean", "poorpeople_percentage"]),
            step=0.1,
            help=f"Minimum data asli: {desc.loc['min', 'poorpeople_percentage']:.2f}%, Maksimum: {desc.loc['max', 'poorpeople_percentage']:.2f}%"
        )
        
        inp_school = st.slider(
            "Rata-rata Lama Sekolah (Tahun)",
            min_value=1.0,
            max_value=16.0,
            value=float(desc.loc["mean", "avg_schooltime"]),
            step=0.1,
            help=f"Minimum data asli: {desc.loc['min', 'avg_schooltime']:.2f} Thn, Maksimum: {desc.loc['max', 'avg_schooltime']:.2f} Thn"
        )
        
    with col_inp2:
        st.markdown("#### 📝 Input Indikator Kesehatan & Ekonomi")
        inp_life = st.slider(
            "Angka Harapan Hidup (Tahun)",
            min_value=50.0,
            max_value=85.0,
            value=float(desc.loc["mean", "life_exp"]),
            step=0.1,
            help=f"Minimum data asli: {desc.loc['min', 'life_exp']:.2f} Thn, Maksimum: {desc.loc['max', 'life_exp']:.2f} Thn"
        )
        
        inp_exp = st.slider(
            "Pengeluaran Per Kapita (Rp per Tahun)",
            min_value=1000,
            max_value=30000,
            value=int(desc.loc["mean", "exp_percap"]),
            step=100,
            help=f"Minimum data asli: Rp {desc.loc['min', 'exp_percap']:,.0f}, Maksimum: Rp {desc.loc['max', 'exp_percap']:,.0f}"
        )
        
    st.markdown("---")
    
    # Jalankan prediksi saat tombol ditekan
    if st.button("🚀 Prediksi Kategori Pembangunan Wilayah", use_container_width=True):
        # Format input dataframe
        input_df = pd.DataFrame([{
            "poorpeople_percentage": inp_poverty,
            "avg_schooltime": inp_school,
            "life_exp": inp_life,
            "exp_percap": inp_exp
        }])
        
        # Lakukan prediksi
        pred_class = rf_model.predict(input_df)[0]
        pred_proba = rf_model.predict_proba(input_df)[0]
        classes = rf_model.classes_
        
        # Tampilkan hasil
        st.markdown("### 🎯 Hasil Prediksi Model:")
        
        # Berikan warna badge sesuai kelas prediksi
        if pred_class == "Rendah":
            st.error(f"## Kategori: Pembangunan **RENDAH**")
        elif pred_class == "Sedang":
            st.warning(f"## Kategori: Pembangunan **SEDANG**")
        else:
            st.success(f"## Kategori: Pembangunan **TINGGI**")
            
        # Plot Probabilitas Kepercayaan Kelas
        prob_df = pd.DataFrame({
            "Kategori": classes,
            "Probabilitas": pred_proba
        })
        
        fig_prob = px.bar(
            prob_df,
            x="Probabilitas",
            y="Kategori",
            orientation='h',
            title="Tingkat Keyakinan Prediksi Model (Probability Confidence)",
            color="Kategori",
            color_discrete_map={
                "Rendah": "#EF553B",
                "Sedang": "#FECB52",
                "Tinggi": "#00CC96"
            },
            range_x=[0, 1]
        )
        fig_prob.update_layout(showlegend=False)
        st.plotly_chart(fig_prob, use_container_width=True)
