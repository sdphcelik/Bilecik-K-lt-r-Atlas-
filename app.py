import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import plotly.express as px
from sklearn.decomposition import PCA
import numpy as np

# --- AYARLAR ---
st.set_page_config(page_title="Bilecik AI Kültür Rehberi", layout="wide")

# Başlık
st.title("🤖 Bilecik Dijital Kültür Atlası: Yapay Zeka Analizi")
st.write("Bu sistem, doğal dil işleme (NLP) kullanarak Bilecik türküleri ve masalları arasında anlamsal bağlar kurar.")

# --- 1. VERİYİ YÜKLE VE HAZIRLA ---
@st.cache_resource # Modeli her seferinde tekrar yüklemesin diye önbellek
def load_model():
    return SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

model = load_model()

# Excel dosyasını oku
try:
    df = pd.read_excel("bilecik_kultur_data.xlsx")
except Exception as e:
    st.error(f"Excel dosyası okunamadı! Hata: {e}")
    st.stop()

# Metinleri vektörlere çevir
if 'isletilecek_veri' not in df.columns:
    df['isletilecek_veri'] = df['baslik'].astype(str) + " " + df['metin'].astype(str) + " " + df['duygu'].astype(str)

embeddings = model.encode(df['isletilecek_veri'].tolist(), convert_to_tensor=True)

# --- 2. SOL TARAF: AKILLI ARAMA ---
col1, col2 = st.columns([1, 2])

with col1:
    st.header("🔍 Duygu Bazlı Arama")
    st.info("Örnek: 'Kardeşine ihanet edenler', 'Sabrın sonu', 'Kahramanlık hikayeleri'")
    
    query = st.text_input("Ne arıyorsun? (Konu, duygu veya kavram yaz)")
    
    if query:
        query_embedding = model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_embedding, embeddings)[0]
        
        # En yüksek skorlu 3 sonucu getir
        top_results = np.argsort(scores.cpu().numpy())[-3:][::-1]
        
        st.write("---")
        st.subheader("💡 Yapay Zeka Önerileri:")
        for idx in top_results:
            score = scores[idx].item()
            row = df.iloc[idx]
            # ... (row satırının altı) ...
    
    # --- BURADAN İTİBAREN YAPIŞTIR ---
    # DİKKAT: Üstteki 'row' satırı ile aynı hizada başlamalı!
            if 'media_link' in row and pd.notna(row['media_link']):
               st.subheader("🎧 Dinle")
               try:
                     st.video(row['media_link'])
               except:
                     st.warning("Medya yüklenemedi.")
    # --- BİTİŞ ---
            st.markdown(f"**{row['baslik']}**")
            st.caption(f"Kategori: {row['kategori']} | Uyumluluk: %{int(score*100)}")
            st.write(f"_{str(row['metin'])[:150]}..._")
            
            # Link varsa butonu göster, yoksa gösterme
            if pd.notna(row['link']) and str(row['link']).startswith('http'):
                 st.markdown(f"[Dinlemek/Okumak için Tıkla]({row['link']})")
            
            st.divider()

# --- 3. SAĞ TARAF: KÜLTÜR HARİTASI ---
with col2:
    st.header("🌌 Kültürel Bağlantı Haritası")
    
    # Boyut İndirgeme (Harita için)
    pca = PCA(n_components=2)
    embeddings_np = embeddings.cpu().numpy()
    components = pca.fit_transform(embeddings_np)
    
    df['x'] = components[:, 0]
    df['y'] = components[:, 1]
    
    fig = px.scatter(df, x='x', y='y', 
                     color='kategori',
                     hover_data=['baslik', 'duygu'],
                     text='baslik',
                     size_max=60,
                     title="Eserlerin Anlamsal Uzayı")
    
    fig.update_traces(textposition='top center')
    fig.update_layout(showlegend=True, height=600)
    

    st.plotly_chart(fig, use_container_width=True)
