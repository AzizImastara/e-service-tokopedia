import os
import re
import pandas as pd
import streamlit as st
import joblib

# =====================================================
# PREPROCESSING
# =====================================================

def remove_emojis(text):
    emoji_pattern = re.compile(
        "[" u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F700-\U0001F77F"
        u"\U0001F780-\U0001F7FF"
        u"\U0001F800-\U0001F8FF"
        u"\U0001F900-\U0001F9FF"
        u"\U0001FA00-\U0001FA6F"
        u"\U0001FA70-\U0001FAFF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


stopwords = {
    'yang', 'di', 'ke', 'dan', 'dari', 'ini', 'itu', 'untuk', 'dengan',
    'atau', 'karena', 'pada', 'jadi', 'sudah', 'belum', 'ada', 'tidak',
    'bukan', 'saya', 'kami', 'kita', 'mereka', 'dia', 'nya', 'akan',
    'dalam', 'jika', 'lagi', 'sebagai', 'oleh', 'bagi', 'tentang',
    'apa', 'mengapa', 'bagaimana', 'adalah', 'saat', 'hingga', 'tp', 'yg'
}


def stemming(word):
    prefixes = ['ber', 'ter', 'me', 'di', 'ke', 'se', 'per']
    suffixes = ['kan', 'an', 'lah', 'kah', 'nya']

    for prefix in prefixes:
        if word.startswith(prefix):
            word = word[len(prefix):]
            break
    for suffix in suffixes:
        if word.endswith(suffix):
            word = word[:-len(suffix)]
            break
    return word


def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = remove_emojis(text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.strip()

    tokens = text.split()
    tokens = [stemming(w) for w in tokens if w not in stopwords]

    return ' '.join(tokens)


# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_artifact(pkl_path="svm_model.pkl"):
    if not os.path.exists(pkl_path):
        st.error(f"Model '{pkl_path}' tidak ditemukan.")
        st.stop()
    return joblib.load(pkl_path)


# =====================================================
# PREDICTION
# =====================================================

def predict_texts(artifact, texts):
    vectorizer = artifact["vectorizer"]
    model = artifact["model"]
    label_encoder = artifact["label_encoder"]

    clean_texts = [clean_text(t) for t in texts]
    X = vectorizer.transform(clean_texts)
    preds = model.predict(X)

    return label_encoder.inverse_transform(preds).tolist()


# =====================================================
# UI PAGES
# =====================================================

def render_home():
    st.markdown("<h2 style='text-align: center;'>Klasifikasikan dan Analisis keluhan Tokopedia Secara Mudah Menggunakan Teknologi Machine Learning Secara Gratis</h2>", unsafe_allow_html=True)    
    st.write("Pilih mode penggunaan:")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Input data sendiri", use_container_width=True):
            st.session_state["page"] = "csv"
            st.rerun()
    with col2:
        if st.button("Coba langsung", use_container_width=True):
            st.session_state["page"] = "demo"
            st.rerun()


def render_csv_page(artifact):
    col1, col2 = st.columns([0.1, 0.9])

    with col1:
        st.write("")
        if st.button("⬅", key="back_csv", use_container_width=True):
            st.session_state["page"] = "home"
            st.rerun()

    with col2:
        st.markdown("## Input data sendiri")

    with st.expander("Panduan pengguna", expanded=True):
        st.markdown(
            "- Siapkan file CSV dengan kolom **content**\n"
            "- Data akan dipreproses sebelum diklasifikasikan\n"
            "- Klik **Proses data** untuk melihat hasil"
        )

    template_df = pd.DataFrame({"content": [
        "aplikasi sering error 😡",
        "cs lama merespon",
        "fitur sangat membantu 👍"
    ]})

    st.download_button(
        "Download template CSV",
        template_df.to_csv(index=False).encode("utf-8"),
        "template_input.csv",
        "text/csv"
    )

    uploaded = st.file_uploader("Unggah file CSV", type=["csv"])

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head(20), use_container_width=True)

        if "content" not in df.columns:
            st.warning("Kolom 'content' tidak ditemukan.")
            return

        if st.button("Proses data", type="primary"):
            texts = df["content"].astype(str).fillna("").tolist()
            clean_texts = [clean_text(t) for t in texts]
            labels = predict_texts(artifact, texts)

            result_df = df.copy()
            result_df["clean_content"] = clean_texts
            result_df["predicted_label"] = labels

            st.success("Proses selesai.")
            st.dataframe(result_df.head(50), use_container_width=True)

            st.download_button(
                "Download hasil (CSV)",
                result_df.to_csv(index=False).encode("utf-8"),
                "hasil_klasifikasi.csv",
                "text/csv"
            )


def render_demo_page(artifact):
    col1, col2 = st.columns([0.1, 0.9])
    with col1:
        st.write("")  # Add vertical space
        if st.button("⬅", key="back_demo", use_container_width=True):
            st.session_state["page"] = "home"
            st.rerun()
    with col2:
        st.header("Coba langsung")

    with st.expander("Panduan pengguna", expanded=True):
        st.markdown(
            "- Masukkan teks ulasan pada kolom di bawah\n"
            "- Teks akan dipreproses terlebih dahulu\n"
            "- Tekan tombol **Input** untuk melihat hasil klasifikasi"
        )

    user_text = st.text_area("Input text", height=150, placeholder="Masukkan ulasan di sini...")

    if st.button("Input", type="primary"):
        if not user_text.strip():
            st.warning("Teks tidak boleh kosong.")
            return

        clean = clean_text(user_text)
        label = predict_texts(artifact, [user_text])[0]

        st.subheader("Hasil Preprocessing")
        st.info(clean)

        st.subheader("Hasil Klasifikasi")
        st.success(f"Label: {label}")


# =====================================================
# MAIN
# =====================================================

def main():
    if "page" not in st.session_state:
        st.session_state["page"] = "home"

    artifact = load_artifact()

    if st.session_state["page"] == "home":
        render_home()
    elif st.session_state["page"] == "csv":
        render_csv_page(artifact)
    elif st.session_state["page"] == "demo":
        render_demo_page(artifact)
    else:
        st.session_state["page"] = "home"
        render_home()


if __name__ == "__main__":
    main()
