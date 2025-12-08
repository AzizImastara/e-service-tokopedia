import os
import pandas as pd
import streamlit as st
import joblib


@st.cache_resource
def load_artifact(pkl_path: str = "svm_model.pkl"):
    if not os.path.exists(pkl_path):
        st.error(f"Model artifact '{pkl_path}' not found. Please run export_model.py first.")
        st.stop()
    return joblib.load(pkl_path)


def predict_texts(artifact, texts: list[str]) -> list[str]:
    vectorizer = artifact["vectorizer"]
    model = artifact["model"]
    label_encoder = artifact["label_encoder"]  

    X = vectorizer.transform(texts)
    preds = model.predict(X)
    labels = label_encoder.inverse_transform(preds)  
    return labels.tolist()


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
        st.write("")  # Add vertical space
        if st.button("⬅", key="back_csv", use_container_width=True):
            st.session_state["page"] = "home"
            st.rerun()

    with col2:
        st.header("Input data sendiri")

    with st.expander("Panduan pengguna", expanded=True):
        st.markdown(
            "- Siapkan file CSV dengan kolom `content` berisi teks yang akan diklasifikasi.\n"
            "- Anda dapat mengunduh template, isi datanya, lalu unggah kembali.\n"
            "- Klik 'Proses data' untuk mendapatkan hasil klasifikasi."
        )

    # Template CSV
    template_df = pd.DataFrame({"content": [
        "ulasan pertama",
        "ulasan kedua",
        "ulasan ketiga"
    ]})
    template_csv = template_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download template file",
        data=template_csv,
        file_name="template_input.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Unggah file CSV", type=["csv"])

    df = None
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Gagal membaca CSV: {e}")
            df = None

    if df is not None:
        st.subheader("Preview data")
        st.dataframe(df.head(20), use_container_width=True)

        if "content" not in df.columns:
            st.warning("Kolom 'content' tidak ditemukan di CSV.")
            return

        if st.button("Proses data", type="primary"):
            texts = df["content"].astype(str).fillna("").tolist()
            labels = predict_texts(artifact, texts)
            result_df = df.copy()
            result_df["predicted_label"] = labels

            st.success("Proses selesai.")
            st.subheader("Hasil klasifikasi (preview)")
            st.dataframe(result_df.head(50), use_container_width=True)

            out_csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download hasil (CSV)",
                data=out_csv,
                file_name="hasil_klasifikasi.csv",
                mime="text/csv",
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
            "- Masukkan teks pada kolom di bawah.\n"
            "- Tekan tombol 'Input' untuk melihat hasil klasifikasi."
        )

    user_text = st.text_area("Input text", height=150)
    if st.button("Input", type="primary"):
        if not user_text.strip():
            st.warning("Silakan masukkan teks terlebih dahulu.")
            return
        label = predict_texts(artifact, [user_text.strip()])[0]
        st.subheader("Hasil output klasifikasi")
        st.success(f"Label: {label}")


def main():
    if "page" not in st.session_state:
        st.session_state["page"] = "home"

    artifact = load_artifact("svm_model.pkl")

    page = st.session_state["page"]
    if page == "home":
        render_home()
    elif page == "csv":
        render_csv_page(artifact)
    elif page == "demo":
        render_demo_page(artifact)
    else:
        st.session_state["page"] = "home"
        render_home()


if __name__ == "__main__":
    main()
