import streamlit as st
import os
import re
import unicodedata
import pandas as pd
import matplotlib.pyplot as plt
import pdfplumber
from docx import Document
from PIL import Image
import nltk
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util
import fitz
import tempfile

nltk.download('punkt')

# Configurações
st.set_page_config(page_title="🔍 Talent Hunter Pro", layout="wide")
st.title("🔍 Talent Hunter Pro - Análise Precisa de Currículos")

# Normalização de texto para lidar com acentos
def normalize_text(text):
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
    return text.lower()

# Carregar modelos com verificação robusta
@st.cache_resource
def load_models():
    try:
        # Carregar modelo de similaridade
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        return model
    except Exception as e:
        st.error(f"Falha ao carregar o modelo: {str(e)}")
        return None

# Processar cada arquivo individualmente
def process_file(file_path, file_type):
    try:
        if file_type == "pdf":
            with pdfplumber.open(file_path) as pdf:
                text = "\n".join([page.extract_text() or "" for page in pdf.pages])
        else:
            doc = Document(file_path)
            text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        
        if text:
            tokens = word_tokenize(text)
            return text.strip()
        else:
            return ""
    except Exception as e:
        st.warning(f"Erro ao processar {os.path.basename(file_path)}: {str(e)}")
        return ""

# Análise semântica com tratamento de acentos
def analyze_resume(text, job_keywords):
    job_text = f"{job_keywords}"
    job_embedding = model.encode(job_text, convert_to_tensor=True)
    resume_embedding = model.encode(text, convert_to_tensor=True)
    score = util.cos_sim(job_embedding, resume_embedding).item()
    matches = sum(1 for keyword in job_keywords if keyword in text)
    return matches, score

# Visualização segura de documentos
def render_document(file_path, file_type):
    try:
        if file_type == "pdf":
            doc = fitz.open(file_path)
            page = doc.load_page(0)
            pix = page.get_pixmap(dpi=130)
            return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        else:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                temp_path = tmp.name
            convert(file_path, temp_path)
            img = render_document(temp_path, "pdf")
            os.unlink(temp_path)
            return img
    except Exception as e:
        st.warning(f"Erro ao renderizar documento: {str(e)}")
        return None

# Interface principal
def main():
    global model
    model = load_models()
    if model is None:
        return

    st.sidebar.header("Configurações")
    min_score = st.sidebar.slider("Score mínimo", 0.0, 1.0, 0.3)
    min_matches = st.sidebar.slider("Mínimo de correspondências", 0, 20, 2)

    col1, col2 = st.columns(2)
    with col1:
        job_title = st.text_input("🎯 Cargo desejado (obrigatório):", placeholder="Analista de Dados")
    with col2:
        job_desc = st.text_area("📋 Descrição do cargo:", placeholder="Principais habilidades...", height=100)

    folder_path = st.text_input("📁 Pasta com currículos (obrigatório):", value="/Users/robson/Documents/Curriculo")

    if st.button("🔍 Analisar Currículos", type="primary") and job_title and folder_path and os.path.exists(folder_path):
        with st.spinner(f"Analisando currículos em {folder_path}..."):
            try:
                job_keywords = [job_title] + [word for word in re.findall(r'\w{4,}', job_desc) if len(word) > 3]
                
                results = []
                valid_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.pdf', '.docx'))]
                
                if not valid_files:
                    st.error("Nenhum arquivo PDF ou DOCX encontrado na pasta!")
                    return

                progress_bar = st.progress(0)
                for i, filename in enumerate(valid_files):
                    file_path = os.path.join(folder_path, filename)
                    file_type = "pdf" if filename.lower().endswith('.pdf') else "docx"
                    
                    text = process_file(file_path, file_type)
                    if not text:
                        continue
                    
                    matches, score = analyze_resume(text, job_keywords)
                    if matches < min_matches:
                        continue
                    
                    if score >= min_score:
                        results.append({
                            "Arquivo": filename,
                            "Caminho": file_path,
                            "Tipo": file_type,
                            "Texto": text,
                            "Score": score,
                            "Matches": matches
                        })
                    
                    progress_bar.progress((i + 1) / len(valid_files))
                
                if not results:
                    st.warning("Nenhum currículo atendeu aos critérios mínimos!")
                    return

                results = sorted(results, key=lambda x: x['Score'], reverse=True)

                st.success(f"✅ {len(results)} currículos relevantes encontrados")

                st.subheader("Distribuição de Adequação")
                fig, ax = plt.subplots()
                ax.hist([r['Score'] for r in results], bins=15, color='skyblue', edgecolor='black')
                ax.set_xlabel('Score de Adequação')
                ax.set_ylabel('Número de Candidatos')
                st.pyplot(fig)

                st.subheader("Ranking de Candidatos")
                df = pd.DataFrame.from_records(results)
                st.dataframe(df[['Arquivo', 'Score', 'Matches']], hide_index=True, use_container_width=True)

            except Exception as e:
                st.error(f"Erro durante a análise: {str(e)}")
    elif not os.path.exists(folder_path):
        st.error("A pasta especificada não existe!")

if __name__ == "__main__":
    main()
