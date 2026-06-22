import os
import base64
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA

# ==================================================
# ASSISTENTE SENAI-SP
# VERSÃO FINAL REVISADA E CORRIGIDA
# ==================================================

load_dotenv()

# ==================================================
# CONFIGURAÇÃO DA PÁGINA
# ==================================================

st.set_page_config(
    page_title="Assistente SENAI-SP",
    page_icon="🔴",
    layout="wide"
)

# ==================================================
# CSS CUSTOMIZADO
# ==================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700&display=swap');

/* Reset e Fonte Global */
html, body, [class*="css"] {
    font-family: 'Montserrat', sans-serif;
}

/* Fundo Principal */
.stApp {
    background: linear-gradient(
        135deg,
        #1A1A1A 0%,
        #2D2D2D 50%,
        #4A0000 100%
    );
}

/* Esconder Elementos Padrão */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* Banner Hero */
.hero {
    background: linear-gradient(
        135deg,
        #E30613,
        #8B0000
    );
    padding: 40px;
    border-radius: 20px;
    text-align: center;
    margin-bottom: 25px;
    color: white;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}

.hero h1 {
    font-weight: 700;
    margin-top: 15px;
    color: white !important;
}

.hero p {
    font-size: 1.1rem;
    opacity: 0.9;
}

/* Área do Chat */
.chat-container {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 20px;
    padding: 10px;
    margin-bottom: 20px;
}

/* Barra Lateral (Sidebar) - Tons de Vermelho */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #4A0000 0%, #1A1A1A 100%) !important;
    border-right: 2px solid #E30613;
}

section[data-testid="stSidebar"] .stMarkdown, 
section[data-testid="stSidebar"] .stText,
section[data-testid="stSidebar"] label {
    color: #FFFFFF !important;
}

/* Botões */
.stButton > button {
    background: #E30613 !important;
    color: white !important;
    border-radius: 12px !important;
    border: none !important;
    padding: 10px 20px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    background: #FF1A1A !important;
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(227, 6, 19, 0.4);
}

/* Caixa de Fontes */
.fonte-box {
    background: #F5F5F5;
    border-left: 5px solid #E30613;
    padding: 15px;
    border-radius: 8px;
    margin-top: 15px;
    color: #333333;
    font-size: 0.9rem;
}

/* Ajuste de inputs */
.stChatInputContainer {
    padding-bottom: 20px !important;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# CABEÇALHO
# ==================================================

logo_html = ""
if os.path.exists("logo_senai.png"):
    with open("logo_senai.png", "rb") as img:
        logo_base64 = base64.b64encode(img.read()).decode()
    logo_html = f'<img src="data:image/png;base64,{logo_base64}" width="240">'

st.markdown(
    f"""
    <div class="hero">
        {logo_html}
        <h1>Assistente SENAI-SP</h1>
        <p>Tire dúvidas sobre o Regimento Comum e Perguntas Frequentes do SENAI-SP.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:
    # GIF do Robô mais interessante
    if os.path.exists("robot.gif"):
        with open("robot.gif", "rb") as f:
            gif_base64 = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <div style="text-align:center; padding: 20px 0;">
                <img src="data:image/gif;base64,{gif_base64}" width="100%" style="border-radius: 15px; border: 2px solid #E30613;">
            </div>
            """,
            unsafe_allow_html=True
        )

    st.success("🤖 Sistema Online")

    st.markdown("### 📚 Base de Conhecimento")
    st.info("📄 Regimento Comum")
    st.info("❓ Perguntas e Respostas")

    quantidade_chunks = st.slider(
        "Precisão da Resposta (Contextos)",
        1, 10, 4
    )

    st.markdown("### 🧠 Inteligência")
    st.caption("Modelo: Llama 3.1 8B")
    st.caption("Embeddings: NV-EmbedQA E5")

    if st.button("🗑️ Limpar Conversa"):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Olá! Sou o Assistente SENAI-SP. Como posso ajudar hoje?"
            }
        ]
        st.rerun()

# ==================================================
# CONFIGURAÇÃO DE API
# ==================================================

nvidia_api_key = os.getenv("NVIDIA_API_KEY")
if not nvidia_api_key:
    try:
        nvidia_api_key = st.secrets["NVIDIA_API_KEY"]
    except Exception:
        pass

if not nvidia_api_key:
    st.warning("⚠️ Chave de API da NVIDIA não configurada. Verifique suas variáveis de ambiente.")
    st.stop()

# ==================================================
# PROCESSAMENTO DE DOCUMENTOS (RAG)
# ==================================================

@st.cache_resource(show_spinner="📘 Sincronizando base de conhecimento SENAI-SP...")
def criar_vectorstore():
    arquivos = [
        "Regimento_comum_extraido.txt",
        "Perguntas_Respostas_SENAI.txt"
    ]
    
    documentos = []
    for arquivo in arquivos:
        if os.path.exists(arquivo):
            loader = TextLoader(arquivo, encoding="utf-8")
            documentos.extend(loader.load())
    
    if not documentos:
        # Fallback para evitar erro se arquivos não existirem no ambiente de teste
        return None, 0

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    docs = splitter.split_documents(documentos)

    embeddings = NVIDIAEmbeddings(
        model="nvidia/nv-embedqa-e5-v5",
        nvidia_api_key=nvidia_api_key,
        model_type="passage"
    )

    vectorstore = FAISS.from_documents(docs, embedding=embeddings)
    return vectorstore, len(docs)

vectorstore, total_chunks = criar_vectorstore()

if vectorstore:
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": quantidade_chunks, "fetch_k": 20}
    )

    # LLM e Chain
    llm = ChatNVIDIA(
        model="meta/llama-3.1-8b-instruct",
        nvidia_api_key=nvidia_api_key,
        temperature=0.2
    )

    template_prompt = """
    Você é um assistente virtual especializado no SENAI-SP.
    Utilize exclusivamente as informações presentes no contexto recuperado.
    Se a resposta não estiver no contexto, diga que não encontrou na base de conhecimento.
    
    Contexto: {context}
    Pergunta: {question}
    Resposta:
    """
    prompt = ChatPromptTemplate.from_template(template_prompt)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

# ==================================================
# HISTÓRICO DE CHAT
# ==================================================

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Olá! Sou o Assistente SENAI-SP. Posso ajudar com dúvidas sobre o Regimento Comum, normas acadêmicas e perguntas frequentes."
        }
    ]

# Exibição das mensagens
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==================================================
# INTERAÇÃO
# ==================================================

if pergunta := st.chat_input("Como posso ajudar com o SENAI-SP?"):
    st.session_state.messages.append({"role": "user", "content": pergunta})
    
    with st.chat_message("user"):
        st.markdown(pergunta)

    with st.chat_message("assistant"):
        if not vectorstore:
            st.error("Base de conhecimento não carregada. Verifique os arquivos no diretório 'documentos/'.")
        else:
            with st.spinner("Buscando informações..."):
                try:
                    resposta = rag_chain.invoke(pergunta)
                    st.markdown(resposta)
                    
                    # Box de fontes formatado sem vazamento de tags
                    st.markdown(
                        f"""
                        <div class="fonte-box">
                            <b>📚 Fontes consultadas:</b><br>
                            • Regimento Comum SENAI-SP<br>
                            • Base de Perguntas e Respostas
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.session_state.messages.append({"role": "assistant", "content": resposta})
                except Exception as e:
                    st.error(f"Ocorreu um erro no processamento: {e}")

# Estatísticas na sidebar (fim do script para garantir carregamento)
with st.sidebar:
    st.divider()
    st.write(f"📊 **Estatísticas da Base**")
    st.write(f"📄 Documentos: 2")
    st.write(f"🧩 Fragmentos: {total_chunks}")
