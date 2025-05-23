import os
from decouple import config
import streamlit as st
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Configura a chave da API da OpenAI a partir do arquivo .env
os.environ['OPENAI_API_KEY'] = config('OPENAI_API_KEY')

# Diretório onde o banco vetorial será salvo
persist_directory = 'db'

# Função para carregar todos os documentos PDF da pasta 'documents'
def load_documents():
    documents = []
    documents_dir = 'documents'
    # Cria a pasta se não existir
    if not os.path.exists(documents_dir):
        os.makedirs(documents_dir)
        return None
    # Lista todos os arquivos PDF
    pdf_files = [f for f in os.listdir(documents_dir) if f.endswith('.pdf')]
    if not pdf_files:
        return None
    # Carrega o conteúdo de cada PDF
    for pdf_file in pdf_files:
        loader = PyPDFLoader(os.path.join(documents_dir, pdf_file))
        documents.extend(loader.load())
    return documents

# Função para dividir os documentos em pedaços menores (chunks)
def process_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=400,
    )
    return text_splitter.split_documents(documents)

# Função para carregar ou criar o banco vetorial (vector store)
def load_or_create_vector_store(chunks):
    # Se já existe, apenas carrega
    if os.path.exists(persist_directory):
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=OpenAIEmbeddings(),
        )
        return vector_store
    # Se não existe, cria um novo
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(),
        persist_directory=persist_directory,
    )
    return vector_store

# Função para obter a resposta do modelo usando RAG
def get_response(query, vector_store):
    # Inicializa o modelo LLM
    llm = ChatOpenAI(model="gpt-4-turbo-preview")
    # Cria o retriever a partir do banco vetorial
    retriever = vector_store.as_retriever()

    # Prompt do sistema para orientar o modelo
    system_prompt = '''
    Você é um assistente especializado em responder perguntas sobre a empresa.
    Use APENAS as informações disponíveis para responder.
    Se a pergunta não estiver relacionada ao contexto ou você não encontrar a informação necessária,
    responda educadamente que não pode ajudar com esse assunto específico.
    Mantenha suas respostas concisas, profissionais e baseadas nos fatos.
    Responda em português do Brasil.
    
    Contexto: {context}
    '''

    # Monta o histórico do chat para dar mais contexto ao modelo
    messages = [("system", system_prompt)]
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            messages.append(("human", msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(("ai", msg["content"]))
    messages.append(("human", "{input}"))

    prompt = ChatPromptTemplate.from_messages(messages)

    # Cria a cadeia de perguntas e respostas
    question_answer_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
    )
    # Cria a cadeia de recuperação (RAG)
    chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=question_answer_chain,
    )
    # Executa a cadeia e retorna a resposta
    response = chain.invoke({'input': query})
    return response.get('answer')

# Inicializa o estado da sessão do Streamlit
def initialize_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False

# Inicializa o sistema de forma discreta (carrega documentos e banco vetorial)
def initialize_system():
    if not st.session_state.initialized:
        documents = load_documents()
        if documents:
            chunks = process_documents(documents)
            st.session_state.vector_store = load_or_create_vector_store(chunks)
        st.session_state.initialized = True

# Função principal da aplicação
def main():
    # Configura a página do Streamlit
    st.set_page_config(
        page_title="Assistente Empresarial",
        page_icon="💬",
        layout="centered"
    )
    # CSS customizado para visual moderno e divertido
    st.markdown(
        '''
        <style>
            body, .stApp {
                background: linear-gradient(120deg, #e3f0ff 0%, #b3d8f8 100%) !important;
            }
            .header-img {
                width: 100%;
                max-width: 420px;
                display: block;
                margin-left: auto;
                margin-right: auto;
                margin-bottom: 0.5rem;
                border-radius: 18px;
                box-shadow: 0 4px 24px #00000022;
            }
            .stChatInput input {
                background-color: #eaf4fb !important;
                border: 2px solid #4F8BF9 !important;
                color: #222 !important;
                border-radius: 12px !important;
                font-size: 1.1em;
            }
            .stChatMessage {
                border-radius: 16px !important;
                margin-bottom: 12px !important;
                box-shadow: 0 2px 12px #4f8bf91a;
                padding: 0.5rem 1rem !important;
            }
            .stMarkdown {
                font-size: 1.13em;
            }
            .stTitle {
                color: #4F8BF9;
                font-weight: 900;
                letter-spacing: 1px;
            }
            .st-bb, .st-cq, .st-cv {
                background: #f0f7ff !important;
            }
        </style>
        ''',
        unsafe_allow_html=True
    )
    # Imagem de cabeçalho pequena do Streamlit (logo oficial)
    st.markdown(
        """
        <div style='text-align:center;'>
            <img src='https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png' width='80' alt='Streamlit Logo' style='margin-bottom:0.5rem;'/>
        </div>
        """,
        unsafe_allow_html=True
    )
    # Inicializa o estado da sessão
    initialize_session_state()
    # Inicializa o sistema (carrega documentos e banco vetorial)
    initialize_system()
    # Título e mensagem de boas-vindas
    st.title("Assistente Empresarial 🤖")
    st.markdown("""
    <div style='font-size:1.2em; color:#222; text-align:center; margin-bottom:1.5em;'>
        Olá! Sou seu assistente virtual. Pergunte o que quiser sobre a empresa e divirta-se explorando!
    </div>
    """, unsafe_allow_html=True)
    # Exibe o histórico de mensagens do chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    # Campo de entrada para novas perguntas
    if prompt := st.chat_input("Digite sua pergunta aqui..."):
        # Adiciona a pergunta do usuário ao histórico
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        # Obtém e exibe a resposta do assistente
        with st.chat_message("assistant"):
            with st.spinner(""):
                if st.session_state.vector_store:
                    response = get_response(prompt, st.session_state.vector_store)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    st.error("Desculpe, não estou conseguindo processar sua pergunta no momento. Por favor, tente novamente mais tarde.")

# Executa a aplicação
if __name__ == "__main__":
    main()
