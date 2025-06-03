import os
from decouple import config
import streamlit as st # Usado para st.session_state se formos simular partes do app

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Importações para RAG Híbrido e outras avaliações
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
# Poderíamos adicionar métricas de bibliotecas como Ragas aqui no futuro, se o usuário instalar

try:
    os.environ['OPENAI_API_KEY'] = config('OPENAI_API_KEY')
except Exception as e:
    print(f"AVISO: Não foi possível carregar OPENAI_API_KEY do .env. Certifique-se de que está configurada. Erro: {e}")
    # Fallback para tentar pegar de os.environ diretamente se já estiver setada no ambiente
    if 'OPENAI_API_KEY' not in os.environ:
        print("ERRO: OPENAI_API_KEY não encontrada. O script não poderá executar chamadas à OpenAI.")
        # exit() # Descomente se quiser que o script pare aqui se não houver chave

persist_directory = 'db'
documents_dir = 'documents' # Pasta de onde os documentos são carregados

# --- 1. FUNÇÕES AUXILIARES (Reutilizadas ou adaptadas do app.py) ---

def load_all_documents_from_source():
    """Carrega e splita todos os documentos PDF da pasta 'documents'."""
    print(f"Carregando documentos da pasta: {documents_dir}")
    docs_for_bm25_and_vectorstore = []
    if not os.path.exists(documents_dir):
        print(f"ERRO: Pasta de documentos '{documents_dir}' não encontrada.")
        return None

    pdf_files = [f for f in os.listdir(documents_dir) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"Nenhum arquivo PDF encontrado em '{documents_dir}'.")
        return None

    for pdf_file in pdf_files:
        loader = PyPDFLoader(os.path.join(documents_dir, pdf_file))
        docs_for_bm25_and_vectorstore.extend(loader.load())
    
    if not docs_for_bm25_and_vectorstore:
        print("Nenhum documento carregado.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=400,
    )
    split_docs = text_splitter.split_documents(documents=docs_for_bm25_and_vectorstore)
    print(f"Total de {len(split_docs)} chunks criados a partir dos documentos.")
    return split_docs

def get_vector_store(documents_for_vectorstore, force_recreate=False):
    """Carrega ou cria o Chroma vector store."""
    if os.path.exists(persist_directory) and not force_recreate:
        print(f"Carregando vector store existente de: {persist_directory}")
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=OpenAIEmbeddings(),
        )
    elif documents_for_vectorstore:
        print(f"Criando novo vector store em: {persist_directory} (forçando recriação ou não existente)")
        vector_store = Chroma.from_documents(
            documents=documents_for_vectorstore,
            embedding=OpenAIEmbeddings(),
            persist_directory=persist_directory,
        )
    else:
        print("ERRO: Nenhum documento para criar vector store e nenhum existente encontrado/permitido carregar.")
        return None
    return vector_store

def get_llm(model_name="gpt-4-turbo-preview"):
    """Retorna o modelo LLM configurado."""
    print(f"Inicializando LLM: {model_name}")
    return ChatOpenAI(model=model_name)

def create_rag_chain(retriever, llm, system_prompt_template):
    """Cria uma cadeia RAG genérica."""
    # Usaremos um histórico de chat simulado ou vazio para testes pontuais
    # Para testes que dependem de histórico, precisaremos de uma lógica mais elaborada
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_template),
        ("human", "{input}") 
        # Nota: Para simular o histórico do app.py, teríamos que passar `st.session_state.messages`
        # ou uma estrutura similar para cá. Para testes unitários de RAG, começamos sem histórico.
    ])

    question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=question_answer_chain)
    return chain

# --- 2. DEFINIÇÃO DO DATASET DE TESTE (FORNECIDO PELO USUÁRIO) ---
EVALUATION_DATASET = [
    {
        "id": "CONTR_OBJ_001",
        "question": "Qual é o objeto principal do Contrato de Participação em Grupo de Consórcio, previsto neste Regulamento?",
        "ideal_answer": "O objeto do Contrato é regulamentar a participação do consorciado em um Grupo de Consórcio específico, permitindo que o participante adquira uma cota com o 'crédito de referência' destinado à aquisição de bens (aqui, veículos), conforme definido na 'Proposta de Adesão' e em conformidade com a Lei 11.795/2008 e a Resolução Bacen nº 285/2023."
    },
    {
        "id": "ADM_OBRIG_002",
        "question": "Quais são as principais obrigações da Administradora em relação ao Grupo de Consórcio?",
        "ideal_answer": "A Administradora deve, entre outras funções: Efetuar controle diário das movimentações financeiras dos Grupos; Disponibilizar em cada Assembleia Geral Ordinária (A.G.O.) o balancete patrimonial, demonstração dos recursos do consórcio e variações das disponibilidades do Grupo; Fornecer informações solicitadas pelos consorciados, desde que autorizadas; Lavrar atas das Assembleias; Encaminhar, junto ao boleto de cobrança, a demonstração dos recursos e variações das disponibilidades do Grupo; Manter sistemas de controle operacional para exame pelo Bacen e pelos representantes dos consorciados."
    },
    {
        "id": "CONS_DEVER_003",
        "question": "Quais são os deveres do Consorciado quanto à atualização cadastral e tratamento de dados?",
        "ideal_answer": "O consorciado deve manter seus dados cadastrais sempre atualizados (endereço, e-mail, telefone, contas bancárias ou chave Pix), mesmo que esteja excluído; autorizar a inscrição dos dados no banco de dados de 'bureau positivo' para decisões de crédito; e zelar pelo sigilo e veracidade das informações, podendo solicitar correção ou exclusão conforme a LGPD."
    },
    {
        "id": "CONTEMP_PROC_004",
        "question": "Como funciona o processo de contemplação no consórcio e quais são suas modalidades?",
        "ideal_answer": "A contemplação atribui ao consorciado ativo o direito de usar o crédito e, ao excluído, restituição das parcelas pagas. Realiza-se exclusivamente por dois mecanismos: Sorteio: aleatório, com base no resultado da Loteria Federal (tabela de equivalência); Lance: oferta de antecipação de parte do saldo devedor, quitando parcelas em ordem inversa ou por percentual. As modalidades (lance livre, lance fixo) e seus percentuais são definidos em Assembleia Geral."
    },
    {
        "id": "CONTEMP_IMPED_005",
        "question": "Quais condições impedem um consorciado de concorrer à contemplação?",
        "ideal_answer": "Estão inapto à contemplação: Consorciado ativo já contemplado; Consorciado ativo em atraso em uma ou mais prestações ou com débitos junto ao Grupo ou Administradora; Consorciado excluído cujo crédito já foi contemplado em Assembleia anterior (para fins de restituição)."
    },
]

# --- 3. IMPLEMENTAÇÃO DOS MODELOS/PIPELINES RAG PARA TESTE ---
SYSTEM_PROMPT_TEMPLATE = """Você é um assistente especializado em responder perguntas sobre o regulamento de consórcio.
Use APENAS as informações disponíveis no contexto fornecido para responder.
Se a pergunta não estiver relacionada ao contexto ou você não encontrar a informação necessária com base no contexto,
responda educadamente que não pode ajudar com esse assunto específico com base no material fornecido.
Mantenha suas respostas concisas, profissionais e baseadas nos fatos do contexto.
Responda em português do Brasil.

Contexto Fornecido: {context}"""

# Modelo RAG Atual (baseado em embeddings)
def get_current_rag_retriever(vector_store):
    return vector_store.as_retriever(search_kwargs={"k": 3}) # k é o número de chunks a recuperar

# Modelo RAG Híbrido (BM25 + Embeddings)
def get_hybrid_rag_retriever(documents_for_bm25, vector_store, bm25_k=2, embedding_k=2, ensemble_weights=[0.5, 0.5]):
    print("Configurando retriever Híbrido (BM25 + Embeddings)...")
    bm25_retriever = BM25Retriever.from_documents(documents_for_bm25)
    bm25_retriever.k = bm25_k
    
    embedding_retriever = vector_store.as_retriever(search_kwargs={"k": embedding_k})
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, embedding_retriever],
        weights=ensemble_weights
    )
    return ensemble_retriever

# --- 4. FUNÇÕES DE AVALIAÇÃO E SCORING ---
def simple_similarity_scorer(generated_answer, ideal_answer):
    """Um scorer muito simples baseado em similaridade de embeddings (conceitual)."""
    if not generated_answer or not ideal_answer:
        return 0.0
    try:
        embeddings = OpenAIEmbeddings()
        gen_emb = embeddings.embed_query(generated_answer)
        ideal_emb = embeddings.embed_query(ideal_answer)
        # Cosine similarity (simple way)
        import numpy as np
        similarity = np.dot(gen_emb, ideal_emb) / (np.linalg.norm(gen_emb) * np.linalg.norm(ideal_emb))
        return max(0, similarity) * 100 # Retorna como porcentagem, não negativo
    except Exception as e:
        print(f"  Erro no simple_similarity_scorer: {e}")
        return 0.0 # Em caso de erro na API ou cálculo

def evaluate_single_query(query_data, rag_chain, retriever_name):
    print(f"  Avaliando com {retriever_name}...")
    response_payload = rag_chain.invoke({'input': query_data["question"]})
    generated_answer = response_payload.get('answer', "Resposta não gerada.")
    retrieved_context_docs = response_payload.get('context', [])
    
    retrieved_context_str = "\n\n---\n\n".join([f"Fonte {i+1}: Trecho do documento (página {doc.metadata.get('page', 'N/A') if hasattr(doc, 'metadata') else 'N/A'}):\n{doc.page_content}" for i, doc in enumerate(retrieved_context_docs)])

    result = {
        "query_id": query_data["id"],
        "question": query_data["question"],
        "ideal_answer": query_data.get("ideal_answer"),
        "retriever_used": retriever_name,
        "generated_answer": generated_answer,
        "retrieved_context": retrieved_context_str
    }

    if result["ideal_answer"]:
        result["similarity_score"] = simple_similarity_scorer(generated_answer, result["ideal_answer"])
        
    # Placeholder para métricas Ragas
    #result["faithfulness_score_ragas"] = calculate_ragas_faithfulness(generated_answer, retrieved_context_docs)
    return result

# --- 5. EXECUÇÃO DOS TESTES ---
def run_evaluation_suite():
    print("--- Iniciando Suite de Avaliação RAG para Regulamento de Consórcio ---")
    print("AVISO: Certifique-se de que a pasta 'documents' contém o PDF do regulamento para teste.")

    all_split_docs = load_all_documents_from_source()
    if not all_split_docs:
        print("ERRO: Documentos não carregados. Encerrando.")
        return

    vector_store = get_vector_store(all_split_docs, force_recreate=True)
    if not vector_store:
        print("ERRO: Vector store não criado/carregado. Encerrando.")
        return

    llm = get_llm()
    if not llm:
        print("ERRO: LLM não inicializado. Encerrando.")
        return

    current_retriever = get_current_rag_retriever(vector_store)
    hybrid_retriever = get_hybrid_rag_retriever(all_split_docs, vector_store)

    retrievers_to_test = {
        "RAG_Atual_Embedding_Only": current_retriever,
        "RAG_Hibrido_BM25_Plus_Embedding": hybrid_retriever,
    }

    all_results = []
    if not EVALUATION_DATASET or not EVALUATION_DATASET[0].get("question"): 
        print("ERRO: `EVALUATION_DATASET` está vazio ou mal configurado. Verifique as perguntas.")
        return
        
    for q_data in EVALUATION_DATASET:
        print(f"\n--- Avaliando Pergunta ID: {q_data['id']} ---")
        print(f"Pergunta: {q_data['question']}")
        print(f"Resposta Esperada: {q_data['ideal_answer']}")
        
        for retriever_name, retriever_instance in retrievers_to_test.items():
            rag_chain = create_rag_chain(retriever_instance, llm, SYSTEM_PROMPT_TEMPLATE)
            eval_result = evaluate_single_query(q_data, rag_chain, retriever_name)
            all_results.append(eval_result)
            
            print(f"\n  Resultados para [{retriever_name}]:")
            print(f"  Resposta Gerada: {eval_result['generated_answer']}")
            if "similarity_score" in eval_result:
                print(f"  Score de Similaridade com Resposta Esperada: {eval_result['similarity_score']:.2f}%")
            print(f"\n  Contexto Recuperado por [{retriever_name}] (primeiros 500 caracteres por fonte):")
            
            # Imprime preview de cada fonte do contexto
            if eval_result['retrieved_context']:
                sources = eval_result['retrieved_context'].split("\n\n---\n\n")
                for i, source_text in enumerate(sources):
                    print(f"    Fonte {i+1} (Preview): {source_text[:500]}..."
                          if len(source_text) > 500 else f"    Fonte {i+1}: {source_text}")            
            else:
                print("    Nenhum contexto recuperado.")
            print("  --- Fim do Contexto --- \n")
        print("------------------------------------------------------")

    print("\n--- Agregação de Scores de Similaridade (Média) ---")
    aggregated_scores = {name: [] for name in retrievers_to_test.keys()}
    for res in all_results:
        if "similarity_score" in res and res.get("similarity_score") is not None:
            aggregated_scores[res["retriever_used"]].append(res["similarity_score"])
    
    for retriever_name, scores in aggregated_scores.items():
        if scores:
            average_score = sum(scores) / len(scores)
            print(f"Média de Similaridade para [{retriever_name}]: {average_score:.2f}%")
        else:
            print(f"Nenhum score de similaridade calculado para [{retriever_name}].")

    print("\n--- Suite de Avaliação RAG Concluída ---")
    print("Analise os resultados impressos acima para comparar o desempenho dos modelos RAG.")
    print("Verifique a fidelidade da Resposta Gerada em relação ao Contexto Recuperado e à Resposta Esperada.")

if __name__ == "__main__":
    if not EVALUATION_DATASET or not EVALUATION_DATASET[0].get("question"): 
        print("ERRO CRÍTICO: `EVALUATION_DATASET` não foi populado corretamente. Verifique o script.")
    else:
        run_evaluation_suite() 