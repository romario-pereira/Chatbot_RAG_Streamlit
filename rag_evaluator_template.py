import os
import numpy as np

# Configuração da API Key da OpenAI - usando múltiplas opções
api_key_loaded = False
try:
    from decouple import config
    os.environ['OPENAI_API_KEY'] = config('OPENAI_API_KEY')
    api_key_loaded = True
    print("✓ API Key carregada via python-decouple")
except ImportError:
    try:
        from dotenv import load_dotenv
        load_dotenv()
        if 'OPENAI_API_KEY' in os.environ:
            api_key_loaded = True
            print("✓ API Key carregada via python-dotenv")
        else:
            print("AVISO: OPENAI_API_KEY não encontrada no .env")
    except ImportError:
        print("AVISO: Nem python-decouple nem python-dotenv estão disponíveis")
except Exception as e:
    print(f"AVISO: Não foi possível carregar OPENAI_API_KEY do .env. Erro: {e}")

if not api_key_loaded and 'OPENAI_API_KEY' not in os.environ:
    print("\n" + "="*60)
    print("⚠️  CONFIGURAÇÃO NECESSÁRIA")
    print("="*60)
    print("Para executar este script, você precisa configurar sua API Key da OpenAI.")
    print("Opções:")
    print("1. Edite o arquivo .env e substitua 'your_openai_api_key_here' pela sua chave")
    print("2. Configure a variável de ambiente: set OPENAI_API_KEY=sua_chave_aqui")
    print("3. Execute: $env:OPENAI_API_KEY='sua_chave_aqui' (PowerShell)")
    print("="*60)
    
    # Dar uma chance ao usuário de configurar manualmente
    user_key = input("Cole sua API Key da OpenAI aqui (ou pressione Enter para sair): ").strip()
    if user_key:
        os.environ['OPENAI_API_KEY'] = user_key
        api_key_loaded = True
        print("✓ API Key configurada manualmente")
    else:
        print("Script cancelado. Configure a API Key e execute novamente.")
        exit(1)

# Importações do LangChain - com tratamento de erros
try:
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.chains.retrieval import create_retrieval_chain
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    
    # Importações para RAG Híbrido (se necessário)
    from langchain_community.retrievers import BM25Retriever
    from langchain.retrievers import EnsembleRetriever
    
    print("✓ Todas as importações do LangChain foram carregadas com sucesso")
except ImportError as e:
    print(f"ERRO CRÍTICO: Falha ao importar bibliotecas do LangChain: {e}")
    print("Certifique-se de que todas as dependências estão instaladas corretamente.")
    print("Execute: pip install -r requirements.txt")
    exit(1)

# --- Configurações Globais Padrão (podem ser sobrescritas) ---
DEFAULT_DOCUMENTS_DIR = 'documents'
DEFAULT_PERSIST_DIR = 'db_eval' # Usar um diretório diferente para não interferir com o app
DEFAULT_LLM_MODEL = "gpt-4o-mini" # Modelo mais econômico e rápido para testes

# --- 1. FUNÇÕES AUXILIARES DE CARREGAMENTO E PREPARAÇÃO ---

def load_documents(documents_dir=DEFAULT_DOCUMENTS_DIR):
    """Carrega e splita todos os documentos PDF da pasta especificada."""
    print(f"Carregando documentos da pasta: {documents_dir}")
    docs_for_processing = []
    if not os.path.exists(documents_dir):
        print(f"ERRO: Pasta de documentos '{documents_dir}' não encontrada.")
        return None, None # Retorna None para docs e splits

    pdf_files = [f for f in os.listdir(documents_dir) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"Nenhum arquivo PDF encontrado em '{documents_dir}'.")
        return None, None

    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(os.path.join(documents_dir, pdf_file))
            docs_for_processing.extend(loader.load())
        except Exception as e:
            print(f"Erro ao carregar o arquivo {pdf_file}: {e}")
            continue # Pula para o próximo arquivo
    
    if not docs_for_processing:
        print("Nenhum documento PDF pôde ser carregado com sucesso.")
        return None, None

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=400,
        length_function=len,
    )
    split_docs = text_splitter.split_documents(documents=docs_for_processing)
    print(f"Total de {len(split_docs)} chunks criados a partir dos documentos.")
    return docs_for_processing, split_docs # Retorna os documentos originais (para BM25) e os splits

def get_vector_store(split_documents, persist_directory=DEFAULT_PERSIST_DIR, force_recreate=False):
    """Carrega ou cria o Chroma vector store."""
    if not split_documents and not (os.path.exists(persist_directory) and not force_recreate):
        print("ERRO: Nenhum documento fornecido para criar o vector store e nenhum existente para carregar (ou recriação forçada).")
        return None

    if os.path.exists(persist_directory) and not force_recreate:
        print(f"Carregando vector store existente de: {persist_directory}")
        try:
            vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=OpenAIEmbeddings(),
            )
            return vector_store
        except Exception as e:
            print(f"Erro ao carregar vector store existente: {e}. Tentando recriar...")
            # Tratar como se fosse para recriar se o carregamento falhar
            if not split_documents:
                 print("ERRO: Não é possível recriar o vector store sem documentos divididos.")
                 return None
            force_recreate = True # Força a recriação

    if split_documents: # Entra aqui se force_recreate é True ou se o diretório não existe
        print(f"Criando novo vector store em: {persist_directory}")
        vector_store = Chroma.from_documents(
            documents=split_documents,
            embedding=OpenAIEmbeddings(),
            persist_directory=persist_directory,
        )
        return vector_store
    else:
        # Este caso não deveria ser alcançado se a lógica anterior estiver correta
        print("ERRO: Incapaz de obter ou criar o vector store.")
        return None

def get_llm(model_name=DEFAULT_LLM_MODEL):
    """Retorna o modelo LLM configurado."""
    print(f"Inicializando LLM: {model_name}")
    return ChatOpenAI(model=model_name, temperature=0) # Temperatura 0 para respostas mais determinísticas

def create_rag_chain_from_retriever(retriever, llm, system_prompt):
    """Cria uma cadeia RAG a partir de um retriever e um system prompt."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=question_answer_chain)
    return chain

# --- 2. DEFINIÇÃO DOS MODELOS/PIPELINES RAG PARA TESTE ---

# Exemplo de System Prompt (pode ser customizado por pipeline)
DEFAULT_SYSTEM_PROMPT = """Você é um assistente de IA. Use o contexto fornecido para responder à pergunta.
Se a informação não estiver no contexto, diga que você não sabe.
Contexto: {context}"""

def get_embedding_retriever(vector_store, k=3):
    """Retorna um retriever baseado em embeddings (Chroma)."""
    if not vector_store: return None
    return vector_store.as_retriever(search_kwargs={"k": k})

def get_hybrid_retriever(raw_documents, vector_store, bm25_k=2, embedding_k=2, ensemble_weights=[0.5, 0.5]):
    """Retorna um retriever híbrido (BM25 + Embeddings)."""
    if not raw_documents or not vector_store: return None
    print("Configurando retriever Híbrido (BM25 + Embeddings)...")
    
    try:
        # BM25Retriever espera documentos com page_content
        # raw_documents já são Document objects, então deve funcionar
        bm25_retriever = BM25Retriever.from_documents(raw_documents)
        bm25_retriever.k = bm25_k
        
        embedding_retriever = vector_store.as_retriever(search_kwargs={"k": embedding_k})
        
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, embedding_retriever],
            weights=ensemble_weights
        )
        return ensemble_retriever
    except Exception as e:
        print(f"Erro ao criar retriever híbrido: {e}")
        return None

# --- 3. DATASET DE AVALIAÇÃO ---
# Este dataset deve ser fornecido ou carregado de um arquivo.
# Formato esperado: lista de dicionários, cada um com "id", "question", "ideal_answer".

# --- 4. FUNÇÕES DE SCORING ---

def semantic_similarity_score(generated_answer, ideal_answer, embedding_model=None):
    """Calcula a similaridade de cosseno entre embeddings das respostas."""
    if not generated_answer or not ideal_answer:
        return 0.0
    
    _embedding_model = embedding_model if embedding_model else OpenAIEmbeddings()
    
    try:
        gen_emb = _embedding_model.embed_query(generated_answer)
        ideal_emb = _embedding_model.embed_query(ideal_answer)
        
        similarity = np.dot(gen_emb, ideal_emb) / (np.linalg.norm(gen_emb) * np.linalg.norm(ideal_emb))
        return max(0, float(similarity)) * 100 # Retorna como porcentagem, não negativo
    except Exception as e:
        print(f"  Erro no semantic_similarity_score: {e}")
        return 0.0

# --- 5. ORQUESTRADOR DA AVALIAÇÃO ---

def run_evaluation(evaluation_dataset, rag_pipelines_to_test, llm, system_prompt=DEFAULT_SYSTEM_PROMPT):
    """
    Executa a avaliação para um conjunto de dados e pipelines RAG.
    
    Args:
        evaluation_dataset (list): Lista de dicts com {"id", "question", "ideal_answer"}.
        rag_pipelines_to_test (dict): Dicionário onde a chave é o nome do pipeline 
                                      e o valor é a instância do retriever.
                                      Ex: {"embedding_rag": retriever1, "hybrid_rag": retriever2}
        llm: A instância do modelo de linguagem.
        system_prompt (str): O prompt de sistema a ser usado.

    Returns:
        dict: Resultados da avaliação, com scores para cada pipeline.
              Ex: {"embedding_rag": {"average_score": 75.0, "details": [...]}, ...}
    """
    if not evaluation_dataset:
        print("Dataset de avaliação está vazio. Nada para testar.")
        return {}
    if not rag_pipelines_to_test:
        print("Nenhum pipeline RAG para testar.")
        return {}

    all_results = {}

    for pipeline_name, retriever in rag_pipelines_to_test.items():
        if retriever is None:
            print(f"Skipping pipeline '{pipeline_name}' pois o retriever não foi inicializado.")
            all_results[pipeline_name] = {"average_score": 0.0, "details": [], "error": "Retriever não inicializado"}
            continue

        print(f"\n--- Avaliando Pipeline: {pipeline_name} ---")
        rag_chain = create_rag_chain_from_retriever(retriever, llm, system_prompt)
        
        pipeline_scores = []
        pipeline_details = []

        for item in evaluation_dataset:
            question = item["question"]
            ideal_answer = item.get("ideal_answer") # .get() para ser flexível se não houver resposta ideal

            print(f"  Avaliando Pergunta ID: {item.get('id', 'N/A')} - \"{question[:50]}...\"")
            
            try:
                response_payload = rag_chain.invoke({'input': question})
                generated_answer = response_payload.get('answer', "Resposta não gerada.")
                
                score = 0.0
                if ideal_answer and generated_answer != "Resposta não gerada.":
                    score = semantic_similarity_score(generated_answer, ideal_answer)
                
                pipeline_scores.append(score)
                pipeline_details.append({
                    "id": item.get("id", "N/A"),
                    "question": question,
                    "ideal_answer": ideal_answer,
                    "generated_answer": generated_answer,
                    "similarity_score": score,
                    "retrieved_context_count": len(response_payload.get('context', []))
                })
                print(f"    Score de Similaridade: {score:.2f}%")

            except Exception as e:
                print(f"    ERRO ao processar pergunta ID {item.get('id', 'N/A')}: {e}")
                pipeline_scores.append(0) # Penaliza em caso de erro
                pipeline_details.append({
                    "id": item.get("id", "N/A"),
                    "question": question,
                    "ideal_answer": ideal_answer,
                    "generated_answer": "ERRO NA GERAÇÃO",
                    "similarity_score": 0.0,
                    "error_message": str(e)
                })

        average_score = np.mean(pipeline_scores) if pipeline_scores else 0.0
        all_results[pipeline_name] = {
            "average_score": average_score,
            "details": pipeline_details
        }
        print(f"  Score Médio para {pipeline_name}: {average_score:.2f}%")
        
    return all_results

def print_summary_report(evaluation_results):
    """Imprime um relatório resumido dos scores."""
    print("\n\n--- Relatório Resumido dos Scores ---")
    if not evaluation_results:
        print("Nenhum resultado de avaliação para reportar.")
        return

    for pipeline_name, results in evaluation_results.items():
        if "error" in results:
            print(f"Pipeline: {pipeline_name} - ERRO: {results['error']}")
        else:
            print(f"Pipeline: {pipeline_name}")
            print(f"  Score Médio de Similaridade: {results['average_score']:.2f}%")
    print("------------------------------------")

# --- 6. EXEMPLO DE USO ---

if __name__ == "__main__":
    print("🚀 Iniciando script de avaliação RAG...")
    print("="*60)

    # ETAPA 1: Carregar e Preparar Documentos
    print("📂 ETAPA 1: Carregando documentos...")
    raw_docs, split_docs = load_documents(documents_dir=DEFAULT_DOCUMENTS_DIR)
    
    if not split_docs:
        print("❌ Não foi possível carregar ou processar documentos. Encerrando avaliação.")
        exit()

    # Use force_recreate=True se quiser reconstruir o vector store do zero
    print("\n🗄️  ETAPA 2: Configurando vector store...")
    vector_store_instance = get_vector_store(split_docs, persist_directory=DEFAULT_PERSIST_DIR, force_recreate=False)

    if not vector_store_instance:
        print("❌ Não foi possível obter o vector store. Encerrando avaliação.")
        exit()

    # ETAPA 2: Configurar LLM
    print(f"\n🤖 ETAPA 3: Configurando LLM ({DEFAULT_LLM_MODEL})...")
    llm_instance = get_llm(model_name=DEFAULT_LLM_MODEL)

    # ETAPA 3: Definir os Pipelines RAG para Testar
    print("\n⚙️  ETAPA 4: Configurando pipelines RAG...")
    pipelines = {}
    
    embedding_ret = get_embedding_retriever(vector_store_instance, k=3)
    if embedding_ret:
        pipelines["RAG_Base_Embeddings"] = embedding_ret
        print("✓ Pipeline RAG Base (Embeddings) configurado")

    # Para o RAG Híbrido, precisamos dos documentos brutos (antes do split para o vector store)
    if raw_docs: # Apenas tente criar o retriever híbrido se raw_docs foram carregados
        hybrid_ret = get_hybrid_retriever(raw_docs, vector_store_instance, bm25_k=2, embedding_k=2)
        if hybrid_ret:
            pipelines["RAG_Hibrido_BM25_Embeddings"] = hybrid_ret
            print("✓ Pipeline RAG Híbrido (BM25 + Embeddings) configurado")
    else:
        print("⚠️  Documentos brutos não carregados, retriever híbrido não será configurado.")

    # ETAPA 4: Definir o Dataset de Avaliação
    print("\n📝 ETAPA 5: Preparando dataset de avaliação...")
    CURRENT_EVALUATION_DATASET = [
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
    ]
    
    print(f"✓ Dataset com {len(CURRENT_EVALUATION_DATASET)} perguntas carregado")
    
    # ETAPA 5: Definir o System Prompt
    custom_system_prompt = """Você é um assistente especializado em responder perguntas sobre o regulamento de consórcio.
Use APENAS as informações disponíveis no contexto fornecido para responder.
Se a pergunta não estiver relacionada ao contexto ou você não encontrar a informação necessária com base no contexto,
responda educadamente que não pode ajudar com esse assunto específico com base no material fornecido.
Mantenha suas respostas concisas, profissionais e baseadas nos fatos do contexto.
Responda em português do Brasil.

Contexto Fornecido: {context}"""

    # ETAPA 6: Executar Avaliação
    print(f"\n🎯 ETAPA 6: Executando avaliação de {len(pipelines)} pipeline(s)...")
    if not pipelines:
        print("❌ Nenhum pipeline RAG foi configurado com sucesso. Encerrando avaliação.")
        exit()

    evaluation_results_data = run_evaluation(
        evaluation_dataset=CURRENT_EVALUATION_DATASET,
        rag_pipelines_to_test=pipelines,
        llm=llm_instance,
        system_prompt=custom_system_prompt 
    )

    # ETAPA 7: Imprimir Relatório
    print_summary_report(evaluation_results_data)

    print("\n✅ Avaliação concluída com sucesso!")
    
    # Salvar resultados detalhados
    try:
        import json
        with open("evaluation_results.json", "w", encoding="utf-8") as f:
            json.dump(evaluation_results_data, f, ensure_ascii=False, indent=4)
        print("💾 Resultados detalhados salvos em evaluation_results.json")
    except Exception as e:
        print(f"❌ Erro ao salvar resultados: {e}") 