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
    
    # Importações para RAG Avançado
    from langchain_community.retrievers import BM25Retriever
    from langchain.retrievers import (
        EnsembleRetriever, 
        ContextualCompressionRetriever,
        ParentDocumentRetriever
    )
    from langchain.retrievers.document_compressors import (
        LLMChainExtractor,
        LLMChainFilter,
        EmbeddingsFilter
    )
    from langchain.retrievers.multi_query import MultiQueryRetriever
    from langchain.storage import InMemoryStore
    
    print("✓ Todas as importações do LangChain foram carregadas com sucesso")
except ImportError as e:
    print(f"ERRO CRÍTICO: Falha ao importar bibliotecas do LangChain: {e}")
    print("Certifique-se de que todas as dependências estão instaladas corretamente.")
    print("Execute: pip install -r requirements.txt")
    exit(1)

# --- CONFIGURAÇÕES GLOBAIS ---
DEFAULT_DOCUMENTS_DIR = 'documents'
DEFAULT_PERSIST_DIR = 'db_advanced' # Diretório específico para testes avançados
DEFAULT_LLM_MODEL = "gpt-4o-mini" # Modelo econômico para testes

# Configurações avançadas para RAG
RAG_CONFIGS = {
    'similarity_threshold': 0.7,
    'small_chunk_size': 400,
    'large_chunk_size': 1500,
    'chunk_overlap': 200,
    'max_docs_for_compression': 10,
    'rerank_top_k': 5
}

# --- 1. FUNÇÕES DE CARREGAMENTO E PREPARAÇÃO ---

def load_documents_multi_size(documents_dir=DEFAULT_DOCUMENTS_DIR):
    """Carrega documentos com múltiplos tamanhos de chunk para diferentes estratégias."""
    print(f"📂 Carregando documentos da pasta: {documents_dir}")
    docs_for_processing = []
    
    if not os.path.exists(documents_dir):
        print(f"❌ ERRO: Pasta de documentos '{documents_dir}' não encontrada.")
        return None, None, None

    pdf_files = [f for f in os.listdir(documents_dir) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"❌ Nenhum arquivo PDF encontrado em '{documents_dir}'.")
        return None, None, None

    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(os.path.join(documents_dir, pdf_file))
            docs_for_processing.extend(loader.load())
            print(f"✅ Carregado: {pdf_file}")
        except Exception as e:
            print(f"❌ Erro ao carregar {pdf_file}: {e}")
            continue

    if not docs_for_processing:
        print("❌ Nenhum documento PDF pôde ser carregado com sucesso.")
        return None, None, None

    print(f"📝 Processando {len(docs_for_processing)} páginas de documentos...")

    # Chunks padrão (1000 chars) - balanceado
    text_splitter_standard = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=400,
        length_function=len,
    )
    standard_docs = text_splitter_standard.split_documents(documents=docs_for_processing)
    
    # Chunks pequenos (400 chars) - busca precisa
    text_splitter_small = RecursiveCharacterTextSplitter(
        chunk_size=RAG_CONFIGS['small_chunk_size'],
        chunk_overlap=RAG_CONFIGS['chunk_overlap'],
        length_function=len,
    )
    small_docs = text_splitter_small.split_documents(documents=docs_for_processing)
    
    # Chunks grandes (1500 chars) - contexto rico
    text_splitter_large = RecursiveCharacterTextSplitter(
        chunk_size=RAG_CONFIGS['large_chunk_size'],
        chunk_overlap=RAG_CONFIGS['chunk_overlap'],
        length_function=len,
    )
    large_docs = text_splitter_large.split_documents(documents=docs_for_processing)
    
    print(f"📊 Chunks criados:")
    print(f"   - Standard (1000 chars): {len(standard_docs)} chunks")
    print(f"   - Small (400 chars): {len(small_docs)} chunks")  
    print(f"   - Large (1500 chars): {len(large_docs)} chunks")
    
    return standard_docs, small_docs, large_docs

def create_vector_stores(standard_docs, small_docs, large_docs, force_recreate=False):
    """Cria múltiplos vector stores otimizados para diferentes estratégias."""
    print("🗄️  Criando vector stores especializados...")
    vector_stores = {}
    
    docs_config = {
        'standard': (standard_docs, "Vector store padrão (1000 chars)"),
        'small': (small_docs, "Vector store para busca precisa (400 chars)"),
        'large': (large_docs, "Vector store para contexto rico (1500 chars)")
    }
    
    for doc_type, (documents, description) in docs_config.items():
        if not documents:
            print(f"⚠️  Pulando {doc_type}: sem documentos")
            continue
            
        store_dir = f"{DEFAULT_PERSIST_DIR}_{doc_type}"
        
        if os.path.exists(store_dir) and not force_recreate:
            print(f"📥 Carregando {description} de: {store_dir}")
            try:
                vector_stores[doc_type] = Chroma(
                    persist_directory=store_dir,
                    embedding_function=OpenAIEmbeddings(),
                )
                print(f"✅ {doc_type} carregado com sucesso")
            except Exception as e:
                print(f"❌ Erro ao carregar {doc_type}: {e}")
                print(f"🔄 Recriando {doc_type}...")
                force_recreate = True
        
        if not os.path.exists(store_dir) or force_recreate:
            print(f"🔨 Criando {description} em: {store_dir}")
            try:
                vector_stores[doc_type] = Chroma.from_documents(
                    documents=documents,
                    embedding=OpenAIEmbeddings(),
                    persist_directory=store_dir,
                )
                print(f"✅ {doc_type} criado com sucesso")
            except Exception as e:
                print(f"❌ Erro ao criar {doc_type}: {e}")
                continue
    
    print(f"🎯 Vector stores criados: {list(vector_stores.keys())}")
    return vector_stores

def get_llm(model_name=DEFAULT_LLM_MODEL):
    """Inicializa o modelo LLM."""
    print(f"🤖 Inicializando LLM: {model_name}")
    return ChatOpenAI(model=model_name, temperature=0)

def create_rag_chain_from_retriever(retriever, llm, system_prompt):
    """Cria uma cadeia RAG a partir de um retriever."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=question_answer_chain)
    return chain

# --- 2. MODELOS RAG AVANÇADOS ---

def get_rag_base_embeddings(vector_store, k=3):
    """🥉 RAG Base - Busca por similaridade de embeddings simples."""
    if not vector_store: 
        return None
    print(f"🔧 Configurando RAG Base (k={k})")
    return vector_store.as_retriever(search_kwargs={"k": k})

def get_rag_hybrid_bm25_embeddings(documents, vector_store, bm25_k=2, embedding_k=2):
    """🥈 RAG Híbrido - Combina BM25 (palavra-chave) + Embeddings (semântica)."""
    if not documents or not vector_store: 
        return None
    print(f"🔧 Configurando RAG Híbrido (BM25: {bm25_k}, Embeddings: {embedding_k})")
    
    try:
        # BM25 para busca por palavras-chave
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = bm25_k
        
        # Embeddings para busca semântica
        embedding_retriever = vector_store.as_retriever(search_kwargs={"k": embedding_k})
        
        # Combina ambos com pesos balanceados
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, embedding_retriever],
            weights=[0.5, 0.5]  # 50% cada método
        )
        return ensemble_retriever
    except Exception as e:
        print(f"❌ Erro ao criar RAG Híbrido: {e}")
        return None

def get_rag_contextual_compression(vector_store, llm, k=6):
    """🧠 RAG com Compressão Contextual - Remove informações irrelevantes."""
    if not vector_store: 
        return None
    print(f"🔧 Configurando RAG com Compressão Contextual (k={k})")
    
    try:
        # Busca mais documentos inicialmente
        base_retriever = vector_store.as_retriever(search_kwargs={"k": k})
        
        # Compressor que extrai apenas informações relevantes usando LLM
        compressor = LLMChainExtractor.from_llm(llm)
        
        # Retriever que filtra e comprime o contexto
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
        return compression_retriever
    except Exception as e:
        print(f"❌ Erro ao criar RAG com Compressão: {e}")
        return None

def get_rag_similarity_threshold(vector_store, threshold=0.7, k=5):
    """🎯 RAG com Threshold - Filtra documentos por score mínimo de relevância."""
    if not vector_store: 
        return None
    print(f"🔧 Configurando RAG com Threshold (score ≥ {threshold})")
    
    try:
        # Busca apenas documentos acima do threshold de similaridade
        base_retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "score_threshold": threshold,
                "k": k
            }
        )
        return base_retriever
    except Exception as e:
        print(f"⚠️  Threshold não suportado, usando busca normal: {e}")
        # Fallback para busca normal
        return vector_store.as_retriever(search_kwargs={"k": k})

def get_rag_multi_query(vector_store, llm, k=3):
    """🔍 RAG com Expansão de Query - Gera múltiplas versões da pergunta."""
    if not vector_store: 
        return None
    print(f"🔧 Configurando RAG com Expansão de Query (k={k})")
    
    try:
        base_retriever = vector_store.as_retriever(search_kwargs={"k": k})
        
        # Gera automaticamente variações da pergunta para busca mais abrangente
        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=llm
        )
        return multi_query_retriever
    except Exception as e:
        print(f"❌ Erro ao criar RAG Multi-Query: {e}")
        return None

def get_rag_parent_document(small_docs, large_docs, vector_store_small, k=3):
    """👨‍👧‍👦 RAG Parent Document - Busca precisa + contexto amplo."""
    if not small_docs or not large_docs or not vector_store_small: 
        return None
    print(f"🔧 Configurando RAG Parent Document (k={k})")
    
    try:
        # Store para mapear documentos pequenos para grandes
        store = InMemoryStore()
        
        # Mapeia cada chunk pequeno para seu documento pai (grande)
        id_key = "doc_id"
        for i, (small_doc, large_doc) in enumerate(zip(small_docs[:len(large_docs)], large_docs)):
            doc_id = f"doc_{i}"
            store.mset([(doc_id, large_doc)])
            small_doc.metadata[id_key] = doc_id
        
        # Busca com chunks pequenos (precisão) mas retorna chunks grandes (contexto)
        parent_retriever = ParentDocumentRetriever(
            vectorstore=vector_store_small,
            docstore=store,
            child_splitter=RecursiveCharacterTextSplitter(
                chunk_size=RAG_CONFIGS['small_chunk_size'],
                chunk_overlap=RAG_CONFIGS['chunk_overlap']
            ),
            parent_splitter=RecursiveCharacterTextSplitter(
                chunk_size=RAG_CONFIGS['large_chunk_size'],
                chunk_overlap=RAG_CONFIGS['chunk_overlap']
            ),
            search_kwargs={"k": k}
        )
        return parent_retriever
    except Exception as e:
        print(f"❌ Erro ao criar RAG Parent Document: {e}")
        return None

def get_rag_fusion_advanced(documents, vector_store, llm, k=3):
    """🔀 RAG Fusion - Combina múltiplas estratégias inteligentemente."""
    if not documents or not vector_store: 
        return None
    print(f"🔧 Configurando RAG Fusion Avançado (k={k})")
    
    try:
        # 1. Embeddings semânticos
        embedding_retriever = vector_store.as_retriever(search_kwargs={"k": k})
        
        # 2. BM25 para palavras-chave
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = k
        
        # 3. Multi-Query para variações da pergunta
        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=embedding_retriever,
            llm=llm
        )
        
        # Combina as 3 estratégias com pesos otimizados
        fusion_retriever = EnsembleRetriever(
            retrievers=[embedding_retriever, bm25_retriever, multi_query_retriever],
            weights=[0.4, 0.3, 0.3]  # Embeddings + BM25 + Multi-Query
        )
        return fusion_retriever
    except Exception as e:
        print(f"❌ Erro ao criar RAG Fusion: {e}")
        return None

def get_rag_high_precision(vector_store, k=2):
    """🎯 RAG Alta Precisão - Poucos documentos, máxima relevância."""
    if not vector_store: 
        return None
    print(f"🔧 Configurando RAG Alta Precisão (k={k})")
    
    # Maximum Marginal Relevance - balanceia relevância vs diversidade
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": k * 4,      # Analisa mais candidatos
            "lambda_mult": 0.9     # Prioriza relevância (0.9) sobre diversidade (0.1)
        }
    )

def get_rag_high_recall(vector_store, k=8):
    """📈 RAG Alta Cobertura - Máximos documentos para não perder informação."""
    if not vector_store: 
        return None
    print(f"🔧 Configurando RAG Alta Cobertura (k={k})")
    
    # Busca mais documentos para garantir cobertura completa
    return vector_store.as_retriever(search_kwargs={"k": k})

# --- 3. SISTEMA DE AVALIAÇÃO ---

def semantic_similarity_score(generated_answer, ideal_answer, embedding_model=None):
    """Calcula similaridade semântica entre respostas usando embeddings."""
    if not generated_answer or not ideal_answer:
        return 0.0
    
    _embedding_model = embedding_model if embedding_model else OpenAIEmbeddings()
    
    try:
        gen_emb = _embedding_model.embed_query(generated_answer)
        ideal_emb = _embedding_model.embed_query(ideal_answer)
        
        # Similaridade de cosseno
        similarity = np.dot(gen_emb, ideal_emb) / (np.linalg.norm(gen_emb) * np.linalg.norm(ideal_emb))
        return max(0, float(similarity)) * 100  # Porcentagem
    except Exception as e:
        print(f"❌ Erro no cálculo de similaridade: {e}")
        return 0.0

def run_advanced_evaluation(evaluation_dataset, rag_pipelines, llm, system_prompt):
    """Executa avaliação completa de todos os pipelines RAG."""
    if not evaluation_dataset or not rag_pipelines:
        print("❌ Dataset ou pipelines vazios")
        return {}

    all_results = {}
    total_pipelines = len(rag_pipelines)
    
    print(f"\n🎯 INICIANDO AVALIAÇÃO DE {total_pipelines} PIPELINES RAG")
    print("=" * 80)

    for i, (pipeline_name, retriever) in enumerate(rag_pipelines.items(), 1):
        if retriever is None:
            print(f"⏭️  {i}/{total_pipelines} - PULANDO {pipeline_name}: Retriever não inicializado")
            all_results[pipeline_name] = {
                "average_score": 0.0, 
                "details": [], 
                "error": "Retriever não inicializado"
            }
            continue

        print(f"\n🔍 {i}/{total_pipelines} - AVALIANDO: {pipeline_name}")
        print("-" * 60)
        
        try:
            rag_chain = create_rag_chain_from_retriever(retriever, llm, system_prompt)
            pipeline_scores = []
            pipeline_details = []

            for j, item in enumerate(evaluation_dataset, 1):
                question = item["question"]
                ideal_answer = item.get("ideal_answer")
                question_id = item.get("id", f"Q{j}")

                print(f"  📝 {j}/{len(evaluation_dataset)} - Pergunta: {question_id}")
                print(f"      \"{question[:60]}...\"")
                
                try:
                    # Executa a pergunta no pipeline RAG
                    response_payload = rag_chain.invoke({'input': question})
                    generated_answer = response_payload.get('answer', "Resposta não gerada.")
                    retrieved_docs = response_payload.get('context', [])
                    
                    # Calcula score de similaridade
                    score = 0.0
                    if ideal_answer and generated_answer != "Resposta não gerada.":
                        score = semantic_similarity_score(generated_answer, ideal_answer)
                    
                    pipeline_scores.append(score)
                    pipeline_details.append({
                        "id": question_id,
                        "question": question,
                        "ideal_answer": ideal_answer,
                        "generated_answer": generated_answer,
                        "similarity_score": score,
                        "retrieved_context_count": len(retrieved_docs),
                        "pipeline": pipeline_name
                    })
                    
                    # Feedback visual do score
                    if score >= 90:
                        emoji = "🟢"
                    elif score >= 75:
                        emoji = "🟡" 
                    elif score >= 60:
                        emoji = "🟠"
                    else:
                        emoji = "🔴"
                    
                    print(f"      {emoji} Score: {score:.1f}% | Docs: {len(retrieved_docs)}")

                except Exception as e:
                    print(f"      ❌ ERRO: {str(e)[:50]}...")
                    pipeline_scores.append(0)
                    pipeline_details.append({
                        "id": question_id,
                        "question": question,
                        "ideal_answer": ideal_answer,
                        "generated_answer": "ERRO NA GERAÇÃO",
                        "similarity_score": 0.0,
                        "error_message": str(e),
                        "pipeline": pipeline_name
                    })

            # Calcula média do pipeline
            average_score = np.mean(pipeline_scores) if pipeline_scores else 0.0
            all_results[pipeline_name] = {
                "average_score": average_score,
                "details": pipeline_details
            }
            
            # Feedback final do pipeline
            if average_score >= 85:
                status = "🏆 EXCELENTE"
            elif average_score >= 75:
                status = "✅ BOM"
            elif average_score >= 65:
                status = "⚠️  REGULAR"
            else:
                status = "❌ BAIXO"
            
            print(f"  📊 {status} - Score Médio: {average_score:.1f}%")
            
        except Exception as e:
            print(f"  ❌ ERRO CRÍTICO no pipeline: {e}")
            all_results[pipeline_name] = {
                "average_score": 0.0,
                "details": [],
                "error": f"Erro crítico: {str(e)}"
            }
        
    return all_results

def print_advanced_report(evaluation_results):
    """Gera relatório detalhado e ranking dos modelos RAG."""
    print("\n\n" + "🏆" * 30)
    print("🏆" + " " * 10 + "RELATÓRIO FINAL - RAG EVALUATION" + " " * 10 + "🏆")
    print("🏆" * 30)
    
    if not evaluation_results:
        print("❌ Nenhum resultado para reportar.")
        return

    # Filtra e ordena resultados válidos
    valid_results = {
        name: results for name, results in evaluation_results.items() 
        if "error" not in results and results.get("average_score", 0) > 0
    }
    
    error_results = {
        name: results for name, results in evaluation_results.items()
        if "error" in results or results.get("average_score", 0) == 0
    }
    
    if valid_results:
        sorted_results = sorted(
            valid_results.items(), 
            key=lambda x: x[1]["average_score"], 
            reverse=True
        )
        
        print(f"\n📊 RANKING DE PERFORMANCE ({len(sorted_results)} modelos)")
        print("=" * 80)
        print(f"{'🏆':<3} {'MODELO RAG':<35} {'📈 SCORE':<12} {'📊 STATUS'}")
        print("-" * 80)
        
        for i, (pipeline_name, results) in enumerate(sorted_results, 1):
            score = results["average_score"]
            
            # Emojis e status baseados no score
            if i == 1:
                rank_emoji = "🥇"
            elif i == 2:
                rank_emoji = "🥈" 
            elif i == 3:
                rank_emoji = "🥉"
            else:
                rank_emoji = f"{i:2d}."
                
            if score >= 85:
                status = "🏆 EXCELENTE"
            elif score >= 75:
                status = "✅ BOM"
            elif score >= 65:
                status = "⚠️  REGULAR"
            else:
                status = "❌ BAIXO"
            
            # Nome limpo do pipeline
            clean_name = pipeline_name.replace("_", " ").title()
            if clean_name.startswith(("1 ", "2 ", "3 ", "4 ", "5 ", "6 ", "7 ", "8 ", "9 ")):
                clean_name = clean_name[2:]  # Remove numeração
            
            print(f"{rank_emoji:<3} {clean_name:<35} {score:>6.1f}%      {status}")
        
        # Estatísticas gerais
        scores = [r["average_score"] for r in valid_results.values()]
        print("\n📈 ESTATÍSTICAS GERAIS")
        print("-" * 40)
        print(f"🏆 Melhor Score:     {max(scores):.1f}%")
        print(f"📊 Score Médio:      {np.mean(scores):.1f}%")
        print(f"📉 Pior Score:       {min(scores):.1f}%")
        print(f"📏 Desvio Padrão:    {np.std(scores):.1f}%")
        
        # Recomendações
        best_model = sorted_results[0][0]
        print(f"\n💡 RECOMENDAÇÃO")
        print("-" * 40)
        print(f"🌟 Modelo Recomendado: {best_model.replace('_', ' ')}")
        print(f"📝 Motivo: Melhor performance geral ({sorted_results[0][1]['average_score']:.1f}%)")
    
    # Erros (se houver)
    if error_results:
        print(f"\n❌ MODELOS COM ERRO ({len(error_results)})")
        print("-" * 40)
        for name, results in error_results.items():
            error_msg = results.get("error", "Erro desconhecido")
            print(f"🔴 {name}: {error_msg}")
    
    print("\n" + "🏆" * 30)
    print("🎯 Avaliação RAG Avançada Concluída!")
    print("🏆" * 30)

# --- 4. CONFIGURAÇÃO DOS TESTES ---

def setup_advanced_rag_pipelines(standard_docs, small_docs, large_docs, vector_stores, llm):
    """Configura todos os pipelines RAG avançados."""
    print("\n⚙️  CONFIGURANDO PIPELINES RAG AVANÇADOS")
    print("=" * 60)
    
    pipelines = {}
    vs_std = vector_stores.get('standard')
    vs_small = vector_stores.get('small')
    vs_large = vector_stores.get('large')
    
    # 1. RAG Base (Baseline)
    if vs_std:
        ret = get_rag_base_embeddings(vs_std, k=3)
        if ret:
            pipelines["1_RAG_Base_Embeddings"] = ret
            print("✅ RAG Base Embeddings")
    
    # 2. RAG Híbrido BM25 + Embeddings
    if standard_docs and vs_std:
        ret = get_rag_hybrid_bm25_embeddings(standard_docs, vs_std)
        if ret:
            pipelines["2_RAG_Hibrido_BM25_Embeddings"] = ret
            print("✅ RAG Híbrido BM25 + Embeddings")
    
    # 3. RAG com Compressão Contextual
    if vs_std:
        ret = get_rag_contextual_compression(vs_std, llm, k=6)
        if ret:
            pipelines["3_RAG_Compressao_Contextual"] = ret
            print("✅ RAG Compressão Contextual")
    
    # 4. RAG com Threshold de Similaridade
    if vs_std:
        ret = get_rag_similarity_threshold(vs_std, RAG_CONFIGS['similarity_threshold'], k=5)
        if ret:
            pipelines["4_RAG_Threshold_Similaridade"] = ret
            print("✅ RAG Threshold Similaridade")
    
    # 5. RAG com Expansão de Query
    if vs_std:
        ret = get_rag_multi_query(vs_std, llm, k=3)
        if ret:
            pipelines["5_RAG_Expansao_Query"] = ret
            print("✅ RAG Expansão Query")
    
    # 6. RAG Parent Document
    if small_docs and large_docs and vs_small:
        ret = get_rag_parent_document(small_docs, large_docs, vs_small, k=3)
        if ret:
            pipelines["6_RAG_Parent_Document"] = ret
            print("✅ RAG Parent Document")
    
    # 7. RAG Fusion Avançado
    if standard_docs and vs_std:
        ret = get_rag_fusion_advanced(standard_docs, vs_std, llm, k=3)
        if ret:
            pipelines["7_RAG_Fusion_Avancado"] = ret
            print("✅ RAG Fusion Avançado")
    
    # 8. RAG Alta Precisão
    if vs_std:
        ret = get_rag_high_precision(vs_std, k=2)
        if ret:
            pipelines["8_RAG_Alta_Precisao"] = ret
            print("✅ RAG Alta Precisão")
    
    # 9. RAG Alta Cobertura
    if vs_std:
        ret = get_rag_high_recall(vs_std, k=8)
        if ret:
            pipelines["9_RAG_Alta_Cobertura"] = ret
            print("✅ RAG Alta Cobertura")
    
    print(f"\n🎯 TOTAL: {len(pipelines)} pipelines RAG configurados com sucesso!")
    return pipelines

# --- 5. EXECUÇÃO PRINCIPAL ---

if __name__ == "__main__":
    print("🚀" * 20)
    print("🚀" + " " * 6 + "SISTEMA AVANÇADO DE AVALIAÇÃO RAG" + " " * 6 + "🚀")
    print("🚀" * 20)
    
    # ETAPA 1: Carregamento de documentos
    print("\n📂 ETAPA 1: CARREGAMENTO DE DOCUMENTOS")
    print("-" * 50)
    standard_docs, small_docs, large_docs = load_documents_multi_size(DEFAULT_DOCUMENTS_DIR)
    
    if not standard_docs:
        print("❌ Falha no carregamento de documentos. Encerrando.")
        exit(1)
    
    # ETAPA 2: Criação de vector stores
    print("\n🗄️  ETAPA 2: CRIAÇÃO DE VECTOR STORES")
    print("-" * 50)
    vector_stores = create_vector_stores(standard_docs, small_docs, large_docs, force_recreate=False)
    
    if not vector_stores:
        print("❌ Falha na criação de vector stores. Encerrando.")
        exit(1)
    
    # ETAPA 3: Configuração do LLM
    print("\n🤖 ETAPA 3: CONFIGURAÇÃO DO LLM")
    print("-" * 50)
    llm_instance = get_llm(DEFAULT_LLM_MODEL)
    
    # ETAPA 4: Configuração dos pipelines RAG
    print("\n⚙️  ETAPA 4: CONFIGURAÇÃO DOS PIPELINES RAG")
    print("-" * 50)
    rag_pipelines = setup_advanced_rag_pipelines(
        standard_docs, small_docs, large_docs, vector_stores, llm_instance
    )
    
    if not rag_pipelines:
        print("❌ Nenhum pipeline RAG configurado. Encerrando.")
        exit(1)
    
    # ETAPA 5: Dataset de avaliação
    print("\n📝 ETAPA 5: PREPARAÇÃO DO DATASET")
    print("-" * 50)
    
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
        }
    ]
    
    print(f"✅ Dataset carregado: {len(EVALUATION_DATASET)} perguntas de teste")
    
    # ETAPA 6: System prompt customizado
    system_prompt = """Você é um assistente especializado em responder perguntas sobre o regulamento de consórcio.

INSTRUÇÕES IMPORTANTES:
- Use APENAS as informações disponíveis no contexto fornecido
- Se a informação não estiver no contexto, diga claramente que não sabe
- Mantenha respostas concisas, profissionais e baseadas nos fatos
- Responda sempre em português do Brasil
- Cite especificamente as leis/resoluções quando mencionadas no contexto

Contexto Fornecido: {context}"""
    
    # ETAPA 7: Execução da avaliação
    print("\n🎯 ETAPA 6: EXECUÇÃO DA AVALIAÇÃO")
    print("-" * 50)
    print(f"🔬 Testando {len(rag_pipelines)} modelos RAG diferentes")
    print(f"📝 Usando {len(EVALUATION_DATASET)} perguntas de teste")
    print(f"🤖 Modelo LLM: {DEFAULT_LLM_MODEL}")
    
    evaluation_results = run_advanced_evaluation(
        evaluation_dataset=EVALUATION_DATASET,
        rag_pipelines=rag_pipelines,
        llm=llm_instance,
        system_prompt=system_prompt
    )
    
    # ETAPA 8: Relatório final
    print_advanced_report(evaluation_results)
    
    # ETAPA 9: Salvamento dos resultados
    print("\n💾 SALVANDO RESULTADOS DETALHADOS")
    print("-" * 50)
    try:
        import json
        timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"rag_evaluation_advanced_{timestamp}.json"
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Resultados salvos em: {filename}")
        print(f"📊 Arquivo contém dados detalhados de {len(evaluation_results)} pipelines")
        
    except Exception as e:
        print(f"❌ Erro ao salvar resultados: {e}")
    
    print("\n🎉 AVALIAÇÃO RAG AVANÇADA CONCLUÍDA COM SUCESSO! 🎉") 