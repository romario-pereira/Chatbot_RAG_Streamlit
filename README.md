# 🤖 Assistente Empresarial - Chatbot RAG com Streamlit

Bem-vindo ao **Assistente Empresarial**, um chatbot inteligente que utiliza a técnica de RAG (Retrieval-Augmented Generation) com modelos GPT da OpenAI para responder perguntas sobre a sua empresa, usando como base documentos PDF previamente inseridos. Tudo isso em uma interface moderna, elegante e fácil de usar, desenvolvida com Streamlit.

## 🚀 Funcionalidades

- Interface de chat moderna e intuitiva (estilo ChatGPT)
- Responde apenas com base nos documentos da empresa (RAG)
- Suporte a múltiplos PDFs (basta colocar na pasta `documents`)
- Histórico de conversa para contexto nas respostas
- Respostas em português do Brasil
- Fácil de rodar localmente
- Segurança: nunca exponha suas chaves ou dados sensíveis

## 🗂️ Estrutura do Projeto

```
Chatbot_RAG_BARE/
│
├── app.py              # Código principal do Streamlit
├── requirements.txt    # Dependências do projeto
├── .gitignore          # Arquivos/pastas ignorados pelo git
├── documents/          # Coloque aqui seus PDFs da empresa
├── db/                 # Banco vetorial (gerado automaticamente)
└── .env                # Sua chave da OpenAI (NÃO subir para o GitHub)
```

## ⚙️ Como rodar o projeto

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/SEU_USUARIO/Chatbot_RAG_Streamlit.git
   cd Chatbot_RAG_Streamlit
   ```

2. **Crie e ative um ambiente virtual (recomendado):**
   ```bash
   python -m venv .venv
   # Ative no Windows:
   .venv\Scripts\activate
   # Ou no Linux/Mac:
   source .venv/bin/activate
   ```

3. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Adicione sua chave da OpenAI no arquivo `.env`:**
   ```env
   OPENAI_API_KEY=sua-chave-aqui
   ```

5. **Coloque seus arquivos PDF na pasta `documents/`.**

6. **Rode o aplicativo:**
   ```bash
   streamlit run app.py
   ```

7. **Acesse no navegador:**  
   Normalmente em [http://localhost:8501](http://localhost:8501)

---

## 🛡️ Segurança

- **NUNCA** suba seu arquivo `.env` para o GitHub.
- Troque sua chave da OpenAI caso ela tenha sido exposta.
- O chatbot só responde perguntas baseadas nos documentos da empresa, garantindo privacidade e foco.

## 💡 Dicas

- Para melhor performance, utilize PDFs bem estruturados e claros.
- O histórico do chat é utilizado para dar mais contexto às respostas.
- Personalize o visual do Streamlit conforme a identidade da sua empresa!

---

## 📄 Licença

Este projeto é livre para uso educacional e corporativo. Sinta-se à vontade para adaptar conforme sua necessidade.

---

Desenvolvido com ❤️ por Romário Aquino. 