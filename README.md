# Agente de Análise de Dados

Análise inteligente de arquivos CSV usando IA. Faça perguntas em linguagem natural e receba análises e gráficos automaticamente.

## Como Rodar

### 1. Clone o repositório
```bash
git clone https://github.com/joaoalcantarabarra/csv-analysis-agent.git
cd csv-analysis-agent
```

### 2. Crie um ambiente virtual
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

### 3. Instale as dependências
```bash
poetry install
```

### 4. Execute a aplicação
```bash
streamlit run app.py
```

### 5. Configure sua API Key
- Acesse `http://localhost:8501`
- Insira sua API Key do Google Gemini
- Obtenha sua chave em: https://aistudio.google.com/api-keys

## Como Usar

1. Faça upload dos seus arquivos CSV
2. Faça perguntas sobre os dados em linguagem natural
3. Receba análises, estatísticas e gráficos automaticamente

## Tecnologias

- Python 3.13
- Streamlit
- LangChain
- LangGraph
- Google Gemini
- Pandas
