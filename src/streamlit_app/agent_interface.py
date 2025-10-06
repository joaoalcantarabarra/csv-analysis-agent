import os
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from src.agents.agent_csv import analyze_csv, create_csv_agent, output_parser
from src.data_processing.file_processor import DataFileProcessor


class AgentInterface:
    """Main class for the agent interface."""

    def __init__(self) -> None:
        """Initialize the agent interface by setting up data directory and session state."""
        self.initialize_data_directory()
        self.initialize_session_state()

    def initialize_data_directory(self) -> None:
        """Initialize the data directory only once per app execution."""

        @st.cache_resource
        def setup_data_directory() -> bool:
            """Set up the data directory and return True if successful."""
            DataFileProcessor.initialize_data_directory()
            return True

        setup_data_directory()

    def initialize_session_state(self) -> None:
        """Initialize session state variables for the application."""
        if 'messages' not in st.session_state:
            st.session_state.messages = []

        if 'datasets' not in st.session_state:
            st.session_state.datasets = {}

        if 'processed_files' not in st.session_state:
            st.session_state.processed_files = []

        if 'agent_ready' not in st.session_state:
            st.session_state.agent_ready = False

        if 'data_files' not in st.session_state:
            st.session_state.data_files = {}

        if 'api_key_configured' not in st.session_state:
            st.session_state.api_key_configured = False

        if 'gemini_api_key' not in st.session_state:
            st.session_state.gemini_api_key = None

    def setup_page_config(self) -> None:
        """Configure the Streamlit page settings."""
        st.set_page_config(
            page_title='Agente de Análise de Dados',
            page_icon='🤖',
            layout='wide',
            initial_sidebar_state='expanded',
        )

    def render_api_key_setup(self) -> bool:
        """
        Render the API Key setup screen and return True if key is configured.

        Returns:
            bool: True if API key is configured, False otherwise.
        """
        st.title('🔐 Configuração da API Key')
        st.markdown('---')

        st.markdown(
            """
        ### Bem-vindo ao Agente de Análise de Dados!
        
        Para começar, você precisa configurar sua **API Key do Google Gemini**.
        
        #### Como obter sua API Key:
        1. Acesse [Google AI Studio](https://aistudio.google.com/api-keys)
        2. Faça login com sua conta Google
        3. Clique em "Create API Key"
        4. Copie a chave gerada
        5. Cole abaixo e clique em "Configurar"
        
        **Nota:** Sua API Key será armazenada apenas durante esta sessão e não será salva permanentemente.
        """
        )

        st.markdown('---')

        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            api_key = st.text_input(
                '🔑 API Key do Google Gemini',
                type='password',
                placeholder='Digite sua API Key aqui...',
                help='Cole aqui a API Key que você obteve no Google AI Studio',
            )

            col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

            with col_btn2:
                if st.button(
                    '✅ Configurar API Key',
                    type='primary',
                    use_container_width=True,
                ):
                    if api_key and len(api_key.strip()) > 0:
                        if len(api_key.strip()) < 20:
                            st.error(
                                '⚠️ A API Key parece muito curta. Verifique se você copiou corretamente.'
                            )
                            return False

                        st.session_state.gemini_api_key = api_key.strip()
                        st.session_state.api_key_configured = True

                        os.environ['GOOGLE_API_KEY'] = api_key.strip()

                        st.success('✅ API Key configurada com sucesso!')
                        st.balloons()
                        st.rerun()
                    else:
                        st.error('⚠️ Por favor, insira uma API Key válida.')
                        return False

        st.markdown('---')

        with st.expander('❓ Precisa de ajuda?'):
            st.markdown(
                """
            **Problemas comuns:**
            
            - **Não consigo encontrar a API Key:** Certifique-se de estar logado no Google AI Studio
            - **A API Key não funciona:** Verifique se você copiou a chave completa, sem espaços extras
            - **Erro de permissões:** Certifique-se de que sua conta Google tem acesso ao Gemini API
            
            **Segurança:**
            - Sua API Key nunca é armazenada em disco
            - A chave existe apenas durante esta sessão
            - Ao fechar o navegador, a chave é removida automaticamente
            """
            )

        return False

    def render_header(self) -> None:
        """Render the application header with title and description."""
        col_title, col_api = st.columns([4, 1])

        with col_title:
            st.title('🤖 Agente de Análise de Dados')

        with col_api:
            if st.button(
                '🔄 Reconfigurar API', help='Alterar a API Key do Gemini'
            ):
                st.session_state.api_key_configured = False
                st.session_state.gemini_api_key = None
                if 'GOOGLE_API_KEY' in os.environ:
                    del os.environ['GOOGLE_API_KEY']
                st.rerun()

        st.markdown('---')

        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(
                '**Faça upload dos seus dados e converse com o agente para análises inteligentes!**'
            )

        with col2:
            if st.session_state.datasets:
                st.success(
                    f'📊 {len(st.session_state.datasets)} dataset(s) carregado(s)'
                )
            else:
                st.info('📁 Nenhum dataset carregado')

    def render_sidebar(self) -> None:
        """Render the sidebar for file upload and dataset management."""
        with st.sidebar:
            st.header('📁 Upload de Dados')

            uploaded_files = st.file_uploader(
                'Selecione arquivos CSV',
                type=['csv'],
                accept_multiple_files=True,
                help='Você pode selecionar múltiplos arquivos CSV',
            )

            uploaded_zip = st.file_uploader(
                'Ou selecione um arquivo ZIP',
                type=['zip'],
                help='ZIP contendo arquivos CSV',
            )

            if uploaded_files:
                self.process_individual_files(uploaded_files)

            if uploaded_zip:
                self.process_zip_file(uploaded_zip)

            if st.session_state.datasets:
                st.markdown('### 📊 Datasets Carregados')
                for dataset_name, df in st.session_state.datasets.items():
                    with st.expander(f'📋 {dataset_name}'):
                        st.write(f'**Shape:** {df.shape}')
                        st.write(f'**Colunas:** {list(df.columns)}')

                        if dataset_name in st.session_state.data_files:
                            st.write(
                                f'**Arquivo:** `{st.session_state.data_files[dataset_name]}`'
                            )

                        if st.checkbox(
                            f'Preview de {dataset_name}',
                            key=f'preview_{dataset_name}',
                        ):
                            st.dataframe(df.head(3), use_container_width=True)

                if st.button('🗑️ Limpar Todos os Datasets', type='secondary'):
                    self.clear_all_datasets()

            self.render_data_directory_info()

    def render_data_directory_info(self) -> None:
        """Render information about the data directory."""
        with st.expander('🗂️ Informações do Diretório de Dados'):
            data_info = DataFileProcessor.get_data_directory_info()

            if data_info['exists']:
                st.write(f"**Caminho:** `{data_info['path']}`")
                st.write(f"**Arquivos:** {len(data_info['files'])}")
                st.write(
                    f"**Tamanho Total:** {data_info['total_size_mb']:.2f} MB"
                )

                if data_info['files']:
                    st.write('**Arquivos no diretório:**')
                    for file_info in data_info['files']:
                        st.write(
                            f"- {file_info['name']} ({file_info['size_mb']:.2f} MB)"
                        )
            else:
                st.write('Diretório de dados não existe ou está vazio')

            if st.button('🗑️ Limpar Diretório de Dados', key='clear_data_dir'):
                DataFileProcessor.cleanup_data_directory()
                DataFileProcessor.initialize_data_directory()
                st.rerun()

    def clear_all_datasets(self) -> None:
        """Clear all loaded datasets and related files from session state."""
        if st.session_state.data_files:
            file_names = list(st.session_state.data_files.keys())
            DataFileProcessor.cleanup_specific_files(file_names)

        st.session_state.datasets = {}
        st.session_state.processed_files = []
        st.session_state.messages = []
        st.session_state.data_files = {}
        st.rerun()

    def process_individual_files(
        self,
        uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile],
    ) -> None:
        """Process individual uploaded CSV files and store them in session state."""
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.processed_files:
                file_content: bytes = uploaded_file.read()
                file_extension: str = uploaded_file.name.lower().split('.')[-1]

                if file_extension == 'csv':
                    df, file_path = DataFileProcessor.read_csv_file(
                        file_content, uploaded_file.name
                    )
                    if df is not None and file_path is not None:
                        st.session_state.datasets[uploaded_file.name] = df
                        st.session_state.data_files[
                            uploaded_file.name
                        ] = file_path
                        st.session_state.processed_files.append(
                            uploaded_file.name
                        )
                        st.success(
                            f'✅ {uploaded_file.name} carregado com sucesso!'
                        )

    def process_zip_file(
        self, uploaded_zip: st.runtime.uploaded_file_manager.UploadedFile
    ) -> None:
        """Process an uploaded ZIP file and extract datasets into session state."""
        if uploaded_zip.name not in st.session_state.processed_files:
            zip_content: bytes = uploaded_zip.read()
            datasets: Dict[
                str, Tuple[pd.DataFrame, str]
            ] = DataFileProcessor.process_zip_file(zip_content)

            if datasets:
                for filename, (df, file_path) in datasets.items():
                    st.session_state.datasets[filename] = df
                    st.session_state.data_files[filename] = file_path

                st.session_state.processed_files.append(uploaded_zip.name)
                st.success(
                    f'✅ ZIP {uploaded_zip.name} processado! {len(datasets)} arquivo(s) carregado(s).'
                )

    def render_response_with_images(self, response_text: str):
        """
        Render the response text with images.

        Args:
            response_text (str): The response text to be rendered.
        """
        image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'

        parts = re.split(image_pattern, response_text)

        i = 0
        while i < len(parts):
            if parts[i] and not (
                i + 2 < len(parts) and parts[i + 1] and parts[i + 2]
            ):
                st.markdown(parts[i])
                i += 1
            elif i + 2 < len(parts):
                if parts[i]:
                    st.markdown(parts[i])

                alt_text = parts[i + 1]
                image_path = parts[i + 2]

                if os.path.exists(image_path):
                    st.image(
                        image_path,
                        caption=alt_text,
                        width=400,
                    )
                else:
                    st.error(f'Imagem não encontrada: {image_path}')

                i += 3
            else:
                i += 1

    def call_agent(self, user_message: str) -> str:
        """
        Call the agent to process user message.

        Args:
            user_message: User's message.

        Returns:
            Agent response as a string.
        """
        try:
            if 'csv_agent' not in st.session_state:
                st.session_state.csv_agent = create_csv_agent()

            agent = st.session_state.csv_agent
            response = analyze_csv(agent, user_message)
            return output_parser(response)
        except Exception as e:
            return f'Erro ao processar solicitação: {str(e)}'

    def render_data_overview(self) -> None:
        """Render an overview of the loaded datasets."""
        if st.session_state.datasets:
            st.header('📊 Visão Geral dos Dados')

            col1, col2, col3, col4 = st.columns(4)

            total_datasets: int = len(st.session_state.datasets)
            total_rows: int = sum(
                df.shape[0] for df in st.session_state.datasets.values()
            )
            total_cols: int = sum(
                df.shape[1] for df in st.session_state.datasets.values()
            )
            memory_usage: float = (
                sum(
                    df.memory_usage(deep=True).sum()
                    for df in st.session_state.datasets.values()
                )
                / 1024
                / 1024
            )

            col1.metric('📋 Datasets', total_datasets)
            col2.metric('📈 Total de Linhas', f'{total_rows:,}')
            col3.metric('📊 Total de Colunas', total_cols)
            col4.metric('💾 Memória (MB)', f'{memory_usage:.1f}')

            selected_dataset: Optional[str] = st.selectbox(
                'Selecione um dataset para visualizar:',
                options=list(st.session_state.datasets.keys()),
                key='dataset_selector',
            )

            if selected_dataset:
                df: pd.DataFrame = st.session_state.datasets[selected_dataset]

                col1, col2 = st.columns([2, 1])

                with col1:
                    st.subheader(f'📋 {selected_dataset}')
                    st.dataframe(df, use_container_width=True)

                with col2:
                    st.subheader('ℹ️ Informações')
                    st.write(f'**Shape:** {df.shape}')

                    if selected_dataset in st.session_state.data_files:
                        st.write(
                            f'**Arquivo:** `{st.session_state.data_files[selected_dataset]}`'
                        )

                    st.write('**Tipos de dados:**')

                    type_counts = df.dtypes.value_counts()
                    for dtype, count in type_counts.items():
                        st.write(f'- {dtype}: {count} coluna(s)')

                    missing_data = df.isnull().sum()
                    if missing_data.any():
                        st.write('**Valores ausentes:**')
                        for col, missing in missing_data[
                            missing_data > 0
                        ].items():
                            st.write(f'- {col}: {missing}')
        else:
            st.info(
                '📁 Nenhum dataset carregado. Faça upload de arquivos CSV na barra lateral para começar.'
            )

    def run(self) -> None:
        """Run the main Streamlit application."""
        self.setup_page_config()

        if not st.session_state.api_key_configured:
            self.render_api_key_setup()
            return

        self.render_header()
        self.render_sidebar()

        tab1, tab2 = st.tabs(['💬 Chat', '📊 Dados'])

        with tab1:
            st.header('💬 Conversa')

            for message in st.session_state.messages:
                with st.chat_message(message['role']):
                    st.markdown(message['content'])

            if not st.session_state.messages:
                st.info(
                    '👋 Olá! Sou seu assistente de análise de dados. Como posso ajudá-lo hoje?'
                )

        with tab2:
            self.render_data_overview()

        st.markdown('---')

        if prompt := st.chat_input('Digite sua pergunta sobre os dados...'):
            st.session_state.messages.append(
                {'role': 'user', 'content': prompt}
            )

            with st.chat_message('user'):
                st.markdown(prompt)

            with st.chat_message('assistant'):
                with st.spinner('🤔 Gerando a resposta...'):
                    try:
                        response: str = self.call_agent(prompt)
                        self.render_response_with_images(response)
                        st.session_state.messages.append(
                            {'role': 'assistant', 'content': response}
                        )
                    except Exception as e:
                        error_message = (
                            f'Erro ao processar solicitação: {str(e)}'
                        )
                        st.markdown(error_message)
                        st.session_state.messages.append(
                            {'role': 'assistant', 'content': error_message}
                        )
