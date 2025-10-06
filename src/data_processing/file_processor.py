import atexit
import io
import os
import re
import shutil
import unicodedata
import zipfile
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st


class DataFileProcessor:
    """Class to process different types of data files."""

    DATA_DIR = 'data'

    @classmethod
    def initialize_data_directory(cls):
        """
        Initialize the data directory by cleaning old files if they exist.
        Also registers the cleanup function to run when the app exits.
        """
        try:
            if os.path.exists(cls.DATA_DIR):
                shutil.rmtree(cls.DATA_DIR)
                print(f'🗑️ Diretório {cls.DATA_DIR} limpo na inicialização')

            os.makedirs(cls.DATA_DIR, exist_ok=True)
            print(f'📁 Diretório {cls.DATA_DIR} criado/inicializado')

            atexit.register(cls.cleanup_data_directory)

        except Exception as e:
            print(f'⚠️ Erro ao inicializar diretório {cls.DATA_DIR}: {str(e)}')

    @classmethod
    def cleanup_data_directory(cls):
        """
        Clean the data directory when the app is terminated.
        This function is registered with atexit.register().
        """
        try:
            if os.path.exists(cls.DATA_DIR):
                shutil.rmtree(cls.DATA_DIR)
                print(f'🗑️ Diretório {cls.DATA_DIR} limpo no encerramento')
        except Exception as e:
            print(f'⚠️ Erro ao limpar diretório {cls.DATA_DIR}: {str(e)}')

    @classmethod
    def cleanup_specific_files(cls, filenames: List[str]):
        """
        Remove specific files from the data directory.

        Args:
            filenames (List[str]): List of file names to remove.
        """
        for filename in filenames:
            file_path = os.path.join(cls.DATA_DIR, filename)
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f'🗑️ Arquivo {filename} removido')
            except Exception as e:
                print(f'⚠️ Erro ao remover {filename}: {str(e)}')

    @staticmethod
    def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize column names by removing accents, spaces, and special characters.

        Args:
            df (pd.DataFrame): DataFrame with columns to normalize.

        Returns:
            pd.DataFrame: DataFrame with normalized column names.
        """

        def normalize_string(text: str) -> str:
            """Normalize a single string into a clean, lowercase, underscore format.

            Args:
                text (str): The string to normalize.

            Returns:
                str: The normalized string.
            """
            if not isinstance(text, str):
                text = str(text)

            normalized = unicodedata.normalize('NFD', text)
            without_accents = ''.join(
                c for c in normalized if unicodedata.category(c) != 'Mn'
            )

            cleaned = re.sub(r'\s+', '_', without_accents.strip())
            cleaned = re.sub(r'[^\w]', '_', cleaned)
            cleaned = re.sub(r'_+', '_', cleaned)
            cleaned = cleaned.strip('_').lower()

            if not cleaned:
                cleaned = 'coluna_sem_nome'

            return cleaned

        df_normalized = df.copy()
        normalized_columns = []
        column_counts: Dict[str, int] = {}

        for col in df.columns:
            normalized_col = normalize_string(col)

            if normalized_col in column_counts:
                column_counts[normalized_col] += 1
                normalized_col = (
                    f'{normalized_col}_{column_counts[normalized_col]}'
                )
            else:
                column_counts[normalized_col] = 0

            normalized_columns.append(normalized_col)

        df_normalized.columns = normalized_columns

        changes = [
            (orig, norm)
            for orig, norm in zip(df.columns, normalized_columns)
            if orig != norm
        ]
        if changes:
            st.info(
                f'🔄 Nomes de colunas normalizados: {len(changes)} alteração(ões)'
            )

        return df_normalized

    @classmethod
    def read_csv_file(
        cls, file_content: bytes, filename: str
    ) -> tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Read a CSV file with multiple encoding attempts and save to data directory.

        Args:
            file_content (bytes): Raw content of the CSV file.
            filename (str): Name of the file being read.

        Returns:
            tuple[Optional[pd.DataFrame], Optional[str]]: Tuple containing (DataFrame, file_path) if successful, otherwise (None, None).
        """
        try:
            os.makedirs(cls.DATA_DIR, exist_ok=True)

            file_path = os.path.join(cls.DATA_DIR, filename)

            with open(file_path, 'wb') as f:
                f.write(file_content)

            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

            for encoding in encodings:
                try:
                    df = pd.read_csv(
                        io.BytesIO(file_content), encoding=encoding
                    )
                    df = cls.normalize_column_names(df)
                    return df, file_path
                except UnicodeDecodeError:
                    continue

            df = pd.read_csv(
                io.BytesIO(file_content), encoding='utf-8', errors='ignore'
            )

            df = cls.normalize_column_names(df)
            return df, file_path

        except Exception as e:
            st.error(f'Erro ao ler CSV {filename}: {str(e)}')
            try:
                if 'file_path' in locals() and os.path.exists(file_path):
                    os.remove(file_path)
            except:
                pass
            return None, None

    @classmethod
    def process_zip_file(
        cls, zip_content: bytes
    ) -> Dict[str, tuple[pd.DataFrame, str]]:
        """
        Process a ZIP file containing CSV files.

        Args:
            zip_content (bytes): Raw content of the ZIP file.

        Returns:
            Dict[str, tuple[pd.DataFrame, str]]: Dictionary of file names to (DataFrame, file_path) tuples.
        """
        datasets: Dict[str, tuple[pd.DataFrame, str]] = {}

        try:
            with zipfile.ZipFile(io.BytesIO(zip_content), 'r') as zip_ref:
                for file_info in zip_ref.infolist():
                    if file_info.is_dir():
                        continue

                    filename = file_info.filename
                    file_extension = filename.lower().split('.')[-1]

                    if file_extension in ['csv']:
                        try:
                            file_content = zip_ref.read(file_info)

                            if file_extension == 'csv':
                                df, file_path = cls.read_csv_file(
                                    file_content, filename
                                )
                                if df is not None and file_path is not None:
                                    datasets[filename] = (df, file_path)
                        except Exception as e:
                            st.warning(
                                f'Erro ao processar {filename}: {str(e)}'
                            )
                            continue

        except Exception as e:
            st.error(f'Erro ao processar arquivo ZIP: {str(e)}')

        return datasets

    @classmethod
    def get_data_directory_info(cls) -> Dict[str, any]:
        """
        Get information about the data directory.

        Returns:
            Dict[str, Any]: Information about the directory (exists, files, total size, etc).
        """
        info = {
            'exists': os.path.exists(cls.DATA_DIR),
            'path': os.path.abspath(cls.DATA_DIR),
            'files': [],
            'total_size_mb': 0,
        }

        if info['exists']:
            try:
                for filename in os.listdir(cls.DATA_DIR):
                    file_path = os.path.join(cls.DATA_DIR, filename)
                    if os.path.isfile(file_path):
                        file_size = os.path.getsize(file_path)
                        info['files'].append(
                            {
                                'name': filename,
                                'size_mb': file_size / (1024 * 1024),
                                'path': file_path,
                            }
                        )
                        info['total_size_mb'] += file_size / (1024 * 1024)
            except Exception as e:
                print(f'Erro ao obter informações do diretório: {str(e)}')

        return info
