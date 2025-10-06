"""
Tool para listar arquivos CSV disponíveis - Versão LangChain
"""
from pathlib import Path
from typing import Type

from langchain.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field


class ListCSVsInput(BaseModel):
    """Input schema for listing CSV files."""

    data_directory: str = Field(
        ...,
        description="Diretório onde os arquivos CSV estão localizados.",
    )

    class Config:
        extra = 'forbid'


class ListAvailableCSVsTool(BaseTool):
    name: str = 'list_available_csvs'
    description: str = """Lista todos os arquivos CSV disponíveis em um diretório especificado.
    IMPORTANTE: Sempre use data_directory='data' a menos que o usuário especifique outro diretório."""
    args_schema: Type[BaseModel] = ListCSVsInput
    return_direct: bool = False

    def _run(
        self,
        data_directory: str,
        run_manager: CallbackManagerForToolRun = None,
    ) -> str:
        try:
            data_dir = Path(data_directory)

            if not data_dir.exists():
                return f"Diretório '{data_directory}' não existe. Certifique-se de que os CSVs estão na pasta correta."

            csv_files = list(data_dir.glob('*.csv'))

            if not csv_files:
                return f"Nenhum arquivo CSV encontrado no diretório '{data_directory}'."

            result = f"CSVs disponíveis no diretório '{data_directory}':\n\n"

            for i, csv_file in enumerate(sorted(csv_files), 1):
                size = csv_file.stat().st_size
                size_mb = size / (1024 * 1024)

                result += f'{i}. {csv_file.name}\n'
                result += f'   Caminho: {csv_file}\n'
                result += f'   Tamanho: {size_mb:.2f} MB\n\n'

            result += f'Total: {len(csv_files)} arquivo(s) CSV encontrado(s)'

            return result

        except Exception as e:
            return f'Erro ao listar CSVs: {str(e)}'

    async def _arun(
        self,
        data_directory: str,
        run_manager: CallbackManagerForToolRun = None,
    ) -> str:
        return self._run(
            data_directory=data_directory, run_manager=run_manager
        )
