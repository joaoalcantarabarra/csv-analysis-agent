from pathlib import Path
from typing import Optional, Type

from langchain.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field


class ListCSVsInput(BaseModel):
    """Input schema for listing CSV files."""

    data_directory: str = Field(
        default="data",
        description="Directory path where CSV files are located",
    )


class ListAvailableCSVsTool(BaseTool):
    name: str = 'list_available_csvs'
    description: str = (
        "List all CSV files available in a specified directory. "
        "Use data_directory='data' by default."
    )
    args_schema: Type[BaseModel] = ListCSVsInput
    return_direct: bool = False

    def _run(
        self,
        data_directory: str = "data",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the tool to list CSV files."""
        try:
            data_dir = Path(data_directory)

            if not data_dir.exists():
                return f"Directory '{data_directory}' does not exist. Please check the path."

            csv_files = list(data_dir.glob('*.csv'))

            if not csv_files:
                return f"No CSV files found in directory '{data_directory}'."

            result = f"Available CSVs in directory '{data_directory}':\n\n"

            for i, csv_file in enumerate(sorted(csv_files), 1):
                size = csv_file.stat().st_size
                size_mb = size / (1024 * 1024)

                result += f'{i}. {csv_file.name}\n'
                result += f'   Path: {csv_file}\n'
                result += f'   Size: {size_mb:.2f} MB\n\n'

            result += f'Total: {len(csv_files)} CSV file(s) found'

            return result

        except Exception as e:
            return f'Error listing CSVs: {str(e)}'

    async def _arun(
        self,
        data_directory: str = "data",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Async version of the tool."""
        return self._run(
            data_directory=data_directory, run_manager=run_manager
        )
