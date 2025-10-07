import json
import os
from typing import Any, Dict, List, Optional, Type, Union

import matplotlib
import matplotlib.pyplot as plt
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool, BaseToolkit
from pydantic import BaseModel, Field


class ChartDataInput(BaseModel):
    data: Union[
        Dict[str, Union[int, float]], List[Dict[str, Any]], str
    ] = Field(
        ...,
        description='Data for the chart. Can be a dictionary, list of dictionaries, or JSON string.',
    )
    title: str = Field(default='Chart', description='Title of the chart')
    x_label: str = Field(default='X-axis', description='Label for x-axis')
    y_label: str = Field(default='Y-axis', description='Label for y-axis')
    filename: Optional[str] = Field(
        default=None,
        description='Custom filename for the chart image (just the filename, not the full path)',
    )


class PieChartInput(BaseModel):
    data: Union[
        Dict[str, Union[int, float]], List[Dict[str, Any]], str
    ] = Field(
        ...,
        description='Data for the chart. Can be a dictionary, list of dictionaries, or JSON string.',
    )
    title: str = Field(default='Pie Chart', description='Title of the chart')
    filename: Optional[str] = Field(
        default=None,
        description='Custom filename for the chart image (just the filename, not the full path)',
    )


class ScatterPlotInput(BaseModel):
    x_data: Optional[List[Union[int, float]]] = Field(
        default=None, description='List of x-values'
    )
    y_data: Optional[List[Union[int, float]]] = Field(
        default=None, description='List of y-values'
    )
    title: str = Field(
        default='Scatter Plot', description='Title of the chart'
    )
    x_label: str = Field(default='X-axis', description='Label for x-axis')
    y_label: str = Field(default='Y-axis', description='Label for y-axis')
    filename: Optional[str] = Field(
        default=None,
        description='Custom filename for the chart image (just the filename, not the full path)',
    )
    x: Optional[List[Union[int, float]]] = Field(
        default=None, description='Alternative parameter for x-values'
    )
    y: Optional[List[Union[int, float]]] = Field(
        default=None, description='Alternative parameter for y-values'
    )
    data: Optional[
        Union[
            List[List[Union[int, float]]], Dict[str, List[Union[int, float]]]
        ]
    ] = Field(
        default=None,
        description="Alternative format - list of [x,y] pairs or dict with 'x' and 'y' keys",
    )


class HistogramInput(BaseModel):
    data: List[Union[int, float]] = Field(
        ..., description='List of numeric values to plot'
    )
    bins: int = Field(
        default=10, description='Number of bins for the histogram'
    )
    title: str = Field(default='Histogram', description='Title of the chart')
    x_label: str = Field(default='Values', description='Label for x-axis')
    y_label: str = Field(default='Frequency', description='Label for y-axis')
    filename: Optional[str] = Field(
        default=None,
        description='Custom filename for the chart image (just the filename, not the full path)',
    )


class BarChartTool(BaseTool):
    name: str = 'create_bar_chart'
    description: str = (
        'Create a bar chart from the provided data. '
        'Data should be a dictionary with categories as keys and values as numbers, '
        'or a list of dictionaries, or a JSON string. '
        'IMPORTANT: For filename, provide ONLY the filename (e.g., "chart.png"), NOT the full path.'
    )
    args_schema: Type[BaseModel] = ChartDataInput
    output_dir: str = Field(default='charts')

    def _normalize_filename(self, filename: Optional[str]) -> str:
        """
        Normalize filename to avoid path duplication.
        Removes any directory prefix to ensure clean filename.
        """
        if filename is None:
            return None

        filename = os.path.basename(filename)

        if '/' in filename:
            filename = filename.split('/')[-1]

        return filename

    def _normalize_data_for_charts(
        self, data: Union[Dict[str, Any], List[Dict[str, Any]], List[Any], str]
    ) -> Dict[str, Union[int, float]]:
        if isinstance(data, dict):
            return {
                str(k): float(v) if isinstance(v, (int, float)) else 0
                for k, v in data.items()
            }
        elif isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], dict):
                result = {}
                for item in data:
                    if isinstance(item, dict):
                        keys = list(item.keys())
                        if len(keys) >= 2:
                            label_key = keys[0]
                            value_key = keys[1]
                            result[str(item[label_key])] = (
                                float(item[value_key])
                                if isinstance(item[value_key], (int, float))
                                else 0
                            )
                return result
            else:
                return {
                    f'Item {i + 1}': float(v)
                    if isinstance(v, (int, float))
                    else 0
                    for i, v in enumerate(data)
                }
        return {'Data': 1.0}

    def _run(
        self,
        data: Union[Dict[str, Union[int, float]], List[Dict[str, Any]], str],
        title: str = 'Bar Chart',
        x_label: str = 'Categories',
        y_label: str = 'Values',
        filename: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            matplotlib.use('Agg')

            os.makedirs(self.output_dir, exist_ok=True)

            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    pass

            normalized_data = self._normalize_data_for_charts(data)
            categories = list(normalized_data.keys())
            values = list(normalized_data.values())

            plt.figure(figsize=(10, 6))
            plt.bar(categories, values)
            plt.title(title)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            filename = self._normalize_filename(filename)

            if filename is None:
                filename = (
                    f'bar_chart_{len(os.listdir(self.output_dir)) + 1}.png'
                )

            file_path = os.path.join(self.output_dir, filename)
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()

            return json.dumps(
                {
                    'chart_type': 'bar_chart',
                    'title': title,
                    'file_path': file_path,
                    'data_points': len(normalized_data),
                    'status': 'success',
                }
            )
        except Exception as e:
            return json.dumps(
                {'chart_type': 'bar_chart', 'error': str(e), 'status': 'error'}
            )


class LineChartTool(BaseTool):
    name: str = 'create_line_chart'
    description: str = (
        'Create a line chart from the provided data. '
        'Data should be a dictionary with x-values as keys and y-values as numbers, '
        'or a list of dictionaries, or a JSON string. '
        'IMPORTANT: For filename, provide ONLY the filename (e.g., "chart.png"), NOT the full path.'
    )
    args_schema: Type[BaseModel] = ChartDataInput
    output_dir: str = Field(default='charts')

    def _normalize_filename(self, filename: Optional[str]) -> str:
        """Normalize filename to avoid path duplication."""
        if filename is None:
            return None
        filename = os.path.basename(filename)
        if '/' in filename:
            filename = filename.split('/')[-1]
        return filename

    def _normalize_data_for_charts(
        self, data: Union[Dict[str, Any], List[Dict[str, Any]], List[Any], str]
    ) -> Dict[str, Union[int, float]]:
        if isinstance(data, dict):
            return {
                str(k): float(v) if isinstance(v, (int, float)) else 0
                for k, v in data.items()
            }
        elif isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], dict):
                result = {}
                for item in data:
                    if isinstance(item, dict):
                        keys = list(item.keys())
                        if len(keys) >= 2:
                            label_key = keys[0]
                            value_key = keys[1]
                            result[str(item[label_key])] = (
                                float(item[value_key])
                                if isinstance(item[value_key], (int, float))
                                else 0
                            )
                return result
            else:
                return {
                    f'Item {i + 1}': float(v)
                    if isinstance(v, (int, float))
                    else 0
                    for i, v in enumerate(data)
                }
        return {'Data': 1.0}

    def _run(
        self,
        data: Union[Dict[str, Union[int, float]], List[Dict[str, Any]], str],
        title: str = 'Line Chart',
        x_label: str = 'X-axis',
        y_label: str = 'Y-axis',
        filename: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            matplotlib.use('Agg')
            os.makedirs(self.output_dir, exist_ok=True)

            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    pass

            normalized_data = self._normalize_data_for_charts(data)
            x_values = list(normalized_data.keys())
            y_values = list(normalized_data.values())

            plt.figure(figsize=(10, 6))
            plt.plot(x_values, y_values, marker='o', linewidth=2, markersize=6)
            plt.title(title)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            filename = self._normalize_filename(filename)
            if filename is None:
                filename = (
                    f'line_chart_{len(os.listdir(self.output_dir)) + 1}.png'
                )

            file_path = os.path.join(self.output_dir, filename)
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()

            return json.dumps(
                {
                    'chart_type': 'line_chart',
                    'title': title,
                    'file_path': file_path,
                    'data_points': len(normalized_data),
                    'status': 'success',
                }
            )
        except Exception as e:
            return json.dumps(
                {
                    'chart_type': 'line_chart',
                    'error': str(e),
                    'status': 'error',
                }
            )


class PieChartTool(BaseTool):
    name: str = 'create_pie_chart'
    description: str = (
        'Create a pie chart from the provided data. '
        'Data should be a dictionary with categories as keys and values as numbers, '
        'or a list of dictionaries, or a JSON string. '
        'IMPORTANT: For filename, provide ONLY the filename (e.g., "chart.png"), NOT the full path.'
    )
    args_schema: Type[BaseModel] = PieChartInput
    output_dir: str = Field(default='charts')

    def _normalize_filename(self, filename: Optional[str]) -> str:
        """Normalize filename to avoid path duplication."""
        if filename is None:
            return None
        filename = os.path.basename(filename)
        if '/' in filename:
            filename = filename.split('/')[-1]
        return filename

    def _normalize_data_for_charts(
        self, data: Union[Dict[str, Any], List[Dict[str, Any]], List[Any], str]
    ) -> Dict[str, Union[int, float]]:
        if isinstance(data, dict):
            return {
                str(k): float(v) if isinstance(v, (int, float)) else 0
                for k, v in data.items()
            }
        elif isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], dict):
                result = {}
                for item in data:
                    if isinstance(item, dict):
                        keys = list(item.keys())
                        if len(keys) >= 2:
                            label_key = keys[0]
                            value_key = keys[1]
                            result[str(item[label_key])] = (
                                float(item[value_key])
                                if isinstance(item[value_key], (int, float))
                                else 0
                            )
                return result
            else:
                return {
                    f'Item {i + 1}': float(v)
                    if isinstance(v, (int, float))
                    else 0
                    for i, v in enumerate(data)
                }
        return {'Data': 1.0}

    def _run(
        self,
        data: Union[Dict[str, Union[int, float]], List[Dict[str, Any]], str],
        title: str = 'Pie Chart',
        filename: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            matplotlib.use('Agg')
            os.makedirs(self.output_dir, exist_ok=True)

            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    pass

            normalized_data = self._normalize_data_for_charts(data)
            labels = list(normalized_data.keys())
            values = list(normalized_data.values())

            plt.figure(figsize=(10, 8))
            plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
            plt.title(title)
            plt.axis('equal')

            filename = self._normalize_filename(filename)
            if filename is None:
                filename = (
                    f'pie_chart_{len(os.listdir(self.output_dir)) + 1}.png'
                )

            file_path = os.path.join(self.output_dir, filename)
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()

            return json.dumps(
                {
                    'chart_type': 'pie_chart',
                    'title': title,
                    'file_path': file_path,
                    'data_points': len(normalized_data),
                    'status': 'success',
                }
            )
        except Exception as e:
            return json.dumps(
                {'chart_type': 'pie_chart', 'error': str(e), 'status': 'error'}
            )


class ScatterPlotTool(BaseTool):
    name: str = 'create_scatter_plot'
    description: str = (
        'Create a scatter plot from the provided data. '
        'Requires x_data and y_data as lists of numeric values, '
        "or alternatively provide data as list of [x,y] pairs or dict with 'x' and 'y' keys. "
        'IMPORTANT: For filename, provide ONLY the filename (e.g., "chart.png"), NOT the full path.'
    )
    args_schema: Type[BaseModel] = ScatterPlotInput
    output_dir: str = Field(default='charts')

    def _normalize_filename(self, filename: Optional[str]) -> str:
        """Normalize filename to avoid path duplication."""
        if filename is None:
            return None
        filename = os.path.basename(filename)
        if '/' in filename:
            filename = filename.split('/')[-1]
        return filename

    def _run(
        self,
        x_data: Optional[List[Union[int, float]]] = None,
        y_data: Optional[List[Union[int, float]]] = None,
        title: str = 'Scatter Plot',
        x_label: str = 'X-axis',
        y_label: str = 'Y-axis',
        filename: Optional[str] = None,
        x: Optional[List[Union[int, float]]] = None,
        y: Optional[List[Union[int, float]]] = None,
        data: Optional[
            Union[
                List[List[Union[int, float]]],
                Dict[str, List[Union[int, float]]],
            ]
        ] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            matplotlib.use('Agg')
            os.makedirs(self.output_dir, exist_ok=True)

            if x_data is None:
                x_data = x
            if y_data is None:
                y_data = y

            if data is not None:
                if isinstance(data, dict):
                    if 'x' in data and 'y' in data:
                        x_data = data['x']
                        y_data = data['y']
                elif isinstance(data, list) and len(data) > 0:
                    if isinstance(data[0], list) and len(data[0]) == 2:
                        x_data = [point[0] for point in data]
                        y_data = [point[1] for point in data]

            if x_data is None or y_data is None:
                raise ValueError('Missing x_data and y_data parameters')
            if len(x_data) != len(y_data):
                raise ValueError('x_data and y_data must have the same length')

            plt.figure(figsize=(10, 6))
            plt.scatter(x_data, y_data, alpha=0.7, s=50)
            plt.title(title)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            filename = self._normalize_filename(filename)
            if filename is None:
                filename = (
                    f'scatter_plot_{len(os.listdir(self.output_dir)) + 1}.png'
                )

            file_path = os.path.join(self.output_dir, filename)
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()

            return json.dumps(
                {
                    'chart_type': 'scatter_plot',
                    'title': title,
                    'file_path': file_path,
                    'data_points': len(x_data),
                    'status': 'success',
                }
            )
        except Exception as e:
            return json.dumps(
                {
                    'chart_type': 'scatter_plot',
                    'error': str(e),
                    'status': 'error',
                }
            )


class HistogramTool(BaseTool):
    name: str = 'create_histogram'
    description: str = (
        'Create a histogram from the provided data. '
        'Data should be a list of numeric values. '
        'IMPORTANT: For filename, provide ONLY the filename (e.g., "chart.png"), NOT the full path.'
    )
    args_schema: Type[BaseModel] = HistogramInput
    output_dir: str = Field(default='charts')

    def _normalize_filename(self, filename: Optional[str]) -> str:
        """Normalize filename to avoid path duplication."""
        if filename is None:
            return None
        filename = os.path.basename(filename)
        if '/' in filename:
            filename = filename.split('/')[-1]
        return filename

    def _run(
        self,
        data: List[Union[int, float]],
        bins: int = 10,
        title: str = 'Histogram',
        x_label: str = 'Values',
        y_label: str = 'Frequency',
        filename: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            matplotlib.use('Agg')
            os.makedirs(self.output_dir, exist_ok=True)

            if not isinstance(data, list) or len(data) == 0:
                raise ValueError('Data must be a non-empty list of numbers')

            numeric_data = []
            for value in data:
                try:
                    numeric_data.append(float(value))
                except (ValueError, TypeError):
                    continue

            if len(numeric_data) == 0:
                raise ValueError('No valid numeric data found')

            plt.figure(figsize=(10, 6))
            plt.hist(numeric_data, bins=bins, alpha=0.7, edgecolor='black')
            plt.title(title)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            filename = self._normalize_filename(filename)
            if filename is None:
                filename = (
                    f'histogram_{len(os.listdir(self.output_dir)) + 1}.png'
                )

            file_path = os.path.join(self.output_dir, filename)
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()

            return json.dumps(
                {
                    'chart_type': 'histogram',
                    'title': title,
                    'file_path': file_path,
                    'data_points': len(numeric_data),
                    'bins': bins,
                    'status': 'success',
                }
            )
        except Exception as e:
            return json.dumps(
                {'chart_type': 'histogram', 'error': str(e), 'status': 'error'}
            )


class VisualizationToolkit(BaseToolkit):
    output_dir: str = Field(default='charts')
    enable_create_bar_chart: bool = Field(default=True)
    enable_create_line_chart: bool = Field(default=True)
    enable_create_pie_chart: bool = Field(default=True)
    enable_create_scatter_plot: bool = Field(default=True)
    enable_create_histogram: bool = Field(default=True)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        try:
            matplotlib.use('Agg')
        except ImportError:
            raise ImportError(
                'matplotlib is not installed. Please install it using: `pip install matplotlib`'
            )

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def get_tools(self) -> List[BaseTool]:
        tools: List[BaseTool] = []

        if self.enable_create_bar_chart:
            tools.append(BarChartTool(output_dir=self.output_dir))
        if self.enable_create_line_chart:
            tools.append(LineChartTool(output_dir=self.output_dir))
        if self.enable_create_pie_chart:
            tools.append(PieChartTool(output_dir=self.output_dir))
        if self.enable_create_scatter_plot:
            tools.append(ScatterPlotTool(output_dir=self.output_dir))
        if self.enable_create_histogram:
            tools.append(HistogramTool(output_dir=self.output_dir))

        return tools
