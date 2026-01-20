# src/regression_analyzer/pipeline.py

from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass, field
import polars as pl

from .loader import DataLoader
from .loader.transpose import transpose_dataframe
from .analyzer import LLMAnalyzer
from .analyzer.models import (
    BusinessContext,
    ColumnAnalysis,
    TransposeDecision,
)
from .stats import StatsEngine
from .stats.models import StatisticsReport
from .charts import ChartGenerator


@dataclass
class AnalysisResult:
    """Complete analysis result."""

    file_path: str
    rows: int
    columns: int
    transposed: bool = False
    transpose_decision: Optional[TransposeDecision] = None
    context: Optional[BusinessContext] = None
    column_analysis: Optional[ColumnAnalysis] = None
    statistics: Optional[StatisticsReport] = None
    charts: List[Path] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "file_path": self.file_path,
            "rows": self.rows,
            "columns": self.columns,
            "transposed": self.transposed,
            "transpose_decision": (
                self.transpose_decision.model_dump()
                if self.transpose_decision else None
            ),
            "context": (
                self.context.model_dump()
                if self.context else None
            ),
            "column_analysis": (
                self.column_analysis.model_dump()
                if self.column_analysis else None
            ),
            "statistics": (
                self.statistics.model_dump()
                if self.statistics else None
            ),
            "charts": [str(p) for p in self.charts],
            "warnings": self.warnings,
        }


class AnalysisPipeline:
    """Orchestrate the full analysis pipeline."""

    def __init__(
        self,
        output_dir: str = "./output",
        use_llm: bool = True,
        model: Optional[str] = None
    ):
        """Initialize pipeline.

        Args:
            output_dir: Directory for output files
            use_llm: Whether to use LLM for analysis
            model: Specific LLM model to use
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.use_llm = use_llm
        self.model = model

        self.df: Optional[pl.DataFrame] = None
        self.file_path: Optional[str] = None

        self._transposed = False
        self._transpose_decision: Optional[TransposeDecision] = None
        self._context: Optional[BusinessContext] = None
        self._column_analysis: Optional[ColumnAnalysis] = None
        self._statistics: Optional[StatisticsReport] = None
        self._charts: List[Path] = []
        self._warnings: List[str] = []

        # Initialize components
        if use_llm:
            try:
                self._llm_analyzer = LLMAnalyzer(model=model)
            except Exception as e:
                self._warnings.append(f"LLM initialization failed: {e}")
                self._llm_analyzer = None
                self.use_llm = False
        else:
            self._llm_analyzer = None

        self._stats_engine = StatsEngine()
        self._chart_generator = ChartGenerator(
            output_dir=self.output_dir / "charts"
        )

    async def load_data(self, file_path: str) -> pl.DataFrame:
        """Load data from file.

        Args:
            file_path: Path to CSV or Excel file

        Returns:
            Loaded DataFrame
        """
        self.file_path = file_path
        loader = DataLoader(file_path)
        self.df = loader.load()

        # Get metadata
        metadata = loader.get_metadata(self.df)
        self._warnings.extend([
            f"Column '{col}' has {count} null values"
            for col, count in metadata["null_counts"].items()
            if count > 0
        ])

        return self.df

    async def check_transpose(self) -> Optional[TransposeDecision]:
        """Check if data needs transposing.

        Returns:
            TransposeDecision if LLM available, None otherwise
        """
        if not self._llm_analyzer or self.df is None:
            return None

        try:
            decision = await self._llm_analyzer.should_transpose(self.df)
            self._transpose_decision = decision

            if decision.should_transpose:
                self.df = transpose_dataframe(self.df)
                self._transposed = True
                self._warnings.append("Data was transposed based on LLM analysis")

            return decision
        except Exception as e:
            self._warnings.append(f"Transpose check failed: {e}")
            return None

    async def identify_context(
        self,
        user_hint: Optional[str] = None
    ) -> Optional[BusinessContext]:
        """Identify business context.

        Args:
            user_hint: Optional user-provided context hint

        Returns:
            BusinessContext if LLM available, None otherwise
        """
        if not self._llm_analyzer or self.df is None:
            return None

        try:
            self._context = await self._llm_analyzer.identify_context(
                self.df, user_hint
            )
            return self._context
        except Exception as e:
            self._warnings.append(f"Context identification failed: {e}")
            return None

    async def analyze_columns(self) -> Optional[ColumnAnalysis]:
        """Analyze column relevance.

        Returns:
            ColumnAnalysis if LLM available, None otherwise
        """
        if not self._llm_analyzer or self.df is None or not self._context:
            return None

        try:
            self._column_analysis = await self._llm_analyzer.analyze_columns(
                self.df, self._context
            )
            return self._column_analysis
        except Exception as e:
            self._warnings.append(f"Column analysis failed: {e}")
            return None

    async def run_statistics(
        self,
        target: Optional[str] = None,
        features: Optional[List[str]] = None
    ) -> StatisticsReport:
        """Run statistical analysis.

        Args:
            target: Target column (uses LLM recommendation if None)
            features: Feature columns (uses LLM recommendation if None)

        Returns:
            StatisticsReport with all results
        """
        if self.df is None:
            raise ValueError("Data not loaded")

        # Use LLM recommendations if available
        if target is None and self._column_analysis:
            target = self._column_analysis.recommended_target

        if features is None and self._column_analysis:
            features = self._column_analysis.recommended_features

        self._statistics = self._stats_engine.analyze(
            self.df,
            target=target,
            features=features
        )

        self._warnings.extend(self._statistics.warnings)
        return self._statistics

    async def generate_charts(self) -> List[Path]:
        """Generate visualization charts.

        Returns:
            List of paths to generated charts
        """
        if self.df is None or self._statistics is None:
            return []

        self._charts = self._chart_generator.generate_all(
            self.df, self._statistics
        )
        return self._charts

    def get_result(self) -> AnalysisResult:
        """Get complete analysis result.

        Returns:
            AnalysisResult with all data
        """
        return AnalysisResult(
            file_path=self.file_path or "",
            rows=self.df.height if self.df is not None else 0,
            columns=self.df.width if self.df is not None else 0,
            transposed=self._transposed,
            transpose_decision=self._transpose_decision,
            context=self._context,
            column_analysis=self._column_analysis,
            statistics=self._statistics,
            charts=self._charts,
            warnings=self._warnings,
        )
