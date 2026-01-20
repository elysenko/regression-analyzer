---
name: cli-interface
id: PRD-05
description: Command-line interface for running analysis pipeline
status: backlog
phase: mvp
priority: P1
complexity: medium
wave: 5

depends_on:
  - PRD-01  # Data Loader
  - PRD-02  # LLM Analyzer
  - PRD-03  # Stats Engine
  - PRD-04  # Chart Generator

creates:
  - src/regression_analyzer/cli.py
  - src/regression_analyzer/pipeline.py
  - tests/test_cli.py
  - tests/test_pipeline.py

modifies:
  - pyproject.toml  # Entry point already defined

database:
  creates: []
  modifies: []

test_command: pytest tests/test_cli.py tests/test_pipeline.py

blocks: []

references: [PRD-01, PRD-02, PRD-03, PRD-04]

created: 2026-01-20T19:06:34Z
updated: 2026-01-20T19:06:34Z
---

# PRD-05: CLI Interface

## Overview

**Feature:** Command-line interface for running analysis pipeline
**Priority:** P1 (User-facing entry point)
**Complexity:** Medium
**Dependencies:** PRD-01, PRD-02, PRD-03, PRD-04 (all core modules)

---

## Problem Statement

Users need a simple way to:
1. Run the complete analysis pipeline from command line
2. Provide context about their data
3. Control which analyses to run
4. Get output in their preferred location

---

## Goals

1. Simple CLI with sensible defaults
2. Support for context hints and target specification
3. Optional transpose detection
4. Configurable output directory
5. Rich terminal output with progress indication
6. JSON output option for programmatic use

---

## Non-Goals

- Interactive mode (batch only for MVP)
- GUI interface
- Configuration file support
- Daemon/server mode

---

## Technical Design

### Architecture

```
src/regression_analyzer/
├── cli.py       # Click command definitions
└── pipeline.py  # Analysis orchestration
```

### CLI Commands

```python
# src/regression_analyzer/cli.py

import asyncio
from pathlib import Path
from typing import Optional
import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel
import json

from .pipeline import AnalysisPipeline, AnalysisResult

console = Console()


@click.group()
@click.version_option()
def main():
    """Regression Analyzer - Automated tabular data analysis with LLM-powered insights."""
    pass


@main.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option(
    '--context', '-c',
    help='Business context hint (e.g., "SaaS company focused on MRR")'
)
@click.option(
    '--target', '-t',
    help='Target column for regression (auto-detected if not specified)'
)
@click.option(
    '--output-dir', '-o',
    default='./output',
    type=click.Path(),
    help='Directory for output files (default: ./output)'
)
@click.option(
    '--check-transpose/--no-transpose',
    default=True,
    help='Check if data needs transposing (default: yes)'
)
@click.option(
    '--skip-llm/--use-llm',
    default=False,
    help='Skip LLM analysis, use heuristics only'
)
@click.option(
    '--model', '-m',
    help='LLM model to use (e.g., sonnet, gpt-4o)'
)
@click.option(
    '--json-output', '-j',
    is_flag=True,
    help='Output results as JSON'
)
def analyze(
    file_path: str,
    context: Optional[str],
    target: Optional[str],
    output_dir: str,
    check_transpose: bool,
    skip_llm: bool,
    model: Optional[str],
    json_output: bool
):
    """Analyze a CSV or Excel file.

    Examples:

        regression-analyzer analyze sales.csv

        regression-analyzer analyze data.xlsx --context "retail, focused on profit"

        regression-analyzer analyze metrics.csv --target revenue --output-dir ./results
    """
    asyncio.run(_analyze_async(
        file_path=file_path,
        context=context,
        target=target,
        output_dir=output_dir,
        check_transpose=check_transpose,
        skip_llm=skip_llm,
        model=model,
        json_output=json_output
    ))


async def _analyze_async(
    file_path: str,
    context: Optional[str],
    target: Optional[str],
    output_dir: str,
    check_transpose: bool,
    skip_llm: bool,
    model: Optional[str],
    json_output: bool
):
    """Async analysis handler."""

    if json_output:
        # Quiet mode for JSON
        result = await _run_analysis_quiet(
            file_path, context, target, output_dir,
            check_transpose, skip_llm, model
        )
        click.echo(json.dumps(result.to_dict(), indent=2, default=str))
        return

    # Rich output mode
    console.print(Panel.fit(
        f"[bold blue]Regression Analyzer[/bold blue]\n"
        f"File: {file_path}",
        border_style="blue"
    ))

    pipeline = AnalysisPipeline(
        output_dir=output_dir,
        use_llm=not skip_llm,
        model=model
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:

        # Load data
        task = progress.add_task("Loading data...", total=None)
        await pipeline.load_data(file_path)
        progress.update(task, description="✓ Data loaded")

        # Transpose check
        if check_transpose and not skip_llm:
            task = progress.add_task("Checking orientation...", total=None)
            transpose_result = await pipeline.check_transpose()
            if transpose_result and transpose_result.should_transpose:
                progress.update(task, description="✓ Data transposed")
            else:
                progress.update(task, description="✓ Orientation OK")

        # Context identification
        if not skip_llm:
            task = progress.add_task("Identifying context...", total=None)
            await pipeline.identify_context(context)
            progress.update(task, description="✓ Context identified")

        # Column analysis
        if not skip_llm:
            task = progress.add_task("Analyzing columns...", total=None)
            await pipeline.analyze_columns()
            progress.update(task, description="✓ Columns analyzed")

        # Statistics
        task = progress.add_task("Running statistics...", total=None)
        await pipeline.run_statistics(target=target)
        progress.update(task, description="✓ Statistics complete")

        # Charts
        task = progress.add_task("Generating charts...", total=None)
        await pipeline.generate_charts()
        progress.update(task, description="✓ Charts generated")

    # Display results
    result = pipeline.get_result()
    _display_results(result)


def _display_results(result: 'AnalysisResult'):
    """Display results in rich format."""

    # Context
    if result.context:
        console.print("\n[bold]Business Context[/bold]")
        table = Table(show_header=False, box=None)
        table.add_row("Company Type:", result.context.company_type)
        table.add_row("Industry:", result.context.industry)
        table.add_row("Focus Areas:", ", ".join(result.context.likely_focus_areas))
        console.print(table)

    # Column Recommendations
    if result.column_analysis:
        console.print("\n[bold]Column Analysis[/bold]")
        if result.column_analysis.recommended_target:
            console.print(f"  Target: [green]{result.column_analysis.recommended_target}[/green]")
        if result.column_analysis.recommended_features:
            console.print(f"  Features: {', '.join(result.column_analysis.recommended_features)}")

    # Regression Results
    if result.statistics and result.statistics.linear_regression:
        reg = result.statistics.linear_regression
        console.print("\n[bold]Linear Regression[/bold]")
        console.print(f"  R² = {reg.r_squared:.3f}")
        console.print(f"  RMSE = {reg.rmse:.2f}")

        # Significant predictors
        sig = [c for c in reg.coefficients if c.is_significant]
        if sig:
            console.print("  Significant predictors:")
            for c in sig:
                direction = "+" if c.coefficient > 0 else ""
                console.print(f"    • {c.feature}: {direction}{c.coefficient:.3f}")

    # Feature Importance
    if result.statistics and result.statistics.feature_importance:
        fi = result.statistics.feature_importance
        console.print("\n[bold]Feature Importance[/bold]")
        console.print(f"  Model R² = {fi.model_r_squared:.3f}")
        for f in fi.features[:5]:
            bar = "█" * int(f.importance * 20)
            console.print(f"    {f.rank}. {f.feature}: {bar} {f.importance:.3f}")

    # Charts
    if result.charts:
        console.print("\n[bold]Generated Charts[/bold]")
        for chart in result.charts:
            console.print(f"  • {chart}")

    # Warnings
    if result.warnings:
        console.print("\n[yellow]Warnings[/yellow]")
        for w in result.warnings:
            console.print(f"  ⚠ {w}")

    console.print("\n[green]Analysis complete![/green]")


async def _run_analysis_quiet(
    file_path: str,
    context: Optional[str],
    target: Optional[str],
    output_dir: str,
    check_transpose: bool,
    use_llm: bool,
    model: Optional[str]
) -> 'AnalysisResult':
    """Run analysis without output (for JSON mode)."""
    pipeline = AnalysisPipeline(
        output_dir=output_dir,
        use_llm=use_llm,
        model=model
    )

    await pipeline.load_data(file_path)

    if check_transpose and use_llm:
        await pipeline.check_transpose()

    if use_llm:
        await pipeline.identify_context(context)
        await pipeline.analyze_columns()

    await pipeline.run_statistics(target=target)
    await pipeline.generate_charts()

    return pipeline.get_result()


@main.command()
def providers():
    """List available LLM providers."""
    from .core.llm_runner import LLMRunner

    console.print("\n[bold]Available LLM Providers[/bold]\n")

    try:
        runner = LLMRunner()
        available = runner.get_available_providers()
        default = runner.get_default_provider()

        for provider in available:
            marker = " (default)" if provider == default else ""
            console.print(f"  ✓ {provider}{marker}")

        console.print("\n[bold]Available Models[/bold]\n")
        for model in runner.get_available_models():
            console.print(f"  • {model}")

    except Exception as e:
        console.print(f"[red]Error initializing LLM runner: {e}[/red]")


if __name__ == "__main__":
    main()
```

### Analysis Pipeline

```python
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
            self._llm_analyzer = LLMAnalyzer(model=model)
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

        decision = await self._llm_analyzer.should_transpose(self.df)
        self._transpose_decision = decision

        if decision.should_transpose:
            self.df = transpose_dataframe(self.df)
            self._transposed = True
            self._warnings.append("Data was transposed based on LLM analysis")

        return decision

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

        self._context = await self._llm_analyzer.identify_context(
            self.df, user_hint
        )
        return self._context

    async def analyze_columns(self) -> Optional[ColumnAnalysis]:
        """Analyze column relevance.

        Returns:
            ColumnAnalysis if LLM available, None otherwise
        """
        if not self._llm_analyzer or self.df is None or not self._context:
            return None

        self._column_analysis = await self._llm_analyzer.analyze_columns(
            self.df, self._context
        )
        return self._column_analysis

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
            rows=self.df.height if self.df else 0,
            columns=self.df.width if self.df else 0,
            transposed=self._transposed,
            transpose_decision=self._transpose_decision,
            context=self._context,
            column_analysis=self._column_analysis,
            statistics=self._statistics,
            charts=self._charts,
            warnings=self._warnings,
        )
```

---

## Implementation Tasks

### Task 1: CLI Commands
- [ ] Create `cli.py` with Click commands
- [ ] Implement `analyze` command with options
- [ ] Implement `providers` command
- [ ] Add version flag

### Task 2: Rich Output
- [ ] Add progress spinners
- [ ] Format results as tables
- [ ] Color-code warnings and errors
- [ ] Display charts paths

### Task 3: JSON Output
- [ ] Implement `--json-output` flag
- [ ] Create `to_dict()` method on AnalysisResult
- [ ] Ensure all nested objects serialize

### Task 4: Analysis Pipeline
- [ ] Create `AnalysisPipeline` class
- [ ] Implement step-by-step execution
- [ ] Handle LLM-free mode
- [ ] Aggregate warnings

### Task 5: Testing
- [ ] Test CLI with mock data
- [ ] Test JSON output format
- [ ] Test pipeline stages
- [ ] Test error handling

---

## Testing Strategy

### CLI Tests
```python
# tests/test_cli.py

import pytest
from click.testing import CliRunner
import json

from regression_analyzer.cli import main

@pytest.fixture
def runner():
    return CliRunner()

@pytest.fixture
def sample_csv(tmp_path):
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("x,y,z\n1,2,3\n4,5,6\n7,8,9\n10,11,12")
    return csv_file

def test_analyze_basic(runner, sample_csv, tmp_path):
    """Test basic analyze command."""
    result = runner.invoke(main, [
        'analyze', str(sample_csv),
        '--output-dir', str(tmp_path / 'output'),
        '--skip-llm'
    ])

    assert result.exit_code == 0
    assert "Analysis complete" in result.output

def test_analyze_json_output(runner, sample_csv, tmp_path):
    """Test JSON output mode."""
    result = runner.invoke(main, [
        'analyze', str(sample_csv),
        '--output-dir', str(tmp_path / 'output'),
        '--skip-llm',
        '--json-output'
    ])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "file_path" in data
    assert "statistics" in data

def test_providers_command(runner):
    """Test providers listing."""
    result = runner.invoke(main, ['providers'])
    assert result.exit_code == 0
    assert "claude-code" in result.output or "Available" in result.output
```

---

## Acceptance Criteria

1. **Basic Usage**: `regression-analyzer analyze file.csv` works
2. **Context**: `--context` option passes hint to LLM
3. **Target Override**: `--target` overrides LLM recommendation
4. **Output Directory**: `--output-dir` controls file location
5. **JSON Mode**: `--json-output` produces valid JSON
6. **Skip LLM**: `--skip-llm` runs statistics only
7. **Progress**: Shows progress during analysis
8. **Results Display**: Formatted, readable output

---

## Dependencies

- `click>=8.0.0` - CLI framework
- `rich>=13.0.0` - Terminal formatting
- All core modules (PRD-01 through PRD-04)

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Long LLM calls timeout | Add configurable timeout, show progress |
| Invalid file path | Clear error message with supported formats |
| Missing API keys | Graceful fallback to skip-llm mode |
