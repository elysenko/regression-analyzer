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
        progress.update(task, description="[green]✓[/green] Data loaded")

        # Transpose check
        if check_transpose and not skip_llm:
            task = progress.add_task("Checking orientation...", total=None)
            transpose_result = await pipeline.check_transpose()
            if transpose_result and transpose_result.should_transpose:
                progress.update(task, description="[green]✓[/green] Data transposed")
            else:
                progress.update(task, description="[green]✓[/green] Orientation OK")

        # Context identification
        if not skip_llm:
            task = progress.add_task("Identifying context...", total=None)
            await pipeline.identify_context(context)
            progress.update(task, description="[green]✓[/green] Context identified")

        # Column analysis
        if not skip_llm:
            task = progress.add_task("Analyzing columns...", total=None)
            await pipeline.analyze_columns()
            progress.update(task, description="[green]✓[/green] Columns analyzed")

        # Statistics
        task = progress.add_task("Running statistics...", total=None)
        await pipeline.run_statistics(target=target)
        progress.update(task, description="[green]✓[/green] Statistics complete")

        # Charts
        task = progress.add_task("Generating charts...", total=None)
        await pipeline.generate_charts()
        progress.update(task, description="[green]✓[/green] Charts generated")

    # Display results
    result = pipeline.get_result()
    _display_results(result)


def _display_results(result: AnalysisResult):
    """Display results in rich format."""

    console.print(f"\n[bold]Data Summary[/bold]")
    console.print(f"  Rows: {result.rows}")
    console.print(f"  Columns: {result.columns}")
    if result.transposed:
        console.print(f"  [yellow]Data was transposed[/yellow]")

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
) -> AnalysisResult:
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
    console.print("\n[bold]Available LLM Providers[/bold]\n")

    try:
        from .core.llm_runner import LLMRunner

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
        console.print("\n[yellow]LLM features require API keys to be set.[/yellow]")
        console.print("Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or other provider keys.")


if __name__ == "__main__":
    main()
