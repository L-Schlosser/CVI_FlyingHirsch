from __future__ import annotations

import logging
from pathlib import Path

import typer

from flyinghirsch.utils.config import load_config
from flyinghirsch.utils.logging import setup_logging
from flyinghirsch.utils.paths import ProjectPaths

app = typer.Typer(add_completion=False, help="FlyingHirsch CV project CLI.")


@app.callback()
def _main(
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    setup_logging(log_level)
    logging.getLogger(__name__).debug("Loaded CLI with config=%s", config)


@app.command()
def info(config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c")) -> None:
    """Print resolved project paths + loaded config keys."""
    cfg = load_config(config)
    paths = ProjectPaths.from_root(Path.cwd())
    typer.echo(f"root: {paths.root}")
    typer.echo(f"data_dir: {paths.data_dir}")
    typer.echo(f"artifacts_dir: {paths.artifacts_dir}")
    typer.echo(f"reports_dir: {paths.reports_dir}")
    typer.echo(f"config_keys: {sorted(cfg.keys())}")


@app.command()
def prepare(
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
) -> None:
    """Create required directories for data/artifacts/reports."""
    _ = load_config(config)
    paths = ProjectPaths.from_root(Path.cwd())
    for p in [paths.data_dir, paths.artifacts_dir, paths.reports_dir]:
        p.mkdir(parents=True, exist_ok=True)
    for p in [
        paths.data_dir / "raw",
        paths.data_dir / "interim",
        paths.data_dir / "processed",
        paths.data_dir / "external",
    ]:
        p.mkdir(parents=True, exist_ok=True)
    typer.echo("Created data/artifacts/reports directories.")

