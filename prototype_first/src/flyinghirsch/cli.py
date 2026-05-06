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


@app.command("stage-dataset")
def stage_dataset(
    src: Path = typer.Option(Path("data/raw/dataset"), "--src", help="Source dataset root (expects train/val/test with images/labels, or images/labels with train/val/test)."),
    dst: Path = typer.Option(None, "--dst", help="Destination root under data/. Default: data/processed/dataset"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing staged files."),
    verify_images: bool = typer.Option(True, "--verify-images/--no-verify-images", help="Try to decode staged images to catch corrupt files."),
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
) -> None:
    """Copy your already-split dataset into the project's data/ structure and write a staging report."""
    _ = load_config(config)
    paths = ProjectPaths.from_root(Path.cwd())
    dst_root = dst if dst is not None else (paths.data_dir / "processed" / "dataset")

    from flyinghirsch.data.stage_dataset import stage_yolo_like_dataset, write_stage_report

    report = stage_yolo_like_dataset(
        src_root=src,
        dst_root=dst_root,
        overwrite=overwrite,
        verify_images=verify_images,
    )
    report_path = paths.reports_dir / "dataset_stage_report.json"
    write_stage_report(report, path=report_path)

    typer.echo(f"staged_to: {report.dst_root}")
    typer.echo(f"images_copied: {report.images_copied}")
    typer.echo(f"labels_copied: {report.labels_copied}")
    typer.echo(f"missing_label_for_image: {report.missing_label_for_image}")
    typer.echo(f"missing_image_for_label: {report.missing_image_for_label}")
    typer.echo(f"unreadable_images: {report.unreadable_images}")
    typer.echo(f"class_histogram: {report.class_histogram}")
    typer.echo(f"report: {report_path}")

