from __future__ import annotations

from pathlib import Path

from scripts._bootstrap import bootstrap_src

bootstrap_src()

from flyinghirsch.data.stage_dataset import stage_yolo_like_dataset, write_stage_report
from flyinghirsch.data.yolo_dataset import (
    discover_class_count_from_labels,
    load_class_names_if_present,
    write_ultralytics_dataset_yaml,
)
from flyinghirsch.utils.config import load_config
from flyinghirsch.utils.paths import ProjectPaths


def main() -> None:
    cfg = load_config(Path("configs/default.yaml"))
    paths = ProjectPaths.from_root(Path.cwd())

    src = Path(cfg.get("dataset", {}).get("default_src", "data/raw/dataset"))
    staged = Path(cfg.get("dataset", {}).get("default_staged", str(paths.data_dir / "processed" / "dataset")))

    report = stage_yolo_like_dataset(
        src_root=src,
        dst_root=staged,
        overwrite=False,
        verify_images=False,
    )
    write_stage_report(report, path=paths.reports_dir / "dataset_stage_report.json")

    # Write Ultralytics dataset YAML into artifacts so training/inference can find it.
    class_names = load_class_names_if_present(staged)
    if class_names is None:
        # Best-effort: infer nc, but Ultralytics prefers explicit names list.
        nc = discover_class_count_from_labels(staged / "labels")
        class_names = [f"class_{i}" for i in range(nc)]

    dataset_yaml = paths.artifacts_dir / "dataset.yaml"
    write_ultralytics_dataset_yaml(path=dataset_yaml, dataset_root=staged, class_names=class_names)

    print("staged_to:", report.dst_root)
    print("dataset_yaml:", dataset_yaml)


if __name__ == "__main__":
    main()

