"""NPU artifact placement and representative calibration selection."""

import csv

import numpy as np
import scipy.io

from apps.npu_service.app.runtime import (
    CalibrationDataReader,
    artifact_paths,
    calibration_windows,
)


def test_npu_artifacts_are_beside_checkpoint(tmp_path):
    checkpoint = tmp_path / "finetuned_resnet18.pth"
    paths = artifact_paths(checkpoint)
    assert paths["onnx"] == tmp_path / "finetuned_resnet18.onnx"
    assert paths["int8"] == tmp_path / "finetuned_resnet18.int8.onnx"
    assert paths["manifest"] == tmp_path / "finetuned_resnet18.int8.json"
    assert paths["vaip_cache"] == tmp_path / "finetuned_resnet18.vaip"


def test_calibration_windows_are_balanced_and_reader_rewinds(tmp_path):
    session = tmp_path / "sessions/model"
    run = session / "train/runs" / ("a" * 32)
    dataset = session / "train/dataset"
    run.mkdir(parents=True)
    checkpoint = run / "finetuned_resnet18.pth"
    checkpoint.touch()
    rows = []
    for label, count, value in [("Static", 3, 1.0), ("Walking", 1, 2.0)]:
        folder = dataset / label
        folder.mkdir(parents=True)
        for index in range(count):
            name = f"sample-{index}.mat"
            # Non-constant data exercises the same global z-score used in deployment.
            array = np.arange(200 * 52, dtype=np.float32).reshape(200, 52) + value
            scipy.io.savemat(folder / name, {"CSIamp": array})
            rows.append({"label": label, "filename": name})
    with (run / "train_split.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["label", "filename"])
        writer.writeheader()
        writer.writerows(rows)

    windows = calibration_windows(checkpoint, ["Static", "Walking"], (200, 52))
    assert len(windows) == 4
    assert all(window.shape == (200, 52) for window in windows)
    reader = CalibrationDataReader(windows)
    assert reader.get_next()["input"].shape == (1, 1, 200, 52)
    reader.rewind()
    assert reader.get_next()["input"].shape == (1, 1, 200, 52)
