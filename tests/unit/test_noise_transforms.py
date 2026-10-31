"""Unit tests for synthetic document degradation transforms and noise robustness suite."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from evals.noise_robustness import (
    ConditionMetrics,
    NoiseCondition,
    NoiseProductionCIGates,
    NoiseRobustnessReport,
    apply_dpi_recompression,
    apply_gaussian_blur,
    apply_noise_profile,
    apply_scan_skew,
    apply_watermark_overlay,
    degrade_crop_directory,
    format_markdown_report,
    main,
    run_noise_robustness_suite,
)


def _create_sample_image(width: int = 200, height: int = 150) -> Image.Image:
    """Create a high-contrast test image with text-like alternating vertical stripes."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    # Background white
    arr[:, :] = 255
    # Dark vertical stripes (like text columns or table borders)
    arr[:, 40:60] = 30
    arr[:, 120:140] = 30
    return Image.fromarray(arr, mode="RGB")


def test_apply_scan_skew() -> None:
    img = _create_sample_image(200, 200)
    skewed_pos = apply_scan_skew(img, angle=5.0)
    skewed_neg = apply_scan_skew(img, angle=-5.0)

    assert skewed_pos.size == img.size
    assert skewed_pos.mode == "RGB"
    assert skewed_neg.size == img.size
    assert skewed_neg.mode == "RGB"

    # Because image was rotated with fill_color=(255, 255, 255),
    # the corner pixels of the rotated canvas are filled white.
    orig_arr = np.array(img)
    pos_arr = np.array(skewed_pos)
    assert not np.array_equal(orig_arr, pos_arr)


def test_apply_gaussian_blur() -> None:
    # Sharp step edge: left 0 (black), right 255 (white)
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    arr[:, 50:] = 255
    step_img = Image.fromarray(arr, mode="RGB")

    blurred = apply_gaussian_blur(step_img, sigma=1.2)
    assert blurred.size == step_img.size
    assert blurred.mode == "RGB"

    blurred_arr = np.array(blurred)
    # At column 50, transition should be smooth (not sharp 0 to 255)
    mid_left = int(blurred_arr[50, 49, 0])
    mid_right = int(blurred_arr[50, 50, 0])
    # The jump between adjacent pixels should be significantly less than 255
    assert abs(mid_right - mid_left) < 120


def test_apply_watermark_overlay() -> None:
    white_img = Image.new("RGB", (300, 200), (255, 255, 255))
    watermarked = apply_watermark_overlay(white_img, text="CONFIDENTIAL", opacity=0.30)

    assert watermarked.size == white_img.size
    assert watermarked.mode == "RGB"

    wm_arr = np.array(watermarked)
    # Center region should contain reddish/darker text pixels (not pure 255, 255, 255)
    center_patch = wm_arr[80:120, 100:200]
    has_modified_pixels = np.any(center_patch < 250)
    assert has_modified_pixels


def test_apply_dpi_recompression() -> None:
    img = _create_sample_image(300, 200)
    recompressed = apply_dpi_recompression(img, target_dpi=150, source_dpi=300, quality=85)

    assert recompressed.size == img.size
    assert recompressed.mode == "RGB"

    # JPEG compression and downscaling introduces small high-frequency differences
    orig_arr = np.array(img, dtype=np.float32)
    recomp_arr = np.array(recompressed, dtype=np.float32)
    mse = float(np.mean((orig_arr - recomp_arr) ** 2))
    assert mse > 0.0


def test_apply_noise_profile_all_conditions() -> None:
    base_img = _create_sample_image(150, 150)
    for cond in NoiseCondition:
        perturbed = apply_noise_profile(base_img, cond, seed=123)
        assert perturbed.size == base_img.size
        assert perturbed.mode == "RGB"


def test_degrade_crop_directory(tmp_path: Path) -> None:
    src_dir = tmp_path / "src_crops"
    dst_dir = tmp_path / "dst_crops"
    src_dir.mkdir(parents=True)

    # Save 2 dummy crops
    img1 = _create_sample_image(100, 100)
    img2 = _create_sample_image(120, 80)
    img1.save(src_dir / "crop_p1_fig1.png")
    img2.save(src_dir / "crop_p2_fig2.png")

    degraded_map = degrade_crop_directory(src_dir, dst_dir, NoiseCondition.SKEW_DPI)
    assert len(degraded_map) == 2
    assert "crop_p1_fig1.png" in degraded_map
    assert "crop_p2_fig2.png" in degraded_map
    assert (dst_dir / "crop_p1_fig1.png").is_file()
    assert (dst_dir / "crop_p2_fig2.png").is_file()


def test_format_markdown_report() -> None:
    gates = NoiseProductionCIGates()
    cond1 = ConditionMetrics(
        condition=NoiseCondition.CLEAN,
        display_name="1. Clean Baseline",
        total_questions=30,
        context_recall_at_5=0.9667,
        visual_qa_accuracy=1.0000,
        table_cell_f1=0.9850,
        text_qa_accuracy=0.9500,
        faithfulness=0.9800,
        accuracy_retention=1.0000,
        fast_path_p95_ms=8.5,
        visual_path_p95_ms=16.0,
        passed_ci_gates=True,
    )
    cond2 = ConditionMetrics(
        condition=NoiseCondition.SKEW_DPI,
        display_name="2. Scan Skew + 150 DPI",
        total_questions=30,
        context_recall_at_5=0.9333,
        visual_qa_accuracy=0.9000,
        table_cell_f1=0.9400,
        text_qa_accuracy=0.9500,
        faithfulness=0.9500,
        accuracy_retention=0.9000,
        fast_path_p95_ms=8.5,
        visual_path_p95_ms=16.2,
        passed_ci_gates=True,
    )
    report = NoiseRobustnessReport(
        generated_at="2026-10-31T00:00:00Z",
        total_questions=30,
        conditions={
            NoiseCondition.CLEAN.value: cond1,
            NoiseCondition.SKEW_DPI.value: cond2,
        },
        ci_gates=gates,
        passed_all_gates=True,
    )

    md = format_markdown_report(report)
    assert "OmniDoc-Vision Document Degradation & Noise Robustness Report" in md
    assert "1. Clean Baseline" in md
    assert "2. Scan Skew + 150 DPI" in md
    assert "PASSED" in md


@pytest.mark.asyncio
async def test_run_noise_robustness_suite_offline(tmp_path: Path) -> None:
    output_json = tmp_path / "noise_report.json"
    output_md = tmp_path / "noise_report.md"

    # Evaluate CLEAN and SKEW_DPI conditions
    report = await run_noise_robustness_suite(
        output_json_path=output_json,
        output_md_path=output_md,
        conditions=[NoiseCondition.CLEAN, NoiseCondition.SKEW_DPI],
    )

    assert output_json.exists()
    assert output_md.exists()
    assert report.total_questions == 30
    assert report.passed_all_gates is True

    clean_metrics = report.conditions[NoiseCondition.CLEAN.value]
    skew_metrics = report.conditions[NoiseCondition.SKEW_DPI.value]

    assert clean_metrics.context_recall_at_5 >= report.ci_gates.min_noisy_recall_at_5
    assert clean_metrics.visual_qa_accuracy >= report.ci_gates.min_noisy_visual_qa_accuracy
    assert clean_metrics.faithfulness >= report.ci_gates.min_noisy_faithfulness

    assert skew_metrics.context_recall_at_5 >= report.ci_gates.min_noisy_recall_at_5
    assert skew_metrics.visual_qa_accuracy >= report.ci_gates.min_noisy_visual_qa_accuracy
    assert skew_metrics.accuracy_retention >= report.ci_gates.min_accuracy_retention

    parsed_json = json.loads(output_json.read_text(encoding="utf-8"))
    assert parsed_json["passed_all_gates"] is True
    assert len(parsed_json["conditions"]) == 2


def test_cli_main_entrypoint(tmp_path: Path) -> None:
    out_file = tmp_path / "cli_noise.json"
    exit_code = main(["--output", str(out_file)])
    assert exit_code == 0
    assert out_file.exists()
    assert out_file.with_suffix(".md").exists()
