
"""
Verify the files needed to deploy the SIH dashboard.

This test checks deployment assets and imports.
It does not upload files or modify datasets.
"""

from pathlib import Path
import ast
import importlib
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "app/streamlit_app.py",
    "requirements.txt",
    "configs/mapping.yaml",
    "data/processed/real_pipeline_validation/pipeline_summary.csv",
    "data/processed/real_pipeline_validation/tracked_objects.csv",
    "data/processed/comparison/integrated_mapping_benchmark_summary.csv",
    "data/processed/comparison/elevation_accuracy_summary.csv",
]

MODULES = [
    "streamlit",
    "numpy",
    "pandas",
    "plotly",
    "yaml",
    "scipy",
]

def main():
    failures = []

    print("\nSIH DEPLOYMENT PACKAGE TEST")
    print("=" * 50)

    print("\n1. Checking required files")

    for relative_path in REQUIRED_FILES:
        path = ROOT / relative_path
        exists = path.is_file()

        print(
            f"{'PASS' if exists else 'FAIL'}: "
            f"{relative_path}"
        )

        if not exists:
            failures.append(
                f"Missing file: {relative_path}"
            )

    print("\n2. Checking Python dependencies")

    for module in MODULES:
        try:
            importlib.import_module(module)
            print(f"PASS: {module}")
        except ImportError as error:
            print(f"FAIL: {module} ({error})")
            failures.append(
                f"Missing dependency: {module}"
            )

    print("\n3. Checking Python syntax")

    python_files = [
        ROOT / "app/streamlit_app.py",
        *(
            ROOT / "src/lidar_mapping/visualization"
        ).glob("*.py"),
    ]

    for path in python_files:
        try:
            ast.parse(
                path.read_text(encoding="utf-8"),
                filename=str(path),
            )
            print(f"PASS: {path.name}")
        except (SyntaxError, OSError) as error:
            print(f"FAIL: {path.name}: {error}")
            failures.append(
                f"Syntax failure: {path}"
            )

    print("\n4. Checking Git exclusion rules")

    excluded_paths = [
        "data/raw/",
        "data/processed/visualizations/",
        ".venv/",
    ]

    for relative_path in excluded_paths:
        result = subprocess.run(
            [
                "git",
                "check-ignore",
                "-q",
                relative_path,
            ],
            cwd=ROOT,
            check=False,
        )

        ignored = result.returncode == 0

        print(
            f"{'PASS' if ignored else 'FAIL'}: "
            f"{relative_path}"
        )

        if not ignored:
            failures.append(
                f"Not ignored: {relative_path}"
            )

    print("\n5. Checking processed frame files")

    result_dir = (
        ROOT
        / "data/processed/real_pipeline_validation"
    )

    for frame in range(10):
        frame_id = f"{frame:06d}"

        for prefix, extension in [
            ("static_map", ".csv"),
            ("object_map", ".csv"),
            ("temporal", ".npz"),
        ]:
            filename = f"{prefix}_{frame_id}{extension}"
            path = result_dir / filename

            if not path.is_file():
                print(f"FAIL: Missing {filename}")
                failures.append(
                    f"Missing processed frame: {filename}"
                )

    if not any(
        "Missing processed frame" in item
        for item in failures
    ):
        print("PASS: All 10 processed frame sets found")

    print("\n" + "=" * 50)

    if failures:
        print("DEPLOYMENT TEST FAILED")

        for failure in failures:
            print("-", failure)

        sys.exit(1)

    print("DEPLOYMENT PACKAGE CHECKS PASSED")
    print(
        "Note: A clean cloud-environment test "
        "is still required."
    )


if __name__ == "__main__":
    main()
