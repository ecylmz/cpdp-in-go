from __future__ import annotations

import argparse
import csv
from importlib.metadata import version
from pathlib import Path
import platform
import shutil
import subprocess


PACKAGES = (
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("scikit-learn", "scikit-learn"),
    ("imbalanced-learn", "imbalanced-learn"),
    ("matplotlib", "matplotlib"),
    ("PyYAML", "PyYAML"),
    ("xgboost", "xgboost"),
    ("tqdm", "tqdm"),
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Record the verified software environment as CSV.")
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "analysis_output" / "generated" / "software_environment.csv",
        help="Destination CSV path.",
    )
    return parser.parse_args()


def command_version(command: str) -> str | None:
    executable = shutil.which(command)
    if executable is None:
        return None
    completed = subprocess.run(
        [command, "--version"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode == 0:
        tokens = completed.stdout.strip().split()
        return tokens[1] if len(tokens) >= 2 else completed.stdout.strip()
    for parent in Path(executable).resolve().parents:
        if parent.name and parent.name[0].isdigit():
            return parent.name
    return "available"


def main() -> None:
    args = parse_args()
    rows = [{"component": "Python", "verified_version": platform.python_version(), "role": "runtime"}]
    rows.extend(
        {
            "component": component,
            "verified_version": version(distribution),
            "role": "Python dependency",
        }
        for component, distribution in PACKAGES
    )
    rows.append({"component": "uv", "verified_version": command_version("uv"), "role": "command runner"})
    rows.append(
        {
            "component": "Tectonic",
            "verified_version": command_version("tectonic"),
            "role": "optional LaTeX compiler",
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("component", "verified_version", "role"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
