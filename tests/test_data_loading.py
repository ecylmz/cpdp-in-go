from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from data_loading import list_available_projects


class ListAvailableProjectsTests(unittest.TestCase):
    def test_excludes_aggregate_directories_and_retains_project_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            for directory_name in ("zeta", "alpha", "all", "combined", "_scratch"):
                (data_dir / directory_name).mkdir()
            (data_dir / "notes.txt").write_text("not a project directory", encoding="utf-8")

            with patch("data_loading.get_schema", return_value=SimpleNamespace(data_dir=data_dir)):
                projects = list_available_projects("commit")

        self.assertEqual(projects, ["alpha", "zeta"])


if __name__ == "__main__":
    unittest.main()
