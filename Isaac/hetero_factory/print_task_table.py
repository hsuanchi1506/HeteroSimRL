"""Print the per-task parameter table (Markdown) from the installed Isaac Lab.

    ./isaaclab.sh -p Isaac/hetero_factory/print_task_table.py --headless > Isaac/hetero_factory/TASK_PARAMS_runtime.md

Run it on the training machine and diff the output with TASK_PARAMS.md (generated offline): both must match,
otherwise the installed Isaac Lab's Factory config differs from the one the table was written against.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import os  # noqa: E402
import sys  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hetero_factory.task_table import build_task_table  # noqa: E402

print(build_task_table("cpu").to_markdown())
app.close()
