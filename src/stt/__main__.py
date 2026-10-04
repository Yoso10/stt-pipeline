"""`python -m stt <command>`: the command line of the pipeline.

How to run (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -m stt --help
    python -m stt smoke
The commands and their options are described in stt/orchestrator/cli.py and in docs/COMPONENTS.md (C13).
"""

import sys

from stt.orchestrator.cli import main

if __name__ == "__main__":
    sys.exit(main())
