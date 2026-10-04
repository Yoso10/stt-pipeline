"""stt: the Whisper fine-tuning pipeline package. Component subfolders are added as each is built.

How to run: the package is used from the source folder, with no installation and no network.
    PowerShell:  $env:PYTHONPATH = "src"; python -c "import stt; print(stt.__version__)"
    cmd:         set PYTHONPATH=src && python -c "import stt; print(stt.__version__)"
Example output:
    0.0.0
Tests: python -m unittest discover -s tests -v   (the test files add src/ to the path themselves)
"""

__version__ = "0.0.0"
