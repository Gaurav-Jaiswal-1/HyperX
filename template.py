import os
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

# Project name
project_name = "hyperx"

# List of file paths to create
list_of_files = [
    f"{project_name}/__init__.py",
    f"{project_name}/activations.py",
    f"{project_name}/utils.py",
    f"{project_name}/versions.py",
    "tests/__init__.py",
    "tests/test_activations.py",
    "examples/example_cnn.py",
    "examples/example_mlp.py",
    "docs/index.md",
    "docs/figures/.gitkeep",        # placeholder to keep empty dir
    "benchmarks/run_cifar10.py",
    "benchmarks/results.md",
    ".gitignore",
    "LICENSE",
    "README.md",
    "setup.py",
    "pyproject.toml",
    "requirements.txt",
    "MANIFEST.in"
]

# Create files and directories
for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass  # Create empty file
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")
