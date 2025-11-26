"""Generate the documentation for the pytrebuchet project using Sphinx."""

if __name__ == "__main__":
    import os
    from pathlib import Path

    # Get directory of the project
    # This file resides in the 'dev' folder, so we go one level up
    project_folder_path = Path(__file__).resolve().parent

    # Change directory to the docs folder
    docs_folder_path = Path(project_folder_path) / "docs"
    os.chdir(docs_folder_path)

    # Create _static and _templates folders under the docs folder if they don't exist
    Path(docs_folder_path / "source" / "_static").mkdir(exist_ok=True)
    Path(docs_folder_path / "source" / "_templates").mkdir(exist_ok=True)

    # Generate rst files from docstrings
    os.system("sphinx-apidoc -o source/ ../src/pytrebuchet")

    # Generate the documentation using Sphinx
    os.system("sphinx-build -b html source build/html")
