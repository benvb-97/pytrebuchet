"""
Generates the documentation for the pytrebuchet project using Sphinx.
"""


if __name__ == "__main__":
    import os

    # Get directory of the project
    # This file resides in the 'dev' folder, so we go one level up
    project_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Change directory to the docs folder
    docs_folder_path = os.path.join(project_folder_path, "docs")
    os.chdir(docs_folder_path)

    # Create _static and _templates folders under the docs folder if they don't exist
    os.makedirs(os.path.join(docs_folder_path, "source", "_static"), exist_ok=True)
    os.makedirs(os.path.join(docs_folder_path, "source", "_templates"), exist_ok=True)

    # Generate rst files from docstrings
    os.system("sphinx-apidoc -o source/ ../src/pytrebuchet")
    
    # Generate the documentation using Sphinx
    os.system("sphinx-build -b html source build/html")