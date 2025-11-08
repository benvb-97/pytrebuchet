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

    # Generate rst files from docstrings
    os.system("sphinx-apidoc -o source/ ../src/pytrebuchet")
    
    # Generate the documentation using Sphinx
    os.system("sphinx-build -b html source build/html")