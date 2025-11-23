"""Small script to quickly reformat all project files using the 'black' and 'isort' package.

You should install black/isort using pip install to your main python installation if you'd like to use this code.
Black/isort is not a requirement for the main code to run.
"""

if __name__ == "__main__":
    import os

    # Get directory of the project
    # This file resides in the 'dev' folder, so we go one level up
    project_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Reformat import statements
    os.system(f"ruff format {project_folder_path}")
