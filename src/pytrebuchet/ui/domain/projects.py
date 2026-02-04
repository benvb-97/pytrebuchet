"""Module for managing projects in the pytrebuchet application.

A project represents a single trebuchet configuration and its associated data.
"""

from os import PathLike
from pathlib import Path


class Project:
    """Represents a single trebuchet project with its configuration and data."""

    def __init__(
        self,
        identifier: int,
        filename: PathLike,
    ) -> None:
        """Initialize a Project instance.

        Args:
            identifier (int): Unique identifier for the project.
            filename (PathLike): Path to the project file.

        """
        self.identifier = identifier
        self.filename = Path(filename)


class ProjectCollection:
    """Represents a collection of Project instances."""

    def __init__(self) -> None:
        """Initialize an empty ProjectCollection."""
        # Mapping of project ID to Project instance
        self._projects: dict[int, Project] = {}

    def __len__(self) -> int:
        """Return the number of projects in the collection."""
        return len(self._projects)

    def create_project(self, filename: PathLike) -> Project:
        """Create a new project and add it to the collection.

        Args:
            filename (PathLike): Path to the new project file.

        Returns:
            Project: The newly created Project instance.

        """
        new_id = max(self._projects.keys(), default=0) + 1
        new_project = Project(identifier=new_id, filename=filename)
        self._projects[new_id] = new_project
        return new_project
