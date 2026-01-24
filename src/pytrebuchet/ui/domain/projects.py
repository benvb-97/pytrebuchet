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
