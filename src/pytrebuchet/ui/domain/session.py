"""Module for managing the user session in the pytrebuchet application."""

from .projects import ProjectCollection


class Session:
    """Represents a user session in the pytrebuchet application.

    The session encapsulates all data and operations related to a single user session.
    """

    def __init__(self) -> None:
        """Initialize the Session."""
        self.projects = ProjectCollection()
