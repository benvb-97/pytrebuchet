"""Qt model for user session in the pytrebuchet application."""

from pytrebuchet.ui.domain.session import Session
from pytrebuchet.ui.models.projects import ProjectCollectionModel


class SessionModel:
    """Qt model representing a user session.

    The SessionModel provides a Qt model interface to the underlying Session data,
    allowing it to be used in Qt views.
    """

    def __init__(self) -> None:
        """Initialize the SessionModel."""
        super().__init__()
        self._session = Session()

        # Initialize sub-models and expose them as attributes
        self.projects_model = ProjectCollectionModel(
            project_collection=self._session.projects
        )
