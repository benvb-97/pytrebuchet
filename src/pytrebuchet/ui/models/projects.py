"""Qt model for project collections in the pytrebuchet application."""

from PySide6.QtCore import QAbstractListModel, Qt

from pytrebuchet.ui.domain.projects import ProjectCollection
from pytrebuchet.ui.qt_type_hints import QtModelIndex


class ProjectCollectionModel(QAbstractListModel):
    """Qt model representing a collection of projects.

    The ProjectCollectionModel provides a Qt model interface to the underlying
    ProjectCollection data, allowing it to be used in Qt views.
    """

    def __init__(self, project_collection: ProjectCollection) -> None:
        """Initialize the ProjectCollectionModel.

        Args:
            project_collection (ProjectCollection): The underlying project collection
                data of the domain layer.

        """
        super().__init__()
        self._project_collection = project_collection

    def rowCount(self, parent: QtModelIndex | None = None) -> int:
        """Return the number of rows in the model."""
        return len(self._project_collection)

    def data(
        self, index: QtModelIndex, role: int = Qt.ItemDataRole.DisplayRole
    ) -> str | None:
        """Return the data for a given index and role."""
        return None
