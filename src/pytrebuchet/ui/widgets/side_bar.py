"""Side bar widget containing a list of open projects."""

from typing import TYPE_CHECKING

from PySide6.QtWidgets import QLabel, QListView, QVBoxLayout, QWidget

from pytrebuchet.ui.models.session import SessionModel

if TYPE_CHECKING:
    from pytrebuchet.ui.widgets.main_window import MainWindow


class SideBar(QWidget):
    """Side bar widget containing a list of open projects."""

    def __init__(self, parent: "MainWindow") -> None:
        """Initialize the side bar."""
        super().__init__(parent)

        self._session_model: SessionModel | None = None

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the UI components of the side bar (widgets and layouts)."""
        self._layout = QVBoxLayout(self)
        self.setLayout(self._layout)
        self.setToolTip(self.tr("Side Bar - List of Currently Opened Projects"))

        self._layout.addWidget(QLabel(self.tr("Open Projects:"), self))

        # Create a list view to display open projects
        self._projects_view = QListView(self)
        self._layout.addWidget(self._projects_view)

    def set_session_model(self, session_model: SessionModel) -> None:
        """Set up the data models for the side bar components."""
        self._session_model = session_model
        self._projects_view.setModel(self._session_model.projects_model)
