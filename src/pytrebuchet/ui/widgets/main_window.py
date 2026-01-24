"""Main window class for the PyTrebuchet application.

It sets up the main UI components including the menu bar, side bar, and
workspace area.
"""

from PySide6.QtWidgets import QMainWindow, QSplitter, QWidget

from pytrebuchet.ui.models.session import SessionModel
from pytrebuchet.ui.widgets.main_menu import MainMenuBar
from pytrebuchet.ui.widgets.side_bar import SideBar


class MainWindow(QMainWindow):
    """Main window class for the PyTrebuchet application.

    It sets up the main UI components including the menu bar, side bar, and
    workspace area.

    :param projects_model: The model containing the list of projects.
    :param settings: The application settings.
    """

    def __init__(self) -> None:
        """Initialize the main window."""
        super().__init__()

        self._setup_ui()
        self._setup_models()
        self._setup_connections()

    def _setup_ui(self) -> None:
        """Set up the main UI components of the main window."""
        self.setWindowTitle("PyTrebuchet User Interface")

        self._menu_bar = MainMenuBar(parent=self)
        self.setMenuBar(self._menu_bar)

        # Create the main splitter to hold side bar and workspace
        self._splitter = QSplitter(self)
        self.setCentralWidget(self._splitter)

        # Add side bar and workspace area to the splitter
        self._side_bar = SideBar(parent=self)
        self._splitter.addWidget(self._side_bar)
        self._splitter.addWidget(QWidget(self))  # Placeholder for workspace area

        self.showMaximized()

    def _setup_models(self) -> None:
        """Set up the data models for the main window components."""
        self._session_model = SessionModel()

        self._side_bar.set_session_model(self._session_model)

    def _setup_connections(self) -> None:
        """Set up the signal-slot connections for the main window."""
