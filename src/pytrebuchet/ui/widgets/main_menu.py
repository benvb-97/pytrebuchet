"""Module for main menu bar."""

from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QFileDialog, QMenu, QMenuBar

from pytrebuchet.ui.widgets.file.new_project_dialog import NewProjectDialog

if TYPE_CHECKING:
    from pytrebuchet.ui.widgets.main_window import MainWindow


class FileMenu(QMenu):
    """File menu for the pytrebuchet application in the main menu bar.

    The FileMenu provides actions for creating, opening, saving,
    and closing projects, as well as accessing application settings.
    """

    def __init__(
        self,
        parent: "MainMenuBar",
    ) -> None:
        """Initialize the abstract menu.

        Args:
            projects_model: The projects data model.
            settings: Application settings.
            parent: Main menu bar parent.

        """
        super().__init__(title=self.tr("File"), parent=parent)

        self._main_menu_bar = parent

        self._setup_actions()

    def _setup_actions(self) -> None:
        """Set up all file menu actions.

        This includes actions for creating, opening, saving, and closing projects,
        as well as accessing application settings and quitting the application.
        """
        # New Project Action
        new_project_action = QAction(self.tr("New Project"), self)
        new_project_action.setStatusTip(self.tr("Create a new project"))
        new_project_action.triggered.connect(self._open_new_project_dialog)
        self.addAction(new_project_action)

        # Open Project Action
        open_project_action = QAction(self.tr("Open Project"), self)
        open_project_action.setStatusTip(self.tr("Open an existing project"))
        open_project_action.triggered.connect(self._open_project)
        self.addAction(open_project_action)

        # ----
        self.addSeparator()

        # Save Current Project Action
        save_current_project_action = QAction(self.tr("Save Current Project"), self)
        save_current_project_action.setStatusTip(self.tr("Saves the active project"))
        save_current_project_action.triggered.connect(self._save_current_project)
        self.addAction(save_current_project_action)

        # Save As Action
        save_as_action = QAction(self.tr("Save As"), self)
        save_as_action.setStatusTip(
            self.tr("Saves the active project with a new directory name")
        )
        save_as_action.triggered.connect(self._save_current_project_as)
        self.addAction(save_as_action)

        # Save All Action
        save_all_action = QAction(self.tr("Save All"), self)
        save_all_action.setStatusTip(self.tr("Saves all opened projects"))
        save_all_action.triggered.connect(self._save_all_projects)
        self.addAction(save_all_action)

        # ----
        self.addSeparator()

        # Close Current Project Action
        close_current_project_action = QAction(self.tr("Close Current Project"), self)
        close_current_project_action.setStatusTip(
            self.tr("Closes the active project without saving")
        )
        close_current_project_action.triggered.connect(self._close_current_project)
        self.addAction(close_current_project_action)

        # Close All Projects Action
        close_all_projects_action = QAction(self.tr("Close All Projects"), self)
        close_all_projects_action.setStatusTip(
            self.tr("Closes all opened projects without saving")
        )
        close_all_projects_action.triggered.connect(self._close_all_projects)
        self.addAction(close_all_projects_action)

        # ----
        self.addSeparator()

        # Settings Action
        settings_action = QAction(self.tr("Settings"), self)
        settings_action.setStatusTip(self.tr("Open application settings"))
        settings_action.triggered.connect(self.open_settings_dialog)
        self.addAction(settings_action)

        # ----
        self.addSeparator()

        # Quit Action
        quit_action = QAction(self.tr("Quit"), self)
        quit_action.setStatusTip(self.tr("Exit the application"))
        quit_action.triggered.connect(
            self._close_application
        )  # Connects to the window's close method
        self.addAction(quit_action)

    def _open_new_project_dialog(self) -> None:
        """Open the new project dialog."""
        new_project_dialog = NewProjectDialog(parent=self)
        new_project_dialog.exec()

    def _open_project(self) -> None:
        """Open an existing project from a filename."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Select Project File"),
            filter=self.tr("Project Files (*.proj)"),
        )
        filename = Path(filename)

        # Check if filename is valid
        if not filename.exists():
            msg = "The selected project file does not exist."
            raise FileNotFoundError(msg)

        raise NotImplementedError  # To be implemented: opening the project in the model

    def _save_current_project(self) -> None:
        """Save the current project."""
        raise NotImplementedError

    def _save_current_project_as(self) -> None:
        """Save the current project to a new file."""
        raise NotImplementedError

    def _save_all_projects(self) -> None:
        """Save all open projects."""
        raise NotImplementedError

    def _close_current_project(self) -> None:
        """Close the current project without saving."""
        raise NotImplementedError

    def _close_all_projects(self) -> None:
        """Close all open projects without saving."""
        raise NotImplementedError

    def open_settings_dialog(self) -> None:
        """Open the application settings dialog."""
        raise NotImplementedError

    def _close_application(self) -> None:
        """Close the application."""
        raise NotImplementedError


class MainMenuBar(QMenuBar):
    """Main menu bar for the pytrebuchet application."""

    def __init__(
        self,
        parent: "MainWindow",
    ) -> None:
        """Initialize the main menu bar.

        Args:
            parent: Main window parent.

        """
        super().__init__(parent)

        self._main_window = parent

        self._create_menus()

    def _create_menus(self) -> None:
        """Create and add all menus to the menu bar."""
        self.addMenu(FileMenu(parent=self))
