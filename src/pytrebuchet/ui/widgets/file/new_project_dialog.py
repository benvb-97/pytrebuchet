"""New Project dialog for creating a new project in RISQ.

It allows the user to select a directory for the new project through a
folder dialog.
"""

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class NewProjectDialog(QDialog):
    """Dialog for creating a new project.

    It allows the user to select a filename for the project
    using a .

    :param projects_model: The model containing the list of projects.
    :param parent: The parent widget.
    """

    create_new_project = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the NewProjectDialog."""
        super().__init__(parent)

        self._setup_ui()
        self._setup_connections()

    def _setup_ui(self) -> None:
        """Set up the widgets for the dialog."""
        # Set dialog properties
        self.setWindowTitle("New Project")
        self._layout = QVBoxLayout(self)

        # Add input widgets for choosing a project filename
        filename_widget = QWidget(self)
        filename_layout = QHBoxLayout(filename_widget)
        self._layout.addWidget(filename_widget)

        self._filename_line_edit = QLineEdit(parent=filename_widget)
        self._filename_line_edit.setPlaceholderText(self.tr("Choose filename"))
        self._filename_button = QPushButton(parent=filename_widget)
        self._filename_button.setIcon(QIcon.fromTheme("document-open"))

        filename_layout.addWidget(QLabel(self.tr("Project Filename")))
        filename_layout.addWidget(self._filename_line_edit)
        filename_layout.addWidget(self._filename_button)

        # Add button box for Create/Cancel project actions
        self._button_box = QDialogButtonBox(Qt.Orientation.Horizontal)
        self._layout.addWidget(self._button_box)

        _ = self._button_box.addButton(
            self.tr("Create"), QDialogButtonBox.ButtonRole.AcceptRole
        )
        _ = self._button_box.addButton(
            self.tr("Cancel"), QDialogButtonBox.ButtonRole.RejectRole
        )

        # Add stretch to push content to the top
        self._layout.addStretch()

    def _setup_connections(self) -> None:
        """Set up signal/slot connections for the dialog."""
        self._filename_button.clicked.connect(self._open_filename_dialog)

        self._button_box.accepted.connect(self._create_new_project)
        self._button_box.rejected.connect(self.reject)

    def _create_new_project(self) -> None:
        """Create a new project with the selected filename. Close the dialog."""
        raise NotImplementedError

    def _open_filename_dialog(self) -> None:
        """Open a file dialog to select a project."""
        filename, _ = QFileDialog.getSaveFileName(
            self,
            self.tr("Select Project File"),
            filter=(self.tr("Project Files (*.proj)")),
        )

        filename = Path(filename)

        # Check if the filename is valid
        if filename.suffix != ".proj":
            filename = filename.with_suffix(".proj")
        self._filename_line_edit.setText(str(filename))
