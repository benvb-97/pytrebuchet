"""Qt-specific type hints."""

from PySide6.QtCore import QModelIndex, QPersistentModelIndex

type QtModelIndex = QModelIndex | QPersistentModelIndex
