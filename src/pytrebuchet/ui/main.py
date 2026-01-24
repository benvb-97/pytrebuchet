"""Job Tracker main application entry point."""

import logging
import logging.config
import sys

from PySide6.QtWidgets import QApplication

from pytrebuchet.ui.widgets.main_window import MainWindow
from pytrebuchet.utilities.logging import LOGGING_CONFIG


def main() -> None:
    """Entry point for the pytrebuchet UI."""
    # Explicitly tell Windows what the correct AppUserModelID is for this application.
    # Prevents windows from grouping multiple instances of the app under "Python.exe"
    # in the taskbar.
    myappid = "pytrebuchet"  # arbitrary string

    try:
        import ctypes  # noqa: PLC0415

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except (AttributeError, ImportError):
        pass  # Not on Windows

    # Initialize logger
    logging.config.dictConfig(LOGGING_CONFIG)
    log = logging.getLogger(__name__)

    # Start the application
    log.info("Starting pytrebuchet user interface")
    application = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    return_code = application.exec()

    # Exit the application
    log.info("Exiting pytrebuchet user interface with return code %d", return_code)
    sys.exit(return_code)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.getLogger(__name__).exception("An unhandled exception occurred")
        sys.exit(1)
