"""Module for managing session-specific data in the web interface.

Note: This Session class is currently not used in main.py, which now uses
NiceGUI's app.storage.user for persistent session management. This class is
retained for potential future use with trebuchet-specific session data.
"""

from pytrebuchet.trebuchet import HingedCounterweightTrebuchet
from pytrebuchet.units import LengthUnit, MassUnit


class Session:
    """Class to hold session-specific data."""

    def __init__(self) -> None:
        """Initialize the session with default units."""
        self.mass_unit: MassUnit = MassUnit.POUND
        self.length_unit: LengthUnit = LengthUnit.MILLIMETER

        self.trebuchet = HingedCounterweightTrebuchet.default()
