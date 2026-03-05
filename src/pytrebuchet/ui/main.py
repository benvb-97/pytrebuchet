"""Main module for the NiceGUI application.

When executed, this module starts a NiceGUI web application at the root path.
"""

import os

from nicegui import app, ui

from pytrebuchet.ui.units import length_unit_labels, mass_unit_labels
from pytrebuchet.units import LengthUnit, MassUnit


def initialize_user_defaults() -> None:
    """Initialize default unit preferences for new users."""
    if "mass_unit" not in app.storage.user:
        app.storage.user["mass_unit"] = MassUnit.POUND
    if "length_unit" not in app.storage.user:
        app.storage.user["length_unit"] = LengthUnit.MILLIMETER


class UnitSelector:
    """UI component for selecting mass and length units."""

    def __init__(self, storage: dict) -> None:
        """Initialize the unit selector with user storage.

        Args:
            storage: Dictionary-like storage object for unit preferences.

        """
        self.storage = storage
        self._build_ui()

    def _build_ui(self) -> None:
        """Construct the unit selection UI components."""
        with ui.row().style("gap: 50px"):
            mass_select = ui.select(
                label="Mass Unit",
                options={
                    unit: f"{label.verbose} ({label.symbol})"
                    for unit, label in mass_unit_labels.items()
                },
                value=self.storage.get("mass_unit"),
            )
            mass_select.classes("w-45")
            mass_select.bind_value(self.storage, "mass_unit")

            length_select = ui.select(
                label="Length Unit",
                options={
                    unit: f"{label.verbose} ({label.symbol})"
                    for unit, label in length_unit_labels.items()
                },
                value=self.storage.get("length_unit"),
            )
            length_select.classes("w-45")
            length_select.bind_value(self.storage, "length_unit")


@ui.page("/")
def root() -> None:
    """Root page for the trebuchet configuration interface."""
    initialize_user_defaults()
    UnitSelector(app.storage.user)


if __name__ in {"__main__", "__mp_main__"}:
    storage_secret = os.environ.get("NICEGUI_STORAGE_SECRET", "")
    ui.run(storage_secret=storage_secret)
