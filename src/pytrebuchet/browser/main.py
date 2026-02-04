"""Main module for the NiceGUI application.

When executed, this module starts a NiceGUI web application at the root path.
"""

from nicegui import ui

from pytrebuchet.browser.units import length_unit_labels, mass_unit_labels
from pytrebuchet.units import LengthUnit, MassUnit


def root() -> None:
    """Root function for the NiceGUI application."""
    with ui.row().style("gap: 50px"):
        ui.select(
            label="Mass Unit",
            options={
                unit: f"{label.verbose} ({label.symbol})"
                for unit, label in mass_unit_labels.items()
            },
            value=MassUnit.KILOGRAM,
        ).classes("w-45")  # Make the select take available space
        ui.select(
            label="Length Unit",
            options={
                unit: f"{label.verbose} ({label.symbol})"
                for unit, label in length_unit_labels.items()
            },
            value=LengthUnit.METER,
        ).classes("w-45")  # Make the select take available space


if __name__ == "__main__":
    ui.run(root)
