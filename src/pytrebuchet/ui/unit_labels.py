"""Display labels for unit enumerations used in the UI.

Labels are defined via exhaustive ``match`` statements. Static type checkers
(mypy, pyright) will raise an error if a new enum member is added without a
corresponding label, catching the omission before the code is run.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import assert_never

from pytrebuchet.units import LengthUnit, MassUnit


@dataclass(frozen=True)
class UnitLabel:
    """Display metadata for a single unit.

    Attrs:
        verbose: Full human-readable name (e.g. ``"Kilogram"``).
        symbol: Short symbol string (e.g. ``"kg"``).
    """

    verbose: str
    symbol: str

    @property
    def display(self) -> str:
        """Return the combined display string shown in the UI.

        Returns:
            Formatted string, e.g. ``"Kilogram (kg)"``.

        """
        return f"{self.verbose} ({self.symbol})"


def _mass_unit_label(unit: MassUnit) -> UnitLabel:
    """Return the display label for *unit*.

    Args:
        unit: The mass unit to look up.

    Returns:
        The corresponding ``UnitLabel``.

    """
    match unit:
        case MassUnit.KILOGRAM:
            return UnitLabel(verbose="Kilogram", symbol="kg")
        case MassUnit.GRAM:
            return UnitLabel(verbose="Gram", symbol="g")
        case MassUnit.POUND:
            return UnitLabel(verbose="Pound", symbol="lb")
        case _ as unreachable:
            assert_never(unreachable)


def _length_unit_label(unit: LengthUnit) -> UnitLabel:
    """Return the display label for *unit*.

    Args:
        unit: The length unit to look up.

    Returns:
        The corresponding ``UnitLabel``.

    """
    match unit:
        case LengthUnit.METER:
            return UnitLabel(verbose="Meter", symbol="m")
        case LengthUnit.CENTIMETER:
            return UnitLabel(verbose="Centimeter", symbol="cm")
        case LengthUnit.MILLIMETER:
            return UnitLabel(verbose="Millimeter", symbol="mm")
        case LengthUnit.FOOT:
            return UnitLabel(verbose="Foot", symbol="ft")
        case LengthUnit.INCH:
            return UnitLabel(verbose="Inch", symbol="in")
        case _ as unreachable:
            assert_never(unreachable)


mass_unit_labels: Mapping[MassUnit, UnitLabel] = {
    unit: _mass_unit_label(unit) for unit in MassUnit
}

length_unit_labels: Mapping[LengthUnit, UnitLabel] = {
    unit: _length_unit_label(unit) for unit in LengthUnit
}
