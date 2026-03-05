"""Enumeration for unit systems."""

from dataclasses import dataclass

from pytrebuchet.units import LengthUnit, MassUnit


@dataclass(frozen=True)
class UnitLabel:
    """Dataclass for unit labels."""

    verbose: str
    symbol: str


mass_unit_labels = {
    MassUnit.KILOGRAM: UnitLabel(verbose="Kilogram", symbol="kg"),
    MassUnit.GRAM: UnitLabel(verbose="Gram", symbol="g"),
    MassUnit.POUND: UnitLabel(verbose="Pound", symbol="lb"),
}

length_unit_labels = {
    LengthUnit.METER: UnitLabel(verbose="Meter", symbol="m"),
    LengthUnit.CENTIMETER: UnitLabel(verbose="Centimeter", symbol="cm"),
    LengthUnit.MILLIMETER: UnitLabel(verbose="Millimeter", symbol="mm"),
    LengthUnit.FOOT: UnitLabel(verbose="Foot", symbol="ft"),
    LengthUnit.INCH: UnitLabel(verbose="Inch", symbol="in"),
}
