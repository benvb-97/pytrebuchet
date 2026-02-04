"""Enumeration definitions for units used in PyTrebuchet."""

from enum import IntEnum


class MassUnit(IntEnum):
    """Enumeration for mass units."""

    KILOGRAM = 1
    GRAM = 0
    POUND = 2


class LengthUnit(IntEnum):
    """Enumeration for length units."""

    METER = 0
    CENTIMETER = 1
    MILLIMETER = 2
    FOOT = 3
    INCH = 4
