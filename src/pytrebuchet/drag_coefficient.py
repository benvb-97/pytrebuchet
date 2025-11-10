import warnings

import numpy as np


def drag_coefficient_smooth_sphere_clift_grace_weber(
    reynolds_number: float | np.ndarray[float],
) -> float | np.ndarray[float]:
    """
    Calculates the drag coefficient for a smooth sphere based on the Reynolds number.

    Source: Clift, Grace, and Weber (Bubbles, Drops, and Particles, Academic Press, 1978)

    :param reynolds_number: Reynolds number of the sphere
    :return: drag coefficient of the sphere (dimensionless)
    """

    # Cast to numpy array for vectorized operations
    if type(reynolds_number) is float:
        is_float = True
        reynolds_number = np.array([reynolds_number])
    else:
        is_float = False

    drag_coefficient = np.zeros_like(reynolds_number)

    # Calculate drag coefficient based on Reynolds number ranges
    w = np.log10(reynolds_number)

    # Re <= 0.01
    idx = reynolds_number <= 1e-2
    drag_coefficient[idx] = 9 / 2 + 24 / reynolds_number[idx]

    # 0.01 < Re <= 20
    idx = (reynolds_number > 1e-2) & (reynolds_number <= 20)
    drag_coefficient[idx] = (
        24
        / reynolds_number[idx]
        * (1 + 0.1315 * np.float_power(reynolds_number[idx], 0.82 - 0.05 * w[idx]))
    )

    # 20 < Re <= 260
    idx = (reynolds_number > 20) & (reynolds_number <= 260)
    drag_coefficient[idx] = (
        24
        / reynolds_number[idx]
        * (1 + 0.1935 * np.float_power(reynolds_number[idx], 0.6305))
    )

    # 260 < Re <= 1.5e3
    idx = (reynolds_number > 260) & (reynolds_number <= 1.5e3)
    drag_coefficient[idx] = np.power(
        10, 1.6435 - 1.1242 * w[idx] + 0.1558 * np.square(w[idx])
    )

    # 1.5e3 < Re <= 1.2e4
    idx = (reynolds_number > 1.5e3) & (reynolds_number <= 1.2e4)
    drag_coefficient[idx] = np.power(
        10,
        -2.4571
        + 2.5558 * w[idx]
        - 0.9295 * np.square(w[idx])
        + 0.1049 * np.power(w[idx], 3),
    )

    # 1.2e4 < Re <= 4.4e4
    idx = (reynolds_number > 1.2e4) & (reynolds_number <= 4.4e4)
    drag_coefficient[idx] = np.power(
        10, -1.9181 + 0.6370 * w[idx] - 0.0636 * np.square(w[idx])
    )

    # 4.4e4 < Re <= 3.38e5
    idx = (reynolds_number > 4.4e4) & (reynolds_number <= 3.38e5)
    drag_coefficient[idx] = np.power(
        10, -4.3390 + 1.5809 * w[idx] - 0.1546 * np.square(w[idx])
    )

    # 3.38e5 < Re <= 4e5
    idx = (reynolds_number > 3.38e5) & (reynolds_number <= 4e5)
    drag_coefficient[idx] = 29.78 - 5.3 * w[idx]

    # 4e5 < Re <= 1e6
    idx = (reynolds_number > 4e5) & (reynolds_number <= 1e6)
    drag_coefficient[idx] = 0.1 * w[idx] - 0.49

    # Re > 1e6
    idx = reynolds_number > 1e6
    drag_coefficient[idx] = 0.19 - 8e4 / reynolds_number[idx]

    # Cast to original format
    if is_float:
        assert drag_coefficient.size == 1
        drag_coefficient = float(drag_coefficient[0])
    return drag_coefficient


def drag_coefficient_smooth_sphere_morrison(
    reynolds_number: float | np.ndarray[float],
) -> float | np.ndarray[float]:
    """
    Calculates the drag coefficient for a smooth sphere based on the Reynolds number.

    Source: Data Correlation for Drag Coefficient for Sphere Faith A. Morrison,
    Department of Chemical Engineering Michigan Technological University, Houghton, MI 49931

    :param reynolds_number: Reynolds number of the sphere
    :return: drag coefficient of the sphere (dimensionless)
    """

    # Correlation is valid for Reynolds numbers between 0.1 and 1e6, consider 0.01 to 1e7 to be safe
    if isinstance(reynolds_number, float):
        if not (reynolds_number >= 0.01 and reynolds_number <= 1e7):
            warnings.warn(f"Reynolds number {reynolds_number} out of range {0.1, 1e7}")
    else:
        min_re, max_re = np.min(reynolds_number), np.max(reynolds_number)
        if (min_re < 0.01) | (max_re > 1e7):
            warnings.warn(f"Reynolds number {min_re, max_re} out of range {0.1, 1e7}")

    # Calculate drag coefficient using the correlation
    drag_coefficient = (
        24 / reynolds_number
        + 2.6
        * (reynolds_number / 5.0)
        / (1 + np.float_power(reynolds_number / 5.0, 1.52))
        + 0.411
        * np.float_power(reynolds_number / 263000, -7.94)
        / (1 + np.power(reynolds_number / 263000, -8))
        + np.float_power(reynolds_number, 0.8) / 461000.0
    )

    return drag_coefficient
