import numpy as np

from src.config import ELEVATION_M

R_DRY = 287.05
RHO_REF = 1.225


def air_density(temp_c, elevation_m: float = ELEVATION_M):
    pressure = 101325.0 * (1 - 2.25577e-5 * elevation_m) ** 5.25588
    return pressure / (R_DRY * (np.asarray(temp_c, dtype=float) + 273.15))


def density_corrected_wind(wind, temp_c, elevation_m: float = ELEVATION_M):
    """IEC 61400-12 density normalisation: cold dense air yields more power at
    the same wind speed, so the curve is fitted against a corrected speed."""
    ratio = air_density(temp_c, elevation_m) / RHO_REF
    return np.asarray(wind, dtype=float) * ratio ** (1.0 / 3.0)


class PowerCurve:
    """Empirical power curve: median normalised power per wind-speed bin,
    forced monotonic, linearly interpolated in between."""

    def __init__(self, bin_width: float = 0.5, max_wind: float = 28.0):
        self.bin_width = bin_width
        self.max_wind = max_wind
        self.centers_ = None
        self.values_ = None

    def fit(self, wind, power):
        wind = np.asarray(wind, dtype=float)
        power = np.asarray(power, dtype=float)
        mask = np.isfinite(wind) & np.isfinite(power)
        wind, power = wind[mask], power[mask]

        edges = np.arange(0.0, self.max_wind + self.bin_width, self.bin_width)
        index = np.digitize(wind, edges) - 1
        index = np.clip(index, 0, len(edges) - 2)

        centers = edges[:-1] + self.bin_width / 2
        values = np.full(len(centers), np.nan)
        for i in range(len(centers)):
            bucket = power[index == i]
            if bucket.size >= 20:
                values[i] = np.median(bucket)

        filled = _fill_nan_1d(values)
        self.centers_ = centers
        self.values_ = np.clip(np.maximum.accumulate(filled), 0.0, 1.0)
        return self

    def predict(self, wind):
        wind = np.asarray(wind, dtype=float)
        return np.interp(wind, self.centers_, self.values_, left=0.0, right=self.values_[-1])


def _fill_nan_1d(values: np.ndarray) -> np.ndarray:
    out = values.copy()
    known = np.flatnonzero(np.isfinite(out))
    if known.size == 0:
        return np.zeros_like(out)
    out[: known[0]] = out[known[0]]
    out[known[-1] + 1 :] = out[known[-1]]
    missing = np.flatnonzero(~np.isfinite(out))
    if missing.size:
        out[missing] = np.interp(missing, known, out[known])
    return out
