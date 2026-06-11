from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd


FINAL_FREQUENCY = "15min"
TIMESTEP_HOURS = 0.25


def _find_column(columns: pd.Index, candidates: list[str]) -> str | None:
    normalized = {str(column).strip().lower(): str(column) for column in columns}
    for candidate in candidates:
        found = normalized.get(candidate.lower())
        if found is not None:
            return found
    return None


def read_demand_file(path: str | Path) -> pd.DataFrame:
    demand_file = Path(path)
    if not demand_file.exists():
        raise FileNotFoundError(f"No se encontro el archivo de demanda: {demand_file}")

    suffix = demand_file.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(demand_file)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(demand_file)
    raise ValueError("La demanda debe venir en un archivo .csv, .xlsx o .xls.")


def read_uploaded_demand(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    raise ValueError("Formato no soportado. Usa CSV, XLSX o XLS.")


def _localize_datetime_series(
    timestamps: pd.Series,
    timezone: str,
    column_name: str,
) -> pd.Series:
    if timestamps.isna().any():
        invalid_count = int(timestamps.isna().sum())
        raise ValueError(
            f"Hay {invalid_count} fechas invalidas en la columna {column_name!r}."
        )

    if timestamps.dt.tz is None:
        return timestamps.dt.tz_localize(
            timezone,
            ambiguous="infer",
            nonexistent="shift_forward",
        )
    return timestamps.dt.tz_convert(timezone)


def _resample_demand_to_15min(demand: pd.DataFrame) -> pd.DataFrame:
    if len(demand) < 2:
        return demand

    diffs = demand.index.to_series().diff().dropna()
    expected = pd.Timedelta(FINAL_FREQUENCY)
    if not diffs.eq(expected).all():
        warnings.warn(
            "La demanda no tiene frecuencia exacta de 15 minutos; se remuestrea "
            "con interpolacion temporal.",
            stacklevel=2,
        )
        target = pd.date_range(
            start=demand.index.min(),
            end=demand.index.max(),
            freq=FINAL_FREQUENCY,
        )
        demand = (
            demand.reindex(demand.index.union(target))
            .interpolate(method="time")
            .reindex(target)
        )
    return demand


def normalize_demand_dataframe(
    raw: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    timezone: str,
    align_by_position: bool = False,
) -> pd.DataFrame:
    raw = raw.copy()
    timestamp_column = _find_column(raw.columns, ["Fecha_Hora", "time", "timestamp"])
    demand_column = _find_column(
        raw.columns,
        ["Demanda_kW", "demanda_kw", "kW", "kw", "Demanda"],
    )
    power_factor_column = _find_column(
        raw.columns,
        ["Factor_Potencia", "factor_potencia", "fp"],
    )

    if timestamp_column is None:
        raise ValueError("Falta columna de fecha: Fecha_Hora, time o timestamp.")
    if demand_column is None:
        raise ValueError("Falta columna de demanda. Usa Demanda_kW.")

    if timestamp_column == "Fecha_Hora":
        timestamps = pd.to_datetime(
            raw[timestamp_column],
            format="%Y-%m-%d %H:%M:%S",
            errors="coerce",
        )
        if timestamps.isna().any():
            timestamps = pd.to_datetime(raw[timestamp_column], errors="coerce")
    else:
        timestamps = pd.to_datetime(raw[timestamp_column], errors="coerce")

    timestamps = _localize_datetime_series(timestamps, timezone, timestamp_column)

    if timestamps.duplicated().any():
        raise ValueError("La curva de demanda contiene fechas duplicadas.")

    demand_values = pd.to_numeric(raw[demand_column], errors="coerce")
    if demand_values.isna().any():
        raise ValueError("La columna de demanda contiene valores no numericos.")

    if power_factor_column is None:
        power_factor = pd.Series(np.nan, index=raw.index)
    else:
        power_factor = pd.to_numeric(raw[power_factor_column], errors="coerce")

    demand = pd.DataFrame(
        {
            "Demanda_kW": demand_values.to_numpy(),
            "Factor_Potencia": power_factor.to_numpy(),
        },
        index=pd.DatetimeIndex(timestamps),
    ).sort_index()
    demand = _resample_demand_to_15min(demand)

    if align_by_position:
        if len(demand) != len(target_index):
            raise ValueError(
                "No se puede alinear demanda y clima por posicion: "
                f"demanda={len(demand):,}, clima={len(target_index):,} intervalos."
            )
        aligned = demand.copy()
        aligned.insert(
            0,
            "Fecha_Hora_Demanda_Original",
            demand.index.tz_convert(timezone).tz_localize(None),
        )
        aligned.index = target_index
        return aligned

    aligned = demand.reindex(target_index)
    if aligned["Demanda_kW"].isna().any():
        raise ValueError(
            "Las fechas de demanda y clima no coinciden. Activa la alineacion por "
            "posicion para usar un ano climatico analogo."
        )
    return aligned


def load_demand(
    path: str | Path,
    target_index: pd.DatetimeIndex,
    timezone: str,
    align_by_position: bool = False,
) -> pd.DataFrame:
    return normalize_demand_dataframe(
        read_demand_file(path),
        target_index=target_index,
        timezone=timezone,
        align_by_position=align_by_position,
    )


def generar_demanda_sintetica_15min(
    target_index: pd.DatetimeIndex,
    consumo_anual_kwh: float | None = None,
    consumos_mensuales_kwh: list[float] | None = None,
    factor_potencia: float = 0.90,
    tipo_perfil: str = "industrial",
) -> pd.DataFrame:
    if len(target_index) == 0:
        raise ValueError("El indice objetivo de demanda esta vacio.")
    if tipo_perfil != "industrial":
        raise ValueError("Solo esta disponible tipo_perfil='industrial'.")
    if consumo_anual_kwh is None and consumos_mensuales_kwh is None:
        raise ValueError("Indica consumo_anual_kwh o consumos_mensuales_kwh.")
    if consumo_anual_kwh is not None and consumos_mensuales_kwh is not None:
        raise ValueError("Usa consumo anual o consumos mensuales, no ambos.")

    local_index = target_index.tz_localize(None) if target_index.tz else target_index
    hour = local_index.hour + local_index.minute / 60
    weekday = local_index.dayofweek < 5
    work_shift = (hour >= 7) & (hour < 19)
    evening = (hour >= 19) & (hour < 22)

    base_profile = (
        0.45
        + 0.55 * work_shift.astype(float)
        + 0.16 * evening.astype(float)
        + 0.12 * weekday.astype(float)
    )
    base_profile = pd.Series(base_profile, index=target_index).clip(lower=0.10)

    if consumo_anual_kwh is not None:
        if consumo_anual_kwh <= 0:
            raise ValueError("El consumo anual debe ser mayor a 0.")
        scale = consumo_anual_kwh / (base_profile.sum() * TIMESTEP_HOURS)
        demand_kw = base_profile * scale
    else:
        if len(consumos_mensuales_kwh or []) != 12:
            raise ValueError("Debes indicar 12 consumos mensuales.")
        if any(value < 0 for value in consumos_mensuales_kwh):
            raise ValueError("Los consumos mensuales deben ser mayores o iguales a 0.")
        if sum(consumos_mensuales_kwh) <= 0:
            raise ValueError("La suma de consumos mensuales debe ser mayor a 0.")

        demand_kw = pd.Series(0.0, index=target_index)
        months = pd.Series(local_index.month, index=target_index)
        for month in range(1, 13):
            mask = months == month
            month_energy = float(consumos_mensuales_kwh[month - 1])
            if not mask.any() or month_energy == 0:
                continue
            month_profile = base_profile.loc[mask]
            scale = month_energy / (month_profile.sum() * TIMESTEP_HOURS)
            demand_kw.loc[mask] = month_profile * scale

    return pd.DataFrame(
        {
            "Fecha_Hora": local_index,
            "Demanda_kW": demand_kw.to_numpy(),
            "Factor_Potencia": float(factor_potencia),
        }
    )


def demand_energy_kwh(demand: pd.DataFrame) -> float:
    return float(pd.to_numeric(demand["Demanda_kW"], errors="coerce").sum() * TIMESTEP_HOURS)
