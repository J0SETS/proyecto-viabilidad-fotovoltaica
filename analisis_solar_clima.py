"""Analisis historico de clima e irradiancia solar para Veracruz.

Descarga datos horarios historicos de Open-Meteo, calcula la irradiancia sobre
el plano del panel con pvlib y guarda CSV + graficas en la carpeta outputs.
"""




from __future__ import annotations




import argparse
import json
import urllib.parse
import urllib.request
import warnings
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")

import calendar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pvlib
import seaborn as sns


OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FINAL_FREQUENCY = "15min"
TIMESTEP_HOURS = 0.25

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "cloud_cover",
    "wind_speed_10m",
    "shortwave_radiation",
    "direct_normal_irradiance",
    "diffuse_radiation",
    "direct_radiation",
]


@dataclass(frozen=True)
class SiteConfig:
    name: str = "Veracruz"
    latitude: float = 19.48
    longitude: float = -96.95
    timezone: str = "America/Mexico_City"
    altitude_m: float = 500.0
    surface_tilt: float = 19.5
    surface_azimuth: float = 180.0
    albedo: float = 0.2


@dataclass(frozen=True)
class ModuleConfig:
    key: str
    manufacturer: str
    model: str
    module_type: str
    technology: str
    pdc0_w: float
    efficiency: float
    area_m2: float
    mass_kg: float
    gamma_pdc_per_c: float
    voc_v: float
    isc_a: float
    vmp_v: float
    imp_a: float
    default_bifacial_gain: float = 0.0
    bifacial_factor: float | None = None


MODULE_OPTIONS = {
    "monofacial": ModuleConfig(
        key="monofacial",
        manufacturer="Jinko Solar",
        model="Tiger Neo 72HC JKM605N-72HL4",
        module_type="Monofacial",
        technology="n-type TOPCon",
        pdc0_w=605.0,
        efficiency=0.2342,
        area_m2=2.583,
        mass_kg=27.0,
        gamma_pdc_per_c=-0.0029,
        voc_v=53.11,
        isc_a=14.31,
        vmp_v=44.23,
        imp_a=13.68,
    ),
    "bifacial": ModuleConfig(
        key="bifacial",
        manufacturer="Jinko Solar",
        model="Tiger Neo N-type JKM625N-78HL4-BDV",
        module_type="Bifacial doble vidrio",
        technology="n-type",
        pdc0_w=625.0,
        efficiency=0.2236,
        area_m2=2.795,
        mass_kg=34.6,
        gamma_pdc_per_c=-0.0029,
        voc_v=55.72,
        isc_a=14.27,
        vmp_v=46.10,
        imp_a=13.56,
        default_bifacial_gain=0.15,
        bifacial_factor=0.80,
    ),
}


def default_date_range(timezone: str) -> tuple[str, str]:
    """Return the last complete calendar year as ISO dates."""
    today = datetime.now(ZoneInfo(timezone)).date()
    year = today.year - 1
    return f"{year}-01-01", f"{year}-12-31"


def parse_iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Fecha invalida: {value!r}. Usa formato YYYY-MM-DD."
        ) from exc


def validate_date_range(start_date: date, end_date: date, timezone: str) -> None:
    if end_date < start_date:
        raise ValueError("La fecha final no puede ser anterior a la fecha inicial.")

    today = datetime.now(ZoneInfo(timezone)).date()
    if end_date > today:
        raise ValueError(
            "Open-Meteo historico no sirve fechas futuras. "
            f"Fecha final recibida: {end_date}; hoy es {today}."
        )


def build_open_meteo_url(site: SiteConfig, start_date: date, end_date: date) -> str:
    params = {
        "latitude": site.latitude,
        "longitude": site.longitude,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": site.timezone,
        "timeformat": "unixtime",
        "wind_speed_unit": "ms",
    }
    return f"{OPEN_METEO_ARCHIVE_URL}?{urllib.parse.urlencode(params)}"


def fetch_historical_weather(
    site: SiteConfig, start_date: date, end_date: date
) -> tuple[pd.DataFrame, str]:
    """Download hourly historical weather from Open-Meteo."""
    validate_date_range(start_date, end_date, site.timezone)
    url = build_open_meteo_url(site, start_date, end_date)

    with urllib.request.urlopen(url, timeout=90) as response:
        payload = json.loads(response.read().decode("utf-8"))

    if payload.get("error"):
        reason = payload.get("reason", "sin detalle")
        raise RuntimeError(f"Open-Meteo rechazo la solicitud: {reason}")

    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        raise RuntimeError("La respuesta de Open-Meteo no contiene datos horarios.")

    weather = pd.DataFrame(hourly)
    weather["time"] = pd.to_datetime(weather["time"], unit="s", utc=True).dt.tz_convert(
        site.timezone
    )
    weather = weather.set_index("time").sort_index()
    weather = weather[~weather.index.duplicated(keep="first")]

    for column in weather.columns:
        weather[column] = pd.to_numeric(weather[column], errors="coerce")

    return weather, url


def build_target_index(
    start_date: date,
    end_date: date,
    timezone: str,
    freq: str = FINAL_FREQUENCY,
) -> pd.DatetimeIndex:
    """Create the weather master index, including the final interval of each day."""
    requested_offset = pd.tseries.frequencies.to_offset(freq)
    final_offset = pd.tseries.frequencies.to_offset(FINAL_FREQUENCY)
    if requested_offset != final_offset:
        warnings.warn(
            f"La frecuencia solicitada {freq!r} se remuestreara a {FINAL_FREQUENCY!r}; "
            "el motor industrial trabaja en intervalos de 15 minutos.",
            stacklevel=2,
        )

    start = pd.Timestamp(start_date, tz=timezone)
    end = pd.Timestamp(end_date + pd.Timedelta(days=1), tz=timezone) - final_offset
    return pd.date_range(start=start, end=end, freq=FINAL_FREQUENCY)


def _interpolate_to_target(
    series: pd.Series,
    target_index: pd.DatetimeIndex,
    fallback: float,
) -> pd.Series:
    expanded_index = series.index.union(target_index)
    return (
        pd.to_numeric(series, errors="coerce")
        .reindex(expanded_index)
        .interpolate(method="time")
        .ffill()
        .bfill()
        .reindex(target_index)
        .fillna(fallback)
    )


def _clearness_ratio_to_target(
    observed: pd.Series,
    clear_sky_hourly: pd.Series,
    target_index: pd.DatetimeIndex,
    upper_limit: float,
) -> pd.Series:
    ratio = observed.clip(lower=0) / clear_sky_hourly.where(clear_sky_hourly > 10)
    ratio = ratio.replace([np.inf, -np.inf], np.nan).clip(lower=0, upper=upper_limit)
    expanded_index = ratio.index.union(target_index)
    return (
        ratio.reindex(expanded_index)
        .interpolate(method="time")
        .ffill()
        .bfill()
        .reindex(target_index)
        .fillna(0)
    )


def resample_weather_to_15min(
    weather: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    site: SiteConfig,
) -> pd.DataFrame:
    """Convert hourly weather to 15 minutes using clear-sky radiation profiles.

    Scalar weather variables are interpolated in time. For GHI, DNI and DHI,
    the hourly ratio against clear sky is interpolated and applied to a
    15-minute clear-sky curve. This keeps sunrise, sunset and nighttime
    physically consistent instead of drawing straight radiation ramps.
    """
    if target_index.tz is None:
        raise ValueError("El indice maestro climatico debe incluir zona horaria.")

    weather = weather.copy()
    if weather.index.tz is None:
        weather.index = weather.index.tz_localize(site.timezone)
    else:
        weather.index = weather.index.tz_convert(site.timezone)

    location = pvlib.location.Location(
        latitude=site.latitude,
        longitude=site.longitude,
        tz=site.timezone,
        altitude=site.altitude_m,
        name=site.name,
    )
    clear_hourly = location.get_clearsky(weather.index, model="ineichen")
    clear_15min = location.get_clearsky(target_index, model="ineichen")
    solpos_15min = location.get_solarposition(target_index)
    daylight = solpos_15min["apparent_elevation"] > 0

    result = pd.DataFrame(index=target_index)
    scalar_fallbacks = {
        "temperature_2m": 25.0,
        "relative_humidity_2m": 0.0,
        "cloud_cover": 0.0,
        "wind_speed_10m": 1.0,
    }
    for column, fallback in scalar_fallbacks.items():
        if column not in weather.columns:
            warnings.warn(
                f"Open-Meteo no entrego {column!r}; se usara fallback {fallback}.",
                stacklevel=2,
            )
            result[column] = fallback
        else:
            result[column] = _interpolate_to_target(
                weather[column], target_index, fallback=fallback
            )

    if "precipitation" in weather.columns:
        # Open-Meteo reports an hourly accumulation. Distribute it over 4 slots.
        result["precipitation"] = (
            pd.to_numeric(weather["precipitation"], errors="coerce")
            .fillna(0)
            .clip(lower=0)
            .reindex(target_index, method="ffill")
            .fillna(0)
            / 4
        )
    else:
        result["precipitation"] = 0.0

    radiation_config = {
        "shortwave_radiation": ("ghi", 1.5),
        "direct_normal_irradiance": ("dni", 1.5),
        "diffuse_radiation": ("dhi", 3.0),
    }
    for weather_column, (clear_column, upper_limit) in radiation_config.items():
        if weather_column not in weather.columns:
            raise ValueError(
                f"Open-Meteo no entrego la variable de radiacion {weather_column!r}."
            )
        ratio = _clearness_ratio_to_target(
            pd.to_numeric(weather[weather_column], errors="coerce").fillna(0),
            clear_hourly[clear_column],
            target_index,
            upper_limit=upper_limit,
        )
        result[weather_column] = (
            clear_15min[clear_column].mul(ratio).where(daylight, 0).clip(lower=0)
        )

    cosine_zenith = np.cos(np.radians(solpos_15min["apparent_zenith"])).clip(lower=0)
    result["direct_radiation"] = (
        result["direct_normal_irradiance"].mul(cosine_zenith).where(daylight, 0)
    )
    return result


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
            timezone, ambiguous="infer", nonexistent="shift_forward"
        )
    return timestamps.dt.tz_convert(timezone)


def _read_demand_file(demand_file: Path) -> pd.DataFrame:
    if not demand_file.exists():
        raise FileNotFoundError(
            f"No se encontro el archivo de demanda: {demand_file.resolve()}"
        )
    suffix = demand_file.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(demand_file)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(demand_file)
    raise ValueError("La demanda debe venir en un archivo .csv, .xlsx o .xls.")


def _find_column(columns: pd.Index, candidates: list[str]) -> str | None:
    normalized = {str(column).strip().lower(): str(column) for column in columns}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    return None


def _resample_demand_to_15min(demand: pd.DataFrame) -> pd.DataFrame:
    if len(demand) < 2:
        return demand
    diffs = demand.index.to_series().diff().dropna()
    expected = pd.Timedelta(FINAL_FREQUENCY)
    if not diffs.eq(expected).all():
        warnings.warn(
            "La demanda no tiene frecuencia exacta de 15 minutos; se remuestreara "
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


def load_demand(
    demand_file: Path,
    target_index: pd.DatetimeIndex,
    timezone: str,
    align_by_position: bool = False,
) -> pd.DataFrame:
    """Load industrial demand and align it with the weather master index."""
    raw = _read_demand_file(demand_file)
    timestamp_column = _find_column(raw.columns, ["Fecha_Hora", "time", "timestamp"])
    demand_column = _find_column(raw.columns, ["Demanda_kW"])
    power_factor_column = _find_column(raw.columns, ["Factor_Potencia"])

    if timestamp_column is None:
        raise ValueError("Falta columna de fecha: Fecha_Hora, time o timestamp.")
    if demand_column is None:
        raise ValueError("Falta la columna obligatoria 'Demanda_kW'.")

    if timestamp_column == "Fecha_Hora":
        timestamps = pd.to_datetime(
            raw[timestamp_column],
            format="%Y-%m-%d %H:%M:%S",
            errors="coerce",
        )
    else:
        timestamps = pd.to_datetime(raw[timestamp_column], errors="coerce")
    timestamps = _localize_datetime_series(timestamps, timezone, timestamp_column)

    if timestamps.duplicated().any():
        raise ValueError("La curva de demanda contiene fechas duplicadas.")

    demand_values = pd.to_numeric(raw[demand_column], errors="coerce")
    if demand_values.isna().any():
        raise ValueError("La columna 'Demanda_kW' contiene NaN o valores no numericos.")

    if power_factor_column is None:
        warnings.warn(
            "No se encontro 'Factor_Potencia'; se conservara como NaN.",
            stacklevel=2,
        )
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

    if len(target_index) == 35040 and len(demand) != 35040:
        raise ValueError(
            "Un periodo de 365 dias a 15 minutos requiere 35,040 registros de "
            f"demanda; se recibieron {len(demand):,}."
        )

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
            "Las fechas de demanda y clima no coinciden. Usa "
            "--align-demand-by-position para aplicar un ano climatico analogo."
        )
    return aligned


def generate_demand(target_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Generate a deterministic industrial profile only when explicitly requested."""
    hour = target_index.hour + target_index.minute / 60
    weekday = target_index.dayofweek < 5
    work_shift = (hour >= 7) & (hour < 19)
    demand_kw = 22 + 22 * work_shift.astype(float) + 5 * weekday.astype(float)
    generated = pd.DataFrame(index=target_index)
    generated["Demanda_kW"] = demand_kw
    generated["Factor_Potencia"] = 0.90
    return generated


def infer_timestep_hours(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 1.0
    seconds = index.to_series().diff().dt.total_seconds().dropna().median()
    return float(seconds / 3600) if pd.notna(seconds) and seconds > 0 else 1.0


def module_gain(module: ModuleConfig, bifacial_gain: float) -> float:
    if module.bifacial_factor is None:
        return 0.0
    # Rear-side gain is an irradiance contribution. The module only converts
    # its bifacial fraction, so 15% rear gain * 0.80 factor becomes 12%.
    return bifacial_gain * module.bifacial_factor


def resolve_system_capacity(
    module: ModuleConfig,
    system_dc_kwp: float,
    num_modules: int | None,
) -> tuple[float, float]:
    if num_modules is not None:
        if num_modules <= 0:
            raise ValueError("--num-modules debe ser mayor que cero.")
        resolved_dc_kwp = num_modules * module.pdc0_w / 1000
        return resolved_dc_kwp, float(num_modules)
    if system_dc_kwp <= 0:
        raise ValueError("--system-dc-kwp debe ser mayor que cero.")
    return system_dc_kwp, system_dc_kwp * 1000 / module.pdc0_w


def add_module_calculations(
    result: pd.DataFrame,
    module: ModuleConfig,
    bifacial_gain: float,
    system_dc_kwp: float,
    num_modules: int | None,
    inverter_efficiency: float,
    system_losses: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Estimate module and complete-system generation with PVWatts."""
    result = result.copy()
    timestep_hours = infer_timestep_hours(result.index)
    if not np.isclose(timestep_hours, TIMESTEP_HOURS):
        raise ValueError(
            f"El calculo FV requiere intervalos de 15 minutos; se detectaron "
            f"{timestep_hours:.4f} horas."
        )

    if "temperature_2m" not in result:
        warnings.warn(
            "Falta temperatura historica; Faiman usara fallback explicito de 25 C.",
            stacklevel=2,
        )
        temp_air = pd.Series(25.0, index=result.index)
    else:
        temp_air = (
            result["temperature_2m"].interpolate(limit_direction="both").fillna(25.0)
        )

    if "wind_speed_10m" not in result:
        warnings.warn(
            "Falta viento historico; Faiman usara fallback explicito de 1 m/s.",
            stacklevel=2,
        )
        wind_speed = pd.Series(1.0, index=result.index)
    else:
        wind_speed = (
            result["wind_speed_10m"]
            .interpolate(limit_direction="both")
            .fillna(1.0)
            .clip(lower=0.0)
        )

    resolved_dc_kwp, equivalent_modules = resolve_system_capacity(
        module, system_dc_kwp, num_modules
    )
    effective_gain = module_gain(module, bifacial_gain)
    effective_poa = result["poa_global"] * (1 + effective_gain)
    cell_temperature = pvlib.temperature.faiman(
        effective_poa,
        temp_air=temp_air,
        wind_speed=wind_speed,
    )
    module_dc_power_w = pvlib.pvsystem.pvwatts_dc(
        effective_irradiance=effective_poa,
        temp_cell=cell_temperature,
        pdc0=module.pdc0_w,
        gamma_pdc=module.gamma_pdc_per_c,
    ).fillna(0).clip(lower=0)
    system_dc_power_kw = (
        pvlib.pvsystem.pvwatts_dc(
            effective_irradiance=effective_poa,
            temp_cell=cell_temperature,
            pdc0=resolved_dc_kwp * 1000,
            gamma_pdc=module.gamma_pdc_per_c,
        )
        .fillna(0)
        .clip(lower=0)
        / 1000
    )
    system_ac_power_kw = (
        system_dc_power_kw * (1 - system_losses) * inverter_efficiency
    )

    result["Ganancia_Bifacial_Efectiva"] = effective_gain
    result["POA_Efectiva_Wm2"] = effective_poa.fillna(0).clip(lower=0)
    result["Temperatura_Celda_C"] = cell_temperature
    result["Generacion_DC_Modulo_W"] = module_dc_power_w
    result["Energia_DC_Modulo_kWh"] = module_dc_power_w * timestep_hours / 1000
    result["Generacion_DC_kW"] = system_dc_power_kw
    result["Generacion_AC_kW"] = system_ac_power_kw
    result["Energia_Solar_DC_kWh"] = system_dc_power_kw * timestep_hours
    result["Energia_Solar_AC_kWh"] = system_ac_power_kw * timestep_hours
    result["Modulo"] = module.model
    result["Tipo_Modulo"] = module.module_type
    result["Potencia_DC_Sistema_kWp"] = resolved_dc_kwp
    result["Num_Modulos_Equivalente"] = equivalent_modules

    scenario = {
        "system_dc_kwp": resolved_dc_kwp,
        "num_modules_equivalent": equivalent_modules,
        "effective_bifacial_gain": effective_gain,
    }
    return result, scenario


def add_demand_balance(result: pd.DataFrame, demand: pd.DataFrame) -> pd.DataFrame:
    """Join demand and compute the AC-side post-injection energy balance."""
    result = result.join(demand, how="left")
    if result["Demanda_kW"].isna().any():
        raise ValueError("La demanda alineada contiene intervalos vacios.")

    result["Demanda_Post_Inyeccion_Solar_kW"] = (
        result["Demanda_kW"] - result["Generacion_AC_kW"]
    ).clip(lower=0)
    result["Energia_Demanda_kWh"] = result["Demanda_kW"] * TIMESTEP_HOURS
    result["Energia_Red_kWh"] = (
        result["Demanda_Post_Inyeccion_Solar_kW"] * TIMESTEP_HOURS
    )
    result["Energia_Autoconsumida_kWh"] = (
        np.minimum(result["Demanda_kW"], result["Generacion_AC_kW"]) * TIMESTEP_HOURS
    )
    result["Energia_Excedente_kWh"] = (
        (result["Generacion_AC_kW"] - result["Demanda_kW"]).clip(lower=0)
        * TIMESTEP_HOURS
    )
    return result


def add_pv_calculations(weather: pd.DataFrame, site: SiteConfig) -> pd.DataFrame:
    """Calculate POA irradiance and clear-sky baseline."""
    result = weather.copy()
    solpos = pvlib.solarposition.get_solarposition(
        result.index,
        latitude=site.latitude,
        longitude=site.longitude,
        altitude=site.altitude_m,
    )

    radiation_columns = [
        "shortwave_radiation",
        "direct_normal_irradiance",
        "diffuse_radiation",
        "direct_radiation",
    ]
    for column in radiation_columns:
        result[column] = result[column].fillna(0).clip(lower=0)

    daylight = solpos["apparent_zenith"] < 90
    ghi = result["shortwave_radiation"].where(daylight, 0)
    dni = result["direct_normal_irradiance"].where(daylight, 0)
    dhi = result["diffuse_radiation"].where(daylight, 0)

    dni_extra = pvlib.irradiance.get_extra_radiation(result.index)
    airmass = pvlib.atmosphere.get_relative_airmass(solpos["apparent_zenith"])

    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=site.surface_tilt,
        surface_azimuth=site.surface_azimuth,
        solar_zenith=solpos["apparent_zenith"],
        solar_azimuth=solpos["azimuth"],
        dni=dni,
        ghi=ghi,
        dhi=dhi,
        dni_extra=dni_extra,
        airmass=airmass,
        albedo=site.albedo,
        model="haydavies",
    )

    location = pvlib.location.Location(
        latitude=site.latitude,
        longitude=site.longitude,
        tz=site.timezone,
        altitude=site.altitude_m,
        name=site.name,
    )
    clearsky = location.get_clearsky(result.index, model="ineichen")
    poa_clearsky = pvlib.irradiance.get_total_irradiance(
        surface_tilt=site.surface_tilt,
        surface_azimuth=site.surface_azimuth,
        solar_zenith=solpos["apparent_zenith"],
        solar_azimuth=solpos["azimuth"],
        dni=clearsky["dni"].where(daylight, 0),
        ghi=clearsky["ghi"].where(daylight, 0),
        dhi=clearsky["dhi"].where(daylight, 0),
        dni_extra=dni_extra,
        airmass=airmass,
        albedo=site.albedo,
        model="haydavies",
    )

    timestep_hours = infer_timestep_hours(result.index)
    result["ghi"] = ghi
    result["dni"] = dni
    result["dhi"] = dhi
    result["solar_zenith"] = solpos["apparent_zenith"]
    result["solar_elevation"] = solpos["apparent_elevation"]
    result["solar_azimuth"] = solpos["azimuth"]
    result["poa_global"] = poa["poa_global"].fillna(0).clip(lower=0)
    result["poa_direct"] = poa["poa_direct"].fillna(0).clip(lower=0)
    result["poa_diffuse"] = poa["poa_diffuse"].fillna(0).clip(lower=0)
    result["ghi_clear_sky"] = clearsky["ghi"].where(daylight, 0).clip(lower=0)
    result["poa_clear_sky"] = poa_clearsky["poa_global"].fillna(0).clip(lower=0)
    result["ghi_energy_kwh_m2"] = result["ghi"] * timestep_hours / 1000
    result["poa_energy_kwh_m2"] = result["poa_global"] * timestep_hours / 1000
    result["poa_clear_sky_energy_kwh_m2"] = (
        result["poa_clear_sky"] * timestep_hours / 1000
    )
    result["clearness_index"] = (
        result["ghi"] / result["ghi_clear_sky"].replace(0, np.nan)
    ).clip(lower=0, upper=1.3)

    return result


def _reporting_month(result: pd.DataFrame) -> pd.Series:
    if "Fecha_Hora_Demanda_Original" in result:
        timestamps = pd.to_datetime(result["Fecha_Hora_Demanda_Original"])
    else:
        timestamps = pd.Series(result.index.tz_localize(None), index=result.index)
    return timestamps.dt.strftime("%Y-%m")


def monthly_summary(result: pd.DataFrame) -> pd.DataFrame:
    """Summarize demand, generation and climate for the technical report."""
    grouped = result.groupby(_reporting_month(result))
    summary = grouped.agg(
        Energia_Demandada_kWh=("Energia_Demanda_kWh", "sum"),
        Energia_Solar_DC_kWh=("Energia_Solar_DC_kWh", "sum"),
        Energia_Solar_AC_kWh=("Energia_Solar_AC_kWh", "sum"),
        Energia_Autoconsumida_kWh=("Energia_Autoconsumida_kWh", "sum"),
        Energia_Red_kWh=("Energia_Red_kWh", "sum"),
        Energia_Excedente_kWh=("Energia_Excedente_kWh", "sum"),
        Demanda_Maxima_Original_kW=("Demanda_kW", "max"),
        Demanda_Maxima_Post_Solar_kW=("Demanda_Post_Inyeccion_Solar_kW", "max"),
        POA_Mensual_kWh_m2=("poa_energy_kwh_m2", "sum"),
        POA_Cielo_Despejado_Mensual_kWh_m2=("poa_clear_sky_energy_kwh_m2", "sum"),
        GHI_Mensual_kWh_m2=("ghi_energy_kwh_m2", "sum"),
        Energia_DC_Modulo_kWh=("Energia_DC_Modulo_kWh", "sum"),
        Temperatura_Media_C=("temperature_2m", "mean"),
        Nubosidad_Media_pct=("cloud_cover", "mean"),
        Precipitacion_mm=("precipitation", "sum"),
    )
    summary.index.name = "Mes"
    summary["Reduccion_Demanda_Maxima_kW"] = (
        summary["Demanda_Maxima_Original_kW"]
        - summary["Demanda_Maxima_Post_Solar_kW"]
    )
    summary["Relacion_POA_Historico_vs_Cielo_Despejado_pct"] = (
        100
        * summary["POA_Mensual_kWh_m2"]
        / summary["POA_Cielo_Despejado_Mensual_kWh_m2"].replace(0, np.nan)
    )
    return summary.round(3)


def annual_summary(
    result: pd.DataFrame,
    module: ModuleConfig,
    scenario: dict[str, float],
) -> pd.DataFrame:
    """Build one annual row per module scenario."""
    hours = len(result) * TIMESTEP_HOURS
    demand_energy = result["Energia_Demanda_kWh"].sum()
    solar_dc_energy = result["Energia_Solar_DC_kWh"].sum()
    solar_ac_energy = result["Energia_Solar_AC_kWh"].sum()
    self_consumed = result["Energia_Autoconsumida_kWh"].sum()
    grid_energy = result["Energia_Red_kWh"].sum()
    surplus = result["Energia_Excedente_kWh"].sum()
    peak_original = result["Demanda_kW"].max()
    peak_post_solar = result["Demanda_Post_Inyeccion_Solar_kW"].max()
    peak_reduction = peak_original - peak_post_solar
    dc_kwp = scenario["system_dc_kwp"]

    row = {
        "Escenario": module.key,
        "Modulo": module.model,
        "Tipo_Modulo": module.module_type,
        "Potencia_DC_Sistema_kWp": dc_kwp,
        "Num_Modulos_Equivalente": scenario["num_modules_equivalent"],
        "Ganancia_Bifacial_Efectiva_pct": 100
        * scenario["effective_bifacial_gain"],
        "Energia_Demandada_Anual_kWh": demand_energy,
        "Energia_Solar_DC_Anual_kWh": solar_dc_energy,
        "Energia_Solar_AC_Anual_kWh": solar_ac_energy,
        "Energia_Autoconsumida_Anual_kWh": self_consumed,
        "Energia_Red_Anual_kWh": grid_energy,
        "Energia_Excedente_Anual_kWh": surplus,
        "Autoconsumo_pct": 100 * self_consumed / solar_ac_energy
        if solar_ac_energy
        else np.nan,
        "Cobertura_Solar_pct": 100 * self_consumed / demand_energy
        if demand_energy
        else np.nan,
        "Demanda_Maxima_Original_kW": peak_original,
        "Demanda_Maxima_Post_Solar_kW": peak_post_solar,
        "Reduccion_Demanda_Maxima_kW": peak_reduction,
        "Reduccion_Demanda_Maxima_pct": 100 * peak_reduction / peak_original
        if peak_original
        else np.nan,
        "Factor_Planta_Demanda_pct": 100 * demand_energy / (peak_original * hours)
        if peak_original and hours
        else np.nan,
        "Capacity_Factor_FV_pct": 100 * solar_ac_energy / (dc_kwp * hours)
        if dc_kwp and hours
        else np.nan,
        "Generacion_Especifica_AC_kWh_kWp": solar_ac_energy / dc_kwp
        if dc_kwp
        else np.nan,
    }
    return pd.DataFrame([row]).round(3)


def module_comparison_summary(annual_summaries: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(annual_summaries, ignore_index=True)


def _representative_week(result: pd.DataFrame) -> pd.DataFrame:
    weekly_energy = result["Energia_Solar_AC_kWh"].resample("7D").sum()
    if weekly_energy.empty:
        return result
    median_energy = weekly_energy.median()
    week_start = (weekly_energy - median_energy).abs().idxmin()
    return result.loc[week_start : week_start + pd.Timedelta(days=7) - pd.Timedelta(minutes=15)]


def save_plots(
    result: pd.DataFrame,
    summary: pd.DataFrame,
    figures_dir: Path,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    daily = result.resample("D").agg(
        poa_energy_kwh_m2=("poa_energy_kwh_m2", "sum"),
        poa_clear_sky_energy_kwh_m2=("poa_clear_sky_energy_kwh_m2", "sum"),
        temperature_mean_c=("temperature_2m", "mean"),
        cloud_cover_mean_pct=("cloud_cover", "mean"),
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    daily[["poa_energy_kwh_m2", "poa_clear_sky_energy_kwh_m2"]].plot(ax=ax)
    ax.set_title("Energia diaria en el plano del panel")
    ax.set_xlabel("Fecha climatica analoga")
    ax.set_ylabel("kWh/m^2 por dia")
    ax.legend(["Historico", "Cielo despejado"])
    fig.tight_layout()
    fig.savefig(figures_dir / "energia_diaria_poa.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        summary.index,
        summary["POA_Mensual_kWh_m2"],
        color="#d97706",
        label="Historico",
    )
    ax.plot(
        summary.index,
        summary["POA_Cielo_Despejado_Mensual_kWh_m2"],
        color="black",
        marker="o",
        label="Cielo despejado",
    )
    ax.set_title("Irradiancia POA mensual historica")
    ax.set_xlabel("Mes")
    ax.set_ylabel("kWh/m^2")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "energia_mensual.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(summary.index, summary["Energia_DC_Modulo_kWh"], color="#0f766e")
    ax.set_title("Energia mensual DC por modulo")
    ax.set_xlabel("Mes")
    ax.set_ylabel("kWh por modulo")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(figures_dir / "energia_mensual_modulos.png", dpi=160)
    plt.close(fig)

    heatmap_source = result.copy()
    heatmap_source["month"] = heatmap_source.index.month
    heatmap_source["clock_time"] = heatmap_source.index.strftime("%H:%M")
    heatmap = heatmap_source.pivot_table(
        index="clock_time", columns="month", values="poa_global", aggfunc="mean"
    )
    heatmap = heatmap.reindex(sorted(heatmap.index))
    heatmap.columns = [calendar.month_abbr[i] for i in heatmap.columns]

    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(
        heatmap,
        ax=ax,
        cmap="inferno",
        cbar_kws={"label": "Irradiancia POA historica (W/m^2)"},
    )
    ax.set_title("Irradiancia promedio por hora y mes")
    ax.set_xlabel("Mes")
    ax.set_ylabel("Hora local")
    fig.tight_layout()
    fig.savefig(figures_dir / "heatmap_poa_historica.png", dpi=160)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(daily.index, daily["temperature_mean_c"], color="#b23a48", label="Temp.")
    ax1.set_ylabel("Temperatura media (C)")
    ax2 = ax1.twinx()
    ax2.plot(
        daily.index,
        daily["cloud_cover_mean_pct"],
        color="#3a6ea5",
        alpha=0.75,
        label="Nubosidad",
    )
    ax2.set_ylabel("Nubosidad media (%)")
    ax1.set_title("Clima diario historico")
    ax1.set_xlabel("Fecha climatica analoga")
    fig.tight_layout()
    fig.savefig(figures_dir / "clima_diario.png", dpi=160)
    plt.close(fig)

    week = _representative_week(result)
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(week.index, week["Demanda_kW"], label="Demanda", color="#1f2937")
    ax.plot(week.index, week["Generacion_AC_kW"], label="Generacion solar AC", color="#d97706")
    ax.set_title("Demanda industrial vs generacion solar AC: semana representativa")
    ax.set_xlabel("Fecha climatica analoga")
    ax.set_ylabel("Potencia (kW)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "demanda_vs_solar.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(week.index, week["Demanda_kW"], label="Demanda original", color="#1f2937")
    ax.plot(
        week.index,
        week["Demanda_Post_Inyeccion_Solar_kW"],
        label="Demanda post-solar",
        color="#2563eb",
    )
    ax.set_title("Demanda original vs demanda post-inyeccion solar")
    ax.set_xlabel("Fecha climatica analoga")
    ax.set_ylabel("Potencia (kW)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "demanda_original_vs_post_solar.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    summary[
        [
            "Energia_Demandada_kWh",
            "Energia_Solar_AC_kWh",
            "Energia_Autoconsumida_kWh",
            "Energia_Red_kWh",
        ]
    ].plot(kind="bar", ax=ax)
    ax.set_title("Energia mensual: demanda, solar, autoconsumo y red")
    ax.set_xlabel("Mes")
    ax.set_ylabel("Energia (kWh)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(figures_dir / "energia_mensual_demanda_solar.png", dpi=160)
    plt.close(fig)

    typical_day = result.groupby(result.index.strftime("%H:%M"))[
        ["Demanda_kW", "Generacion_AC_kW", "Demanda_Post_Inyeccion_Solar_kW"]
    ].mean()
    fig, ax = plt.subplots(figsize=(12, 5))
    typical_day.plot(ax=ax)
    ax.set_title("Perfil promedio diario")
    ax.set_xlabel("Hora local")
    ax.set_ylabel("Potencia media (kW)")
    ax.set_xticks(np.arange(0, len(typical_day), 8))
    ax.set_xticklabels(typical_day.index[::8], rotation=45)
    fig.tight_layout()
    fig.savefig(figures_dir / "perfil_dia_tipico.png", dpi=160)
    plt.close(fig)


def write_outputs(
    result: pd.DataFrame,
    summary: pd.DataFrame,
    annual: pd.DataFrame,
    module: ModuleConfig,
    scenario: dict[str, float],
    site: SiteConfig,
    source_url: str,
    output_dir: Path,
    start_date: date,
    end_date: date,
    bifacial_gain: float,
    inverter_efficiency: float,
    system_losses: float,
    demand_file: Path | None,
    align_demand_by_position: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figuras"

    export = result.rename(
        columns={
            "ghi": "GHI_Wm2",
            "dni": "DNI_Wm2",
            "dhi": "DHI_Wm2",
            "poa_global": "Gtot_POA_Wm2",
            "temperature_2m": "Temperatura_Ambiente_C",
            "wind_speed_10m": "Velocidad_Viento_ms",
        }
    ).copy()
    ordered_columns = [
        "Fecha_Hora_Demanda_Original",
        "Demanda_kW",
        "Factor_Potencia",
        "GHI_Wm2",
        "DNI_Wm2",
        "DHI_Wm2",
        "Gtot_POA_Wm2",
        "Temperatura_Ambiente_C",
        "Velocidad_Viento_ms",
        "Generacion_DC_kW",
        "Generacion_AC_kW",
        "Energia_Solar_DC_kWh",
        "Energia_Solar_AC_kWh",
        "Demanda_Post_Inyeccion_Solar_kW",
        "Energia_Demanda_kWh",
        "Energia_Red_kWh",
        "Energia_Autoconsumida_kWh",
        "Energia_Excedente_kWh",
        "Modulo",
        "Tipo_Modulo",
        "Potencia_DC_Sistema_kWp",
        "Num_Modulos_Equivalente",
        "Ganancia_Bifacial_Efectiva",
        "POA_Efectiva_Wm2",
        "Temperatura_Celda_C",
        "Generacion_DC_Modulo_W",
        "Energia_DC_Modulo_kWh",
        "relative_humidity_2m",
        "cloud_cover",
        "precipitation",
        "ghi_clear_sky",
        "poa_clear_sky",
        "clearness_index",
    ]
    export = export[[column for column in ordered_columns if column in export.columns]]
    export.index = export.index.tz_localize(None)
    export.index.name = "Fecha_Hora"
    export.to_csv(output_dir / "clima_solar_demanda_15min.csv")
    summary.to_csv(output_dir / "resumen_mensual.csv")
    annual.to_csv(output_dir / "resumen_anual.csv", index=False)
    save_plots(result, summary, figures_dir)

    metadata = {
        "generated_at": datetime.now(ZoneInfo(site.timezone)).isoformat(),
        "weather_date_range": {
            "weather_start_date": start_date.isoformat(),
            "weather_end_date": end_date.isoformat(),
        },
        "site": asdict(site),
        "module": asdict(module),
        "system": {
            **scenario,
            "requested_bifacial_gain": bifacial_gain,
            "inverter_efficiency": inverter_efficiency,
            "system_losses": system_losses,
        },
        "demand": {
            "file": str(demand_file.resolve()) if demand_file else None,
            "align_by_position": align_demand_by_position,
            "intervals": len(result),
            "frequency": FINAL_FREQUENCY,
        },
        "source": {
            "name": "Open-Meteo Historical Weather API",
            "url": source_url,
        },
        "notes": [
            "GHI, DNI y DHI historicos horarios se remuestrean a 15 minutos.",
            "El remuestreo aplica indices de claridad horarios a curvas de cielo despejado de 15 minutos.",
            "POA se calcula con pvlib.irradiance.get_total_irradiance.",
            "La generacion DC del modulo y sistema se calcula con pvlib.pvsystem.pvwatts_dc.",
            "La temperatura de celda se estima con el modelo Faiman.",
            "La demanda industrial se descuenta usando generacion AC despues de inversor y perdidas.",
            "La bifacialidad escala irradiancia POA, no la potencia nominal STC.",
        ],
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8"
    )


def selected_modules(selection: str) -> list[ModuleConfig]:
    if selection == "all":
        return list(MODULE_OPTIONS.values())
    return [MODULE_OPTIONS[selection]]


def parse_args() -> argparse.Namespace:
    default_start, default_end = default_date_range(SiteConfig.timezone)
    parser = argparse.ArgumentParser(
        description="Motor solar-industrial con clima historico y demanda a 15 minutos."
    )
    parser.add_argument("--name", default=SiteConfig.name, help="Nombre del sitio.")
    parser.add_argument("--lat", type=float, default=SiteConfig.latitude, help="Latitud.")
    parser.add_argument(
        "--lon",
        type=float,
        default=SiteConfig.longitude,
        help="Longitud. Para Veracruz debe ser negativa.",
    )
    parser.add_argument("--tz", default=SiteConfig.timezone, help="Zona horaria IANA.")
    parser.add_argument(
        "--altitude",
        type=float,
        default=SiteConfig.altitude_m,
        help="Altitud en metros.",
    )
    parser.add_argument(
        "--tilt",
        type=float,
        default=SiteConfig.surface_tilt,
        help="Inclinacion del panel en grados. 0 es horizontal.",
    )
    parser.add_argument(
        "--azimuth",
        type=float,
        default=SiteConfig.surface_azimuth,
        help="Azimut pvlib del panel. 180 es sur, 90 este, 270 oeste.",
    )
    parser.add_argument(
        "--albedo",
        type=float,
        default=SiteConfig.albedo,
        help="Reflectancia del suelo.",
    )
    demand_group = parser.add_mutually_exclusive_group()
    demand_group.add_argument(
        "--demand-file",
        type=Path,
        help="Curva industrial .csv, .xlsx o .xls.",
    )
    demand_group.add_argument(
        "--generate-demand",
        action="store_true",
        help="Genera un perfil industrial sintetico explicito para pruebas.",
    )
    parser.add_argument(
        "--align-demand-by-position",
        action="store_true",
        help="Alinea demanda y clima por posicion aunque pertenezcan a anos distintos.",
    )
    parser.add_argument(
        "--module",
        choices=["all", *MODULE_OPTIONS.keys()],
        default="all",
        help="Modulo a evaluar. Default: all.",
    )
    parser.add_argument(
        "--bifacial-gain",
        type=float,
        default=MODULE_OPTIONS["bifacial"].default_bifacial_gain,
        help="Ganancia bifacial como fraccion. Ejemplo: 0.15 para 15%%.",
    )
    parser.add_argument(
        "--system-dc-kwp",
        type=float,
        default=399.3,
        help="Potencia DC total para comparacion justa entre escenarios.",
    )
    parser.add_argument(
        "--num-modules",
        type=int,
        help="Numero fisico de modulos. Si se define, recalcula la potencia DC total.",
    )
    parser.add_argument(
        "--inverter-efficiency",
        type=float,
        default=0.96,
        help="Eficiencia del inversor como fraccion. Default: 0.96.",
    )
    parser.add_argument(
        "--system-losses",
        type=float,
        default=0.0,
        help="Perdidas adicionales del sistema como fraccion. Default: 0.",
    )
    parser.add_argument(
        "--weather-start-date",
        "--start-date",
        dest="weather_start_date",
        type=parse_iso_date,
        default=parse_iso_date(default_start),
        help=f"Fecha climatica inicial YYYY-MM-DD. Default: {default_start}.",
    )
    parser.add_argument(
        "--weather-end-date",
        "--end-date",
        dest="weather_end_date",
        type=parse_iso_date,
        default=parse_iso_date(default_end),
        help=f"Fecha climatica final YYYY-MM-DD. Default: {default_end}.",
    )
    parser.add_argument(
        "--freq",
        default=FINAL_FREQUENCY,
        help="Frecuencia solicitada. El motor final remuestrea a 15min.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Carpeta de salida.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bifacial_gain < 0:
        raise ValueError("--bifacial-gain no puede ser negativo.")
    if not 0 < args.inverter_efficiency <= 1:
        raise ValueError("--inverter-efficiency debe estar entre 0 y 1.")
    if not 0 <= args.system_losses < 1:
        raise ValueError("--system-losses debe estar entre 0 y 1.")
    if not args.demand_file and not args.generate_demand:
        raise ValueError(
            "Indica --demand-file o activa explicitamente --generate-demand."
        )

    site = SiteConfig(
        name=args.name,
        latitude=args.lat,
        longitude=args.lon,
        timezone=args.tz,
        altitude_m=args.altitude,
        surface_tilt=args.tilt,
        surface_azimuth=args.azimuth,
        albedo=args.albedo,
    )
    modules = selected_modules(args.module)

    target_index = build_target_index(
        args.weather_start_date,
        args.weather_end_date,
        site.timezone,
        freq=args.freq,
    )
    hourly_weather, source_url = fetch_historical_weather(
        site, args.weather_start_date, args.weather_end_date
    )
    weather = resample_weather_to_15min(hourly_weather, target_index, site)
    solar_resource = add_pv_calculations(weather, site)

    if args.demand_file:
        demand = load_demand(
            args.demand_file,
            target_index,
            timezone=site.timezone,
            align_by_position=args.align_demand_by_position,
        )
    else:
        demand = generate_demand(target_index)

    annual_summaries = []
    for module in modules:
        result, scenario = add_module_calculations(
            solar_resource,
            module=module,
            bifacial_gain=args.bifacial_gain,
            system_dc_kwp=args.system_dc_kwp,
            num_modules=args.num_modules,
            inverter_efficiency=args.inverter_efficiency,
            system_losses=args.system_losses,
        )
        result = add_demand_balance(result, demand)
        summary = monthly_summary(result)
        annual = annual_summary(result, module, scenario)
        annual_summaries.append(annual)

        scenario_output_dir = (
            args.output_dir / module.key if len(modules) > 1 else args.output_dir
        )
        write_outputs(
            result=result,
            summary=summary,
            annual=annual,
            module=module,
            scenario=scenario,
            site=site,
            source_url=source_url,
            output_dir=scenario_output_dir,
            start_date=args.weather_start_date,
            end_date=args.weather_end_date,
            bifacial_gain=args.bifacial_gain,
            inverter_efficiency=args.inverter_efficiency,
            system_losses=args.system_losses,
            demand_file=args.demand_file,
            align_demand_by_position=args.align_demand_by_position,
        )

    module_comparison = module_comparison_summary(annual_summaries)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    module_comparison.to_csv(args.output_dir / "comparacion_modulos.csv", index=False)

    total_energy = solar_resource["poa_energy_kwh_m2"].sum()
    clear_sky_energy = solar_resource["poa_clear_sky_energy_kwh_m2"].sum()
    ratio = 100 * total_energy / clear_sky_energy if clear_sky_energy else np.nan

    print("Analisis terminado")
    print(f"Sitio: {site.name} ({site.latitude}, {site.longitude})")
    print(
        f"Periodo climatico: {args.weather_start_date} a {args.weather_end_date} "
        f"({len(target_index):,} intervalos)"
    )
    print(f"Inclinacion / azimut: {site.surface_tilt} / {site.surface_azimuth}")
    print(f"Energia POA historica: {total_energy:.1f} kWh/m^2")
    print(f"Equivalente vs cielo despejado: {ratio:.1f}%")
    for _, row in module_comparison.iterrows():
        print(
            f"{row['Escenario']}: {row['Energia_Solar_AC_Anual_kWh']:.1f} kWh AC, "
            f"autoconsumo={row['Autoconsumo_pct']:.1f}%, "
            f"cobertura={row['Cobertura_Solar_pct']:.1f}%"
        )
    print(f"Salidas: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
