from __future__ import annotations

import json
import urllib.parse
import urllib.request
import warnings
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pvlib

from demanda import (
    FINAL_FREQUENCY,
    TIMESTEP_HOURS,
    load_demand,
    normalize_demand_dataframe,
)


OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

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
    name: str = "Consolapan / Xalapa, Veracruz"
    latitude: float = 19.54
    longitude: float = -96.91
    timezone: str = "America/Mexico_City"
    altitude_m: float = 1400.0
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


LOCATIONS_MEXICO = {
    "Consolapan / Xalapa, Veracruz": {
        "name": "Consolapan / Xalapa, Veracruz",
        "lat": 19.54,
        "lon": -96.91,
        "altitude": 1400.0,
        "timezone": "America/Mexico_City",
    },
    "Coatepec, Veracruz": {
        "name": "Coatepec, Veracruz",
        "lat": 19.4521,
        "lon": -96.9615,
        "altitude": 1200.0,
        "timezone": "America/Mexico_City",
    },
    "Veracruz, Veracruz": {
        "name": "Veracruz, Veracruz",
        "lat": 19.1738,
        "lon": -96.1342,
        "altitude": 10.0,
        "timezone": "America/Mexico_City",
    },
    "Monterrey, Nuevo Leon": {
        "name": "Monterrey, Nuevo Leon",
        "lat": 25.6866,
        "lon": -100.3161,
        "altitude": 538.0,
        "timezone": "America/Monterrey",
    },
    "Ciudad de Mexico": {
        "name": "Ciudad de Mexico",
        "lat": 19.4326,
        "lon": -99.1332,
        "altitude": 2240.0,
        "timezone": "America/Mexico_City",
    },
}


def default_weather_range() -> tuple[date, date]:
    return date(2024, 12, 21), date(2025, 12, 20)


def validate_date_range(start_date: date, end_date: date, timezone: str) -> None:
    if end_date < start_date:
        raise ValueError("La fecha final no puede ser anterior a la fecha inicial.")
    if (end_date - start_date).days > 370:
        raise ValueError("Usa un periodo maximo de 370 dias por corrida.")

    today = datetime.now(ZoneInfo(timezone)).date()
    if end_date >= today:
        raise ValueError(
            "Open-Meteo Archive solo acepta clima historico. "
            f"Usa una fecha final anterior a {today.isoformat()}."
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
    site: SiteConfig,
    start_date: date,
    end_date: date,
) -> tuple[pd.DataFrame, str]:
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
    requested_offset = pd.tseries.frequencies.to_offset(freq)
    final_offset = pd.tseries.frequencies.to_offset(FINAL_FREQUENCY)
    if requested_offset != final_offset:
        warnings.warn(
            f"La frecuencia solicitada {freq!r} se remuestreara a {FINAL_FREQUENCY!r}.",
            stacklevel=2,
        )

    start = pd.Timestamp(start_date, tz=timezone)
    end = pd.Timestamp(end_date + timedelta(days=1), tz=timezone) - final_offset
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
            result[column] = fallback
        else:
            result[column] = _interpolate_to_target(
                weather[column],
                target_index,
                fallback=fallback,
            )

    if "precipitation" in weather.columns:
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


def infer_timestep_hours(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return TIMESTEP_HOURS
    seconds = index.to_series().diff().dt.total_seconds().dropna().median()
    return float(seconds / 3600) if pd.notna(seconds) and seconds > 0 else TIMESTEP_HOURS


def module_gain(module: ModuleConfig, bifacial_gain: float) -> float:
    if module.bifacial_factor is None:
        return 0.0
    return bifacial_gain * module.bifacial_factor


def resolve_system_capacity(
    module: ModuleConfig,
    system_dc_kwp: float,
    num_modules: int | None,
) -> tuple[float, float]:
    if num_modules is not None:
        if num_modules <= 0:
            raise ValueError("num_modules debe ser mayor que cero.")
        resolved_dc_kwp = num_modules * module.pdc0_w / 1000
        return resolved_dc_kwp, float(num_modules)
    if system_dc_kwp <= 0:
        raise ValueError("system_dc_kwp debe ser mayor que cero.")
    return system_dc_kwp, system_dc_kwp * 1000 / module.pdc0_w


def add_pv_calculations(weather: pd.DataFrame, site: SiteConfig) -> pd.DataFrame:
    result = weather.copy()
    solpos = pvlib.solarposition.get_solarposition(
        result.index,
        latitude=site.latitude,
        longitude=site.longitude,
        altitude=site.altitude_m,
    )

    for column in [
        "shortwave_radiation",
        "direct_normal_irradiance",
        "diffuse_radiation",
        "direct_radiation",
    ]:
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
    result["poa_clear_sky_energy_kwh_m2"] = result["poa_clear_sky"] * timestep_hours / 1000
    result["clearness_index"] = (
        result["ghi"] / result["ghi_clear_sky"].replace(0, np.nan)
    ).clip(lower=0, upper=1.3)
    return result


def add_module_calculations(
    result: pd.DataFrame,
    module: ModuleConfig,
    bifacial_gain: float,
    system_dc_kwp: float,
    num_modules: int | None,
    inverter_efficiency: float,
    system_losses: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    result = result.copy()
    timestep_hours = infer_timestep_hours(result.index)
    if not np.isclose(timestep_hours, TIMESTEP_HOURS):
        raise ValueError("El calculo FV requiere intervalos de 15 minutos.")

    temp_air = result.get("temperature_2m", pd.Series(25.0, index=result.index))
    temp_air = pd.to_numeric(temp_air, errors="coerce").interpolate(limit_direction="both").fillna(25.0)
    wind_speed = result.get("wind_speed_10m", pd.Series(1.0, index=result.index))
    wind_speed = pd.to_numeric(wind_speed, errors="coerce").interpolate(limit_direction="both").fillna(1.0).clip(lower=0.0)

    resolved_dc_kwp, equivalent_modules = resolve_system_capacity(
        module,
        system_dc_kwp,
        num_modules,
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
    system_ac_power_kw = system_dc_power_kw * (1 - system_losses) * inverter_efficiency
    system_ac_power_ref_kw = (
        pvlib.pvsystem.pvwatts_dc(
            effective_irradiance=effective_poa,
            temp_cell=25.0,
            pdc0=resolved_dc_kwp * 1000,
            gamma_pdc=module.gamma_pdc_per_c,
        )
        .fillna(0)
        .clip(lower=0)
        / 1000
        * (1 - system_losses)
        * inverter_efficiency
    )
    thermal_penalty_kw = system_ac_power_ref_kw - system_ac_power_kw

    result["Ganancia_Bifacial_Efectiva"] = effective_gain
    result["POA_Efectiva_Wm2"] = effective_poa.fillna(0).clip(lower=0)
    result["Temperatura_Celda_C"] = cell_temperature
    result["Generacion_DC_Modulo_W"] = module_dc_power_w
    result["Energia_DC_Modulo_kWh"] = module_dc_power_w * timestep_hours / 1000
    result["Generacion_DC_kW"] = system_dc_power_kw
    result["Generacion_AC_kW"] = system_ac_power_kw
    result["Energia_Solar_DC_kWh"] = system_dc_power_kw * timestep_hours
    result["Energia_Solar_AC_kWh"] = system_ac_power_kw * timestep_hours
    result["Generacion_AC_Sin_Temp_kW"] = system_ac_power_ref_kw
    result["Energia_Solar_AC_Sin_Temp_kWh"] = system_ac_power_ref_kw * timestep_hours
    result["Generacion_Solar_kW"] = result["Generacion_AC_kW"]
    result["Generacion_Solar_Sin_Temp_kW"] = result["Generacion_AC_Sin_Temp_kW"]
    result["Penalizacion_Temperatura_kW"] = thermal_penalty_kw
    result["Perdida_Temperatura_kWh"] = thermal_penalty_kw * timestep_hours
    result["Penalizacion_Temperatura_pct_inst"] = np.where(
        result["Generacion_AC_Sin_Temp_kW"] > 0,
        result["Penalizacion_Temperatura_kW"] / result["Generacion_AC_Sin_Temp_kW"] * 100,
        0.0,
    )
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
    result = result.copy()
    demand_aligned = demand.reindex(result.index)
    if demand_aligned["Demanda_kW"].isna().any():
        raise ValueError("La demanda alineada contiene intervalos vacios.")

    timestep_hours = infer_timestep_hours(result.index)
    result["Demanda_kW"] = demand_aligned["Demanda_kW"]
    result["Factor_Potencia"] = demand_aligned["Factor_Potencia"]
    if "Fecha_Hora_Demanda_Original" in demand_aligned.columns:
        result["Fecha_Hora_Demanda_Original"] = demand_aligned["Fecha_Hora_Demanda_Original"]

    result["Demanda_Post_Inyeccion_Solar_kW"] = (
        result["Demanda_kW"] - result["Generacion_AC_kW"]
    ).clip(lower=0)
    result["Energia_Demanda_kWh"] = result["Demanda_kW"] * timestep_hours
    result["Energia_Red_kWh"] = result["Demanda_Post_Inyeccion_Solar_kW"] * timestep_hours
    result["Energia_Autoconsumida_kWh"] = (
        np.minimum(result["Demanda_kW"], result["Generacion_AC_kW"]).clip(lower=0) * timestep_hours
    )
    result["Energia_Excedente_kWh"] = (
        result["Generacion_AC_kW"] - result["Demanda_kW"]
    ).clip(lower=0) * timestep_hours
    return result


def _reporting_month(result: pd.DataFrame) -> pd.Series:
    if "Fecha_Hora_Demanda_Original" in result:
        timestamps = pd.to_datetime(result["Fecha_Hora_Demanda_Original"])
    else:
        timestamps = pd.Series(result.index.tz_localize(None), index=result.index)
    return timestamps.dt.strftime("%Y-%m")


def monthly_summary(result: pd.DataFrame) -> pd.DataFrame:
    grouped = result.groupby(_reporting_month(result))
    summary = grouped.agg(
        Energia_Demandada_kWh=("Energia_Demanda_kWh", "sum"),
        Energia_Solar_DC_kWh=("Energia_Solar_DC_kWh", "sum"),
        Energia_Solar_AC_kWh=("Energia_Solar_AC_kWh", "sum"),
        Energia_Solar_AC_Sin_Temp_kWh=("Energia_Solar_AC_Sin_Temp_kWh", "sum"),
        Perdida_Temperatura_kWh=("Perdida_Temperatura_kWh", "sum"),
        Energia_Autoconsumida_kWh=("Energia_Autoconsumida_kWh", "sum"),
        Energia_Red_kWh=("Energia_Red_kWh", "sum"),
        Energia_Excedente_kWh=("Energia_Excedente_kWh", "sum"),
        Demanda_Maxima_Original_kW=("Demanda_kW", "max"),
        Demanda_Maxima_Post_Solar_kW=("Demanda_Post_Inyeccion_Solar_kW", "max"),
        POA_Mensual_kWh_m2=("poa_energy_kwh_m2", "sum"),
        POA_Cielo_Despejado_Mensual_kWh_m2=("poa_clear_sky_energy_kwh_m2", "sum"),
        GHI_Mensual_kWh_m2=("ghi_energy_kwh_m2", "sum"),
        Temperatura_Media_C=("temperature_2m", "mean"),
        Temperatura_Celda_Prom_C=("Temperatura_Celda_C", "mean"),
        Temperatura_Celda_Max_C=("Temperatura_Celda_C", "max"),
        Nubosidad_Media_pct=("cloud_cover", "mean"),
        Precipitacion_mm=("precipitation", "sum"),
    )
    summary.index.name = "Mes"
    summary["Reduccion_Demanda_Maxima_kW"] = (
        summary["Demanda_Maxima_Original_kW"] - summary["Demanda_Maxima_Post_Solar_kW"]
    )
    summary["Relacion_POA_Historico_vs_Cielo_Despejado_pct"] = (
        100
        * summary["POA_Mensual_kWh_m2"]
        / summary["POA_Cielo_Despejado_Mensual_kWh_m2"].replace(0, np.nan)
    )
    summary["Penalizacion_Temperatura_pct"] = (
        100
        * summary["Perdida_Temperatura_kWh"]
        / summary["Energia_Solar_AC_Sin_Temp_kWh"].replace(0, np.nan)
    )
    return summary.round(3)


def annual_summary(
    result: pd.DataFrame,
    module: ModuleConfig,
    scenario: dict[str, float],
) -> pd.DataFrame:
    hours = len(result) * TIMESTEP_HOURS
    demand_energy = result["Energia_Demanda_kWh"].sum()
    solar_ac_energy = result["Energia_Solar_AC_kWh"].sum()
    solar_ac_no_temp_energy = result["Energia_Solar_AC_Sin_Temp_kWh"].sum()
    thermal_penalty_energy = result["Perdida_Temperatura_kWh"].sum()
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
        "Ganancia_Bifacial_Efectiva_pct": 100 * scenario["effective_bifacial_gain"],
        "Energia_Demandada_Anual_kWh": demand_energy,
        "Energia_Solar_AC_Anual_kWh": solar_ac_energy,
        "Energia_Solar_AC_Sin_Temp_Anual_kWh": solar_ac_no_temp_energy,
        "Perdida_Temperatura_Anual_kWh": thermal_penalty_energy,
        "Penalizacion_Temperatura_Anual_pct": (
            100 * thermal_penalty_energy / solar_ac_no_temp_energy
            if solar_ac_no_temp_energy
            else np.nan
        ),
        "Energia_Autoconsumida_Anual_kWh": self_consumed,
        "Energia_Red_Anual_kWh": grid_energy,
        "Energia_Excedente_Anual_kWh": surplus,
        "Autoconsumo_pct": 100 * self_consumed / solar_ac_energy if solar_ac_energy else np.nan,
        "Cobertura_Solar_pct": 100 * self_consumed / demand_energy if demand_energy else np.nan,
        "Demanda_Maxima_Original_kW": peak_original,
        "Demanda_Maxima_Post_Solar_kW": peak_post_solar,
        "Reduccion_Demanda_Maxima_kW": peak_reduction,
        "Reduccion_Demanda_Maxima_pct": (
            100 * peak_reduction / peak_original if peak_original else np.nan
        ),
        "Capacity_Factor_FV_pct": 100 * solar_ac_energy / (dc_kwp * hours) if dc_kwp and hours else np.nan,
        "Generacion_Especifica_AC_kWh_kWp": solar_ac_energy / dc_kwp if dc_kwp else np.nan,
    }
    return pd.DataFrame([row]).round(3)


def module_comparison_summary(annual_summaries: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(annual_summaries, ignore_index=True)


def selected_modules(selection: str) -> list[ModuleConfig]:
    if selection == "all":
        return list(MODULE_OPTIONS.values())
    if selection not in MODULE_OPTIONS:
        raise ValueError(f"Modulo no reconocido: {selection}")
    return [MODULE_OPTIONS[selection]]


def _prepare_export_df(result: pd.DataFrame) -> pd.DataFrame:
    export = result.copy()
    export.insert(0, "Fecha_Hora", export.index.tz_convert(export.index.tz).tz_localize(None))
    return export.reset_index(drop=True)


def run_solar_demand_simulation(
    site: SiteConfig,
    demand_df: pd.DataFrame | None = None,
    demand_file: str | Path | None = None,
    weather_start_date: date | None = None,
    weather_end_date: date | None = None,
    module_selection: str = "all",
    bifacial_gain: float = 0.15,
    system_dc_kwp: float = 399.3,
    num_modules: int | None = None,
    inverter_efficiency: float = 0.96,
    system_losses: float = 0.0,
    align_demand_by_position: bool = True,
) -> dict[str, object]:
    if demand_df is not None and demand_file is not None:
        raise ValueError("Usa demand_df o demand_file, no ambos al mismo tiempo.")
    if demand_df is None and demand_file is None:
        raise ValueError("La simulacion requiere demand_df o demand_file.")
    if bifacial_gain < 0:
        raise ValueError("bifacial_gain no puede ser negativo.")
    if not 0 < inverter_efficiency <= 1:
        raise ValueError("inverter_efficiency debe estar entre 0 y 1.")
    if not 0 <= system_losses < 1:
        raise ValueError("system_losses debe estar entre 0 y 1.")

    if weather_start_date is None or weather_end_date is None:
        default_start, default_end = default_weather_range()
        weather_start_date = weather_start_date or default_start
        weather_end_date = weather_end_date or default_end

    target_index = build_target_index(weather_start_date, weather_end_date, site.timezone)
    hourly_weather, source_url = fetch_historical_weather(site, weather_start_date, weather_end_date)
    weather = resample_weather_to_15min(hourly_weather, target_index, site)
    solar_resource = add_pv_calculations(weather, site)

    if demand_df is not None:
        demand = normalize_demand_dataframe(
            demand_df,
            target_index=target_index,
            timezone=site.timezone,
            align_by_position=align_demand_by_position,
        )
    else:
        demand = load_demand(
            demand_file,
            target_index=target_index,
            timezone=site.timezone,
            align_by_position=align_demand_by_position,
        )

    scenarios = {}
    annual_summaries = []
    for module in selected_modules(module_selection):
        result, scenario = add_module_calculations(
            solar_resource,
            module=module,
            bifacial_gain=bifacial_gain,
            system_dc_kwp=system_dc_kwp,
            num_modules=num_modules,
            inverter_efficiency=inverter_efficiency,
            system_losses=system_losses,
        )
        result = add_demand_balance(result, demand)
        summary = monthly_summary(result)
        annual = annual_summary(result, module, scenario)
        annual_summaries.append(annual)
        scenarios[module.key] = {
            "result": result,
            "export_result": _prepare_export_df(result),
            "summary": summary,
            "annual": annual,
            "module": module,
            "scenario": scenario,
        }

    module_comparison = module_comparison_summary(annual_summaries)
    primary_key = "bifacial" if "bifacial" in scenarios else next(iter(scenarios))

    return {
        "scenarios": scenarios,
        "primary_key": primary_key,
        "primary": scenarios[primary_key],
        "module_comparison": module_comparison,
        "site": site,
        "site_dict": asdict(site),
        "source_url": source_url,
        "weather_start_date": weather_start_date,
        "weather_end_date": weather_end_date,
    }
