# =============================================================================
# motor_calculo_mono_bi.py — Motor de cálculo fotovoltaico mono/bifacial
# =============================================================================

import requests
import pandas as pd
import numpy as np
import pvlib
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS


# =============================================================================
# CATÁLOGO DE PANELES
# =============================================================================

# =============================================================================
# CATÁLOGO DE PANELES
# =============================================================================

PANELES = {
    "monofacial": {
        "nombre": "Monofacial",
        "fabricante": "Jinko Solar",
        "modelo": "Tiger Neo 72HC — JKM605N-72HL4",
        "potencia_w": 605,
        "gamma_pdc": -0.0029,
        "bifacial": False,
        "area_m2": 2.58,  # Dimensiones aprox: 2.278m x 1.134m
    },
    "bifacial_jinko": {
        "nombre": "Bifacial",
        "fabricante": "Jinko Solar",
        "modelo": "Tiger Neo N-type — JKM625N-78HL4-BDV",
        "potencia_w": 625,
        "gamma_pdc": -0.0029,
        "bifacial": True,
        "gb_default": 0.15,
        "area_m2": 2.79,  # Dimensiones aprox: 2.465m x 1.134m
    },
}

# =============================================================================
# FUNCIONES DE CLIMA HISTÓRICO
# =============================================================================

def descargar_clima_open_meteo(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    timezone: str = "America/Mexico_City",
) -> pd.DataFrame:
    """
    Descarga clima histórico horario desde Open-Meteo Historical Weather API.

    Para este MVP solo se usan temperatura ambiente y velocidad de viento.
    La irradiancia POA se conserva como la calcula el motor con cielo despejado.
    """
    endpoint = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": float(lat),
        "longitude": float(lon),
        "start_date": str(start_date),
        "end_date": str(end_date),
        "hourly": "temperature_2m,wind_speed_10m",
        "timezone": timezone,
        "wind_speed_unit": "ms",
        "temperature_unit": "celsius",
        # Preparado para futuro, sin usarse todavía en el MVP:
        # "hourly": "temperature_2m,wind_speed_10m,shortwave_radiation,direct_normal_irradiance,diffuse_radiation",
    }

    try:
        response = requests.get(endpoint, params=params, timeout=45)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"No se pudo descargar clima histórico desde Open-Meteo: {exc}") from exc

    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError("Open-Meteo respondió, pero el contenido no es JSON válido.") from exc

    if data.get("error"):
        reason = data.get("reason", "Error no especificado por Open-Meteo.")
        raise RuntimeError(f"Open-Meteo rechazó la consulta: {reason}")

    hourly = data.get("hourly")
    if not isinstance(hourly, dict):
        raise RuntimeError("Respuesta inválida de Open-Meteo: falta la llave 'hourly'.")

    expected_keys = ["time", "temperature_2m", "wind_speed_10m"]
    missing = [key for key in expected_keys if key not in hourly]
    if missing:
        raise RuntimeError(f"Respuesta inválida de Open-Meteo: faltan columnas {missing}.")

    lengths = {key: len(hourly[key]) for key in expected_keys}
    if len(set(lengths.values())) != 1:
        raise RuntimeError(f"Respuesta inválida de Open-Meteo: longitudes inconsistentes {lengths}.")

    df_clima = pd.DataFrame(
        {
            "Fecha_Hora_Clima": pd.to_datetime(hourly["time"], errors="coerce"),
            "Temperatura_Ambiente_C": pd.to_numeric(hourly["temperature_2m"], errors="coerce"),
            "Velocidad_Viento_ms": pd.to_numeric(hourly["wind_speed_10m"], errors="coerce"),
        }
    )

    if df_clima["Fecha_Hora_Clima"].isna().any():
        raise RuntimeError("Respuesta inválida de Open-Meteo: hay timestamps no interpretables.")

    df_clima = df_clima.set_index("Fecha_Hora_Clima").sort_index()
    df_clima = df_clima[~df_clima.index.duplicated(keep="first")]

    if df_clima.empty:
        raise RuntimeError("Open-Meteo no devolvió datos horarios para el rango solicitado.")

    if df_clima[["Temperatura_Ambiente_C", "Velocidad_Viento_ms"]].isna().all().any():
        raise RuntimeError("Open-Meteo devolvió una variable climática completamente vacía.")

    return df_clima


def preparar_clima_15min(
    df_clima_horario: pd.DataFrame,
    tiempos_naive: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Convierte clima horario a 15 minutos y lo alinea con el índice del motor.

    Si el año climático histórico no coincide con el año simulado, se mapea por
    posición temporal. Esto permite usar un año climático análogo, por ejemplo
    clima 2025 para una simulación con índice 2026.
    """
    required_cols = ["Temperatura_Ambiente_C", "Velocidad_Viento_ms"]
    missing = [col for col in required_cols if col not in df_clima_horario.columns]
    if missing:
        raise ValueError(f"df_clima_horario no contiene las columnas requeridas: {missing}")

    clima = df_clima_horario[required_cols].copy()
    clima.index = pd.to_datetime(clima.index)
    if getattr(clima.index, "tz", None) is not None:
        clima.index = clima.index.tz_localize(None)
    clima = clima.sort_index()
    clima = clima[~clima.index.duplicated(keep="first")]

    tiempos_objetivo = pd.DatetimeIndex(pd.to_datetime(tiempos_naive))
    if getattr(tiempos_objetivo, "tz", None) is not None:
        tiempos_objetivo = tiempos_objetivo.tz_localize(None)

    # Open-Meteo entrega datos horarios. Se extiende 45 min para cubrir el último
    # día completo en resolución de 15 min (23:00, 23:15, 23:30, 23:45).
    indice_15 = pd.date_range(
        start=clima.index.min(),
        end=clima.index.max() + pd.Timedelta(minutes=45),
        freq="15min",
    )

    clima_15 = clima.reindex(indice_15)
    clima_15 = clima_15.interpolate(method="time").ffill().bfill()

    alineado = clima_15.reindex(tiempos_objetivo)
    missing_ratio = float(alineado.isna().mean().max()) if not alineado.empty else 1.0

    if missing_ratio > 0.25:
        if len(clima_15) != len(tiempos_objetivo):
            raise ValueError(
                "No se pudo alinear el clima histórico con la simulación. "
                f"Clima 15min: {len(clima_15):,} intervalos; "
                f"simulación: {len(tiempos_objetivo):,} intervalos. "
                "Usa un rango climático con la misma cantidad de días que el año simulado."
            )
        # Año climático análogo: se preserva la forma temporal y se reasigna al año simulado.
        alineado = clima_15.copy()
        alineado.index = tiempos_objetivo
    else:
        alineado = alineado.interpolate(method="time").ffill().bfill()

    alineado = alineado[required_cols]
    alineado.index = tiempos_objetivo
    return alineado


# =============================================================================
# FUNCIONES DE SOPORTE MATEMÁTICO Y PERFILADO
# =============================================================================

def calcular_tilt_optimo(lat: float, lon: float, altura: float) -> float:
    """
    Barrido heurístico rápido para encontrar el tilt que maximiza la irradiancia
    anual de cielo despejado, asumiendo acimut al sur.
    """
    tiempos = pd.date_range("2026-01-01", "2026-12-31 23:00", freq="1h", tz="America/Mexico_City")

    sol = pvlib.solarposition.get_solarposition(tiempos, lat, lon)
    airmass = pvlib.atmosphere.get_relative_airmass(sol["apparent_zenith"])
    airmass_abs = pvlib.atmosphere.get_absolute_airmass(airmass, pvlib.atmosphere.alt2pres(altura))
    cs = pvlib.clearsky.ineichen(
        sol["apparent_zenith"],
        airmass_absolute=airmass_abs,
        linke_turbidity=3,
        altitude=altura,
    )

    best_tilt = lat
    max_poa = 0.0
    rango_tilt = np.arange(max(0, lat - 15), lat + 15, 1.0)

    for t in rango_tilt:
        poa = pvlib.irradiance.get_total_irradiance(
            surface_tilt=t,
            surface_azimuth=180,  # 180° = Sur en pvlib.
            solar_zenith=sol["apparent_zenith"],
            solar_azimuth=sol["azimuth"],
            dni=cs["dni"],
            ghi=cs["ghi"],
            dhi=cs["dhi"],
        )
        total = poa["poa_global"].sum()
        if total > max_poa:
            max_poa = total
            best_tilt = t

    return float(best_tilt)

def calcular_dia_tipico_horizontal(lat: float, lon: float, altura: float, fecha: str,
                                    tilt: float = 0.0, acimut_usuario: float = 0.0,
                                    panel_key: str = "monofacial"):
    """
    Irradiancia de cielo despejado (GHI, DNI, DHI) para un día específico y la
    irradiancia POA + producción instantánea de un panel de referencia con la
    orientación (tilt, acimut) indicada, evaluada a T_celda = 25 °C (sin corrección
    térmica, sin dependencia de clima histórico).
    """
    tz = "America/Mexico_City"
    tiempos = pd.date_range(start=f"{fecha} 00:00", end=f"{fecha} 23:45", freq="15min", tz=tz)

    sol = pvlib.solarposition.get_solarposition(tiempos, lat, lon)
    airmass = pvlib.atmosphere.get_relative_airmass(sol["apparent_zenith"])
    airmass_abs = pvlib.atmosphere.get_absolute_airmass(airmass, pvlib.atmosphere.alt2pres(altura))
    cs = pvlib.clearsky.ineichen(
        sol["apparent_zenith"], airmass_absolute=airmass_abs, linke_turbidity=3, altitude=altura,
    )

    # Misma convención que calcular_viabilidad: UI 0°=Sur -> pvlib 180°=Sur
    acimut_pvlib = (acimut_usuario + 180) % 360
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt, surface_azimuth=acimut_pvlib,
        solar_zenith=sol["apparent_zenith"], solar_azimuth=sol["azimuth"],
        dni=cs["dni"], ghi=cs["ghi"], dhi=cs["dhi"],
    )["poa_global"].clip(lower=0).fillna(0)

    panel = PANELES[panel_key]
    pdc = pvlib.pvsystem.pvwatts_dc(
        effective_irradiance=poa, temp_cell=25.0,
        pdc0=panel["potencia_w"], gamma_pdc=panel["gamma_pdc"],
    ).clip(lower=0)

    df = pd.DataFrame({
        "Hora": tiempos.tz_localize(None),
        "GHI_Wm2": cs["ghi"].clip(lower=0).fillna(0).values,
        "DNI_Wm2": cs["dni"].clip(lower=0).fillna(0).values,
        "DHI_Wm2": cs["dhi"].clip(lower=0).fillna(0).values,
        "POA_Wm2": poa.values,
        "Potencia_Panel_W": pdc.values,
    })
    energia_kwh = float((df["Potencia_Panel_W"] / 1000.0 * 0.25).sum())
    return df, energia_kwh

def generar_perfil_demanda(
    tipo_demanda: str,
    kwh_mensuales: list,
    kwh_anual: float,
    tiempos_naive: pd.DatetimeIndex,
) -> np.ndarray:
    """
    Distribuye energía de consumo en potencia kW a resolución de 15 minutos.
    """
    df_d = pd.DataFrame(index=tiempos_naive)
    df_d["mes"] = df_d.index.month

    if tipo_demanda == "Anual":
        kw_constante = kwh_anual / (365 * 24)
        df_d["Demanda_kW"] = kw_constante
    else:
        df_d["Demanda_kW"] = 0.0
        for mes in range(1, 13):
            mask = df_d["mes"] == mes
            horas_mes = mask.sum() * 0.25
            if horas_mes > 0:
                df_d.loc[mask, "Demanda_kW"] = kwh_mensuales[mes - 1] / horas_mes

    return df_d["Demanda_kW"].values


def resumen_penalizacion_temperatura_mensual(df_motor: pd.DataFrame) -> pd.DataFrame:
    """
    Resume generación de referencia, generación con temperatura y pérdida térmica por mes.
    """
    df = df_motor.copy()
    if "Fecha_Hora" in df.columns:
        fecha = pd.to_datetime(df["Fecha_Hora"])
    else:
        fecha = pd.to_datetime(df.index)
    df["Mes"] = fecha.dt.month

    resumen = (
        df.groupby("Mes", as_index=False)
        .agg(
            Generacion_Sin_Temp_kWh=("Energia_Solar_Sin_Temp_kWh", "sum"),
            Generacion_Con_Temp_kWh=("Energia_Solar_kWh", "sum"),
            Perdida_Temperatura_kWh=("Perdida_Temperatura_kWh", "sum"),
            Temperatura_Ambiente_Prom_C=("Temperatura_Ambiente_C", "mean"),
            Temperatura_Celda_Prom_C=("Temperatura_Celda_C", "mean"),
            Temperatura_Celda_Max_C=("Temperatura_Celda_C", "max"),
        )
        .sort_values("Mes")
    )

    resumen["Penalizacion_Temperatura_pct"] = np.where(
        resumen["Generacion_Sin_Temp_kWh"] > 0,
        resumen["Perdida_Temperatura_kWh"] / resumen["Generacion_Sin_Temp_kWh"] * 100.0,
        0.0,
    )

    cols = [
        "Mes",
        "Generacion_Sin_Temp_kWh",
        "Generacion_Con_Temp_kWh",
        "Perdida_Temperatura_kWh",
        "Penalizacion_Temperatura_pct",
        "Temperatura_Ambiente_Prom_C",
        "Temperatura_Celda_Prom_C",
        "Temperatura_Celda_Max_C",
    ]
    return resumen[cols]


# =============================================================================
# MOTOR PRINCIPAL DE SIMULACIÓN
# =============================================================================

def calcular_viabilidad(
    lat: float,
    lon: float,
    altura: float,
    tipo_demanda: str,
    kwh_mensuales: list,
    kwh_anual: float,
    tilt: float,
    acimut_usuario: float,
    tipo_panel: str = "monofacial",
    gb: float = 0.15,
    usar_clima_historico: bool = True,
    weather_start_date: str = "2025-01-01",
    weather_end_date: str = "2025-12-31",
    df_clima_horario: pd.DataFrame | None = None,
    num_paneles: int = 1, # NUEVO PARÁMETRO
):
    # 1. SELECCIÓN DE PANEL
    if tipo_panel not in PANELES:
        raise ValueError(
            f"tipo_panel='{tipo_panel}' no reconocido. "
            f"Opciones válidas: {list(PANELES.keys())}"
        )
    panel = PANELES[tipo_panel]

    # 2. ÍNDICE TEMPORAL DE SIMULACIÓN
    tz = "America/Mexico_City"
    tiempos = pd.date_range(
        start="2026-01-01 00:00",
        end="2026-12-31 23:45",
        freq="15min",
        tz=tz,
    )
    tiempos_naive = tiempos.tz_localize(None)

    # 3. POSICIÓN SOLAR Y CIELO DESPEJADO
    sol = pvlib.solarposition.get_solarposition(tiempos, lat, lon)
    airmass = pvlib.atmosphere.get_relative_airmass(sol["apparent_zenith"])
    airmass_abs = pvlib.atmosphere.get_absolute_airmass(airmass, pvlib.atmosphere.alt2pres(altura))
    clearsky = pvlib.clearsky.ineichen(
        sol["apparent_zenith"],
        airmass_absolute=airmass_abs,
        linke_turbidity=3,
        altitude=altura,
    )

    # 4. IRRADIANCIA POA
    # pvlib usa 0=Norte, 90=Este, 180=Sur, 270=Oeste.
    # La UI usa 0=Sur, -90=Este, 90=Oeste.
    acimut_pvlib = (acimut_usuario + 180) % 360
    irradiance_poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=acimut_pvlib,
        solar_zenith=sol["apparent_zenith"],
        solar_azimuth=sol["azimuth"],
        dni=clearsky["dni"],
        ghi=clearsky["ghi"],
        dhi=clearsky["dhi"],
    )

    # 5. DATAFRAME BASE Y DEMANDA
    df_motor = pd.DataFrame(index=tiempos_naive)
    df_motor["Gtot_POA_Wm2"] = irradiance_poa["poa_global"].clip(lower=0).fillna(0).values
    df_motor["Demanda_kW"] = generar_perfil_demanda(tipo_demanda, kwh_mensuales, kwh_anual, tiempos_naive)

    # 6. CLIMA PARA TEMPERATURA DE CELDA
    if usar_clima_historico:
        if df_clima_horario is None:
            df_clima_horario = descargar_clima_open_meteo(
                lat=lat,
                lon=lon,
                start_date=weather_start_date,
                end_date=weather_end_date,
                timezone=tz,
            )
        df_clima_15min = preparar_clima_15min(df_clima_horario, tiempos_naive)
        df_motor["Temperatura_Ambiente_C"] = df_clima_15min["Temperatura_Ambiente_C"].values
        df_motor["Velocidad_Viento_ms"] = df_clima_15min["Velocidad_Viento_ms"].values
    else:
        # Fallback simplificado, útil para pruebas sin internet o comparación rápida.
        df_motor["Temperatura_Ambiente_C"] = 20.0
        df_motor["Velocidad_Viento_ms"] = 1.5

    # 7. MODELADO ENERGÉTICO CON Y SIN PENALIZACIÓN TÉRMICA
    gamma_pdc = panel["gamma_pdc"]
    potencia_pico_dc = panel["potencia_w"] * num_paneles
    
    if panel["bifacial"]:
        gb_efectivo = float(np.clip(gb, 0.0, 1.0))
        potencia_ef_dc = potencia_pico_dc * (1.0 + gb_efectivo)
    else:
        potencia_ef_dc = potencia_pico_dc

    temp_params = TEMPERATURE_MODEL_PARAMETERS["sapm"]["close_mount_glass_glass"]
    df_motor["Temperatura_Celda_C"] = pvlib.temperature.sapm_cell(
        poa_global=df_motor["Gtot_POA_Wm2"],
        temp_air=df_motor["Temperatura_Ambiente_C"],
        wind_speed=df_motor["Velocidad_Viento_ms"],
        **temp_params,
    )

    pdc_con_temp = pvlib.pvsystem.pvwatts_dc(
        effective_irradiance=df_motor["Gtot_POA_Wm2"],
        temp_cell=df_motor["Temperatura_Celda_C"],
        pdc0=potencia_ef_dc,
        gamma_pdc=gamma_pdc,
    )
    pdc_ref_25 = pvlib.pvsystem.pvwatts_dc(
        effective_irradiance=df_motor["Gtot_POA_Wm2"],
        temp_cell=25.0,
        pdc0=potencia_ef_dc,
        gamma_pdc=gamma_pdc,
    )

    df_motor["Generacion_Solar_Sin_Temp_kW"] = (pdc_ref_25 / 1_000.0).clip(lower=0)
    df_motor["Generacion_Solar_kW"] = (pdc_con_temp / 1_000.0).clip(lower=0)
    df_motor["Penalizacion_Temperatura_kW"] = (
        df_motor["Generacion_Solar_Sin_Temp_kW"] - df_motor["Generacion_Solar_kW"]
    )

    df_motor["Energia_Solar_Sin_Temp_kWh"] = df_motor["Generacion_Solar_Sin_Temp_kW"] * 0.25
    df_motor["Energia_Solar_kWh"] = df_motor["Generacion_Solar_kW"] * 0.25
    df_motor["Perdida_Temperatura_kWh"] = df_motor["Penalizacion_Temperatura_kW"] * 0.25
    df_motor["Penalizacion_Temperatura_pct_inst"] = np.where(
        df_motor["Generacion_Solar_Sin_Temp_kW"] > 0,
        df_motor["Penalizacion_Temperatura_kW"]
        / df_motor["Generacion_Solar_Sin_Temp_kW"]
        * 100.0,
        0.0,
    )

    df_motor["Demanda_Post_Inyeccion_Solar_kW"] = (
        df_motor["Demanda_kW"] - df_motor["Generacion_Solar_kW"]
    ).clip(lower=0)

    # 8. ENERGÍA ANUAL GENERADA CON TEMPERATURA REAL
    energia_anual = float(df_motor["Energia_Solar_kWh"].sum())

    # 9. SALIDA
    df_motor.index.name = "Fecha_Hora"
    df_motor.reset_index(inplace=True)

    columnas_salida = [
        "Fecha_Hora",
        "Demanda_kW",
        "Gtot_POA_Wm2",
        "Temperatura_Ambiente_C",
        "Velocidad_Viento_ms",
        "Temperatura_Celda_C",
        "Generacion_Solar_Sin_Temp_kW",
        "Generacion_Solar_kW",
        "Penalizacion_Temperatura_kW",
        "Energia_Solar_Sin_Temp_kWh",
        "Energia_Solar_kWh",
        "Perdida_Temperatura_kWh",
        "Penalizacion_Temperatura_pct_inst",
        "Demanda_Post_Inyeccion_Solar_kW",
    ]
    return df_motor[columnas_salida], energia_anual


import math

import math

def dimensionar_y_simular(
    lat: float,
    lon: float,
    altura: float,
    tipo_demanda: str,
    kwh_mensuales: list,
    kwh_anual: float,
    tilt: float,
    acimut_usuario: float,
    tipo_panel: str = "monofacial",
    gb: float = 0.15,
    usar_clima_historico: bool = True,
    weather_start_date: str = "2025-01-01",
    weather_end_date: str = "2025-12-31",
    df_clima_horario: pd.DataFrame | None = None,
    porcentaje_cobertura: float = 1.0  # 1.0 = Cubrir el 100% de la demanda
):
    """
    Dimensiona la cantidad de paneles requeridos para cubrir la demanda indicada,
    calcula el área necesaria y ejecuta la simulación del sistema.
    """
    # 1. Determinar el objetivo de generación anual
    if tipo_demanda != "Anual":
        demanda_objetivo = sum(kwh_mensuales) * porcentaje_cobertura
    else:
        demanda_objetivo = kwh_anual * porcentaje_cobertura

    if demanda_objetivo <= 0:
        raise ValueError("La demanda anual objetivo debe ser mayor a 0 para dimensionar el sistema.")

    # 2. Primera pasada: Simular el rendimiento de 1 solo panel
    _, energia_1_panel = calcular_viabilidad(
        lat=lat, lon=lon, altura=altura,
        tipo_demanda=tipo_demanda, kwh_mensuales=kwh_mensuales, kwh_anual=kwh_anual,
        tilt=tilt, acimut_usuario=acimut_usuario,
        tipo_panel=tipo_panel, gb=gb,
        usar_clima_historico=usar_clima_historico,
        weather_start_date=weather_start_date, weather_end_date=weather_end_date,
        df_clima_horario=df_clima_horario,
        num_paneles=1 # Forzamos 1 panel
    )

    if energia_1_panel <= 0:
        raise RuntimeError("La simulación de 1 panel resultó en 0 kWh. Revisa la ubicación, tilt o datos climáticos.")

    # 3. Dimensionamiento matemático
    # Redondeamos hacia arriba para asegurar la cobertura de la demanda
    num_paneles_requeridos = math.ceil(demanda_objetivo / energia_1_panel)

    
    # 4. Segunda pasada: Simular el sistema real dimensionado
    df_motor_final, energia_anual_final = calcular_viabilidad(
        lat=lat, lon=lon, altura=altura,
        tipo_demanda=tipo_demanda, kwh_mensuales=kwh_mensuales, kwh_anual=kwh_anual,
        tilt=tilt, acimut_usuario=acimut_usuario,
        tipo_panel=tipo_panel, gb=gb,
        usar_clima_historico=usar_clima_historico,
        weather_start_date=weather_start_date, weather_end_date=weather_end_date,
        df_clima_horario=df_clima_horario,
        num_paneles=num_paneles_requeridos # Usamos los paneles calculados
    )

    potencia_instalada_kwp = (PANELES[tipo_panel]["potencia_w"] * num_paneles_requeridos) / 1000.0
    
    # NUEVO: Cálculo del área total
    area_unitaria = PANELES[tipo_panel].get("area_m2", 2.5) # 2.5 es un fallback por seguridad
    area_total_m2 = num_paneles_requeridos * area_unitaria

    # 5. Entregar resultados completos
    return {
        "num_paneles": num_paneles_requeridos,
        "potencia_instalada_kwp": potencia_instalada_kwp,
        "area_total_m2": float(area_total_m2),
        "demanda_objetivo_kwh": demanda_objetivo,
        "generacion_estimada_kwh": energia_anual_final, # Corregido el typo aquí
        "df_simulacion": df_motor_final
    }