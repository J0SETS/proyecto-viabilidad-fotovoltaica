import pandas as pd
import numpy as np
import pvlib
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

# =============================================================================
#  CATÁLOGO DE PANELES
# =============================================================================
PANELES = {
    "monofacial": {
        "nombre":           "Monofacial",
        "fabricante":       "Jinko Solar",
        "modelo":           "Tiger Neo 72HC — JKM605N-72HL4",
        "potencia_w":       605,
        "potencia_pico_dc": 399_300.0,
        "gamma_pdc":        -0.0029,
        "bifacial":         False,
    },
    "bifacial_jinko": {
        "nombre":           "Bifacial",
        "fabricante":       "Jinko Solar",
        "modelo":           "Tiger Neo N-type — JKM625N-78HL4-BDV",
        "potencia_w":       625,
        "potencia_pico_dc": 399_300.0,
        "gamma_pdc":        -0.0029,
        "bifacial":         True,
        "gb_default":       0.15,
    },
}

# =============================================================================
#  FUNCIONES DE SOPORTE MATEMÁTICO Y PERFILADO
# =============================================================================
def calcular_tilt_optimo(lat: float, lon: float, altura: float) -> float:
    """
    Realiza un barrido heurístico rápido para encontrar el ángulo de inclinación (tilt)
    que maximiza la irradiancia anual, asumiendo un acimut orientado al sur (180° en pvlib).
    """
    # Usamos una resolución menor (1 hora) para que el barrido sea instantáneo
    tiempos = pd.date_range("2026-01-01", "2026-12-31 23:00", freq="1h", tz="America/Mexico_City")
    sol = pvlib.solarposition.get_solarposition(tiempos, lat, lon)
    
    airmass = pvlib.atmosphere.get_relative_airmass(sol["apparent_zenith"])
    airmass_abs = pvlib.atmosphere.get_absolute_airmass(airmass, pvlib.atmosphere.alt2pres(altura))
    
    cs = pvlib.clearsky.ineichen(
        sol["apparent_zenith"], 
        airmass_absolute=airmass_abs, 
        linke_turbidity=3, 
        altitude=altura
    )

    best_tilt = lat
    max_poa = 0.0
    
    # Barrido heurístico de +/- 15 grados alrededor de la latitud local
    rango_tilt = np.arange(max(0, lat - 15), lat + 15, 1.0)
    
    for t in rango_tilt:
        poa = pvlib.irradiance.get_total_irradiance(
            surface_tilt=t, 
            surface_azimuth=180,  # 180° = Sur en el estándar de pvlib
            solar_zenith=sol["apparent_zenith"], 
            solar_azimuth=sol["azimuth"],
            dni=cs["dni"], 
            ghi=cs["ghi"], 
            dhi=cs["dhi"]
        )
        total = poa['poa_global'].sum()
        if total > max_poa:
            max_poa = total
            best_tilt = t
            
    return float(best_tilt)

def generar_perfil_demanda(tipo_demanda: str, kwh_mensuales: list, kwh_anual: float, tiempos_naive: pd.DatetimeIndex) -> np.ndarray:
    """
    Distribuye el consumo de energía (kWh) en intervalos de potencia (kW) de 15 minutos
    para mantener la compatibilidad con el motor de alta resolución.
    """
    df_d = pd.DataFrame(index=tiempos_naive)
    df_d['mes'] = df_d.index.month

    if tipo_demanda == "Anual":
        # Distribución plana a lo largo de todo el año
        kw_constante = kwh_anual / (365 * 24)
        df_d['Demanda_kW'] = kw_constante
    else:
        # Distribución plana, pero segmentada por los días que tiene cada mes
        df_d['Demanda_kW'] = 0.0
        for mes in range(1, 13):
            mask = df_d['mes'] == mes
            horas_mes = mask.sum() * 0.25  # Cada fila representa 15 min (0.25 horas)
            if horas_mes > 0:
                df_d.loc[mask, 'Demanda_kW'] = kwh_mensuales[mes-1] / horas_mes
                
    return df_d['Demanda_kW'].values

# =============================================================================
#  MOTOR PRINCIPAL DE SIMULACIÓN
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
):
    # 1. SELECCIÓN DE PANEL
    if tipo_panel not in PANELES:
        raise ValueError(
            f"tipo_panel='{tipo_panel}' no reconocido. "
            f"Opciones válidas: {list(PANELES.keys())}"
        )
    panel = PANELES[tipo_panel]

    # 2. ÍNDICE TEMPORAL (Resolución 15 min)
    tz = 'America/Mexico_City'
    tiempos = pd.date_range(
        start='2026-01-01 00:00',
        end='2026-12-31 23:45',
        freq='15min',
        tz=tz,
    )
    tiempos_naive = tiempos.tz_localize(None)

    # 3. POSICIÓN SOLAR Y CIELO DESPEJADO
    sol = pvlib.solarposition.get_solarposition(tiempos, lat, lon)
    airmass = pvlib.atmosphere.get_relative_airmass(sol['apparent_zenith'])
    airmass_abs = pvlib.atmosphere.get_absolute_airmass(airmass, pvlib.atmosphere.alt2pres(altura))

    clearsky = pvlib.clearsky.ineichen(
        sol['apparent_zenith'],
        airmass_absolute=airmass_abs,
        linke_turbidity=3,
        altitude=altura,
    )

    # 4. TRANSFORMACIÓN DE ACIMUT
    # pvlib usa 0=Norte, 90=Este, 180=Sur, 270=Oeste.
    # El usuario ingresa: 0=Sur, -90=Este, 90=Oeste.
    acimut_pvlib = (acimut_usuario + 180) % 360

    irradiance_poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=acimut_pvlib,
        solar_zenith=sol['apparent_zenith'],
        solar_azimuth=sol['azimuth'],
        dni=clearsky['dni'],
        ghi=clearsky['ghi'],
        dhi=clearsky['dhi'],
    )

    # 5. CONSTRUCCIÓN DEL DATAFRAME BASE Y DEMANDA SINTÉTICA
    df_motor = pd.DataFrame(index=tiempos_naive)
    df_motor['Gtot_POA_Wm2'] = irradiance_poa['poa_global'].clip(lower=0).fillna(0).values
    df_motor['Demanda_kW'] = generar_perfil_demanda(tipo_demanda, kwh_mensuales, kwh_anual, tiempos_naive)

    # 6. MODELADO ENERGÉTICO
    gamma_pdc = panel['gamma_pdc']
    potencia_pico_dc = panel['potencia_pico_dc']

    if panel['bifacial']:
        gb_efectivo = float(np.clip(gb, 0.0, 1.0))
        potencia_ef_dc = potencia_pico_dc * (1.0 + gb_efectivo)
    else:
        potencia_ef_dc = potencia_pico_dc

    temp_params = TEMPERATURE_MODEL_PARAMETERS['sapm']['close_mount_glass_glass']
    temp_celda = pvlib.temperature.sapm_cell(
        poa_global=df_motor['Gtot_POA_Wm2'],
        temp_air=20.0,
        wind_speed=1.5,
        **temp_params,
    )

    pdc = pvlib.pvsystem.pvwatts_dc(
        effective_irradiance=df_motor['Gtot_POA_Wm2'],
        temp_cell=temp_celda,
        pdc0=potencia_ef_dc,
        gamma_pdc=gamma_pdc,
    )

    df_motor['Generacion_Solar_kW'] = (pdc / 1_000.0).clip(lower=0)
    df_motor['Demanda_Post_Inyeccion_Solar_kW'] = (
        df_motor['Demanda_kW'] - df_motor['Generacion_Solar_kW']
    ).clip(lower=0)

    # 7. ENERGÍA ANUAL GENERADA
    energia_anual = float((df_motor['Generacion_Solar_kW'] * 0.25).sum())

    # 8. SALIDA DE DATOS
    df_motor.index.name = 'Fecha_Hora'
    df_motor.reset_index(inplace=True)

    columnas_salida = [
        'Fecha_Hora',
        'Demanda_kW',
        'Gtot_POA_Wm2',
        'Generacion_Solar_kW',
        'Demanda_Post_Inyeccion_Solar_kW',
    ]
    df_motor = df_motor[columnas_salida]

    return df_motor, energia_anual