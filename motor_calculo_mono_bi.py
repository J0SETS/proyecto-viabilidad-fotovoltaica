import pandas as pd
import numpy as np
import pvlib
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS


#  CATÁLOGO DE PANELES
#  Cada entrada define los parámetros que consume calcular_viabilidad().

PANELES = {
    "monofacial": {
        "nombre":           "Monofacial",
        "fabricante":       "Jinko Solar",
        "modelo":           "Tiger Neo 72HC — JKM605N-72HL4",
        "potencia_w":       605,
        "potencia_pico_dc": 399_300.0,   # Wp totales del sistema (sin cambio)
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


def calcular_viabilidad(
    lat: float,
    lon: float,
    altura: float,
    df_demanda: pd.DataFrame,
    tipo_panel: str = "monofacial",
    gb: float = 0.15,
):

    # SELECCIÓN DE PANEL

    if tipo_panel not in PANELES:
        raise ValueError(
            f"tipo_panel='{tipo_panel}' no reconocido. "
            f"Opciones válidas: {list(PANELES.keys())}"
        )
    panel = PANELES[tipo_panel]


    # ÍNDICE TEMPORAL  (año completo, intervalos de 15 min, zona horaria local)

    tz = 'America/Mexico_City'
    tiempos = pd.date_range(
        start='2026-12-21 00:00',
        end='2027-12-20 23:45',
        freq='15min',
        tz=tz,
    )
    # Versión sin zona horaria para el DataFrame final (más amigable para Plotly)
    tiempos_naive = tiempos.tz_localize(None)


    # POSICIÓN SOLAR

    sol = pvlib.solarposition.get_solarposition(tiempos, lat, lon)


    # IRRADIANCIA EN CIELO DESPEJADO (modelo Ineichen)
    airmass = pvlib.atmosphere.get_relative_airmass(sol['apparent_zenith'])
    airmass_abs = pvlib.atmosphere.get_absolute_airmass(
        airmass, pvlib.atmosphere.alt2pres(altura)
    )

    clearsky = pvlib.clearsky.ineichen(
        sol['apparent_zenith'],
        airmass_absolute=airmass_abs,
        linke_turbidity=3,
        altitude=altura,
    )

    irradiance_poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=25,
        surface_azimuth=180,
        solar_zenith=sol['apparent_zenith'],
        solar_azimuth=sol['azimuth'],
        dni=clearsky['dni'],
        ghi=clearsky['ghi'],
        dhi=clearsky['dhi'],
    )

    # CONSTRUCCIÓN DEL DATAFRAME BASE
    df_motor = pd.DataFrame(index=tiempos_naive)
    df_motor['Gtot_POA_Wm2'] = irradiance_poa['poa_global'].values

    # ALINEACIÓN DE LA CURVA DE DEMANDA
    try:
        if isinstance(df_demanda, str):
            df_demanda = pd.read_csv(df_demanda, index_col=0, parse_dates=True)

        col_demanda = 'Demanda_kW' if 'Demanda_kW' in df_demanda.columns else df_demanda.columns[0]
        serie_demanda = pd.to_numeric(df_demanda[col_demanda], errors='coerce').fillna(0).values

        if len(serie_demanda) == len(tiempos_naive):
            df_motor['Demanda_kW'] = serie_demanda
        else:
            s_tmp = pd.Series(serie_demanda)
            s_resampled = (
                s_tmp.reindex(pd.RangeIndex(len(tiempos_naive)))
                     .interpolate(method='linear')
                     .bfill()
                     .ffill()
            )
            df_motor['Demanda_kW'] = s_resampled.values

    except Exception:
        rng = np.random.default_rng(seed=42)
        df_motor['Demanda_kW'] = rng.uniform(150, 300, len(tiempos_naive))

    # 6. MODELADO ENERGÉTICO
    gamma_pdc        = panel['gamma_pdc']
    potencia_pico_dc = panel['potencia_pico_dc']   # Wp base del sistema

    # --- Ganancia bifacial ---------------------------------------------------
    # Para el panel bifacial se escala la potencia pico DC efectiva antes de
    # pasarla a pvwatts_dc, aplicando la fórmula:
    #
    #   P_ef = P_STC × (1 + G_b)
    #
    # donde G_b es la ganancia trasera configurada por el usuario.
    # Para el monofacial G_b = 0, por lo que potencia_ef == potencia_pico_dc.
    if panel['bifacial']:
        gb_efectivo    = float(np.clip(gb, 0.0, 1.0))   # Seguridad: acota entre 0 y 100 %
        potencia_ef_dc = potencia_pico_dc * (1.0 + gb_efectivo)
    else:
        potencia_ef_dc = potencia_pico_dc

    temp_params = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
    temp_celda = pvlib.temperature.sapm_cell(
        poa_global=df_motor['Gtot_POA_Wm2'],
        temp_air=20.0,
        wind_speed=1.5,
        **temp_params,
    )

    pdc = pvlib.pvsystem.pvwatts_dc(
        effective_irradiance=df_motor['Gtot_POA_Wm2'],
        temp_cell=temp_celda,
        pdc0=potencia_ef_dc,          # ← usa la potencia efectiva (bifacial o no)
        gamma_pdc=gamma_pdc,
    )

    df_motor['Generacion_Solar_kW']           = (pdc / 1_000.0).clip(lower=0)
    df_motor['Demanda_Post_Inyeccion_Solar_kW'] = (
        df_motor['Demanda_kW'] - df_motor['Generacion_Solar_kW']
    ).clip(lower=0)


    # 7. ENERGÍA ANUAL GENERADA  (kWh = kW × 0.25 h por intervalo de 15 min)
    energia_anual = float((df_motor['Generacion_Solar_kW'] * 0.25).sum())


    # 8. PREPARAR DATAFRAME FINAL
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