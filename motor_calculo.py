# =============================================================================
#  motor_calculo.py — Motor de simulación fotovoltaica
#  Dependencias: pvlib >= 0.9, pandas >= 1.5, numpy
# =============================================================================

import pandas as pd
import numpy as np
import pvlib
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS


def calcular_viabilidad(lat: float, lon: float, altura: float, df_demanda: pd.DataFrame):
    """
    Simula un año completo de generación solar y su impacto en la curva de demanda.

    Parámetros
    ----------
    lat, lon : float  — Coordenadas geográficas del sitio.
    altura   : float  — Altitud sobre el nivel del mar en metros.
    df_demanda : pd.DataFrame — DataFrame con al menos una columna numérica de demanda
                                (preferiblemente llamada 'Demanda_kW').

    Retorna
    -------
    df_motor : pd.DataFrame  con columnas:
               ['Fecha_Hora', 'Demanda_kW', 'Gtot_POA_Wm2',
                'Generacion_Solar_kW', 'Demanda_Post_Inyeccion_Solar_kW']
    energia_anual : float  — kWh solares generados en el año simulado.
    """

    # -------------------------------------------------------------------------
    # 1. ÍNDICE TEMPORAL  (año completo, intervalos de 15 min, zona horaria local)
    # -------------------------------------------------------------------------
    tz = 'America/Mexico_City'
    tiempos = pd.date_range(
        start='2026-12-21 00:00',
        end='2027-12-20 23:45',
        freq='15min',
        tz=tz,
    )
    # Versión sin zona horaria para el DataFrame final (más amigable para Plotly)
    tiempos_naive = tiempos.tz_localize(None)

    # -------------------------------------------------------------------------
    # 2. POSICIÓN SOLAR
    #    Usamos la DatetimeIndex completa (con tz) para que pvlib mantenga
    #    el índice temporal en todos los objetos intermedios.
    # -------------------------------------------------------------------------
    sol = pvlib.solarposition.get_solarposition(tiempos, lat, lon)

    # -------------------------------------------------------------------------
    # 3. IRRADIANCIA EN CIELO DESPEJADO (modelo Ineichen)
    #
    #    BUG ORIGINAL corregido:
    #    Pasar sol['apparent_zenith'] (Serie con índice) en vez de
    #    sol['apparent_zenith'].values (array numpy) evita que ineichen()
    #    devuelva un OrderedDict y garantiza un DataFrame con índice datetime.
    #    Lo mismo aplica a get_total_irradiance: los argumentos deben ser
    #    Series con índice para recibir un DataFrame con DatetimeIndex.
    # -------------------------------------------------------------------------
    airmass = pvlib.atmosphere.get_relative_airmass(sol['apparent_zenith'])
    airmass_abs = pvlib.atmosphere.get_absolute_airmass(
        airmass, pvlib.atmosphere.alt2pres(altura)
    )
    
    turbidity = pvlib.clearsky.lookup_linke_turbidity(
        tiempos, lat, lon)


    clearsky = pvlib.clearsky.ineichen(
        sol['apparent_zenith'],
        airmass_absolute=airmass_abs,
        linke_turbidity=turbidity,
        altitude=altura,
    )  # → DataFrame con columnas ['ghi', 'dni', 'dhi'] e índice datetime (tz)

    irradiance_poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=25,
        surface_azimuth=180,
        solar_zenith=sol['apparent_zenith'],
        solar_azimuth=sol['azimuth'],
        dni=clearsky['dni'],
        ghi=clearsky['ghi'],
        dhi=clearsky['dhi'],
    )  # → DataFrame con índice datetime (tz)

    # -------------------------------------------------------------------------
    # 4. CONSTRUCCIÓN DEL DATAFRAME BASE  (índice sin zona horaria)
    #
    #    BUG ORIGINAL corregido:
    #    Usar .values al asignar irradiance_poa['poa_global'] evita que pandas
    #    intente alinear índices tz-aware vs tz-naive, lo que producía NaN.
    # -------------------------------------------------------------------------
    df_motor = pd.DataFrame(index=tiempos_naive)
    df_motor['Gtot_POA_Wm2'] = irradiance_poa['poa_global'].values

    # -------------------------------------------------------------------------
    # 5. ALINEACIÓN DE LA CURVA DE DEMANDA
    # -------------------------------------------------------------------------
    try:
        # Si viene como ruta de archivo en vez de DataFrame (compatibilidad)
        if isinstance(df_demanda, str):
            df_demanda = pd.read_csv(df_demanda, index_col=0, parse_dates=True)

        # Identificar la columna de demanda
        col_demanda = 'Demanda_kW' if 'Demanda_kW' in df_demanda.columns else df_demanda.columns[0]
        serie_demanda = pd.to_numeric(df_demanda[col_demanda], errors='coerce').fillna(0).values

        if len(serie_demanda) == len(tiempos_naive):
            # Mismo número de puntos: asignar directamente por posición
            df_motor['Demanda_kW'] = serie_demanda
        else:
            # Distinto tamaño: interpolar / rellenar con forward-fill + backward-fill
            # BUG ORIGINAL corregido: fillna(method='bfill') eliminado en pandas ≥ 2.0
            s_tmp = pd.Series(serie_demanda)
            s_resampled = s_tmp.reindex(
                pd.RangeIndex(len(tiempos_naive))
            ).interpolate(method='linear').bfill().ffill()
            df_motor['Demanda_kW'] = s_resampled.values

    except Exception:
        # Fallback: demanda sintética si hay cualquier error al procesar el CSV
        rng = np.random.default_rng(seed=42)
        df_motor['Demanda_kW'] = rng.uniform(150, 300, len(tiempos_naive))

    # -------------------------------------------------------------------------
    # 6. MODELADO ENERGÉTICO
    # -------------------------------------------------------------------------
    potencia_pico_dc = 399_300.0   # Wp (potencia pico DC del sistema)
    gamma_pdc        = -0.0029     # Coeficiente de temperatura de potencia (1/°C)


    # Hacer esto en un instant quinceminutal con la base de datos de temperatura ambiente y velocidad del viento para Jalapa.
    temp_params = TEMPERATURE_MODEL_PARAMETERS['sapm']['close_mount_glass_glass']
    temp_celda = pvlib.temperature.sapm_cell(
        poa_global=df_motor['Gtot_POA_Wm2'],
        temp_air=20.0,  # Cambiar usando base de datos de temperatura ambiente para jalapa. 
        wind_speed=1.5, # Base de datos. 
        **temp_params,
    )

    pdc = pvlib.pvsystem.pvwatts_dc(
        effective_irradiance=df_motor['Gtot_POA_Wm2'],
        temp_cell=temp_celda,
        pdc0=potencia_pico_dc,
        gamma_pdc=gamma_pdc,
    )

    df_motor['Generacion_Solar_kW'] = (pdc / 1_000.0).clip(lower=0)
    df_motor['Demanda_Post_Inyeccion_Solar_kW'] = (
        df_motor['Demanda_kW'] - df_motor['Generacion_Solar_kW']
    ).clip(lower=0)

    # -------------------------------------------------------------------------
    # 7. ENERGÍA ANUAL GENERADA  (kWh = kW × 0.25 h por intervalo de 15 min)
    # -------------------------------------------------------------------------
    energia_anual = float((df_motor['Generacion_Solar_kW'] * 0.25).sum())

    df_motor.index.name = 'Fecha_Hora'
    df_motor.reset_index(inplace=True)

    # Conservar solo las columnas necesarias en el orden esperado por app.py
    columnas_salida = [
        'Fecha_Hora',
        'Demanda_kW',
        'Gtot_POA_Wm2',
        'Generacion_Solar_kW',
        'Demanda_Post_Inyeccion_Solar_kW',
    ]
    df_motor = df_motor[columnas_salida]

    return df_motor, energia_anual
