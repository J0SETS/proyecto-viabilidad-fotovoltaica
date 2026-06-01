# =============================================================================
#  pronostico.py — Pronóstico meteorológico y generación solar a 5 días
#
#  Fuente     : Open-Meteo (https://open-meteo.com) — gratuito, sin API key
#  Física     : descomposición GHI → DNI/DHI con modelo Erbs (pvlib)
#  Resolución : datos horarios de la API, interpolados a 15 min
# =============================================================================

import requests
import numpy as np
import pandas as pd
import pvlib
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

# ---------------------------------------------------------------------------
# CATÁLOGO WMO
# Mapea código meteorológico → (descripción legible, nivel de alerta)
# Niveles posibles: ok | leve | moderado | alto | critico
# ---------------------------------------------------------------------------
WMO_CODES: dict[int, tuple[str, str]] = {
    0:  ("Cielo despejado",                          "ok"),
    1:  ("Principalmente despejado",                 "ok"),
    2:  ("Parcialmente nublado",                     "leve"),
    3:  ("Nublado / cubierto",                       "leve"),
    45: ("Niebla",                                   "moderado"),
    48: ("Niebla con depósitos de escarcha",         "moderado"),
    51: ("Llovizna ligera",                          "leve"),
    53: ("Llovizna moderada",                        "moderado"),
    55: ("Llovizna densa",                           "moderado"),
    61: ("Lluvia ligera",                            "moderado"),
    63: ("Lluvia moderada",                          "alto"),
    65: ("Lluvia fuerte",                            "alto"),
    71: ("Nieve ligera",                             "moderado"),
    73: ("Nieve moderada",                           "alto"),
    75: ("Nieve fuerte",                             "alto"),
    80: ("Chubascos de lluvia ligera",               "moderado"),
    81: ("Chubascos de lluvia moderada",             "moderado"),
    82: ("Chubascos de lluvia fuerte",               "alto"),
    95: ("Tormenta eléctrica",                       "critico"),
    96: ("Tormenta eléctrica con granizo ligero",    "critico"),
    99: ("Tormenta eléctrica con granizo fuerte",    "critico"),
}

# Configuración visual por nivel — usada por app.py para tarjetas y alertas
NIVEL_CONFIG: dict[str, dict] = {
    "ok":       {"emoji": "☀️",  "label": "Normal",   "color": "#4ecdc4"},
    "leve":     {"emoji": "🌤️", "label": "Leve",     "color": "#ffe033"},
    "moderado": {"emoji": "🌥️", "label": "Moderado", "color": "#f5a623"},
    "alto":     {"emoji": "🌧️", "label": "Alto",     "color": "#ff6b6b"},
    "critico":  {"emoji": "⛈️", "label": "Crítico",  "color": "#ff2244"},
}

# Orden ascendente de severidad (se usa para determinar el nivel máximo del día)
_ORDEN_NIVEL = ["ok", "leve", "moderado", "alto", "critico"]


# =============================================================================
# 1. DESCARGA DE PRONÓSTICO METEOROLÓGICO
# =============================================================================

def obtener_pronostico(lat: float, lon: float, dias: int = 5) -> dict:
    """
    Descarga el pronóstico horario desde la API pública de Open-Meteo.

    Variables horarias descargadas
    --------------------------------
    shortwave_radiation  → Radiación de onda corta = GHI (W/m²)
    cloud_cover          → Cobertura de nubosidad (%)
    temperature_2m       → Temperatura del aire a 2 m (°C)
    windspeed_10m        → Velocidad del viento a 10 m (km/h)  ← la API entrega km/h
    precipitation        → Precipitación acumulada por hora (mm)
    weathercode          → Código WMO del estado del tiempo

    Parámetros
    ----------
    lat, lon : float — Coordenadas del sistema fotovoltaico.
    dias     : int   — Horizonte de pronóstico (máx. 16). Por defecto 5.

    Retorna
    -------
    dict con la respuesta JSON completa de la API.
    """
    params = {
        "latitude":      lat,
        "longitude":     lon,
        "hourly":        ",".join([
            "shortwave_radiation",
            "cloud_cover",
            "temperature_2m",
            "windspeed_10m",
            "precipitation",
            "weathercode",
        ]),
        "timezone":      "America/Mexico_City",
        "forecast_days": dias,
    }
    resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params=params,
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


# =============================================================================
# 2. CÁLCULO DE GENERACIÓN SOLAR PRONOSTICADA
# =============================================================================

def calcular_generacion_pronostico(
    datos_json: dict,
    lat:        float,
    lon:        float,
    altura:     float,
    pdc0:       float = 399_300.0,
    gamma_pdc:  float = -0.0029,
) -> pd.DataFrame:
    """
    Transforma la respuesta JSON de Open-Meteo en una curva de generación solar
    a resolución de 15 min.

    Pipeline físico
    ---------------
    1. Construye un DataFrame horario con las variables meteorológicas.
       · windspeed_10m se convierte de km/h → m/s (÷ 3.6) porque Open-Meteo
         entrega km/h pero pvlib.temperature.sapm_cell espera m/s.
    2. Calcula la posición solar con pvlib (DatetimeIndex con zona horaria).
    3. Descompone GHI en DNI y DHI usando pvlib.irradiance.erbs().
       · Firma correcta: erbs(ghi, zenith, datetime_or_doy)
         NO existe pvlib.irradiance.decomposition.erbs en pvlib ≥ 0.9.
       · Con entradas numpy devuelve OrderedDict de arrays numpy;
         se accede directamente con la clave, sin .values.
    4. Calcula la irradiancia en el plano del array (POA) a 25° de inclinación,
       azimut 180° (orientación sur).
    5. Calcula temperatura de celda (modelo SAPM, montaje close_mount_glass_glass).
    6. Calcula potencia DC con el modelo PVWatts.
    7. Interpola a 15 min (lineal para variables numéricas, forward-fill para
       el código WMO que es categórico).

    Precisión de la interpolación
    ------------------------------
    La interpolación lineal 1 h → 15 min es válida para planificación operativa.
    No representa sub-variabilidad de nubosidad. La fiabilidad del pronóstico
    decae de forma notable después de las 72 h.

    Parámetros
    ----------
    datos_json : dict     — Respuesta de obtener_pronostico().
    lat, lon   : float    — Coordenadas del sistema.
    altura     : float    — Altitud en metros s.n.m.
    pdc0       : float    — Potencia pico DC del sistema (W).
    gamma_pdc  : float    — Coeficiente de temperatura de potencia (1/°C).

    Retorna
    -------
    pd.DataFrame con columnas:
        Fecha_Hora, GHI_Wm2, Nubosidad_pct, Temp_Aire_C, Viento_ms,
        Precipitacion_mm, Gtot_POA_Wm2, Generacion_Solar_kW, Codigo_Clima
    """
    h = datos_json["hourly"]

    # ── 1. DataFrame horario ──────────────────────────────────────────────────
    df = pd.DataFrame({
        "Fecha_Hora":       pd.to_datetime(h["time"]),
        "GHI_Wm2":          np.array(h["shortwave_radiation"], dtype=float),
        "Nubosidad_pct":    np.array(h["cloud_cover"],         dtype=float),
        "Temp_Aire_C":      np.array(h["temperature_2m"],      dtype=float),
        # Open-Meteo entrega windspeed_10m en km/h; sapm_cell necesita m/s
        "Viento_ms":        np.array(h["windspeed_10m"],       dtype=float) / 3.6,
        "Precipitacion_mm": np.array(h["precipitation"],       dtype=float),
        "Codigo_Clima":     np.array(h["weathercode"],         dtype=int),
    })

    # ── 2. Posición solar ─────────────────────────────────────────────────────
    # Open-Meteo devuelve tiempos sin offset; se localiza explícitamente.
    times_tz = pd.DatetimeIndex(df["Fecha_Hora"]).tz_localize(
        "America/Mexico_City",
        ambiguous="infer",           # resuelve cambios de horario de verano
        nonexistent="shift_forward",
    )
    sol = pvlib.solarposition.get_solarposition(times_tz, lat, lon)

    # ── 3. Descomposición GHI → DNI + DHI  (modelo Erbs) ─────────────────────
    # Ruta correcta en pvlib ≥ 0.9: pvlib.irradiance.erbs()
    # Parámetro temporal: datetime_or_doy (NO "times")
    # Con arrays numpy como entrada, devuelve OrderedDict de arrays numpy.
    decomp = pvlib.irradiance.erbs(
        ghi=df["GHI_Wm2"].values,
        zenith=sol["apparent_zenith"].values,
        datetime_or_doy=times_tz.dayofyear,
    )
    dni = np.clip(decomp["dni"], 0, None)
    dhi = np.clip(decomp["dhi"], 0, None)

    # ── 4. Irradiancia POA ────────────────────────────────────────────────────
    # Con arrays numpy, get_total_irradiance también devuelve OrderedDict.
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=25,
        surface_azimuth=180,
        solar_zenith=sol["apparent_zenith"].values,
        solar_azimuth=sol["azimuth"].values,
        dni=dni,
        ghi=df["GHI_Wm2"].values,
        dhi=dhi,
    )
    poa_global = np.clip(poa["poa_global"], 0, None)
    df["Gtot_POA_Wm2"] = poa_global

    # ── 5. Temperatura de celda ───────────────────────────────────────────────
    temp_params = TEMPERATURE_MODEL_PARAMETERS["sapm"]["close_mount_glass_glass"]
    temp_celda = pvlib.temperature.sapm_cell(
        poa_global=poa_global,
        temp_air=df["Temp_Aire_C"].values,
        wind_speed=df["Viento_ms"].values,   # ya en m/s tras la conversión
        **temp_params,
    )

    # ── 6. Potencia DC (PVWatts) ──────────────────────────────────────────────
    pdc_arr = pvlib.pvsystem.pvwatts_dc(
        effective_irradiance=poa_global,
        temp_cell=temp_celda,
        pdc0=pdc0,
        gamma_pdc=gamma_pdc,
    )
    df["Generacion_Solar_kW"] = np.clip(pdc_arr / 1_000.0, 0, None)

    # ── 7. Interpolación 1 h → 15 min ────────────────────────────────────────
    df_idx   = df.set_index("Fecha_Hora")
    codigo_h = df_idx["Codigo_Clima"].copy()                  # categórica → ffill
    df_num   = df_idx.drop(columns=["Codigo_Clima"])
    df_15    = df_num.resample("15min").interpolate(method="linear")
    df_15["Generacion_Solar_kW"] = df_15["Generacion_Solar_kW"].clip(lower=0)
    df_15["Codigo_Clima"]        = codigo_h.resample("15min").ffill()
    df_15 = df_15.reset_index().rename(columns={"index": "Fecha_Hora"})

    return df_15


# =============================================================================
# 3. RESUMEN DIARIO Y CLASIFICACIÓN DE ALERTAS
# =============================================================================

def _nivel_maximo(codigos: pd.Series, nubosidad_media: float) -> tuple[str, str]:
    """
    Devuelve el nivel de alerta más severo del día y su descripción.
    Si no hay código extremo pero la nubosidad supera el 70%, escala a moderado.
    """
    nivel, desc = "ok", "Cielo despejado"
    for cod in codigos.dropna().unique():
        d, n = WMO_CODES.get(int(cod), ("Desconocido", "leve"))
        if _ORDEN_NIVEL.index(n) > _ORDEN_NIVEL.index(nivel):
            nivel, desc = n, d

    if nivel == "ok" and nubosidad_media > 70:
        nivel = "moderado"
        desc  = f"Nublado ({nubosidad_media:.0f}% de cobertura)"

    return nivel, desc


def resumen_diario(df: pd.DataFrame) -> list[dict]:
    """
    Genera un resumen por día a partir del DataFrame a 15 min.

    Para estadísticas puntuales (temperatura, precipitación) se usan solo las
    filas correspondientes a horas exactas, evitando sobre-ponderar los puntos
    interpolados.

    Retorna
    -------
    Lista de dicts con claves:
        fecha, nivel, emoji, label, color, descripcion,
        generacion_kwh, nubosidad_pct, temp_max_c, precipitacion_mm
    """
    df = df.copy()
    df["_dia"] = df["Fecha_Hora"].dt.date
    df_h = df[df["Fecha_Hora"].dt.minute == 0].copy()   # filas de hora en punto

    resultado = []
    for dia, g_15 in df.groupby("_dia"):
        g_h = df_h[df_h["_dia"] == dia]

        nub_media   = float(g_h["Nubosidad_pct"].mean()) if not g_h.empty else 0.0
        nivel, desc = _nivel_maximo(g_h["Codigo_Clima"], nub_media)
        cfg         = NIVEL_CONFIG[nivel]

        resultado.append({
            "fecha":            dia,
            "nivel":            nivel,
            "emoji":            cfg["emoji"],
            "label":            cfg["label"],
            "color":            cfg["color"],
            "descripcion":      desc,
            "generacion_kwh":   float(g_15["Generacion_Solar_kW"].sum() * 0.25),
            "nubosidad_pct":    nub_media,
            "temp_max_c":       float(g_h["Temp_Aire_C"].max()) if not g_h.empty else float("nan"),
            "precipitacion_mm": float(g_h["Precipitacion_mm"].sum()) if not g_h.empty else 0.0,
        })
    return resultado