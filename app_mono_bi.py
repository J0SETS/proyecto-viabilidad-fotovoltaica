# =============================================================================
#  app.py — Análisis de Viabilidad Energética Fotovoltaica
#
#  Archivo único: contiene el motor de simulación (pvlib) y la interfaz
#  Streamlit en un solo módulo. Sin dependencias externas adicionales.
#
#  Dependencias: streamlit, pvlib >= 0.9, pandas >= 1.5, numpy, plotly
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
#  SECCIÓN 1 — IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import traceback
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pvlib
import streamlit as st
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS


# ─────────────────────────────────────────────────────────────────────────────
#  SECCIÓN 2 — CATÁLOGO DE PANELES
#
#  Centraliza todos los parámetros de cada módulo. Agregar un nuevo panel
#  solo requiere añadir una entrada aquí; el motor y la UI lo leen dinámicamente.
# ─────────────────────────────────────────────────────────────────────────────
PANELES = {
    "Monofacial": {
        "label":            "Monofacial",
        "fabricante":       "Jinko Solar",
        "modelo_corto":     "JKM605N-72HL4",
        "modelo_completo":  "Tiger Neo 72HC (JKM605N-72HL4)",
        "potencia_w":       605,
        "potencia_pico_dc": 399_300.0,   # Wp — potencia total del array
        "gamma_pdc":        -0.0029,     # Coeficiente de temperatura (1/°C) = −0.29%/°C
        "bifacial":         False,
        # Datos técnicos de referencia (no usados en pvwatts_dc)
        # η = 23.42 %  |  A_unitario = 2.583 m²  |  P_STC_unitario = 605 W
    },
    "Bifacial": {
        "label":            "Bifacial",
        "fabricante":       "Jinko Solar",
        "modelo_corto":     "JKM625N-78HL4-BDV",
        "modelo_completo":  "Tiger Neo N-type (JKM625N-78HL4-BDV)",
        "potencia_w":       625,
        "potencia_pico_dc": 399_300.0,
        "gamma_pdc":        -0.0029,
        "bifacial":         True,
        "gb_default":       0.15,
        # Datos técnicos de referencia (no usados en pvwatts_dc)
        # η = 22.36 %  |  A_unitario = 2.795 m²  |  P_STC_unitario = 625 W
    },
}


# ─────────────────────────────────────────────────────────────────────────────
#  SECCIÓN 3 — MOTOR DE SIMULACIÓN FOTOVOLTAICA
# ─────────────────────────────────────────────────────────────────────────────
def calcular_viabilidad(
    lat: float,
    lon: float,
    altura: float,
    df_demanda: pd.DataFrame,
    tipo_panel: str = "Monofacial",
    gb: float = 0.0,
):
    """
    Simula un año completo de generación solar y su impacto en la curva de demanda.

    Parámetros
    ----------
    lat, lon    : float          Coordenadas geográficas del sitio.
    altura      : float          Altitud sobre el nivel del mar en metros.
    df_demanda  : pd.DataFrame   DataFrame con una columna numérica de demanda (kW).
    tipo_panel  : str            Clave del diccionario PANELES: 'Monofacial' | 'Bifacial'.
    gb          : float          Ganancia trasera bifacial como fracción decimal (ej. 0.15).
                                 Solo se aplica cuando tipo_panel == 'Bifacial'.

    Retorna
    -------
    df_motor      : pd.DataFrame con columnas:
                    ['Fecha_Hora', 'Demanda_kW', 'Gtot_POA_Wm2',
                     'Generacion_Solar_kW', 'Demanda_Post_Inyeccion_Solar_kW']
    energia_anual : float  kWh solares generados en el año simulado.
    """

    if tipo_panel not in PANELES:
        raise ValueError(
            f"tipo_panel='{tipo_panel}' no reconocido. "
            f"Opciones válidas: {list(PANELES.keys())}"
        )
    panel = PANELES[tipo_panel]

    # ── 3.1  Índice temporal: año completo a intervalos de 15 min ─────────────
    tz = "America/Mexico_City"
    tiempos = pd.date_range(
        start="2026-12-21 00:00",
        end="2027-12-20 23:45",
        freq="15min",
        tz=tz,
    )
    tiempos_naive = tiempos.tz_localize(None)

    # ── 3.2  Posición solar ────────────────────────────────────────────────────
    sol = pvlib.solarposition.get_solarposition(tiempos, lat, lon)

    # ── 3.3  Irradiancia en cielo despejado (modelo Ineichen) ─────────────────
    airmass     = pvlib.atmosphere.get_relative_airmass(sol["apparent_zenith"])
    airmass_abs = pvlib.atmosphere.get_absolute_airmass(
        airmass, pvlib.atmosphere.alt2pres(altura)
    )
    clearsky = pvlib.clearsky.ineichen(
        sol["apparent_zenith"],
        airmass_absolute=airmass_abs,
        linke_turbidity=3,
        altitude=altura,
    )
    irradiance_poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=25,
        surface_azimuth=180,
        solar_zenith=sol["apparent_zenith"],
        solar_azimuth=sol["azimuth"],
        dni=clearsky["dni"],
        ghi=clearsky["ghi"],
        dhi=clearsky["dhi"],
    )

    # ── 3.4  DataFrame base ────────────────────────────────────────────────────
    df_motor = pd.DataFrame(index=tiempos_naive)
    df_motor["Gtot_POA_Wm2"] = irradiance_poa["poa_global"].values

    # ── 3.5  Alineación de la curva de demanda ────────────────────────────────
    try:
        if isinstance(df_demanda, str):
            df_demanda = pd.read_csv(df_demanda, index_col=0, parse_dates=True)

        col = "Demanda_kW" if "Demanda_kW" in df_demanda.columns else df_demanda.columns[0]
        serie = pd.to_numeric(df_demanda[col], errors="coerce").fillna(0).values

        if len(serie) == len(tiempos_naive):
            df_motor["Demanda_kW"] = serie
        else:
            s_resampled = (
                pd.Series(serie)
                .reindex(pd.RangeIndex(len(tiempos_naive)))
                .interpolate(method="linear")
                .bfill()
                .ffill()
            )
            df_motor["Demanda_kW"] = s_resampled.values

    except Exception:
        rng = np.random.default_rng(seed=42)
        df_motor["Demanda_kW"] = rng.uniform(150, 300, len(tiempos_naive))

    # ── 3.6  Modelado energético ───────────────────────────────────────────────
    #
    #  Para el panel BIFACIAL se escala la potencia pico DC efectiva antes de
    #  pasarla a pvwatts_dc, aplicando:
    #
    #      P_ef = P_STC × (1 + G_b)
    #
    #  donde G_b es la ganancia trasera configurada por el usuario (fracción decimal).
    #  Para el MONOFACIAL, G_b = 0  →  P_ef == P_STC (sin cambio).
    # ──────────────────────────────────────────────────────────────────────────
    potencia_base = panel["potencia_pico_dc"]
    gamma_pdc     = panel["gamma_pdc"]

    if panel["bifacial"]:
        gb_ef       = float(np.clip(gb, 0.0, 1.0))
        potencia_ef = potencia_base * (1.0 + gb_ef)
    else:
        potencia_ef = potencia_base

    temp_params = TEMPERATURE_MODEL_PARAMETERS["sapm"]["open_rack_glass_glass"]
    temp_celda  = pvlib.temperature.sapm_cell(
        poa_global=df_motor["Gtot_POA_Wm2"],
        temp_air=20.0,
        wind_speed=1.5,
        **temp_params,
    )
    pdc = pvlib.pvsystem.pvwatts_dc(
        effective_irradiance=df_motor["Gtot_POA_Wm2"],
        temp_cell=temp_celda,
        pdc0=potencia_ef,
        gamma_pdc=gamma_pdc,
    )

    df_motor["Generacion_Solar_kW"] = (pdc / 1_000.0).clip(lower=0)
    df_motor["Demanda_Post_Inyeccion_Solar_kW"] = (
        df_motor["Demanda_kW"] - df_motor["Generacion_Solar_kW"]
    ).clip(lower=0)

    # ── 3.7  Energía anual generada (kWh = kW × 0.25 h por intervalo de 15 min)
    energia_anual = float((df_motor["Generacion_Solar_kW"] * 0.25).sum())

    # ── 3.8  DataFrame final ──────────────────────────────────────────────────
    df_motor.index.name = "Fecha_Hora"
    df_motor.reset_index(inplace=True)
    df_motor = df_motor[[
        "Fecha_Hora",
        "Demanda_kW",
        "Gtot_POA_Wm2",
        "Generacion_Solar_kW",
        "Demanda_Post_Inyeccion_Solar_kW",
    ]]
    return df_motor, energia_anual


# ─────────────────────────────────────────────────────────────────────────────
#  SECCIÓN 4 — HELPERS DE INTERFAZ
# ─────────────────────────────────────────────────────────────────────────────
def _etiqueta_superficie(pct: int) -> str:
    """Devuelve el texto de entorno según el porcentaje de ganancia bifacial."""
    if pct <= 10:
        return "🌱  Suelo oscuro / Tierra o Techo asfáltico convencional"
    elif pct <= 15:
        return "🏢  Pasto / Techo industrial gris estándar"
    elif pct <= 20:
        return "🚗  Concreto claro / Estacionamientos (Carport)"
    else:
        return "☀️  Superficie blanca altamente reflectiva (Máximo rendimiento)"


def _make_fig(title: str = "") -> go.Figure:
    """Figura Plotly con el tema oscuro corporativo ya aplicado."""
    _PLOT_BG   = "#12171f"
    _GRID_COL  = "#1e2535"
    _AXIS_COL  = "#2a3040"
    _FONT_COL  = "#8892a4"
    _LABEL_COL = "#c8bfae"

    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=_PLOT_BG,
        font=dict(family="IBM Plex Mono", color=_FONT_COL, size=11),
        xaxis=dict(gridcolor=_GRID_COL, linecolor=_AXIS_COL,
                   tickfont=dict(color=_FONT_COL), title_font=dict(color=_LABEL_COL)),
        yaxis=dict(gridcolor=_GRID_COL, linecolor=_AXIS_COL,
                   tickfont=dict(color=_FONT_COL), title_font=dict(color=_LABEL_COL)),
        hovermode="x unified",
        margin=dict(l=55, r=20, t=40, b=50),
        height=300,
        hoverlabel=dict(bgcolor="#1e2535", bordercolor="#3a4a5c",
                        font=dict(family="IBM Plex Mono", color="#e8e0d0", size=11)),
        title=dict(text=title, font=dict(color=_LABEL_COL, size=12), x=0),
    )
    return fig


def _filtrar_rango(df: pd.DataFrame, desde, hasta) -> pd.DataFrame:
    """Filtra el DataFrame al rango de fechas (ambos extremos inclusivos)."""
    ts_desde = pd.Timestamp(desde)
    ts_hasta = pd.Timestamp(hasta) + timedelta(days=1) - timedelta(seconds=1)
    return df.loc[(df["Fecha_Hora"] >= ts_desde) & (df["Fecha_Hora"] <= ts_hasta)]


# ─────────────────────────────────────────────────────────────────────────────
#  SECCIÓN 5 — CONFIGURACIÓN STREAMLIT
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Análisis Fotovoltaico",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"]  { font-family: 'IBM Plex Sans', sans-serif; }
    .stApp                      { background-color: #0f1117; color: #e8e0d0; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #161b27;
        border-right: 1px solid #2a3040;
    }
    [data-testid="stSidebar"] .stMarkdown p {
        color: #8892a4; font-size: 0.78rem;
        letter-spacing: 0.06em; text-transform: uppercase;
    }

    /* ── Títulos ── */
    h1 {
        font-family: 'IBM Plex Mono', monospace !important;
        color: #f5a623 !important; letter-spacing: -0.02em; font-size: 1.8rem !important;
    }
    h3 {
        font-family: 'IBM Plex Mono', monospace !important;
        color: #c8bfae !important; font-size: 0.85rem !important;
        letter-spacing: 0.12em; text-transform: uppercase;
        border-bottom: 1px solid #2a3040; padding-bottom: 6px; margin-top: 2rem !important;
    }

    /* ── Métricas ── */
    [data-testid="metric-container"] {
        background-color: #161b27; border: 1px solid #2a3040;
        border-radius: 8px; padding: 18px 20px;
    }
    [data-testid="metric-container"] label {
        color: #8892a4 !important; font-size: 0.72rem !important;
        text-transform: uppercase; letter-spacing: 0.1em;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #f5a623 !important; font-family: 'IBM Plex Mono', monospace;
        font-size: 1.6rem !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricDelta"] {
        color: #4ecdc4 !important; font-size: 0.8rem !important;
    }

    /* ── Botón principal ── */
    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #f5a623, #e8860d);
        color: #0f1117; font-family: 'IBM Plex Mono', monospace;
        font-weight: 600; font-size: 0.82rem; letter-spacing: 0.08em;
        text-transform: uppercase; border: none; border-radius: 6px;
        padding: 14px 0; width: 100%; transition: opacity 0.2s;
    }
    div[data-testid="stButton"] > button:hover { opacity: 0.85; }

    /* ── Inputs ── */
    [data-testid="stNumberInput"] input,
    [data-testid="stDateInput"]   input {
        background-color: #1e2535; border: 1px solid #2a3040;
        color: #e8e0d0; border-radius: 5px;
    }
    [data-testid="stNumberInput"] label,
    [data-testid="stDateInput"]   label { color: #8892a4 !important; font-size: 0.78rem !important; }

    [data-testid="stFileUploader"] {
        background-color: #1e2535; border: 1px dashed #3a4a5c;
        border-radius: 8px; padding: 10px;
    }
    [data-testid="stFileUploader"] label { color: #8892a4 !important; font-size: 0.78rem !important; }

    hr { border-color: #2a3040; }

    [data-testid="stAlert"] {
        background-color: #1e2535; border-radius: 6px;
        border-left: 3px solid #f5a623; color: #c8bfae; font-size: 0.83rem;
    }

    /* ── Ficha de panel ── */
    .panel-card {
        background: #1a2235;
        border: 1px solid #2a3a50;
        border-left: 3px solid #f5a623;
        border-radius: 8px;
        padding: 14px 16px;
        margin-top: 8px;
        font-family: 'IBM Plex Sans', sans-serif;
        line-height: 1.8;
    }
    .pc-tipo   { color: #f5a623; font-weight: 600; font-size: 0.70rem;
                 letter-spacing: 0.12em; text-transform: uppercase; }
    .pc-fabr   { color: #8892a4; font-size: 0.78rem; margin-top: 2px; }
    .pc-modelo { color: #e8e0d0; font-size: 0.85rem; font-weight: 600; margin-bottom: 8px; }
    .pc-tag    { display: inline-block; background: #0f1117;
                 border: 1px solid #2a3a50; border-radius: 4px;
                 padding: 3px 10px; font-size: 0.72rem;
                 color: #4ecdc4; font-family: 'IBM Plex Mono', monospace; }

    /* ── Etiqueta de superficie bajo el slider ── */
    .superficie-label {
        margin-top: 6px; padding: 9px 13px;
        background: #1a2235; border-radius: 6px;
        border-left: 3px solid #4ecdc4;
        font-size: 0.80rem; color: #c8bfae;
        font-family: 'IBM Plex Sans', sans-serif;
    }

    /* ── Badge de panel simulado ── */
    .panel-badge {
        display: inline-block;
        background: #1a2235;
        border: 1px solid #2a3a50;
        border-left: 3px solid #4ecdc4;
        border-radius: 6px;
        padding: 6px 14px;
        font-size: 0.80rem;
        color: #4ecdc4;
        font-family: 'IBM Plex Mono', monospace;
        margin-bottom: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  SECCIÓN 6 — SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
for _key, _default in [
    ("df_motor",      None),
    ("energia_anual", None),
    ("sim_ok",        False),
    ("panel_usado",   "Monofacial"),
    ("gb_usado",      0.0),
]:
    if _key not in st.session_state:
        st.session_state[_key] = _default


# ─────────────────────────────────────────────────────────────────────────────
#  SECCIÓN 7 — TÍTULO PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# ☀️ Análisis de Viabilidad Energética Fotovoltaica")
st.markdown(
    "<p style='color:#8892a4;font-size:0.82rem;margin-top:-12px;"
    "font-family:IBM Plex Mono;'>"
    "Simulación anual · Resolución 15 min · Motor pvlib</p>",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
#  SECCIÓN 8 — BARRA LATERAL
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:

    # ── Ubicación ────────────────────────────────────────────────────────────
    st.markdown("### Ubicación del Sistema")

    latitud = st.number_input(
        "Latitud (°)", min_value=-90.0, max_value=90.0,
        value=25.6866, step=0.0001, format="%.4f",
        help="Rango: −90 (polo sur) a 90 (polo norte).",
    )
    longitud = st.number_input(
        "Longitud (°)", min_value=-180.0, max_value=180.0,
        value=-100.3161, step=0.0001, format="%.4f",
        help="Rango: −180 a 180.",
    )
    altura = st.number_input(
        "Altitud sobre nivel del mar (m)", min_value=0.0,
        value=538.0, step=1.0, format="%.0f",
        help="Altitud del sitio de instalación en metros sobre el nivel del mar.",
    )

    st.markdown("---")

    # ── Selector de panel ─────────────────────────────────────────────────────
    # key único y explícito → resuelve StreamlitDuplicateElementId
    st.markdown("### Panel Solar")

    tipo_panel = st.selectbox(
        "Tipo de módulo",
        options=list(PANELES.keys()),
        format_func=lambda k: PANELES[k]["label"],
        key="select_tipo_panel_bifacial_unique",
    )

    p = PANELES[tipo_panel]

    # ── Ficha ejecutiva (sin datos eléctricos) ────────────────────────────────
    st.markdown(
        f"<div class='panel-card'>"
        f"  <div class='pc-tipo'>{p['label']}</div>"
        f"  <div class='pc-fabr'>{p['fabricante']}</div>"
        f"  <div class='pc-modelo'>{p['modelo_completo']}</div>"
        f"  <span class='pc-tag'>{p['potencia_w']} W</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Slider de superficie (solo bifacial) ──────────────────────────────────
    if tipo_panel == "Bifacial":
        st.markdown("<div style='margin-top:14px'></div>", unsafe_allow_html=True)

        gb_pct = st.slider(
            "Tipo de Superficie (Beneficio Bifacial %)",
            min_value=5,
            max_value=25,
            value=15,
            step=1,
            format="%d%%",
            key="slider_gb_bifacial_unique",
            help="El beneficio adicional de la cara trasera depende del albedo del suelo.",
        )
        gb = gb_pct / 100.0

        st.markdown(
            f"<div class='superficie-label'>{_etiqueta_superficie(gb_pct)}</div>",
            unsafe_allow_html=True,
        )
    else:
        gb     = 0.0
        gb_pct = 0

    st.markdown("---")

    # ── Carga del CSV de demanda ───────────────────────────────────────────────
    st.markdown("### Curva de Demanda")

    archivo_csv = st.file_uploader(
        "Sube el CSV de demanda de la empresa",
        type=["csv", "txt"],
        key="uploader_demanda_unique",
        help=(
            "Columna numérica de potencia (kW). "
            "Idealmente 35 040 filas (año completo a 15 min). "
            "El motor interpola automáticamente si el tamaño difiere."
        ),
    )

    df_cargado = None
    if archivo_csv is not None:
        try:
            df_cargado = pd.read_csv(archivo_csv)
            st.success(
                f"✓ {archivo_csv.name}  ·  "
                f"{df_cargado.shape[0]:,} filas × {df_cargado.shape[1]} cols"
            )
        except Exception as e:
            st.error(f"Error al leer el CSV: {e}")

    st.markdown("---")
    boton_ejecutar = st.button("▶ Ejecutar Simulación", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
#  SECCIÓN 9 — EJECUCIÓN DE LA SIMULACIÓN
# ─────────────────────────────────────────────────────────────────────────────
if boton_ejecutar:
    if df_cargado is None:
        st.warning("Sube un archivo CSV de demanda antes de ejecutar la simulación.")
    else:
        with st.spinner("Ejecutando simulación anual — puede tardar unos segundos…"):
            try:
                df_motor, energia_anual = calcular_viabilidad(
                    lat=latitud,
                    lon=longitud,
                    altura=altura,
                    df_demanda=df_cargado,
                    tipo_panel=tipo_panel,
                    gb=gb,
                )
                df_motor["Fecha_Hora"] = pd.to_datetime(df_motor["Fecha_Hora"])

                st.session_state["df_motor"]      = df_motor
                st.session_state["energia_anual"] = energia_anual
                st.session_state["sim_ok"]        = True
                st.session_state["panel_usado"]   = tipo_panel
                st.session_state["gb_usado"]      = gb

            except Exception as e:
                st.session_state["sim_ok"] = False
                st.error(f"Error en el motor de cálculo: {e}")
                with st.expander("Ver detalles del error"):
                    st.code(traceback.format_exc(), language="python")


# ─────────────────────────────────────────────────────────────────────────────
#  SECCIÓN 10 — RESULTADOS
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state["sim_ok"] and st.session_state["df_motor"] is not None:

    df_motor: pd.DataFrame = st.session_state["df_motor"]
    energia_anual: float   = st.session_state["energia_anual"]
    panel_usado: str       = st.session_state["panel_usado"]
    gb_usado: float        = st.session_state["gb_usado"]
    p_usado                = PANELES[panel_usado]

    # ── Badge "Panel simulado" ────────────────────────────────────────────────
    if panel_usado == "Bifacial":
        badge_txt = (
            f"Panel simulado: Bifacial — Jinko Solar {p_usado['modelo_corto']}"
            f"  ·  Superficie: {gb_usado * 100:.0f}%"
        )
    else:
        badge_txt = (
            f"Panel simulado: Monofacial — Jinko Solar {p_usado['modelo_corto']}"
        )

    st.markdown(
        f"<div class='panel-badge'>{badge_txt}</div>",
        unsafe_allow_html=True,
    )

    fecha_min = df_motor["Fecha_Hora"].min().date()
    fecha_max = df_motor["Fecha_Hora"].max().date()

    # ── Filtro temporal ───────────────────────────────────────────────────────
    st.markdown("### Ventana de Visualización")

    col_fi, col_ff = st.columns(2)
    with col_fi:
        fecha_inicio = st.date_input(
            "Desde", value=fecha_min,
            min_value=fecha_min, max_value=fecha_max,
            key="date_desde",
        )
    with col_ff:
        fecha_fin = st.date_input(
            "Hasta",
            value=min(fecha_min + timedelta(days=6), fecha_max),
            min_value=fecha_min, max_value=fecha_max,
            key="date_hasta",
        )

    if fecha_inicio > fecha_fin:
        st.warning("La fecha de inicio no puede ser posterior a la de fin.")
        st.stop()

    df_vis = _filtrar_rango(df_motor, fecha_inicio, fecha_fin)
    if df_vis.empty:
        st.warning("No hay datos en el rango seleccionado.")
        st.stop()

    # ── Indicadores anuales ───────────────────────────────────────────────────
    st.markdown("### Indicadores Anuales")

    pico_kw      = float(df_motor["Generacion_Solar_kW"].max())
    dem_orig_kwh = float(df_motor["Demanda_kW"].sum() * 0.25)
    dem_post_kwh = float(df_motor["Demanda_Post_Inyeccion_Solar_kW"].sum() * 0.25)
    ahorro_kwh   = dem_orig_kwh - dem_post_kwh
    pct_ahorro   = (ahorro_kwh / dem_orig_kwh * 100) if dem_orig_kwh > 0 else 0.0
    hsp          = energia_anual / pico_kw if pico_kw > 0 else 0.0

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Energía Solar Generada",  f"{energia_anual:,.0f} kWh", delta="Anual")
    with m2:
        st.metric("Pico Máximo",             f"{pico_kw:,.1f} kW",        delta="Instantáneo")
    with m3:
        st.metric("Energía Ahorrada en Red", f"{ahorro_kwh:,.0f} kWh",
                  delta=f"{pct_ahorro:.1f}% de la demanda")
    with m4:
        st.metric("Horas Pico Equivalentes", f"{hsp:,.0f} h/año",         delta="HSP anuales")

    st.markdown("---")

    # ── Gráfica 1: Irradiancia POA ────────────────────────────────────────────
    st.markdown("### Irradiancia Global en el Plano del Array  ·  Gtot POA (W/m²)")

    fig1 = _make_fig()
    fig1.add_trace(go.Scatter(
        x=df_vis["Fecha_Hora"], y=df_vis["Gtot_POA_Wm2"],
        name="Irradiancia POA", mode="lines",
        line=dict(color="#f5a623", width=1.5),
        hovertemplate="%{x|%d %b %H:%M}<br><b>%{y:.1f} W/m²</b><extra></extra>",
    ))
    fig1.update_layout(yaxis_title="W/m²", xaxis_title="")
    st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})

    # ── Gráfica 2: Generación solar ───────────────────────────────────────────
    st.markdown("### Generación Solar  ·  (kW)")

    fig2 = _make_fig()
    fig2.add_trace(go.Scatter(
        x=df_vis["Fecha_Hora"], y=df_vis["Generacion_Solar_kW"],
        name="Generación Solar", mode="lines", fill="tozeroy",
        line=dict(color="#ffe033", width=1.5),
        fillcolor="rgba(255,224,51,0.12)",
        hovertemplate="%{x|%d %b %H:%M}<br><b>%{y:.2f} kW</b><extra></extra>",
    ))
    fig2.update_layout(yaxis_title="kW", xaxis_title="")
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    # ── Gráfica 3: Comparativa de carga ──────────────────────────────────────
    st.markdown("### Comparativa de Carga  ·  Demanda Original vs. Post-Inyección (kW)")

    fig3 = _make_fig()
    fig3.add_trace(go.Scatter(
        x=df_vis["Fecha_Hora"], y=df_vis["Demanda_Post_Inyeccion_Solar_kW"],
        name="Post-Inyección Solar", mode="lines", fill="tozeroy",
        line=dict(color="#4ecdc4", width=1.8),
        fillcolor="rgba(78,205,196,0.12)",
        hovertemplate="%{x|%d %b %H:%M}<br>Post-Inyección: <b>%{y:.2f} kW</b><extra></extra>",
    ))
    fig3.add_trace(go.Scatter(
        x=df_vis["Fecha_Hora"], y=df_vis["Demanda_kW"],
        name="Demanda Original", mode="lines",
        line=dict(color="#ff6b6b", width=1.8, dash="dot"),
        hovertemplate="%{x|%d %b %H:%M}<br>Demanda: <b>%{y:.2f} kW</b><extra></extra>",
    ))
    fig3.update_layout(
        yaxis_title="kW", xaxis_title="Fecha / Hora",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(color="#c8bfae", size=11), bgcolor="rgba(0,0,0,0)",
        ),
    )
    st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    # ── Tabla y descarga ───────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("Ver datos tabulares del período seleccionado"):
        st.dataframe(df_vis.reset_index(drop=True), use_container_width=True, height=280)

    st.download_button(
        label="⬇ Descargar resultados anuales completos (CSV)",
        data=df_motor.to_csv(index=False).encode("utf-8"),
        file_name="resultados_fotovoltaicos_anuales.csv",
        mime="text/csv",
    )

# ─────────────────────────────────────────────────────────────────────────────
#  ESTADO INICIAL — antes de la primera simulación
# ─────────────────────────────────────────────────────────────────────────────
else:
    st.info(
        "Configura la ubicación y sube tu curva de demanda en la barra lateral. "
        "Luego presiona **▶ Ejecutar Simulación** para ver los resultados."
    )