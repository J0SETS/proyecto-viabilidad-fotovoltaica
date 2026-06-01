"""Interfaz Streamlit para el motor solar-industrial."""

from __future__ import annotations

import hashlib
import traceback
from datetime import date, timedelta
from io import BytesIO, StringIO

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analisis_solar_clima import SiteConfig, run_solar_demand_simulation


LOCATIONS_MEXICO = {
    "Consolapan / Xalapa, Veracruz": {
        "name": "Consolapan / Xalapa, Veracruz",
        "lat": 19.54,
        "lon": -96.91,
        "altitude": 1400.0,
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
    "Guadalajara, Jalisco": {
        "name": "Guadalajara, Jalisco",
        "lat": 20.6597,
        "lon": -103.3496,
        "altitude": 1566.0,
        "timezone": "America/Mexico_City",
    },
    "Merida, Yucatan": {
        "name": "Merida, Yucatan",
        "lat": 20.9674,
        "lon": -89.5926,
        "altitude": 10.0,
        "timezone": "America/Merida",
    },
    "Cancun, Quintana Roo": {
        "name": "Cancun, Quintana Roo",
        "lat": 21.1619,
        "lon": -86.8515,
        "altitude": 10.0,
        "timezone": "America/Cancun",
    },
    "Tijuana, Baja California": {
        "name": "Tijuana, Baja California",
        "lat": 32.5149,
        "lon": -117.0382,
        "altitude": 20.0,
        "timezone": "America/Tijuana",
    },
}

PANELES_UI = {
    "Monofacial": {
        "key": "monofacial",
        "label": "Monofacial",
        "fabricante": "Jinko Solar",
        "modelo_corto": "JKM605N-72HL4",
        "modelo_completo": "Tiger Neo 72HC (JKM605N-72HL4)",
        "potencia_w": 605,
    },
    "Bifacial": {
        "key": "bifacial",
        "label": "Bifacial",
        "fabricante": "Jinko Solar",
        "modelo_corto": "JKM625N-78HL4-BDV",
        "modelo_completo": "Tiger Neo N-type (JKM625N-78HL4-BDV)",
        "potencia_w": 625,
    },
}


def _etiqueta_superficie(pct: int) -> str:
    if pct <= 10:
        return "Suelo oscuro / tierra o techo asfaltico"
    if pct <= 15:
        return "Pasto / techo industrial gris"
    if pct <= 20:
        return "Concreto claro / estacionamiento"
    return "Superficie blanca altamente reflectiva"


def _make_fig(title: str = "", height: int = 300) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#12171f",
        font=dict(family="IBM Plex Mono", color="#8892a4", size=11),
        xaxis=dict(
            gridcolor="#1e2535",
            linecolor="#2a3040",
            tickfont=dict(color="#8892a4"),
            title_font=dict(color="#c8bfae"),
        ),
        yaxis=dict(
            gridcolor="#1e2535",
            linecolor="#2a3040",
            tickfont=dict(color="#8892a4"),
            title_font=dict(color="#c8bfae"),
        ),
        hovermode="x unified",
        margin=dict(l=55, r=20, t=40, b=50),
        height=height,
        hoverlabel=dict(
            bgcolor="#1e2535",
            bordercolor="#3a4a5c",
            font=dict(family="IBM Plex Mono", color="#e8e0d0", size=11),
        ),
        title=dict(text=title, font=dict(color="#c8bfae", size=12), x=0),
    )
    return fig


def _filtrar_rango(df: pd.DataFrame, desde: date, hasta: date) -> pd.DataFrame:
    ts_desde = pd.Timestamp(desde)
    ts_hasta = pd.Timestamp(hasta) + timedelta(days=1) - timedelta(seconds=1)
    return df.loc[(df["Fecha_Hora"] >= ts_desde) & (df["Fecha_Hora"] <= ts_hasta)]


def _read_uploaded_demand(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(BytesIO(raw))
    return pd.read_excel(BytesIO(raw))


def _prepare_result_for_ui(result: pd.DataFrame) -> pd.DataFrame:
    result = result.copy()
    result.index = result.index.tz_localize(None)
    result.index.name = "Fecha_Hora"
    return result.reset_index().rename(
        columns={
            "ghi": "GHI_Wm2",
            "dni": "DNI_Wm2",
            "dhi": "DHI_Wm2",
            "poa_global": "Gtot_POA_Wm2",
            "temperature_2m": "Temperatura_Ambiente_C",
            "wind_speed_10m": "Velocidad_Viento_ms",
        }
    )


@st.cache_data(show_spinner=False)
def _run_cached_simulation(
    demand_csv: str,
    demand_hash: str,
    site_name: str,
    latitude: float,
    longitude: float,
    altitude_m: float,
    timezone: str,
    surface_tilt: float,
    surface_azimuth: float,
    albedo: float,
    weather_start_date: date,
    weather_end_date: date,
    module_key: str,
    bifacial_gain: float,
    system_dc_kwp: float,
    inverter_efficiency: float,
    system_losses: float,
    align_demand_by_position: bool,
) -> dict[str, object]:
    del demand_hash
    demand_df = pd.read_csv(StringIO(demand_csv))
    site = SiteConfig(
        name=site_name,
        latitude=latitude,
        longitude=longitude,
        timezone=timezone,
        altitude_m=altitude_m,
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        albedo=albedo,
    )
    return run_solar_demand_simulation(
        site=site,
        demand_df=demand_df,
        weather_start_date=weather_start_date,
        weather_end_date=weather_end_date,
        module_key=module_key,
        bifacial_gain=bifacial_gain,
        system_dc_kwp=system_dc_kwp,
        inverter_efficiency=inverter_efficiency,
        system_losses=system_losses,
        align_demand_by_position=align_demand_by_position,
        write_files=False,
    )


st.set_page_config(
    page_title="Analisis Fotovoltaico Industrial",
    page_icon="☀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .stApp { background-color: #0f1117; color: #e8e0d0; }
    [data-testid="stSidebar"] { background-color: #161b27; border-right: 1px solid #2a3040; }
    [data-testid="stSidebar"] .stMarkdown p { color: #8892a4; font-size: 0.78rem; }
    h1 { font-family: 'IBM Plex Mono', monospace !important; color: #f5a623 !important; font-size: 1.8rem !important; }
    h3 {
        font-family: 'IBM Plex Mono', monospace !important; color: #c8bfae !important;
        font-size: 0.85rem !important; letter-spacing: 0.12em; text-transform: uppercase;
        border-bottom: 1px solid #2a3040; padding-bottom: 6px; margin-top: 2rem !important;
    }
    [data-testid="metric-container"] {
        background-color: #161b27; border: 1px solid #2a3040; border-radius: 8px; padding: 18px 20px;
    }
    [data-testid="metric-container"] label {
        color: #8892a4 !important; font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.08em;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #f5a623 !important; font-family: 'IBM Plex Mono', monospace; font-size: 1.35rem !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricDelta"] { color: #4ecdc4 !important; font-size: 0.8rem !important; }
    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #f5a623, #e8860d); color: #0f1117;
        font-family: 'IBM Plex Mono', monospace; font-weight: 600; font-size: 0.82rem;
        text-transform: uppercase; border: none; border-radius: 6px; padding: 14px 0; width: 100%;
    }
    [data-testid="stNumberInput"] input, [data-testid="stDateInput"] input {
        background-color: #1e2535; border: 1px solid #2a3040; color: #e8e0d0; border-radius: 5px;
    }
    [data-testid="stFileUploader"] {
        background-color: #1e2535; border: 1px dashed #3a4a5c; border-radius: 8px; padding: 10px;
    }
    hr { border-color: #2a3040; }
    [data-testid="stAlert"] {
        background-color: #1e2535; border-radius: 6px; border-left: 3px solid #f5a623;
        color: #c8bfae; font-size: 0.83rem;
    }
    .panel-card, .panel-badge, .superficie-label {
        background: #1a2235; border: 1px solid #2a3a50; border-radius: 8px; padding: 12px 14px; margin-top: 8px;
    }
    .panel-card { border-left: 3px solid #f5a623; line-height: 1.8; }
    .panel-badge { display: inline-block; border-left: 3px solid #4ecdc4; color: #4ecdc4; font-family: 'IBM Plex Mono'; margin-bottom: 1rem; }
    .superficie-label { border-left: 3px solid #4ecdc4; color: #c8bfae; font-size: 0.80rem; }
    .pc-tipo { color: #f5a623; font-weight: 600; font-size: 0.70rem; letter-spacing: 0.12em; text-transform: uppercase; }
    .pc-fabr { color: #8892a4; font-size: 0.78rem; }
    .pc-modelo { color: #e8e0d0; font-size: 0.85rem; font-weight: 600; margin-bottom: 8px; }
    .pc-tag {
        display: inline-block; background: #0f1117; border: 1px solid #2a3a50; border-radius: 4px;
        padding: 3px 10px; font-size: 0.72rem; color: #4ecdc4; font-family: 'IBM Plex Mono';
    }
</style>
""",
    unsafe_allow_html=True,
)

for key, default in [
    ("simulation", None),
    ("sim_ok", False),
    ("panel_usado", "Bifacial"),
    ("gb_usado", 0.15),
]:
    if key not in st.session_state:
        st.session_state[key] = default

st.markdown("# ☀ Analisis de Viabilidad Energetica Fotovoltaica")
st.markdown(
    "<p style='color:#8892a4;font-size:0.82rem;margin-top:-12px;font-family:IBM Plex Mono;'>"
    "Simulacion industrial anual · Clima historico analogo · Resolucion 15 min</p>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Ubicacion del sistema")
    location_mode = st.radio(
        "Modo de ubicacion",
        ["Ubicacion predefinida", "Coordenadas personalizadas"],
        key="location_mode",
    )

    if location_mode == "Ubicacion predefinida":
        location_label = st.selectbox(
            "Ubicacion en Mexico",
            options=list(LOCATIONS_MEXICO),
            key="predefined_location",
        )
        location = LOCATIONS_MEXICO[location_label]
        site_name = location["name"]
        latitude = location["lat"]
        longitude = location["lon"]
        altitude = location["altitude"]
        timezone = location["timezone"]
        st.caption(
            f"{latitude:.4f}, {longitude:.4f} · {altitude:.0f} m · {timezone}"
        )
    else:
        site_name = st.text_input("Nombre del sitio", value="Sitio industrial")
        latitude = st.number_input("Latitud", value=19.54, step=0.0001, format="%.4f")
        longitude = st.number_input("Longitud", value=-96.91, step=0.0001, format="%.4f")
        altitude = st.number_input("Altitud (m)", min_value=0.0, value=1400.0, step=1.0)
        timezone = st.text_input("Zona horaria IANA", value="America/Mexico_City")
        if not 14 <= latitude <= 33:
            st.warning("La latitud esta fuera del rango habitual de Mexico (14 a 33).")
        if longitude >= 0:
            st.warning("Para Mexico la longitud normalmente debe ser negativa.")

    st.markdown("---")
    st.markdown("### Panel solar")
    panel_label = st.selectbox("Tipo de modulo", options=list(PANELES_UI), key="panel")
    panel_ui = PANELES_UI[panel_label]
    module_key = panel_ui["key"]
    st.markdown(
        f"<div class='panel-card'>"
        f"<div class='pc-tipo'>{panel_ui['label']}</div>"
        f"<div class='pc-fabr'>{panel_ui['fabricante']}</div>"
        f"<div class='pc-modelo'>{panel_ui['modelo_completo']}</div>"
        f"<span class='pc-tag'>{panel_ui['potencia_w']} W</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if module_key == "bifacial":
        bifacial_gain_pct = st.slider(
            "Beneficio bifacial trasero (%)",
            min_value=5,
            max_value=25,
            value=15,
            step=1,
            help="El motor aplica tambien el factor bifacial del modulo: 80%.",
        )
        bifacial_gain = bifacial_gain_pct / 100
        st.markdown(
            f"<div class='superficie-label'>{_etiqueta_superficie(bifacial_gain_pct)}</div>",
            unsafe_allow_html=True,
        )
    else:
        bifacial_gain_pct = 0
        bifacial_gain = 0.0

    st.markdown("---")
    st.markdown("### Sistema FV")
    system_dc_kwp = st.number_input(
        "Potencia DC del sistema (kWp)", min_value=0.1, value=399.3, step=1.0
    )
    inverter_efficiency = st.number_input(
        "Eficiencia del inversor", min_value=0.01, max_value=1.0, value=0.96, step=0.01
    )
    system_losses = st.number_input(
        "Perdidas adicionales", min_value=0.0, max_value=0.99, value=0.0, step=0.01
    )
    surface_tilt = st.number_input(
        "Inclinacion del panel (grados)", min_value=0.0, max_value=90.0, value=19.5, step=0.5
    )
    surface_azimuth = st.number_input(
        "Azimut del panel (grados)", min_value=0.0, max_value=360.0, value=180.0, step=1.0
    )
    albedo = st.number_input(
        "Albedo frontal", min_value=0.0, max_value=1.0, value=0.2, step=0.05
    )

    st.markdown("---")
    with st.expander("Configuracion climatica", expanded=False):
        weather_start_date = st.date_input(
            "Inicio de clima historico", value=date(2024, 12, 21)
        )
        weather_end_date = st.date_input(
            "Fin de clima historico", value=date(2025, 12, 20)
        )
        align_demand_by_position = st.checkbox(
            "Alinear demanda por posicion",
            value=True,
            help="Usa el patron industrial con un ano climatico historico analogo.",
        )
        st.caption(
            "El clima historico analogo evita solicitar fechas futuras a Open-Meteo."
        )

    st.markdown("---")
    st.markdown("### Curva de demanda")
    uploaded_file = st.file_uploader(
        "Sube CSV o XLSX de demanda",
        type=["csv", "xlsx", "xls"],
        key="demand_uploader",
        help="Debe incluir Fecha_Hora y Demanda_kW. Factor_Potencia se conserva si existe.",
    )

    demand_df = None
    if uploaded_file is not None:
        try:
            demand_df = _read_uploaded_demand(uploaded_file)
            st.success(
                f"{uploaded_file.name} · {demand_df.shape[0]:,} filas x {demand_df.shape[1]} columnas"
            )
        except Exception as exc:
            st.error(f"No se pudo leer la demanda: {exc}")

    st.markdown("---")
    run_button = st.button("Ejecutar simulacion", width="stretch")

if run_button:
    if demand_df is None:
        st.warning("Sube una curva de demanda antes de ejecutar la simulacion.")
    elif weather_start_date > weather_end_date:
        st.error("La fecha climatica inicial no puede ser posterior a la final.")
    else:
        demand_csv = demand_df.to_csv(index=False)
        demand_hash = hashlib.sha256(demand_csv.encode("utf-8")).hexdigest()
        with st.spinner("Descargando clima historico y ejecutando simulacion anual..."):
            try:
                simulation = _run_cached_simulation(
                    demand_csv=demand_csv,
                    demand_hash=demand_hash,
                    site_name=site_name,
                    latitude=float(latitude),
                    longitude=float(longitude),
                    altitude_m=float(altitude),
                    timezone=timezone,
                    surface_tilt=float(surface_tilt),
                    surface_azimuth=float(surface_azimuth),
                    albedo=float(albedo),
                    weather_start_date=weather_start_date,
                    weather_end_date=weather_end_date,
                    module_key=module_key,
                    bifacial_gain=float(bifacial_gain),
                    system_dc_kwp=float(system_dc_kwp),
                    inverter_efficiency=float(inverter_efficiency),
                    system_losses=float(system_losses),
                    align_demand_by_position=align_demand_by_position,
                )
                st.session_state["simulation"] = simulation
                st.session_state["sim_ok"] = True
                st.session_state["panel_usado"] = panel_label
                st.session_state["gb_usado"] = bifacial_gain
            except Exception as exc:
                st.session_state["sim_ok"] = False
                st.error(f"Error en el motor solar-industrial: {exc}")
                with st.expander("Ver detalles del error"):
                    st.code(traceback.format_exc(), language="python")

if st.session_state["sim_ok"] and st.session_state["simulation"] is not None:
    simulation = st.session_state["simulation"]
    df_motor = _prepare_result_for_ui(simulation["result"])
    summary = simulation["summary"].copy()
    annual = simulation["annual"].iloc[0]
    panel_usado = st.session_state["panel_usado"]
    gb_usado = st.session_state["gb_usado"]
    panel_ui = PANELES_UI[panel_usado]

    if panel_usado == "Bifacial":
        badge = (
            f"Panel simulado: Bifacial · Jinko Solar {panel_ui['modelo_corto']} · "
            f"Ganancia trasera: {gb_usado * 100:.0f}% · "
            f"Ganancia efectiva: {annual['Ganancia_Bifacial_Efectiva_pct']:.1f}%"
        )
    else:
        badge = f"Panel simulado: Monofacial · Jinko Solar {panel_ui['modelo_corto']}"
    st.markdown(f"<div class='panel-badge'>{badge}</div>", unsafe_allow_html=True)

    fecha_min = df_motor["Fecha_Hora"].min().date()
    fecha_max = df_motor["Fecha_Hora"].max().date()
    st.markdown("### Ventana de visualizacion")
    col_start, col_end = st.columns(2)
    with col_start:
        display_start = st.date_input(
            "Desde",
            value=fecha_min,
            min_value=fecha_min,
            max_value=fecha_max,
            key="display_start",
        )
    with col_end:
        display_end = st.date_input(
            "Hasta",
            value=min(fecha_min + timedelta(days=6), fecha_max),
            min_value=fecha_min,
            max_value=fecha_max,
            key="display_end",
        )

    if display_start > display_end:
        st.warning("La fecha inicial no puede ser posterior a la fecha final.")
        st.stop()
    df_vis = _filtrar_rango(df_motor, display_start, display_end)

    st.markdown("### Indicadores anuales")
    metric_rows = [
        [
            ("Solar AC anual", annual["Energia_Solar_AC_Anual_kWh"], "kWh"),
            ("Demanda anual", annual["Energia_Demandada_Anual_kWh"], "kWh"),
            ("Autoconsumida", annual["Energia_Autoconsumida_Anual_kWh"], "kWh"),
            ("Tomada de red", annual["Energia_Red_Anual_kWh"], "kWh"),
        ],
        [
            ("Excedentes", annual["Energia_Excedente_Anual_kWh"], "kWh"),
            ("Cobertura solar", annual["Cobertura_Solar_pct"], "%"),
            ("Autoconsumo", annual["Autoconsumo_pct"], "%"),
            ("Generacion especifica", annual["Generacion_Especifica_AC_kWh_kWp"], "kWh/kWp"),
        ],
        [
            ("Demanda maxima original", annual["Demanda_Maxima_Original_kW"], "kW"),
            ("Demanda maxima post-solar", annual["Demanda_Maxima_Post_Solar_kW"], "kW"),
            ("Reduccion de demanda maxima", annual["Reduccion_Demanda_Maxima_kW"], "kW"),
            ("Potencia DC instalada", annual["Potencia_DC_Sistema_kWp"], "kWp"),
        ],
    ]
    for row in metric_rows:
        columns = st.columns(4)
        for column, (label, value, unit) in zip(columns, row):
            with column:
                st.metric(label, f"{value:,.1f} {unit}")

    st.markdown("---")
    st.markdown("### Irradiancia global en el plano del array · Gtot POA")
    fig1 = _make_fig()
    fig1.add_trace(
        go.Scatter(
            x=df_vis["Fecha_Hora"],
            y=df_vis["Gtot_POA_Wm2"],
            name="Irradiancia POA",
            mode="lines",
            line=dict(color="#f5a623", width=1.5),
        )
    )
    fig1.update_layout(yaxis_title="W/m2")
    st.plotly_chart(fig1, width="stretch", config={"displayModeBar": False})

    st.markdown("### Generacion solar AC")
    fig2 = _make_fig()
    fig2.add_trace(
        go.Scatter(
            x=df_vis["Fecha_Hora"],
            y=df_vis["Generacion_AC_kW"],
            name="Generacion AC",
            mode="lines",
            fill="tozeroy",
            line=dict(color="#ffe033", width=1.5),
            fillcolor="rgba(255,224,51,0.12)",
        )
    )
    fig2.update_layout(yaxis_title="kW")
    st.plotly_chart(fig2, width="stretch", config={"displayModeBar": False})

    st.markdown("### Comparativa de carga · Demanda original vs post-inyeccion")
    fig3 = _make_fig()
    fig3.add_trace(
        go.Scatter(
            x=df_vis["Fecha_Hora"],
            y=df_vis["Demanda_Post_Inyeccion_Solar_kW"],
            name="Post-inyeccion solar",
            mode="lines",
            fill="tozeroy",
            line=dict(color="#4ecdc4", width=1.8),
            fillcolor="rgba(78,205,196,0.12)",
        )
    )
    fig3.add_trace(
        go.Scatter(
            x=df_vis["Fecha_Hora"],
            y=df_vis["Demanda_kW"],
            name="Demanda original",
            mode="lines",
            line=dict(color="#ff6b6b", width=1.8, dash="dot"),
        )
    )
    fig3.update_layout(yaxis_title="kW")
    st.plotly_chart(fig3, width="stretch", config={"displayModeBar": False})

    st.markdown("### Energia mensual · Demanda, solar, autoconsumo y red")
    fig4 = _make_fig(height=360)
    monthly_colors = {
        "Energia_Demandada_kWh": "#ff6b6b",
        "Energia_Solar_AC_kWh": "#ffe033",
        "Energia_Autoconsumida_kWh": "#4ecdc4",
        "Energia_Red_kWh": "#5b8def",
    }
    monthly_labels = {
        "Energia_Demandada_kWh": "Demanda",
        "Energia_Solar_AC_kWh": "Solar AC",
        "Energia_Autoconsumida_kWh": "Autoconsumo",
        "Energia_Red_kWh": "Red",
    }
    for column, color in monthly_colors.items():
        fig4.add_trace(
            go.Bar(
                x=summary.index,
                y=summary[column],
                name=monthly_labels[column],
                marker_color=color,
            )
        )
    fig4.update_layout(barmode="group", yaxis_title="kWh")
    st.plotly_chart(fig4, width="stretch", config={"displayModeBar": False})

    st.markdown("---")
    with st.expander("Ver datos tabulares del periodo seleccionado"):
        visible_columns = [
            "Fecha_Hora",
            "Demanda_kW",
            "Factor_Potencia",
            "Gtot_POA_Wm2",
            "Generacion_DC_kW",
            "Generacion_AC_kW",
            "Demanda_Post_Inyeccion_Solar_kW",
        ]
        st.dataframe(df_vis[visible_columns], width="stretch", height=280)

    download_columns = df_motor
    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button(
            "Descargar serie 15 min",
            data=download_columns.to_csv(index=False).encode("utf-8"),
            file_name="clima_solar_demanda_15min.csv",
            mime="text/csv",
        )
    with d2:
        st.download_button(
            "Descargar resumen mensual",
            data=summary.to_csv().encode("utf-8"),
            file_name="resumen_mensual.csv",
            mime="text/csv",
        )
    with d3:
        st.download_button(
            "Descargar resumen anual",
            data=simulation["annual"].to_csv(index=False).encode("utf-8"),
            file_name="resumen_anual.csv",
            mime="text/csv",
        )

    st.caption(f"Fuente climatica: {simulation['source_url']}")
else:
    st.info(
        "Configura el sistema y sube una curva de demanda en la barra lateral. "
        "Luego ejecuta la simulacion para ver resultados."
    )
