# =============================================================================
#  app.py — Interfaz de Viabilidad Fotovoltaica
#  Requiere: streamlit, pandas, plotly, motor_calculo.py en el mismo directorio
# =============================================================================

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import timedelta
from motor_calculo import calcular_viabilidad

# -----------------------------------------------------------------------------
# CONFIGURACIÓN GLOBAL DE PÁGINA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Análisis Fotovoltaico",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# ESTILOS CSS — dashboard técnico/industrial
# Paleta: fondo carbón oscuro, acento ámbar solar, tipografía monoespaciada.
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .stApp { background-color: #0f1117; color: #e8e0d0; }

    [data-testid="stSidebar"] {
        background-color: #161b27;
        border-right: 1px solid #2a3040;
    }
    [data-testid="stSidebar"] .stMarkdown p {
        color: #8892a4; font-size: 0.78rem;
        letter-spacing: 0.06em; text-transform: uppercase;
    }

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

    [data-testid="metric-container"] {
        background-color: #161b27; border: 1px solid #2a3040;
        border-radius: 8px; padding: 18px 20px;
    }
    [data-testid="metric-container"] label {
        color: #8892a4 !important; font-size: 0.72rem !important;
        text-transform: uppercase; letter-spacing: 0.1em;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #f5a623 !important; font-family: 'IBM Plex Mono', monospace; font-size: 1.6rem !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricDelta"] {
        color: #4ecdc4 !important; font-size: 0.8rem !important;
    }

    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #f5a623, #e8860d);
        color: #0f1117; font-family: 'IBM Plex Mono', monospace;
        font-weight: 600; font-size: 0.82rem; letter-spacing: 0.08em;
        text-transform: uppercase; border: none; border-radius: 6px;
        padding: 14px 0; width: 100%; transition: opacity 0.2s;
    }
    div[data-testid="stButton"] > button:hover { opacity: 0.85; }

    [data-testid="stNumberInput"] input {
        background-color: #1e2535; border: 1px solid #2a3040; color: #e8e0d0; border-radius: 5px;
    }
    [data-testid="stNumberInput"] label { color: #8892a4 !important; font-size: 0.78rem !important; }
    [data-testid="stDateInput"] input {
        background-color: #1e2535; border: 1px solid #2a3040; color: #e8e0d0; border-radius: 5px;
    }
    [data-testid="stFileUploader"] {
        background-color: #1e2535; border: 1px dashed #3a4a5c; border-radius: 8px; padding: 10px;
    }
    [data-testid="stFileUploader"] label { color: #8892a4 !important; font-size: 0.78rem !important; }
    hr { border-color: #2a3040; }
    [data-testid="stAlert"] {
        background-color: #1e2535; border-radius: 6px;
        border-left: 3px solid #f5a623; color: #c8bfae; font-size: 0.83rem;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONSTANTES DE ESTILO PLOTLY  (reutilizadas en las 3 gráficas)
# =============================================================================
PLOT_BG  = "#12171f"
GRID_COL = "#1e2535"
AXIS_COL = "#2a3040"
FONT_COL = "#8892a4"
LABEL_COL = "#c8bfae"

BASE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=PLOT_BG,
    font=dict(family="IBM Plex Mono", color=FONT_COL, size=11),
    xaxis=dict(gridcolor=GRID_COL, linecolor=AXIS_COL,
               tickfont=dict(color=FONT_COL), title_font=dict(color=LABEL_COL)),
    yaxis=dict(gridcolor=GRID_COL, linecolor=AXIS_COL,
               tickfont=dict(color=FONT_COL), title_font=dict(color=LABEL_COL)),
    hovermode="x unified",
    margin=dict(l=55, r=20, t=40, b=50),
    height=300,
    hoverlabel=dict(bgcolor="#1e2535", bordercolor="#3a4a5c",
                    font=dict(family="IBM Plex Mono", color="#e8e0d0", size=11)),
)

# =============================================================================
# HELPERS
# =============================================================================

def make_fig(title: str = "") -> go.Figure:
    """Devuelve un Figure base con el layout oscuro ya aplicado."""
    fig = go.Figure()
    fig.update_layout(**BASE_LAYOUT,
                      title=dict(text=title, font=dict(color=LABEL_COL, size=12), x=0))
    return fig


def filtrar_rango(df: pd.DataFrame, desde, hasta) -> pd.DataFrame:
    """Filtra el DataFrame al rango de fechas seleccionado (ambos extremos inclusivos)."""
    ts_desde = pd.Timestamp(desde)
    ts_hasta = pd.Timestamp(hasta) + timedelta(days=1) - timedelta(seconds=1)
    return df.loc[(df["Fecha_Hora"] >= ts_desde) & (df["Fecha_Hora"] <= ts_hasta)]

# =============================================================================
# SESSION STATE
# Se persiste el resultado de la simulación para que el filtro de fechas
# no vuelva a llamar al motor de cálculo con cada interacción.
# =============================================================================
for key, default in [("df_motor", None), ("energia_anual", None), ("sim_ok", False)]:
    if key not in st.session_state:
        st.session_state[key] = default

# =============================================================================
# TÍTULO PRINCIPAL
# =============================================================================
st.markdown("# ☀️ Análisis de Viabilidad Energética Fotovoltaica")
st.markdown(
    "<p style='color:#8892a4;font-size:0.82rem;margin-top:-12px;font-family:IBM Plex Mono;'>"
    "Simulación anual · Resolución 15 min · Motor pvlib</p>",
    unsafe_allow_html=True,
)

# =============================================================================
# BARRA LATERAL
# =============================================================================
with st.sidebar:
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
        "Altura del Panel (m)", min_value=0.0,
        value=1.5, step=0.1, format="%.2f",
        help="Altura del panel sobre el nivel del suelo.",
    )

    st.markdown("---")
    st.markdown("### Curva de Demanda")

    archivo_csv = st.file_uploader(
        "Sube el CSV de demanda de la empresa",
        type=["csv", "txt"],
        help="Debe contener una columna numérica de potencia (kW) con 35 040 filas "
             "para un año completo a intervalos de 15 min, o cualquier número de filas "
             "(el motor interpolará automáticamente).",
    )

    # Vista previa compacta del archivo cargado
    df_cargado = None
    if archivo_csv is not None:
        try:
            df_cargado = pd.read_csv(archivo_csv)
            st.success(f"✓ {archivo_csv.name}  ·  {df_cargado.shape[0]:,} filas × {df_cargado.shape[1]} cols")
        except Exception as e:
            st.error(f"Error al leer el CSV: {e}")

    st.markdown("---")
    boton_ejecutar = st.button("▶ Ejecutar Simulación", use_container_width=True)

# =============================================================================
# EJECUCIÓN DE LA SIMULACIÓN
# =============================================================================
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
                )
                # Garantizar tipo datetime (por si el motor lo devuelve como string)
                df_motor["Fecha_Hora"] = pd.to_datetime(df_motor["Fecha_Hora"])

                # Persistir en session_state
                st.session_state["df_motor"]    = df_motor
                st.session_state["energia_anual"] = energia_anual
                st.session_state["sim_ok"]       = True

            except Exception as e:
                st.session_state["sim_ok"] = False
                st.error(f"Error en el motor de cálculo: {e}")
                # Mostrar traceback completo para facilitar depuración
                import traceback
                with st.expander("Ver detalles del error"):
                    st.code(traceback.format_exc(), language="python")

# =============================================================================
# SECCIÓN DE RESULTADOS
# Se renderiza si hay datos válidos en session_state.
# =============================================================================
if st.session_state["sim_ok"] and st.session_state["df_motor"] is not None:

    df_motor: pd.DataFrame = st.session_state["df_motor"]
    energia_anual: float   = st.session_state["energia_anual"]

    fecha_min = df_motor["Fecha_Hora"].min().date()
    fecha_max = df_motor["Fecha_Hora"].max().date()

    # -------------------------------------------------------------------------
    # FILTRO TEMPORAL
    # Por defecto muestra la primera semana para no saturar con 35 040 puntos.
    # -------------------------------------------------------------------------
    st.markdown("### Ventana de Visualización")

    col_fi, col_ff = st.columns(2)
    with col_fi:
        fecha_inicio = st.date_input(
            "Desde", value=fecha_min,
            min_value=fecha_min, max_value=fecha_max,
        )
    with col_ff:
        fecha_fin = st.date_input(
            "Hasta",
            value=min(fecha_min + timedelta(days=6), fecha_max),
            min_value=fecha_min, max_value=fecha_max,
        )

    if fecha_inicio > fecha_fin:
        st.warning("La fecha de inicio no puede ser posterior a la de fin.")
        st.stop()

    df_vis = filtrar_rango(df_motor, fecha_inicio, fecha_fin)
    if df_vis.empty:
        st.warning("No hay datos en el rango seleccionado.")
        st.stop()

    # -------------------------------------------------------------------------
    # MÉTRICAS  (valores anuales completos, no del rango filtrado)
    # -------------------------------------------------------------------------
    st.markdown("### Indicadores Anuales")

    pico_kw       = float(df_motor["Generacion_Solar_kW"].max())
    dem_orig_kwh  = float(df_motor["Demanda_kW"].sum() * 0.25)
    dem_post_kwh  = float(df_motor["Demanda_Post_Inyeccion_Solar_kW"].sum() * 0.25)
    ahorro_kwh    = dem_orig_kwh - dem_post_kwh
    pct_ahorro    = (ahorro_kwh / dem_orig_kwh * 100) if dem_orig_kwh > 0 else 0.0
    hsp           = energia_anual / pico_kw if pico_kw > 0 else 0.0

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Energía Solar Generada", f"{energia_anual:,.0f} kWh", delta="Anual")
    with m2:
        st.metric("Pico Máximo", f"{pico_kw:,.1f} kW", delta="Instantáneo")
    with m3:
        st.metric("Energía Ahorrada en Red", f"{ahorro_kwh:,.0f} kWh",
                  delta=f"{pct_ahorro:.1f}% de la demanda")
    with m4:
        st.metric("Horas Pico Equivalentes", f"{hsp:,.0f} h/año", delta="HSP anuales")

    st.markdown("---")

    # =========================================================================
    # GRÁFICA 1 — Irradiancia Global en el Plano (POA)  ·  Línea naranja
    # =========================================================================
    st.markdown("### Irradiancia Global en el Plano del Array  ·  Gtot POA (W/m²)")

    fig1 = make_fig()
    fig1.add_trace(go.Scatter(
        x=df_vis["Fecha_Hora"], y=df_vis["Gtot_POA_Wm2"],
        name="Irradiancia POA", mode="lines",
        line=dict(color="#f5a623", width=1.5),
        hovertemplate="%{x|%d %b %H:%M}<br><b>%{y:.1f} W/m²</b><extra></extra>",
    ))
    fig1.update_layout(yaxis_title="W/m²", xaxis_title="")
    st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})

    # =========================================================================
    # GRÁFICA 2 — Generación Solar Pura  ·  Línea amarilla con área rellena
    # =========================================================================
    st.markdown("### Generación Solar Pura  ·  (kW)")

    fig2 = make_fig()
    fig2.add_trace(go.Scatter(
        x=df_vis["Fecha_Hora"], y=df_vis["Generacion_Solar_kW"],
        name="Generación Solar", mode="lines", fill="tozeroy",
        line=dict(color="#ffe033", width=1.5),
        fillcolor="rgba(255,224,51,0.12)",
        hovertemplate="%{x|%d %b %H:%M}<br><b>%{y:.2f} kW</b><extra></extra>",
    ))
    fig2.update_layout(yaxis_title="kW", xaxis_title="")
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    # =========================================================================
    # GRÁFICA 3 — Comparativa de Carga
    #   · Demanda Original: línea roja punteada
    #   · Post-Inyección Solar: línea verde con área rellena
    # =========================================================================
    st.markdown("### Comparativa de Carga  ·  Demanda Original vs. Post-Inyección (kW)")

    fig3 = make_fig()

    # Traza 1: Post-inyección (va detrás, con relleno)
    fig3.add_trace(go.Scatter(
        x=df_vis["Fecha_Hora"], y=df_vis["Demanda_Post_Inyeccion_Solar_kW"],
        name="Post-Inyección Solar", mode="lines", fill="tozeroy",
        line=dict(color="#4ecdc4", width=1.8),
        fillcolor="rgba(78,205,196,0.12)",
        hovertemplate="%{x|%d %b %H:%M}<br>Post-Inyección: <b>%{y:.2f} kW</b><extra></extra>",
    ))
    # Traza 2: Demanda original (encima, punteada roja)
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

    # -------------------------------------------------------------------------
    # TABLA Y DESCARGA
    # -------------------------------------------------------------------------
    st.markdown("---")
    with st.expander("Ver datos tabulares del período seleccionado"):
        st.dataframe(df_vis.reset_index(drop=True), use_container_width=True, height=280)

    st.download_button(
        label="⬇ Descargar resultados anuales completos (CSV)",
        data=df_motor.to_csv(index=False).encode("utf-8"),
        file_name="resultados_fotovoltaicos_anuales.csv",
        mime="text/csv",
    )

# =============================================================================
# ESTADO INICIAL — antes de ejecutar cualquier simulación
# =============================================================================
else:
    st.info(
        "Configura la ubicación y sube tu curva de demanda en la barra lateral. "
        "Luego presiona **▶ Ejecutar Simulación** para ver los resultados."
    )
