# =============================================================================
# app_mono_bi.py — Interfaz de Usuario Fotovoltaica
# =============================================================================

import traceback
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from motor_calculo_mono_bi import (
    PANELES,
    calcular_viabilidad,
    calcular_tilt_optimo,
    descargar_clima_open_meteo,
    resumen_penalizacion_temperatura_mensual,
)


# ─────────────────────────────────────────────────────────────────────────────
# CATÁLOGOS Y CONSTANTES DE UI
# ─────────────────────────────────────────────────────────────────────────────

ESTADOS_MEXICO = {
    "Personalizado": (25.6866, -100.3161, 540.0),
    "Nuevo León (Monterrey)": (25.6866, -100.3161, 540.0),
    "Jalisco (Guadalajara)": (20.6597, -103.3496, 1566.0),
    "Chihuahua (Chihuahua)": (28.6329, -106.0691, 1415.0),
    "Ciudad de México": (19.4326, -99.1332, 2240.0),
    "Querétaro (Querétaro)": (20.5888, -100.3899, 1820.0),
    "Yucatán (Mérida)": (20.9674, -89.6237, 10.0),
    "Baja California (Tijuana)": (32.5149, -117.0382, 20.0),
    "Sonora (Hermosillo)": (29.0729, -110.9559, 210.0),
}

MESES_STR = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS DE VISUALIZACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def _make_fig(title: str = "") -> go.Figure:
    """Genera una figura Plotly base con el tema oscuro corporativo."""
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#12171f",
        font=dict(family="IBM Plex Mono", color="#8892a4", size=11),
        xaxis=dict(gridcolor="#1e2535", linecolor="#2a3040"),
        yaxis=dict(gridcolor="#1e2535", linecolor="#2a3040"),
        hovermode="x unified",
        height=300,
        margin=dict(l=55, r=20, t=40, b=50),
        title=dict(text=title, font=dict(color="#c8bfae", size=12), x=0),
    )
    return fig


def _filtrar_rango(df: pd.DataFrame, desde, hasta) -> pd.DataFrame:
    ts_desde = pd.Timestamp(desde)
    ts_hasta = pd.Timestamp(hasta) + timedelta(days=1) - timedelta(seconds=1)
    return df.loc[(df["Fecha_Hora"] >= ts_desde) & (df["Fecha_Hora"] <= ts_hasta)]


def _matriz_irradiancia(df: pd.DataFrame):
    df = df.copy()
    df["_mes"] = df["Fecha_Hora"].dt.month
    df["_slot"] = df["Fecha_Hora"].dt.hour * 4 + df["Fecha_Hora"].dt.minute // 15
    pivot = df.pivot_table(index="_slot", columns="_mes", values="Gtot_POA_Wm2", aggfunc="mean")
    pivot = pivot.reindex(index=range(96), columns=range(1, 13)).fillna(0.0)
    y_labels = [f"{h:02d}:{m:02d}" for h in range(24) for m in range(0, 60, 15)]
    return pivot.values, y_labels, MESES_STR


@st.cache_data(show_spinner=False)
def get_tilt_optimo(lat, lon, altura):
    return calcular_tilt_optimo(lat, lon, altura)


@st.cache_data(show_spinner=False)
def get_clima_open_meteo_cached(lat, lon, start_date, end_date, timezone="America/Mexico_City"):
    """Cachea solo la descarga climática; no depende de tilt, acimut ni panel."""
    return descargar_clima_open_meteo(lat, lon, start_date, end_date, timezone)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN STREAMLIT & UI
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Análisis Fotovoltaico",
    page_icon="☀️",
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
 h1 { font-family: 'IBM Plex Mono', monospace !important; color: #f5a623 !important; font-size: 1.8rem !important; }
 h3 { font-family: 'IBM Plex Mono', monospace !important; color: #c8bfae !important; font-size: 0.85rem !important; text-transform: uppercase; border-bottom: 1px solid #2a3040; padding-bottom: 6px; }
 [data-testid="metric-container"] { background-color: #161b27; border: 1px solid #2a3040; border-radius: 8px; padding: 18px 20px; }
 [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #f5a623 !important; font-family: 'IBM Plex Mono', monospace; }
 div[data-testid="stButton"] > button { background: linear-gradient(135deg, #f5a623, #e8860d); color: #0f1117; font-weight: 600; }
 .panel-card { background: #1a2235; border: 1px solid #2a3a50; border-left: 3px solid #f5a623; border-radius: 8px; padding: 14px 16px; margin-top: 8px; }
 .pc-tipo { color: #f5a623; font-weight: 600; font-size: 0.70rem; text-transform: uppercase; }
 .pc-modelo { color: #e8e0d0; font-size: 0.85rem; font-weight: 600; }
 .pc-tag { display: inline-block; background: #0f1117; border: 1px solid #2a3a50; border-radius: 4px; padding: 3px 10px; font-size: 0.72rem; color: #4ecdc4; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown("# ☀️ Análisis de Viabilidad Energética Fotovoltaica")


# ── BARRA LATERAL ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 1. Ubicación")
    estado_sel = st.selectbox("Estado de la República", options=list(ESTADOS_MEXICO.keys()))
    lat_def, lon_def, alt_def = ESTADOS_MEXICO[estado_sel]

    col1, col2 = st.columns(2)
    latitud = col1.number_input("Latitud (°)", value=lat_def if lat_def else 25.6866, format="%.4f")
    longitud = col2.number_input("Longitud (°)", value=lon_def if lon_def else -100.3161, format="%.4f")
    altura = st.number_input("Altitud (msnm)", value=alt_def if alt_def else 540.0, format="%.0f")

    st.markdown("### 2. Orientación del Arreglo")
    tilt_opt = get_tilt_optimo(latitud, longitud, altura)
    st.info(f"Ángulo óptimo calculado: **{tilt_opt:.1f}°**")

    c_tilt, c_az = st.columns(2)
    tilt = c_tilt.number_input("Inclinación (°)", value=float(tilt_opt), step=1.0)
    acimut = c_az.number_input(
        "Acimut (°)",
        value=0.0,
        help="0° = Sur, 90° = Oeste, -90° = Este",
        step=5.0,
    )

    st.markdown("### 3. Perfil de Consumo")
    tipo_demanda = st.radio("Formato de entrada", ["Anual", "Mensual"], horizontal=True)
    kwh_anual = 0.0
    kwh_mensuales = [0.0] * 12

    if tipo_demanda == "Anual":
        kwh_anual = st.number_input("Consumo Anual Total (kWh)", value=100_000.0, step=5000.0)
    else:
        st.caption("Consumo mensual (kWh)")
        for i in range(12):
            kwh_mensuales[i] = st.number_input(MESES_STR[i], value=8500.0, step=500.0, key=f"mes_{i}")

    st.markdown("### 4. Tecnología")
    tipo_panel = st.selectbox(
        "Tipo de módulo",
        options=["monofacial", "bifacial_jinko"],
        format_func=lambda k: PANELES[k]["nombre"],
    )
    p = PANELES[tipo_panel]
    st.markdown(
        f"<div class='panel-card'><div class='pc-tipo'>{p['nombre']}</div>"
        f"<div class='pc-modelo'>{p['modelo']}</div>"
        f"<span class='pc-tag'>{p['potencia_w']} W</span></div>",
        unsafe_allow_html=True,
    )

    gb = 0.0
    if p["bifacial"]:
        gb = st.slider("Beneficio Bifacial (%)", 5, 25, 15) / 100.0

    st.markdown("### 5. Clima histórico")
    usar_clima_historico = st.checkbox("Usar clima histórico Open-Meteo", value=True)
    col_w1, col_w2 = st.columns(2)
    weather_start_date = col_w1.date_input("Inicio", value=date(2025, 1, 1))
    weather_end_date = col_w2.date_input("Fin", value=date(2025, 12, 31))

    if usar_clima_historico:
        st.caption(
            "Se usa como año climático análogo para estimar temperatura de celda "
            "y penalización térmica."
        )
    else:
        st.warning(
            "Modo simplificado: se usará temperatura ambiente fija de 20 °C y viento fijo de 1.5 m/s."
        )

    st.markdown("---")
    boton_ejecutar = st.button("▶ Ejecutar Simulación", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# EJECUCIÓN
# ─────────────────────────────────────────────────────────────────────────────

if boton_ejecutar:
    with st.spinner("Ejecutando cálculo de irradiancia POA y penalización térmica..."):
        try:
            if weather_end_date < weather_start_date:
                raise ValueError("La fecha final del clima histórico no puede ser anterior a la fecha inicial.")

            df_clima_horario = None
            if usar_clima_historico:
                df_clima_horario = get_clima_open_meteo_cached(
                    latitud,
                    longitud,
                    weather_start_date.isoformat(),
                    weather_end_date.isoformat(),
                    "America/Mexico_City",
                )

            df_motor, energia_anual = calcular_viabilidad(
                latitud,
                longitud,
                altura,
                tipo_demanda,
                kwh_mensuales,
                kwh_anual,
                tilt,
                acimut,
                tipo_panel,
                gb,
                usar_clima_historico=usar_clima_historico,
                weather_start_date=weather_start_date.isoformat(),
                weather_end_date=weather_end_date.isoformat(),
                df_clima_horario=df_clima_horario,
            )
            df_temp_mensual = resumen_penalizacion_temperatura_mensual(df_motor)

            st.session_state.update(
                {
                    "df_motor": df_motor,
                    "energia_anual": energia_anual,
                    "df_temp_mensual": df_temp_mensual,
                    "usar_clima_historico": usar_clima_historico,
                    "sim_ok": True,
                }
            )
        except Exception as e:
            st.session_state["sim_ok"] = False
            st.error(f"Error en el motor de cálculo: {e}")
            st.code(traceback.format_exc(), language="python")


# ─────────────────────────────────────────────────────────────────────────────
# RESULTADOS
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.get("sim_ok"):
    df_motor = st.session_state["df_motor"].copy()
    df_temp_mensual = st.session_state["df_temp_mensual"].copy()

    dem_orig_kwh = float(df_motor["Demanda_kW"].sum() * 0.25)
    dem_post_kwh = float(df_motor["Demanda_Post_Inyeccion_Solar_kW"].sum() * 0.25)
    ahorro_kwh = dem_orig_kwh - dem_post_kwh

    perdida_anual_temp = float(df_motor["Perdida_Temperatura_kWh"].sum())
    gen_sin_temp_anual = float(df_motor["Energia_Solar_Sin_Temp_kWh"].sum())
    penalizacion_anual_pct = perdida_anual_temp / gen_sin_temp_anual * 100.0 if gen_sin_temp_anual > 0 else 0.0
    temp_celda_prom = float(df_motor["Temperatura_Celda_C"].mean())
    temp_celda_max = float(df_motor["Temperatura_Celda_C"].max())

    m1, m2, m3 = st.columns(3)
    m1.metric("Consumo Anual Original", f"{dem_orig_kwh:,.0f} kWh")
    m2.metric("Generación Solar Anual", f"{st.session_state['energia_anual']:,.0f} kWh")
    m3.metric(
        "Consumo Anual Residual",
        f"{dem_post_kwh:,.0f} kWh",
        delta=f"{-ahorro_kwh / dem_orig_kwh * 100:.1f}%" if dem_orig_kwh > 0 else "0.0%",
        delta_color="inverse",
    )

    mt1, mt2, mt3, mt4 = st.columns(4)
    mt1.metric("Pérdida anual por temperatura", f"{perdida_anual_temp:,.0f} kWh")
    mt2.metric("Penalización térmica anual (%)", f"{penalizacion_anual_pct:.2f}%")
    mt3.metric("Temperatura celda promedio", f"{temp_celda_prom:.1f} °C")
    mt4.metric("Temperatura celda máxima", f"{temp_celda_max:.1f} °C")

    if not st.session_state.get("usar_clima_historico", True):
        st.info("Resultados térmicos calculados con supuesto simplificado: 20 °C ambiente y 1.5 m/s de viento.")

    st.markdown("---")

    # ── GRÁFICO 1: BALANCE MENSUAL DE ENERGÍA ───────────────────────────────
    st.markdown("### Balance Energético Mensual (kWh)")
    df_motor["Mes"] = df_motor["Fecha_Hora"].dt.month
    df_mes = df_motor.groupby("Mes")[["Demanda_kW", "Generacion_Solar_kW", "Demanda_Post_Inyeccion_Solar_kW"]].sum() * 0.25
    df_mes = df_mes.reindex(range(1, 13)).fillna(0.0)

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(x=MESES_STR, y=df_mes["Demanda_kW"], name="Consumo Total", marker_color="#ff6b6b"))
    fig_bar.add_trace(go.Bar(x=MESES_STR, y=df_mes["Generacion_Solar_kW"], name="Generación Solar", marker_color="#f5a623"))
    fig_bar.add_trace(go.Bar(x=MESES_STR, y=df_mes["Demanda_Post_Inyeccion_Solar_kW"], name="Consumo Residual (Red)", marker_color="#4ecdc4"))
    fig_bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#12171f",
        barmode="group",
        font=dict(family="IBM Plex Mono", color="#8892a4"),
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # ── NUEVA SECCIÓN: PENALIZACIÓN POR TEMPERATURA MENSUAL ─────────────────
    st.markdown("### Penalización por Temperatura Mensual")
    df_temp_plot = df_temp_mensual.copy()
    df_temp_plot["Mes_Str"] = df_temp_plot["Mes"].map(lambda m: MESES_STR[int(m) - 1])

    fig_temp_bar = go.Figure()
    fig_temp_bar.add_trace(
        go.Bar(
            x=df_temp_plot["Mes_Str"],
            y=df_temp_plot["Generacion_Sin_Temp_kWh"],
            name="Generación sin temp. (25 °C)",
            marker_color="#4ecdc4",
        )
    )
    fig_temp_bar.add_trace(
        go.Bar(
            x=df_temp_plot["Mes_Str"],
            y=df_temp_plot["Generacion_Con_Temp_kWh"],
            name="Generación con temp.",
            marker_color="#f5a623",
        )
    )
    fig_temp_bar.add_trace(
        go.Bar(
            x=df_temp_plot["Mes_Str"],
            y=df_temp_plot["Perdida_Temperatura_kWh"],
            name="Pérdida térmica",
            marker_color="#ff6b6b",
        )
    )
    fig_temp_bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#12171f",
        barmode="group",
        font=dict(family="IBM Plex Mono", color="#8892a4"),
        height=360,
        yaxis_title="kWh/mes",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_temp_bar, use_container_width=True)

    fig_temp_pct = _make_fig("Penalización térmica mensual (%)")
    fig_temp_pct.add_trace(
        go.Scatter(
            x=df_temp_plot["Mes_Str"],
            y=df_temp_plot["Penalizacion_Temperatura_pct"],
            name="Penalización térmica",
            mode="lines+markers",
            line=dict(color="#ffe033", width=2),
            marker=dict(size=7),
        )
    )
    fig_temp_pct.update_layout(yaxis_title="%", height=300)
    st.plotly_chart(fig_temp_pct, use_container_width=True)

    fig_temp_doble = make_subplots(specs=[[{"secondary_y": True}]])
    fig_temp_doble.add_trace(
        go.Scatter(
            x=df_temp_plot["Mes_Str"],
            y=df_temp_plot["Temperatura_Ambiente_Prom_C"],
            name="Temp. ambiente prom.",
            mode="lines+markers",
            line=dict(color="#4ecdc4", width=2),
        ),
        secondary_y=False,
    )
    fig_temp_doble.add_trace(
        go.Scatter(
            x=df_temp_plot["Mes_Str"],
            y=df_temp_plot["Temperatura_Celda_Prom_C"],
            name="Temp. celda prom.",
            mode="lines+markers",
            line=dict(color="#f5a623", width=2),
        ),
        secondary_y=True,
    )
    fig_temp_doble.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#12171f",
        font=dict(family="IBM Plex Mono", color="#8892a4"),
        hovermode="x unified",
        height=320,
        margin=dict(l=60, r=60, t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig_temp_doble.update_yaxes(title_text="Temp. ambiente (°C)", gridcolor="#1e2535", secondary_y=False)
    fig_temp_doble.update_yaxes(title_text="Temp. celda (°C)", showgrid=False, secondary_y=True)
    st.plotly_chart(fig_temp_doble, use_container_width=True)

    st.dataframe(df_temp_mensual, use_container_width=True)
    st.download_button(
        label="⬇ Descargar resumen mensual de temperatura (CSV)",
        data=df_temp_mensual.to_csv(index=False).encode("utf-8-sig"),
        file_name="resumen_penalizacion_temperatura_mensual.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown("---")

    # ── FILTRO TEMPORAL PARA SERIES DE TIEMPO ────────────────────────────────
    st.markdown("### Análisis Dinámico en Alta Resolución (15 min)")
    fecha_min = df_motor["Fecha_Hora"].min().date()
    fecha_max = df_motor["Fecha_Hora"].max().date()

    col_fi, col_ff = st.columns(2)
    with col_fi:
        fecha_inicio = st.date_input("Desde", value=fecha_min, min_value=fecha_min, max_value=fecha_max)
    with col_ff:
        fecha_fin = st.date_input(
            "Hasta",
            value=min(fecha_min + timedelta(days=6), fecha_max),
            min_value=fecha_min,
            max_value=fecha_max,
        )

    df_vis = _filtrar_rango(df_motor, fecha_inicio, fecha_fin)

    # ── GRÁFICO 2: DEMANDA VS IRRADIANCIA ───────────────────────────────────
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    fig1.add_trace(
        go.Scatter(
            x=df_vis["Fecha_Hora"],
            y=df_vis["Demanda_kW"],
            name="Demanda (kW)",
            mode="lines",
            line=dict(color="#ff6b6b", width=1.8),
        ),
        secondary_y=False,
    )
    fig1.add_trace(
        go.Scatter(
            x=df_vis["Fecha_Hora"],
            y=df_vis["Gtot_POA_Wm2"],
            name="Gtot POA (W/m²)",
            mode="lines",
            line=dict(color="#f5a623", width=1.5),
        ),
        secondary_y=True,
    )
    fig1.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#12171f",
        font=dict(family="IBM Plex Mono", color="#8892a4", size=11),
        hovermode="x unified",
        height=320,
        margin=dict(l=60, r=60, t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig1.update_yaxes(title_text="Demanda (kW)", title_font=dict(color="#ff6b6b"), tickfont=dict(color="#ff6b6b"), gridcolor="#1e2535", secondary_y=False)
    fig1.update_yaxes(title_text="Irradiancia POA (W/m²)", title_font=dict(color="#f5a623"), tickfont=dict(color="#f5a623"), showgrid=False, secondary_y=True)
    st.plotly_chart(fig1, use_container_width=True)

    # ── GRÁFICO 3: GENERACIÓN SOLAR ─────────────────────────────────────────
    fig2 = _make_fig()
    fig2.add_trace(
        go.Scatter(
            x=df_vis["Fecha_Hora"],
            y=df_vis["Generacion_Solar_kW"],
            name="Generación Solar",
            mode="lines",
            fill="tozeroy",
            line=dict(color="#ffe033", width=1.5),
            fillcolor="rgba(255,224,51,0.12)",
        )
    )
    fig2.update_layout(yaxis_title="kW", xaxis_title="", height=280, margin=dict(t=20, b=20))
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # ── GRÁFICO 4: MATRIZ DE IRRADIANCIA POA ────────────────────────────────
    st.markdown("### Matriz de Irradiancia POA Anual (W/m²)")
    _z, _y_lbl, _x_lbl = _matriz_irradiancia(df_motor)
    _custom = np.array([[_y_lbl[r] for _ in range(12)] for r in range(96)])

    escala_inferno_negro = [
        [0.00, "#12171f"],
        [0.11, "#1b0c41"],
        [0.22, "#4a0c6b"],
        [0.33, "#781c6d"],
        [0.44, "#a52c60"],
        [0.55, "#cf4446"],
        [0.66, "#ed6925"],
        [0.77, "#fb9b06"],
        [0.88, "#f7d13d"],
        [1.00, "#fcffa4"],
    ]

    fig_hm = go.Figure(
        go.Heatmap(
            z=_z,
            x=_x_lbl,
            y=list(range(96)),
            colorscale=escala_inferno_negro,
            zmin=0,
            zmax=np.max(_z),
            customdata=_custom,
            hovertemplate="Mes: <b>%{x}</b><br>Hora: <b>%{customdata}</b><br>Irradiancia: <b>%{z:.1f} W/m²</b><extra></extra>",
        )
    )
    _tick_slots = list(range(0, 96, 8))
    fig_hm.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#12171f",
        font=dict(family="IBM Plex Mono", color="#8892a4"),
        xaxis=dict(side="bottom", linecolor="#2a3040"),
        yaxis=dict(
            autorange="reversed",
            tickvals=_tick_slots,
            ticktext=[_y_lbl[i] for i in _tick_slots],
            gridcolor="#1e2535",
        ),
        height=500,
        margin=dict(l=60, r=20, t=20, b=30),
    )
    st.plotly_chart(fig_hm, use_container_width=True)
