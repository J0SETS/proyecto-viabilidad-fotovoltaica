# =============================================================================
# unified_app.py — Interfaz de Usuario Fotovoltaica + BESS + Clima Histórico
# =============================================================================

import traceback
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Importamos la lógica matemática y catálogos desde el motor solar
from motor_calculo_mono_bi import (
    PANELES,
    calcular_tilt_optimo,
    calcular_dia_tipico_horizontal,
    descargar_clima_open_meteo,
    resumen_penalizacion_temperatura_mensual,
    dimensionar_y_simular,
)

# Importamos el motor de cálculo BESS
from baterias import MotorBESS
from analisis_financiero import InputsFinancieros, calcular_cashflow_20_anios, fmt_mxn, fmt_anios


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
    "Veracruz (Streger)": (25.686, -100.3, 540.0)
}

MESES_STR = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS DE VISUALIZACIÓN
# ─────────────────────────────────────────────────────────────────────────────
def _make_fig(title: str = "") -> go.Figure:
    """Genera una figura Plotly base con el tema oscuro corporativo."""
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#12171f",
        font=dict(family="IBM Plex Mono", color="#8892a4", size=11),
        xaxis=dict(gridcolor="#1e2535", linecolor="#2a3040"),
        yaxis=dict(gridcolor="#1e2535", linecolor="#2a3040"),
        hovermode="x unified", height=300, margin=dict(l=55, r=20, t=40, b=50),
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
    """Cachea la descarga climática para no repetir la llamada a la API innecesariamente."""
    return descargar_clima_open_meteo(lat, lon, start_date, end_date, timezone)

@st.cache_data(show_spinner=False)
def get_dia_tipico(lat, lon, altura, fecha_str, tilt, acimut_usuario, panel_key="monofacial"):
    return calcular_dia_tipico_horizontal(lat, lon, altura, fecha_str, tilt, acimut_usuario, panel_key)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN STREAMLIT & ESTILOS
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Análisis Fotovoltaico Integral",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
    html, body, [class*="css"]  { font-family: 'IBM Plex Sans', sans-serif; }
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

    /* Estilos BESS & Riesgos */
    .bess-card { background: #1a2235; border: 1px solid #2a3a50; border-left: 3px solid #4ecdc4; border-radius: 8px; padding: 16px 18px; margin-top: 8px; font-family: 'IBM Plex Mono', monospace; }
    .bess-card-label { color: #4ecdc4; font-weight: 600; font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 10px; }
    .bess-card-row { display: flex; justify-content: space-between; align-items: baseline; border-bottom: 1px solid #1e2535; padding: 5px 0; font-size: 0.78rem; }
    .bess-card-row:last-child { border-bottom: none; }
    .bess-card-key { color: #8892a4; }
    .bess-card-val { color: #e8e0d0; font-weight: 600; }

    .riesgo-card { border-radius: 8px; padding: 16px 18px; margin-top: 8px; font-family: 'IBM Plex Mono', monospace; }
    .riesgo-alto   { background: rgba(231,76,60,0.12);  border: 1px solid rgba(231,76,60,0.35);  border-left: 3px solid #e74c3c; }
    .riesgo-medio  { background: rgba(245,166,35,0.10); border: 1px solid rgba(245,166,35,0.30); border-left: 3px solid #f5a623; }
    .riesgo-bajo   { background: rgba(78,205,196,0.10); border: 1px solid rgba(78,205,196,0.30); border-left: 3px solid #4ecdc4; }
    .riesgo-badge { display: inline-block; border-radius: 4px; padding: 4px 12px; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 10px; }
    .badge-alto  { background: rgba(231,76,60,0.25);  color: #ff6b6b; }
    .badge-medio { background: rgba(245,166,35,0.25); color: #f5a623; }
    .badge-bajo  { background: rgba(78,205,196,0.25); color: #4ecdc4; }
    .riesgo-justif { color: #8892a4; font-size: 0.75rem; line-height: 1.55; margin-top: 8px; font-family: 'IBM Plex Sans', sans-serif; }
    .riesgo-autonomia-label { color: #c8bfae; font-size: 0.70rem; text-transform: uppercase; margin-top: 10px; }
    .riesgo-autonomia-val   { font-size: 1.4rem; font-weight: 700; margin-top: 2px; }
    .autonomia-alto  { color: #ff6b6b; }
    .autonomia-medio { color: #f5a623; }
    .autonomia-bajo  { color: #4ecdc4; }

    .impacto-table { width: 100%; border-collapse: collapse; font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; margin-top: 8px; }
    .impacto-table th { color: #8892a4; font-weight: 600; text-transform: uppercase; font-size: 0.65rem; letter-spacing: 0.06em; padding: 8px 12px; border-bottom: 1px solid #2a3040; text-align: left; }
    .impacto-table td { padding: 9px 12px; border-bottom: 1px solid #1e2535; color: #c8bfae; vertical-align: middle; }
    .impacto-table tr:last-child td { border-bottom: none; }
    .impacto-table tr:hover td { background: rgba(255,255,255,0.025); }
    .cat-badge { display: inline-block; border-radius: 3px; padding: 2px 8px; font-size: 0.65rem; font-weight: 700; letter-spacing: 0.05em; }
    .cat-largo  { background: rgba(231,76,60,0.20);  color: #ff6b6b; }
    .cat-medio  { background: rgba(245,166,35,0.20); color: #f5a623; }
    .cat-corto  { background: rgba(78,205,196,0.20); color: #4ecdc4; }
    .impacto-section-wrap { background: #161b27; border: 1px solid #2a3040; border-radius: 8px; padding: 16px 18px; margin-top: 8px; }
    .impacto-section-label { color: #c8bfae; font-family: 'IBM Plex Mono', monospace; font-size: 0.70rem; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

st.markdown("# ☀️ Análisis de Viabilidad Energética Fotovoltaica y BESS")

# ── BARRA LATERAL ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 1. Ubicación")
    estado_sel = st.selectbox("Estado de la República", options=list(ESTADOS_MEXICO.keys()))
    lat_def, lon_def, alt_def = ESTADOS_MEXICO[estado_sel]

    col1, col2 = st.columns(2)
    latitud   = col1.number_input("Latitud (°)",   value=lat_def if lat_def else 25.6866, format="%.4f")
    longitud  = col2.number_input("Longitud (°)",  value=lon_def if lon_def else -100.3161, format="%.4f")
    altura    = st.number_input("Altitud (msnm)",  value=alt_def if alt_def else 540.0, format="%.0f")

    st.markdown("### 2. Orientación del Arreglo")
    tilt_opt = get_tilt_optimo(latitud, longitud, altura)
    st.info(f"📐 Ángulo óptimo calculado para un año: **{tilt_opt:.1f}°**")

    c_tilt, c_az = st.columns(2)
    tilt   = c_tilt.number_input("Inclinación (°)", value=float(tilt_opt), step=1.0)
    acimut = c_az.number_input("Acimut (°)", value=0.0, help="0° = Sur, 90° = Oeste, -90° = Este", step=5.0)

    st.markdown("### 3. Perfil de Consumo")
    tipo_demanda = st.radio("Formato de entrada", ["Anual", "Mensual"], horizontal=True)
    kwh_anual    = 0.0
    kwh_mensuales = [0.0] * 12

    if tipo_demanda == "Anual":
        kwh_anual = st.number_input("Consumo Anual Total (kWh)", value=100_000.0, step=5000.0)
    else:
        st.caption("Consumo mensual (kWh)")
        for i in range(12):
            kwh_mensuales[i] = st.number_input(MESES_STR[i], value=8500.0, step=500.0, key=f"mes_{i}")

    st.markdown("### 4. Tecnología y Dimensionamiento")
    tipo_panel = st.selectbox(
        "Tipo de módulo",
        options=["monofacial", "bifacial_jinko"],
        format_func=lambda k: PANELES[k]["nombre"],
    )
    p = PANELES[tipo_panel]
    st.markdown(
        f"<div class='panel-card'>"
        f"<div class='pc-tipo'>{p['nombre']}</div>"
        f"<div class='pc-modelo'>{p['modelo']}</div>"
        f"<span class='pc-tag'>{p['potencia_w']} W</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    gb = 0.0
    if p["bifacial"]:
        gb = st.slider("Beneficio Bifacial (%)", 5, 25, 15) / 100.0

    porcentaje_cobertura = st.slider(
        "Porcentaje de cobertura objetivo (%)", 
        min_value=10, max_value=100, value=100, step=5,
        help="Porcentaje del consumo total que deseas cubrir con generación solar."
    ) / 100.0

    st.markdown("### 5. Clima Histórico (Open-Meteo)")
    usar_clima_historico = st.checkbox("Usar clima histórico para temp. de celda", value=True)
    col_w1, col_w2 = st.columns(2)
    weather_start_date = col_w1.date_input("Inicio", value=date(2025, 1, 1))
    weather_end_date = col_w2.date_input("Fin", value=date(2025, 12, 31))

    if not usar_clima_historico:
        st.warning("Modo simplificado: se usará temperatura ambiente fija de 20 °C.")

    # ── SECCIÓN 6: ALMACENAMIENTO (BESS) ─────────────────────────────────────
    st.markdown("### 6. Almacenamiento de Energía (BESS)")
    b1, b2 = st.columns(2)
    carga_critica_kw = b1.number_input("Carga Crítica (kW)", value=300.0, step=25.0)
    horas_respaldo = b2.number_input("Autonomía (horas)", value=4.0, step=0.5, min_value=0.5)

    with st.expander("⚡ Configuración de Apagones (Historial CFE)"):
        frecuencia_largos = st.number_input("Apagones Largos (>1 hr / año)", value=3, step=1, min_value=0)
        frecuencia_medios = st.number_input("Apagones Medios (1–10 min / año)", value=12, step=1, min_value=0)
        frecuencia_cortos = st.number_input("Microapagones (<1 min / año)", value=24, step=1, min_value=0)

    st.markdown("---")
    boton_ejecutar = st.button("▶ Ejecutar Simulación Integral", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# VISTA RÁPIDA (SIEMPRE VISIBLE) — IRRADIANCIA DÍA TÍPICO + PRODUCCIÓN ESTIMADA
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### ⚡ Vista Rápida — Irradiancia de Día Típico y Producción Estimada")

df_dia, energia_dia_kwh = get_dia_tipico(latitud, longitud, altura, date.today().isoformat(), tilt, acimut)
panel_default = PANELES["monofacial"]

col_qv1, col_qv2 = st.columns([3, 1])

with col_qv1:
    fig_qv = _make_fig("Irradiancia de Cielo Despejado — GHI / DNI / DHI / POA")
    fig_qv.add_trace(go.Scatter(x=df_dia["Hora"], y=df_dia["GHI_Wm2"], name="GHI", mode="lines", line=dict(color="#f5a623", width=1.5)))
    fig_qv.add_trace(go.Scatter(x=df_dia["Hora"], y=df_dia["DNI_Wm2"], name="DNI", mode="lines", line=dict(color="#ff6b6b", width=1.5)))
    fig_qv.add_trace(go.Scatter(x=df_dia["Hora"], y=df_dia["DHI_Wm2"], name="DHI", mode="lines", line=dict(color="#4ecdc4", width=1.5)))
    fig_qv.add_trace(go.Scatter(x=df_dia["Hora"], y=df_dia["POA_Wm2"], name=f"POA (tilt {tilt:.0f}°, acimut {acimut:.0f}°)", mode="lines", line=dict(color="#ffe033", width=2)))
    fig_qv.update_layout(yaxis_title="W/m²", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_qv, use_container_width=True)

with col_qv2:
    st.metric("Irradiancia pico (POA)", f"{df_dia['POA_Wm2'].max():,.0f} W/m²")
    st.metric(f"Producción estimada ({panel_default['potencia_w']} W)", f"{energia_dia_kwh:.2f} kWh/día")
    st.caption(
        f"Panel de referencia: **{panel_default['modelo']}**, tilt = {tilt:.1f}°, acimut = {acimut:.1f}° "
        f"(0°=Sur), T_celda = 25 °C (sin corrección térmica). Cielo despejado, Ineichen — "
        f"{date.today().strftime('%d/%m/%Y')}."
    )
# ─────────────────────────────────────────────────────────────────────────────
# EJECUCIÓN
# ─────────────────────────────────────────────────────────────────────────────
if boton_ejecutar:
    with st.spinner("Ejecutando dimensionamiento, penalización térmica y cálculo BESS..."):
        try:
            if weather_end_date < weather_start_date:
                raise ValueError("La fecha final del clima no puede ser anterior a la fecha inicial.")

            # ── 1. Descarga Climática ──
            df_clima_horario = None
            if usar_clima_historico:
                df_clima_horario = get_clima_open_meteo_cached(
                    latitud, longitud,
                    weather_start_date.isoformat(),
                    weather_end_date.isoformat(),
                    "America/Mexico_City",
                )

            # ── 2. Motor Solar y Dimensionamiento de Paneles ──
            resultados_solar = dimensionar_y_simular(
                lat=latitud, lon=longitud, altura=altura,
                tipo_demanda=tipo_demanda, kwh_mensuales=kwh_mensuales, kwh_anual=kwh_anual,
                tilt=tilt, acimut_usuario=acimut,
                tipo_panel=tipo_panel, gb=gb,
                usar_clima_historico=usar_clima_historico,
                weather_start_date=weather_start_date.isoformat(),
                weather_end_date=weather_end_date.isoformat(),
                df_clima_horario=df_clima_horario,
                porcentaje_cobertura=porcentaje_cobertura
            )
            
            df_motor = resultados_solar["df_simulacion"]
            df_temp_mensual = resumen_penalizacion_temperatura_mensual(df_motor)

            # ── 3. Motor BESS ──
            motor_bess = MotorBESS()
            res_dim = motor_bess.calcular_dimensionamiento(
                horas_respaldo=horas_respaldo,
                carga_critica_kw=carga_critica_kw,
            )
            res_cortes = motor_bess.analizar_historico_cortes(
                frecuencia_largos=int(frecuencia_largos),
                frecuencia_medios=int(frecuencia_medios),
                frecuencia_cortos=int(frecuencia_cortos),
            )

            # ── 4. Guardar en Session State ──
            st.session_state.update({
                "df_motor": df_motor,
                "df_temp_mensual": df_temp_mensual,
                "res_solar": resultados_solar,
                "res_dim": res_dim,
                "res_cortes": res_cortes,
                "bess_specs": motor_bess.specs,
                "usar_clima_historico": usar_clima_historico,
                "sim_ok": True,
            })

        except Exception as e:
            st.error(f"Error en el motor de cálculo: {e}")
            st.code(traceback.format_exc(), language="python")

# ─────────────────────────────────────────────────────────────────────────────
# RESULTADOS
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.get("sim_ok"):
    df_motor = st.session_state["df_motor"]
    df_temp_mensual = st.session_state["df_temp_mensual"]
    res_solar = st.session_state["res_solar"]
    res_dim = st.session_state["res_dim"]
    res_cortes = st.session_state["res_cortes"]
    bess_specs = st.session_state["bess_specs"]

    tab_solar, tab_bess, tab_fin = st.tabs(["☀️ Análisis Solar y Térmico", "🔋 Almacenamiento y Resiliencia (BESS)", "💵 Análisis financiero"])

    # =========================================================================
    # TAB 1 — ANÁLISIS SOLAR Y TÉRMICO
    # =========================================================================
    with tab_solar:
        # ── MÉTRICAS SUPERIORES ──────────────────────────────────────────────
        dem_orig_kwh = float(df_motor["Demanda_kW"].sum() * 0.25)
        dem_post_kwh = float(df_motor["Demanda_Post_Inyeccion_Solar_kW"].sum() * 0.25)
        ahorro_kwh = dem_orig_kwh - dem_post_kwh

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Consumo Anual Original", f"{dem_orig_kwh:,.0f} kWh")
        m2.metric("Generación Solar Anual", f"{res_solar['generacion_estimada_kwh']:,.0f} kWh")
        m3.metric(
            "Consumo Anual Residual", f"{dem_post_kwh:,.0f} kWh",
            delta=f"{-ahorro_kwh / dem_orig_kwh * 100:.1f}%" if dem_orig_kwh > 0 else "0.0%",
            delta_color="inverse",
        )
        
        # ── MÉTRICAS DE DIMENSIONAMIENTO ──
        m4.metric(
            "Paneles Requeridos", f"{res_solar['num_paneles']} uds.",
            delta=f"{res_solar['potencia_instalada_kwp']:,.1f} kWp instalados",
            delta_color="off"
        )
        st.caption(f"**Área total estimada del arreglo fotovoltaico:** {res_solar['area_total_m2']:,.1f} m²")

        # ── MÉTRICAS TÉRMICAS ──
        perdida_anual_temp = float(df_motor["Perdida_Temperatura_kWh"].sum())
        gen_sin_temp_anual = float(df_motor["Energia_Solar_Sin_Temp_kWh"].sum())
        penalizacion_anual_pct = perdida_anual_temp / gen_sin_temp_anual * 100.0 if gen_sin_temp_anual > 0 else 0.0
        temp_celda_prom = float(df_motor["Temperatura_Celda_C"].mean())
        temp_celda_max = float(df_motor["Temperatura_Celda_C"].max())

        st.markdown("---")
        mt1, mt2, mt3, mt4 = st.columns(4)
        mt1.metric("Pérdida anual por temperatura", f"{perdida_anual_temp:,.0f} kWh")
        mt2.metric("Penalización térmica anual (%)", f"{penalizacion_anual_pct:.2f}%")
        mt3.metric("Temperatura celda promedio", f"{temp_celda_prom:.1f} °C")
        mt4.metric("Temperatura celda máxima", f"{temp_celda_max:.1f} °C")

        if not st.session_state.get("usar_clima_historico", True):
            st.info("Resultados térmicos calculados con supuesto simplificado: 20 °C ambiente y 1.5 m/s de viento.")

        st.markdown("---")

        # ── GRÁFICOS SOLARES Y TÉRMICOS ──────────────────────────────────────
        col_g1, col_g2 = st.columns(2)

        with col_g1:
            st.markdown("### Balance Energético Mensual (kWh)")
            df_motor["Mes"] = df_motor["Fecha_Hora"].dt.month
            df_mes = df_motor.groupby("Mes")[["Demanda_kW", "Generacion_Solar_kW", "Demanda_Post_Inyeccion_Solar_kW"]].sum() * 0.25
            df_mes = df_mes.reindex(range(1, 13)).fillna(0.0)

            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(x=MESES_STR, y=df_mes["Demanda_kW"], name="Consumo Total", marker_color="#ff6b6b"))
            fig_bar.add_trace(go.Bar(x=MESES_STR, y=df_mes["Generacion_Solar_kW"], name="Generación Solar", marker_color="#f5a623"))
            fig_bar.add_trace(go.Bar(x=MESES_STR, y=df_mes["Demanda_Post_Inyeccion_Solar_kW"], name="Consumo Residual", marker_color="#4ecdc4"))
            fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#12171f", barmode="group", font=dict(family="IBM Plex Mono", color="#8892a4"), height=350, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_g2:
            st.markdown("### Temperaturas de Operación (°C)")
            df_temp_plot = df_temp_mensual.copy()
            df_temp_plot["Mes_Str"] = df_temp_plot["Mes"].map(lambda m: MESES_STR[int(m) - 1])
            
            fig_temp_un_eje = _make_fig()
            fig_temp_un_eje.add_trace(go.Scatter(x=df_temp_plot["Mes_Str"], y=df_temp_plot["Temperatura_Ambiente_Prom_C"], name="T. amb prom.", mode="lines+markers", line=dict(color="#4ecdc4", width=2)))
            fig_temp_un_eje.add_trace(go.Scatter(x=df_temp_plot["Mes_Str"], y=df_temp_plot["Temperatura_Celda_Prom_C"], name="T. celda prom.", mode="lines+markers", line=dict(color="#f5a623", width=2)))
            fig_temp_un_eje.add_trace(go.Scatter(x=df_temp_plot["Mes_Str"], y=df_temp_plot["Temperatura_Celda_Max_C"], name="T. celda máx.", mode="lines+markers", line=dict(color="#ff6b6b", width=2, dash="dash")))
            fig_temp_un_eje.update_layout(yaxis_title="°C", height=350, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_temp_un_eje, use_container_width=True)

        st.markdown("---")

        # ── FILTRO TEMPORAL PARA SERIES DE TIEMPO ────────────────────────────
        st.markdown("### Análisis Dinámico en Alta Resolución (15 min)")
        fecha_min = df_motor["Fecha_Hora"].min().date()
        fecha_max = df_motor["Fecha_Hora"].max().date()

        col_fi, col_ff = st.columns(2)
        with col_fi:
            fecha_inicio = st.date_input("Desde", value=fecha_min, min_value=fecha_min, max_value=fecha_max)
        with col_ff:
            fecha_fin = st.date_input("Hasta", value=min(fecha_min + timedelta(days=6), fecha_max), min_value=fecha_min, max_value=fecha_max)

        df_vis = _filtrar_rango(df_motor, fecha_inicio, fecha_fin)
        # ── GRÁFICO: DEMANDA ENERGÉTICA VS PRODUCCIÓN FV CORREGIDA TÉRMICAMENTE ─────
        st.markdown("### Demanda Energética vs Producción Fotovoltaica Corregida Térmicamente")

        fig_demanda_fv = _make_fig()

        fig_demanda_fv.add_trace(
            go.Scatter(
                x=df_vis["Fecha_Hora"],
                y=df_vis["Demanda_kW"],
                name="Demanda energética (kW)",
                mode="lines",
                line=dict(color="#ff6b6b", width=1.8),
            )
        )

        fig_demanda_fv.add_trace(
            go.Scatter(
                x=df_vis["Fecha_Hora"],
                y=df_vis["Generacion_Solar_kW"],
                name="Producción FV corregida térmicamente (kW)",
                mode="lines",
                fill="tozeroy",
                line=dict(color="#ffe033", width=1.6),
                fillcolor="rgba(255,224,51,0.12)",
            )
        )

        fig_demanda_fv.add_trace(
            go.Scatter(
                x=df_vis["Fecha_Hora"],
                y=df_vis["Demanda_Post_Inyeccion_Solar_kW"],
                name="Demanda residual después de FV (kW)",
                mode="lines",
                line=dict(color="#4ecdc4", width=1.4, dash="dot"),
            )
        )

        fig_demanda_fv.update_layout(
            yaxis_title="Potencia (kW)",
            xaxis_title="",
            height=360,
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
            margin=dict(l=60, r=25, t=40, b=35),
        )

        st.plotly_chart(fig_demanda_fv, use_container_width=True)
        fig1 = make_subplots(specs=[[{"secondary_y": True}]])
        fig1.add_trace(go.Scatter(x=df_vis["Fecha_Hora"], y=df_vis["Demanda_kW"], name="Demanda (kW)", mode="lines", line=dict(color="#ff6b6b", width=1.8)), secondary_y=False)
        fig1.add_trace(go.Scatter(x=df_vis["Fecha_Hora"], y=df_vis["Gtot_POA_Wm2"], name="Gtot POA (W/m²)", mode="lines", line=dict(color="#f5a623", width=1.5)), secondary_y=True)
        fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#12171f", font=dict(family="IBM Plex Mono", color="#8892a4", size=11), hovermode="x unified", height=320, margin=dict(l=60, r=60, t=40, b=30), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig1.update_yaxes(title_text="Demanda (kW)", title_font=dict(color="#ff6b6b"), tickfont=dict(color="#ff6b6b"), gridcolor="#1e2535", secondary_y=False)
        fig1.update_yaxes(title_text="Irradiancia POA", title_font=dict(color="#f5a623"), tickfont=dict(color="#f5a623"), showgrid=False, secondary_y=True)
        st.plotly_chart(fig1, use_container_width=True)

        # ── MATRIZ POA ───────────────────────────────────────────────────────
        st.markdown("### Matriz de Irradiancia POA Anual (W/m²)")
        _z, _y_lbl, _x_lbl = _matriz_irradiancia(df_motor)
        _custom = np.array([[_y_lbl[r] for _ in range(12)] for r in range(96)])
        escala_inferno_negro = [[0.00, "#12171f"], [0.11, "#1b0c41"], [0.22, "#4a0c6b"], [0.33, "#781c6d"], [0.44, "#a52c60"], [0.55, "#cf4446"], [0.66, "#ed6925"], [0.77, "#fb9b06"], [0.88, "#f7d13d"], [1.00, "#fcffa4"]]

        fig_hm = go.Figure(go.Heatmap(z=_z, x=_x_lbl, y=list(range(96)), colorscale=escala_inferno_negro, zmin=0, zmax=np.max(_z), customdata=_custom, hovertemplate="Mes: <b>%{x}</b><br>Hora: <b>%{customdata}</b><br>Irradiancia: <b>%{z:.1f} W/m²</b><extra></extra>"))
        _tick_slots = list(range(0, 96, 8))
        fig_hm.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#12171f", font=dict(family="IBM Plex Mono", color="#8892a4"), xaxis=dict(side="bottom", linecolor="#2a3040"), yaxis=dict(autorange="reversed", tickvals=_tick_slots, ticktext=[_y_lbl[i] for i in _tick_slots], gridcolor="#1e2535"), height=500, margin=dict(l=60, r=20, t=20, b=30))
        st.plotly_chart(fig_hm, use_container_width=True)

    # =========================================================================
    # TAB 2 — BESS
    # =========================================================================
    with tab_bess:
        st.markdown("### Dimensionamiento del Arreglo BESS")
        mb1, mb2, mb3 = st.columns(3)

        mb1.metric("Gabinetes Requeridos", f"{res_dim.unidades_bess} unidades", help="Gabinetes Sungrow PowerStack 255CS en paralelo.")
        mb2.metric("Capacidad Total Instalada", f"{res_dim.capacidad_total_kwh:,.0f} kWh", delta=f"+{res_dim.sobredimensionamiento_pct:.1f}% sobre requerido", delta_color="off")
        mb3.metric("Potencia AC del Arreglo", f"{res_dim.potencia_total_kw:,.0f} kW", help="Potencia AC nominal total del arreglo de gabinetes.")

        st.markdown("---")

        st.markdown("### Equipo de Referencia y Evaluación de Riesgo")
        col_ficha, col_riesgo = st.columns(2, gap="medium")

        with col_ficha:
            st.markdown(
                f"""
                <div class="bess-card">
                    <div class="bess-card-label">Ficha Técnica del Gabinete</div>
                    <div class="bess-card-row"><span class="bess-card-key">Modelo</span><span class="bess-card-val">{bess_specs.modelo}</span></div>
                    <div class="bess-card-row"><span class="bess-card-key">Química</span><span class="bess-card-val">{bess_specs.quimica}</span></div>
                    <div class="bess-card-row"><span class="bess-card-key">Capacidad nominal</span><span class="bess-card-val">{bess_specs.capacidad_nominal_kwh:.0f} kWh / gab</span></div>
                    <div class="bess-card-row"><span class="bess-card-key">Potencia nominal AC</span><span class="bess-card-val">{bess_specs.potencia_nominal_ac_kw:.0f} kW / gab</span></div>
                    <div class="bess-card-row"><span class="bess-card-key">RTE (Efficiency)</span><span class="bess-card-val">{bess_specs.rte * 100:.0f} %</span></div>
                    <div class="bess-card-row"><span class="bess-card-key">Ciclos garantizados</span><span class="bess-card-val">{bess_specs.ciclos_minimos:,} – {bess_specs.ciclos_maximos:,}</span></div>
                    <div class="bess-card-row"><span class="bess-card-key">kWh requeridos</span><span class="bess-card-val">{res_dim.capacidad_requerida_kwh:,.2f} kWh</span></div>
                </div>
                """, unsafe_allow_html=True
            )

        with col_riesgo:
            nivel = res_cortes.nivel_riesgo
            nivel_lower = nivel.lower()
            riesgo_color_map = {
                "alto":  ("#ff6b6b", "badge-alto",  "riesgo-alto",  "autonomia-alto"),
                "medio": ("#f5a623", "badge-medio", "riesgo-medio", "autonomia-medio"),
                "bajo":  ("#4ecdc4", "badge-bajo",  "riesgo-bajo",  "autonomia-bajo"),
            }
            _, badge_cls, card_cls, auto_cls = riesgo_color_map[nivel_lower]

            st.markdown(
                f"""
                <div class="riesgo-card {card_cls}">
                    <span class="riesgo-badge {badge_cls}">Riesgo Operativo — {nivel.upper()}</span>
                    <div class="riesgo-autonomia-label">Autonomía Mínima Recomendada</div>
                    <div class="riesgo-autonomia-val {auto_cls}">{res_cortes.autonomia_recomendada_hrs:.0f} horas</div>
                    <div class="riesgo-justif">{res_cortes.justificacion_riesgo}</div>
                </div>
                """, unsafe_allow_html=True
            )

        st.markdown("---")
        st.markdown("### Desglose del Impacto Anual por Cortes de Energía")

        impacto = res_cortes.impacto_anual_estimado
        cat_config = {"largos": ("LARGOS > 1 hr", "cat-largo"), "medios": ("MEDIOS 1–10 min", "cat-medio"), "cortos": ("CORTOS < 1 min", "cat-corto")}
        rows_html = ""
        total_eventos = total_horas = 0.0

        for key, (label_txt, badge_cls) in cat_config.items():
            d = impacto[key]
            eventos, dur_repr, horas_acum, desc = d["eventos"], d["duracion_representativa_hrs"], d["horas_desabasto_acumuladas"], d["descripcion"]
            total_eventos += eventos
            total_horas += horas_acum
            dur_str = f"{dur_repr:.1f} hr" if dur_repr >= 1 else (f"{dur_repr * 60:.0f} min" if dur_repr >= (1/60) else f"{dur_repr * 3600:.0f} seg")
            
            # Nota cómo el HTML está pegado a la izquierda sin espacios
            rows_html += f"""<tr>
<td><span class="cat-badge {badge_cls}">{label_txt}</span></td>
<td style="text-align:center; color:#e8e0d0; font-weight:600;">{eventos}</td>
<td style="text-align:center; color:#8892a4;">{dur_str}</td>
<td style="text-align:center; color:#f5a623; font-weight:600;">{horas_acum:.2f} hr</td>
<td style="color:#6b7585; font-size:0.72rem;">{desc}</td>
</tr>"""

        # Fila de totales sin sangría
        rows_html += f"""<tr style="background:rgba(255,255,255,0.04);">
<td style="color:#c8bfae; font-weight:700; font-size:0.72rem; text-transform:uppercase;">TOTAL ANUAL</td>
<td style="text-align:center; color:#e8e0d0; font-weight:700;">{total_eventos}</td>
<td style="text-align:center; color:#8892a4;">—</td>
<td style="text-align:center; color:#f5a623; font-weight:700;">{total_horas:.2f} hr</td>
<td style="color:#8892a4; font-size:0.72rem;">Autonomía recomendada: <strong>{res_cortes.autonomia_recomendada_hrs:.0f} horas</strong></td>
</tr>"""

        # Contenedor final también pegado a la izquierda
        st.markdown(
            f"""
<div class="impacto-section-wrap">
    <div class="impacto-section-label">Historial CFE — Resumen estadístico anual</div>
    <table class="impacto-table">
        <thead>
            <tr><th style="width:22%;">Categoría</th><th style="width:10%; text-align:center;">Eventos / año</th><th style="width:14%; text-align:center;">Duración repr.</th><th style="width:14%; text-align:center;">Horas desabasto</th><th>Impacto operativo</th></tr>
        </thead>
        <tbody>{rows_html}</tbody>
    </table>
</div>
            """, unsafe_allow_html=True
        )
        
    with tab_fin:

        st.markdown("### Parámetros Financieros del Proyecto")
    
        with st.form("form_financiero"):
            fc1, fc2, fc3 = st.columns(3)
    
            capex = fc1.number_input(
                "CAPEX — Inversión Inicial (MXN)",
                value=float(
                    res_solar.get("num_paneles", 0) * 4_500          # ~$4,500 MXN/panel como placeholder
                    + res_dim.unidades_bess * 850_000                 # ~$850K MXN/gabinete Sungrow placeholder
                ),
                step=50_000.0, format="%.0f",
                help="Costo total de equipos + ingeniería + instalación.",
            )
            opex = fc1.number_input(
                "OPEX — Mantenimiento Anual (MXN)",
                value=max(capex * 0.01, 20_000.0),
                step=5_000.0, format="%.0f",
                help="Limpieza, revisiones, seguros, monitoreo.",
            )
    
            perdidas_apagon = fc2.number_input(
                "Pérdidas por Apagones (MXN/año)",
                value=float(res_cortes.impacto_anual_estimado.get("largos", {}).get("horas_desabasto_acumuladas", 0) * 50_000),
                step=10_000.0, format="%.0f",
                help="Pérdida económica estimada por paros de producción / daños.",
            )
            tarifa_cfe = fc2.number_input(
                "Tarifa CFE Actual (MXN/kWh)",
                value=2.85, step=0.05, format="%.2f",
                help="Tarifa media tensión GDMTH o equivalente.",
            )
    
            inflacion_tarifa = fc3.slider(
                "Inflación Tarifaria Anual (%)", 4, 15, 8,
                help="Incremento histórico promedio CFE: 7–10 % anual.",
            ) / 100.0
            degradacion = fc3.slider(
                "Degradación Anual de Paneles (%)", 0, 2, 1,
                help="Típico LID + degradación lineal: 0.5 – 0.7 % / año.",
            ) / 100.0 / 2   # el slider va de 0–2 en pasos de 0.5 implícito
    
            boton_fin = st.form_submit_button("📊 Calcular ROI", use_container_width=True)
    
        if boton_fin:
            consumo_cubierto = res_solar.get("generacion_estimada_kwh", 0.0)
    
            inputs_fin = InputsFinancieros(
                capex_mxn=capex,
                opex_anual_mxn=opex,
                perdidas_apagon_mxn=perdidas_apagon,
                tarifa_cfe_mxn_kwh=tarifa_cfe,
                consumo_cubierto_kwh=consumo_cubierto,
                inflacion_tarifa_pct=inflacion_tarifa,
                degradacion_panel_pct=degradacion,
                horizonte_anios=20,
            )
    
            df_cf, roi_anios = calcular_cashflow_20_anios(inputs_fin)
            st.session_state["df_cf"]    = df_cf
            st.session_state["roi_anios"] = roi_anios
            st.session_state["capex_fin"] = capex
    
        if "df_cf" in st.session_state:
            df_cf    = st.session_state["df_cf"]
            roi_anios = st.session_state["roi_anios"]
            capex_fin = st.session_state["capex_fin"]
    
            # ── MÉTRICAS RESUMEN ─────────────────────────────────────────────────
            van_20 = df_cf["Flujo_Acumulado_MXN"].iloc[-1]
            ahorro_total = df_cf["Ahorro_Tarifa_MXN"].sum() + df_cf["Ahorro_Apagones_MXN"].sum()
            opex_total   = df_cf["OPEX_MXN"].sum()
    
            mf1, mf2, mf3, mf4 = st.columns(4)
            mf1.metric("Inversión Inicial",        fmt_mxn(capex_fin))
            mf2.metric("Payback (ROI)",            fmt_anios(roi_anios))
            mf3.metric("Valor Neto a 20 años",     fmt_mxn(van_20),
                    delta="positivo" if van_20 >= 0 else "negativo",
                    delta_color="normal" if van_20 >= 0 else "inverse")
            mf4.metric("Ahorro Acumulado Bruto",   fmt_mxn(ahorro_total))
    
            st.markdown("---")
    
            col_roi, col_bar = st.columns(2)
    
            # ── GRÁFICA 1: Curva de Flujo Acumulado (ROI) ───────────────────────
            with col_roi:
                st.markdown("### Curva de Recuperación de Inversión")
    
                acum_con_capex = [-capex_fin] + df_cf["Flujo_Acumulado_MXN"].tolist()
                anios_eje      = list(range(0, 21))
    
                color_acum = [
                    "#ff6b6b" if v < 0 else "#4ecdc4"
                    for v in acum_con_capex
                ]
    
                fig_roi = go.Figure()
    
                # Zona de pérdida / ganancia
                fig_roi.add_hrect(
                    y0=min(acum_con_capex) * 1.05, y1=0,
                    fillcolor="rgba(231,76,60,0.06)", line_width=0,
                )
                fig_roi.add_hrect(
                    y0=0, y1=max(acum_con_capex) * 1.05,
                    fillcolor="rgba(78,205,196,0.06)", line_width=0,
                )
    
                # Línea de recuperación cero
                fig_roi.add_hline(y=0, line=dict(color="#2a3040", width=1.5, dash="dot"))
    
                # Marcador ROI vertical
                if roi_anios != float("inf"):
                    fig_roi.add_vline(
                        x=roi_anios,
                        line=dict(color="#f5a623", width=1.5, dash="dash"),
                        annotation_text=f"ROI ≈ {roi_anios:.1f} años",
                        annotation_font=dict(color="#f5a623", size=10, family="IBM Plex Mono"),
                        annotation_position="top left",
                    )
    
                # Área rellena bajo la curva
                fig_roi.add_trace(go.Scatter(
                    x=anios_eje, y=acum_con_capex,
                    fill="tozeroy",
                    fillcolor="rgba(78,205,196,0.08)",
                    line=dict(color="#4ecdc4", width=2.5),
                    mode="lines+markers",
                    marker=dict(color=color_acum, size=7, line=dict(color="#0f1117", width=1)),
                    name="Flujo Acumulado",
                    hovertemplate="Año %{x}<br>Acumulado: $%{y:,.0f} MXN<extra></extra>",
                ))
    
                fig_roi.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#12171f",
                    font=dict(family="IBM Plex Mono", color="#8892a4", size=11),
                    xaxis=dict(title="Año", gridcolor="#1e2535", linecolor="#2a3040", dtick=2),
                    yaxis=dict(title="MXN", gridcolor="#1e2535", linecolor="#2a3040",
                            tickformat="$,.0f"),
                    height=380, margin=dict(l=70, r=20, t=30, b=50),
                    showlegend=False,
                )
                st.plotly_chart(fig_roi, use_container_width=True)
    
            # ── GRÁFICA 2: Barras de Cashflow Anual ─────────────────────────────
            with col_bar:
                st.markdown("### Desglose de Flujo de Caja Anual")
    
                anios_str = [str(a) for a in df_cf["Año"]]
                bar_colors = [
                    "#4ecdc4" if v >= 0 else "#ff6b6b"
                    for v in df_cf["Flujo_Neto_MXN"]
                ]
    
                fig_bar = go.Figure()
    
                fig_bar.add_trace(go.Bar(
                    x=anios_str, y=df_cf["Ahorro_Tarifa_MXN"],
                    name="Ahorro Tarifa CFE",
                    marker_color="#f5a623",
                    hovertemplate="Año %{x}<br>Ahorro Tarifa: $%{y:,.0f}<extra></extra>",
                ))
                fig_bar.add_trace(go.Bar(
                    x=anios_str, y=df_cf["Ahorro_Apagones_MXN"],
                    name="Ahorro Apagones",
                    marker_color="#4ecdc4",
                    hovertemplate="Año %{x}<br>Ahorro Apagones: $%{y:,.0f}<extra></extra>",
                ))
                fig_bar.add_trace(go.Bar(
                    x=anios_str, y=-df_cf["OPEX_MXN"],
                    name="OPEX (egreso)",
                    marker_color="#ff6b6b",
                    hovertemplate="Año %{x}<br>OPEX: $%{y:,.0f}<extra></extra>",
                ))
                fig_bar.add_trace(go.Scatter(
                    x=anios_str, y=df_cf["Flujo_Neto_MXN"],
                    name="Flujo Neto",
                    mode="lines+markers",
                    line=dict(color="#e8e0d0", width=2, dash="dot"),
                    marker=dict(size=5),
                    hovertemplate="Año %{x}<br>Flujo Neto: $%{y:,.0f}<extra></extra>",
                ))
    
                fig_bar.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#12171f",
                    font=dict(family="IBM Plex Mono", color="#8892a4", size=11),
                    barmode="relative",
                    xaxis=dict(title="Año", gridcolor="#1e2535", linecolor="#2a3040"),
                    yaxis=dict(title="MXN", gridcolor="#1e2535", linecolor="#2a3040",
                            tickformat="$,.0f"),
                    height=380, margin=dict(l=70, r=20, t=30, b=50),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                xanchor="right", x=1, font=dict(size=10)),
                    hovermode="x unified",
                )
                st.plotly_chart(fig_bar, use_container_width=True)
    
            # ── TABLA DETALLADA ──────────────────────────────────────────────────
            with st.expander("📋 Ver tabla de flujo de caja completa"):
                df_display = df_cf.copy()
                for col in ["Ahorro_Tarifa_MXN", "Ahorro_Apagones_MXN", "OPEX_MXN",
                            "Flujo_Neto_MXN", "Flujo_Acumulado_MXN"]:
                    df_display[col] = df_display[col].map(lambda v: f"${v:,.0f}")
                df_display.columns = ["Año", "Ahorro Tarifa", "Ahorro Apagones",
                                    "OPEX", "Flujo Neto", "Flujo Acumulado"]
                st.dataframe(df_display, use_container_width=True, hide_index=True)