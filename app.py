from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from bess import MotorBESS
from demanda import (
    demand_energy_kwh,
    generar_demanda_sintetica_15min,
    read_demand_file,
    read_uploaded_demand,
)
from exportaciones import dataframe_to_csv_bytes, dict_to_json_bytes
from finanzas import (
    calcular_finanzas_desde_simulacion,
    detalle_apagones_financiero_df,
    escenarios_financieros_df,
    formato_mxn,
    formato_pct,
    formato_anios,
)
from motor_solar import (
    LOCATIONS_MEXICO,
    MODULE_OPTIONS,
    SiteConfig,
    build_target_index,
    default_weather_range,
    run_solar_demand_simulation,
)


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
DEMANDA_EJEMPLO = DATA_DIR / "demanda_ejemplo.csv"


st.set_page_config(
    page_title="Streger - Viabilidad FV, BESS y Finanzas",
    layout="wide",
)


st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; }
    .metric-card { border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def make_fig(title: str = "", height: int = 360) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=20, r=20, t=55, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def metric_row(columns: list[tuple[str, str, str | None]]) -> None:
    cols = st.columns(len(columns))
    for col, (label, value, help_text) in zip(cols, columns):
        col.metric(label, value, help=help_text)


def build_demand_dataframe(
    demand_mode: str,
    target_index: pd.DatetimeIndex,
    consumo_anual_kwh: float,
    consumos_mensuales_kwh: list[float],
    factor_potencia: float,
    uploaded_file,
) -> pd.DataFrame:
    if demand_mode == "CSV/XLSX cargado":
        if uploaded_file is None:
            raise ValueError("Sube un archivo de demanda o elige otro modo.")
        return read_uploaded_demand(uploaded_file)
    if demand_mode == "CSV ejemplo del ZIP":
        return read_demand_file(DEMANDA_EJEMPLO)
    if demand_mode == "Consumo mensual":
        return generar_demanda_sintetica_15min(
            target_index=target_index,
            consumos_mensuales_kwh=consumos_mensuales_kwh,
            factor_potencia=factor_potencia,
        )
    return generar_demanda_sintetica_15min(
        target_index=target_index,
        consumo_anual_kwh=consumo_anual_kwh,
        factor_potencia=factor_potencia,
    )


def annual_fig(comparison: pd.DataFrame) -> go.Figure:
    fig = make_fig("Energia anual por escenario")
    fig.add_trace(
        go.Bar(
            x=comparison["Escenario"],
            y=comparison["Energia_Solar_AC_Anual_kWh"],
            name="Generacion AC",
        )
    )
    fig.add_trace(
        go.Bar(
            x=comparison["Escenario"],
            y=comparison["Energia_Autoconsumida_Anual_kWh"],
            name="Autoconsumo",
        )
    )
    fig.update_layout(barmode="group", yaxis_title="kWh/ano")
    return fig


def monthly_energy_fig(summary: pd.DataFrame) -> go.Figure:
    plot_df = summary.reset_index()
    fig = make_fig("Balance mensual")
    fig.add_trace(go.Bar(x=plot_df["Mes"], y=plot_df["Energia_Demandada_kWh"], name="Demanda"))
    fig.add_trace(go.Bar(x=plot_df["Mes"], y=plot_df["Energia_Solar_AC_kWh"], name="Solar AC"))
    fig.add_trace(go.Bar(x=plot_df["Mes"], y=plot_df["Energia_Autoconsumida_kWh"], name="Autoconsumo"))
    fig.add_trace(go.Bar(x=plot_df["Mes"], y=plot_df["Energia_Red_kWh"], name="Red"))
    fig.update_layout(barmode="group", yaxis_title="kWh")
    return fig


def timeseries_fig(result: pd.DataFrame) -> go.Figure:
    sample = result.copy()
    if len(sample) > 96 * 21:
        sample = sample.iloc[: 96 * 21]
    x_values = sample.index.tz_localize(None)
    fig = make_fig("Demanda vs generacion solar (primeras semanas)", height=420)
    fig.add_trace(go.Scatter(x=x_values, y=sample["Demanda_kW"], name="Demanda kW", mode="lines"))
    fig.add_trace(go.Scatter(x=x_values, y=sample["Generacion_AC_kW"], name="Generacion AC kW", mode="lines"))
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=sample["Demanda_Post_Inyeccion_Solar_kW"],
            name="Demanda post solar kW",
            mode="lines",
        )
    )
    fig.update_layout(yaxis_title="kW")
    return fig


st.title("Viabilidad fotovoltaica, BESS y finanzas - Streger")
st.caption(
    "App final modular para demo. Integra clima historico, demanda, FV mono/bifacial, "
    "respaldo BESS y analisis financiero preliminar."
)


with st.sidebar:
    st.header("Parametros")

    location_mode = st.radio("Ubicacion", ["Predefinida", "Manual"], horizontal=True)
    if location_mode == "Predefinida":
        location_label = st.selectbox(
            "Sitio",
            options=list(LOCATIONS_MEXICO.keys()),
        )
        loc = LOCATIONS_MEXICO[location_label]
        site_name = loc["name"]
        latitude = float(loc["lat"])
        longitude = float(loc["lon"])
        altitude = float(loc["altitude"])
        timezone = loc["timezone"]
    else:
        site_name = st.text_input("Nombre del sitio", value="Streger")
        latitude = st.number_input("Latitud", value=19.54, step=0.0001, format="%.4f")
        longitude = st.number_input("Longitud", value=-96.91, step=0.0001, format="%.4f")
        altitude = st.number_input("Altitud (m)", min_value=0.0, value=1400.0, step=1.0)
        timezone = st.text_input("Zona horaria", value="America/Mexico_City")

    default_start, default_end = default_weather_range()
    col_d1, col_d2 = st.columns(2)
    weather_start = col_d1.date_input("Clima inicio", value=default_start)
    weather_end = col_d2.date_input("Clima fin", value=default_end)
    align_by_position = st.checkbox(
        "Alinear demanda por posicion",
        value=True,
        help="Util para usar demanda futura con un ano climatico historico analogo.",
    )

    st.divider()
    module_selection = st.selectbox(
        "Escenarios FV",
        options=["all", "monofacial", "bifacial"],
        format_func=lambda value: {
            "all": "Comparar monofacial y bifacial",
            "monofacial": "Solo monofacial",
            "bifacial": "Solo bifacial",
        }[value],
    )
    system_dc_kwp = st.number_input("Potencia DC sistema (kWp)", min_value=1.0, value=399.3, step=10.0)
    inverter_efficiency = st.number_input("Eficiencia inversor", min_value=0.01, max_value=1.0, value=0.96, step=0.01)
    system_losses_pct = st.number_input("Perdidas adicionales (%)", min_value=0.0, max_value=50.0, value=0.0, step=0.5)
    bifacial_gain_pct = st.number_input("Ganancia trasera bifacial (%)", min_value=0.0, max_value=50.0, value=15.0, step=1.0)
    surface_tilt = st.number_input("Tilt (grados)", min_value=0.0, max_value=90.0, value=19.5, step=0.5)
    surface_azimuth = st.number_input("Azimut pvlib (180 = sur)", min_value=0.0, max_value=360.0, value=180.0, step=5.0)
    albedo = st.number_input("Albedo", min_value=0.0, max_value=1.0, value=0.20, step=0.01)

    st.divider()
    demand_mode = st.selectbox(
        "Modo de demanda",
        ["CSV ejemplo del ZIP", "Consumo anual", "Consumo mensual", "CSV/XLSX cargado"],
    )
    consumo_anual = st.number_input("Consumo anual (kWh)", min_value=1.0, value=100_000.0, step=5_000.0)
    factor_potencia = st.number_input("Factor de potencia", min_value=0.01, max_value=1.0, value=0.90, step=0.01)
    uploaded_file = None
    monthly_values = [8500.0] * 12
    if demand_mode == "Consumo mensual":
        st.caption("Consumo mensual kWh")
        for month in range(12):
            monthly_values[month] = st.number_input(
                f"Mes {month + 1}",
                min_value=0.0,
                value=monthly_values[month],
                step=500.0,
                key=f"mes_{month + 1}",
            )
    if demand_mode == "CSV/XLSX cargado":
        uploaded_file = st.file_uploader("Archivo demanda", type=["csv", "xlsx", "xls"])

    st.divider()
    st.subheader("BESS y finanzas")
    carga_critica_kw = st.number_input("Carga critica BESS (kW)", min_value=1.0, value=30.0, step=5.0)
    horas_respaldo = st.number_input("Horas respaldo", min_value=0.5, value=4.0, step=0.5)
    frecuencia_largos = st.number_input("Apagones largos / ano", min_value=0, value=3, step=1)
    frecuencia_medios = st.number_input("Apagones medios / ano", min_value=0, value=12, step=1)
    frecuencia_cortos = st.number_input("Microapagones / ano", min_value=0, value=24, step=1)
    escenario_costo = st.selectbox("Costo por apagon", ["conservador", "medio", "alto"], index=1)
    escenario_mitigacion = st.selectbox("Mitigacion BESS/UPS", ["conservadora", "media", "alta"], index=1)
    inversion_fv = st.number_input("Inversion FV estimada (MXN)", min_value=0.0, value=1_500_000.0, step=50_000.0)
    inversion_bess = st.number_input("Inversion BESS/UPS estimada (MXN)", min_value=0.0, value=1_200_000.0, step=50_000.0)

    run_button = st.button("Ejecutar simulacion", type="primary", width="stretch")


if run_button:
    try:
        site = SiteConfig(
            name=site_name,
            latitude=latitude,
            longitude=longitude,
            timezone=timezone,
            altitude_m=altitude,
            surface_tilt=surface_tilt,
            surface_azimuth=surface_azimuth,
            albedo=albedo,
        )
        target_index = build_target_index(weather_start, weather_end, timezone)
        demand_df = build_demand_dataframe(
            demand_mode=demand_mode,
            target_index=target_index,
            consumo_anual_kwh=consumo_anual,
            consumos_mensuales_kwh=monthly_values,
            factor_potencia=factor_potencia,
            uploaded_file=uploaded_file,
        )
        with st.spinner("Descargando clima historico y ejecutando motor solar..."):
            simulation = run_solar_demand_simulation(
                site=site,
                demand_df=demand_df,
                weather_start_date=weather_start,
                weather_end_date=weather_end,
                module_selection=module_selection,
                bifacial_gain=bifacial_gain_pct / 100,
                system_dc_kwp=system_dc_kwp,
                inverter_efficiency=inverter_efficiency,
                system_losses=system_losses_pct / 100,
                align_demand_by_position=align_by_position,
            )
        st.session_state["simulation"] = simulation
        st.session_state["demand_df"] = demand_df
        st.session_state["inputs"] = {
            "demand_mode": demand_mode,
            "system_dc_kwp": system_dc_kwp,
            "bifacial_gain_pct": bifacial_gain_pct,
            "inverter_efficiency": inverter_efficiency,
            "system_losses_pct": system_losses_pct,
            "carga_critica_kw": carga_critica_kw,
            "horas_respaldo": horas_respaldo,
            "frecuencia_largos": frecuencia_largos,
            "frecuencia_medios": frecuencia_medios,
            "frecuencia_cortos": frecuencia_cortos,
            "escenario_costo": escenario_costo,
            "escenario_mitigacion": escenario_mitigacion,
            "inversion_fv": inversion_fv,
            "inversion_bess": inversion_bess,
        }
        st.success("Simulacion terminada.")
    except Exception as exc:
        st.error(f"No se pudo ejecutar la simulacion: {exc}")


simulation = st.session_state.get("simulation")
demand_df_state = st.session_state.get("demand_df")

tabs = st.tabs(
    [
        "Configuracion",
        "Demanda electrica",
        "Simulacion solar FV",
        "Comparacion mono/bifacial",
        "BESS / respaldo",
        "Analisis financiero Streger",
        "Exportar resultados",
    ]
)


with tabs[0]:
    st.subheader("Configuracion del proyecto")
    st.write("Ruta de trabajo:", str(APP_DIR))
    st.write("Demanda ejemplo:", str(DEMANDA_EJEMPLO))
    config_df = pd.DataFrame(
        [
            ["Sitio", site_name],
            ["Latitud", latitude],
            ["Longitud", longitude],
            ["Altitud m", altitude],
            ["Zona horaria", timezone],
            ["Clima", f"{weather_start} a {weather_end}"],
            ["Potencia DC kWp", system_dc_kwp],
            ["Escenarios FV", module_selection],
        ],
        columns=["Parametro", "Valor"],
    )
    config_df["Valor"] = config_df["Valor"].astype(str)
    st.dataframe(config_df, width="stretch", hide_index=True)


with tabs[1]:
    st.subheader("Demanda electrica")
    if demand_df_state is not None:
        st.write(f"Modo usado: {st.session_state['inputs']['demand_mode']}")
        st.dataframe(demand_df_state.head(200), width="stretch")
        try:
            st.metric("Energia de demanda cargada", f"{demand_energy_kwh(demand_df_state):,.1f} kWh")
        except Exception:
            st.caption("La energia se calcula despues de normalizar la demanda en el motor.")
    else:
        st.info("Ejecuta la simulacion para ver la demanda cargada o generada.")


with tabs[2]:
    st.subheader("Simulacion solar FV")
    if simulation:
        primary = simulation["primary"]
        annual = primary["annual"].iloc[0]
        metric_row(
            [
                ("Escenario principal", str(annual["Escenario"]), None),
                ("Generacion AC", f"{annual['Energia_Solar_AC_Anual_kWh']:,.0f} kWh", None),
                ("Autoconsumo", f"{annual['Energia_Autoconsumida_Anual_kWh']:,.0f} kWh", None),
                ("Cobertura solar", f"{annual['Cobertura_Solar_pct']:.1f}%", None),
            ]
        )
        st.plotly_chart(monthly_energy_fig(primary["summary"]), width="stretch")
        st.plotly_chart(timeseries_fig(primary["result"]), width="stretch")
        st.dataframe(primary["summary"], width="stretch")
    else:
        st.info("Ejecuta la simulacion para ver resultados solares.")


with tabs[3]:
    st.subheader("Comparacion monofacial vs bifacial")
    if simulation:
        comparison = simulation["module_comparison"]
        st.plotly_chart(annual_fig(comparison), width="stretch")
        st.dataframe(comparison, width="stretch", hide_index=True)
        module_specs = pd.DataFrame([vars(module) for module in MODULE_OPTIONS.values()])
        st.markdown("Especificaciones de modulos")
        st.dataframe(module_specs, width="stretch", hide_index=True)
    else:
        st.info("Elige comparar ambos escenarios y ejecuta la simulacion.")


with tabs[4]:
    st.subheader("BESS / respaldo")
    try:
        motor_bess = MotorBESS()
        dim_bess = motor_bess.calcular_dimensionamiento(horas_respaldo, carga_critica_kw)
        ana_bess = motor_bess.analizar_historico_cortes(
            int(frecuencia_largos),
            int(frecuencia_medios),
            int(frecuencia_cortos),
        )
        metric_row(
            [
                ("Gabinetes BESS", f"{dim_bess.unidades_bess}", None),
                ("Capacidad instalada", f"{dim_bess.capacidad_total_kwh:,.0f} kWh", None),
                ("Potencia AC", f"{dim_bess.potencia_total_kw:,.0f} kW", None),
                ("Riesgo operativo", ana_bess.nivel_riesgo, None),
            ]
        )
        st.write(ana_bess.justificacion_riesgo)
        bess_df = pd.DataFrame([dim_bess.to_dict()]).T.reset_index()
        bess_df.columns = ["Metrica", "Valor"]
        bess_df["Valor"] = bess_df["Valor"].astype(str)
        st.dataframe(bess_df, width="stretch", hide_index=True)
        st.download_button(
            "Descargar resumen BESS",
            data=motor_bess.generar_resumen_ejecutivo(
                horas_respaldo,
                carga_critica_kw,
                int(frecuencia_largos),
                int(frecuencia_medios),
                int(frecuencia_cortos),
            ).encode("utf-8"),
            file_name="resumen_bess.txt",
            mime="text/plain",
        )
    except Exception as exc:
        st.error(f"No se pudo calcular BESS: {exc}")


with tabs[5]:
    st.subheader("Analisis financiero Streger")
    if simulation:
        primary_result = simulation["primary"]["result"]
        resumen_fin = calcular_finanzas_desde_simulacion(
            primary_result,
            escenario_costo_apagon=escenario_costo,
            escenario_mitigacion=escenario_mitigacion,
            carga_critica_kw=carga_critica_kw,
            horas_respaldo=horas_respaldo,
            inversion_fv_mxn=inversion_fv,
            inversion_bess_mxn=inversion_bess,
        )
        diagnostico = resumen_fin["diagnostico_recibo"]
        ahorro = resumen_fin["ahorro_solar_express"]
        perdidas = resumen_fin["perdidas_apagones"]
        beneficio = resumen_fin["beneficio_respaldo"]
        comparacion = resumen_fin["comparacion_escenarios"]

        metric_row(
            [
                ("Tarifa", diagnostico["tarifa"], None),
                ("Precio medio", f"${diagnostico['precio_medio_total_mxn_kwh']:.2f}/kWh", None),
                ("Ahorro solar anual", formato_mxn(ahorro["ahorro_solar_anual_mxn"]), None),
                ("Beneficio evitable", formato_mxn(beneficio["beneficio_anual_evitable_mxn"]), None),
            ]
        )
        metric_row(
            [
                ("Perdida por apagones", formato_mxn(perdidas["perdida_anual_estimada_mxn"]), None),
                ("Mitigacion total", formato_pct(beneficio["porcentaje_total_evitable"]), None),
                ("Autoconsumo usado", f"{ahorro['energia_autoconsumida_kwh_anual']:,.0f} kWh", None),
                ("Referencia perdida", formato_mxn(perdidas["perdida_anual_reportada_referencia_mxn"]), None),
            ]
        )

        detalle_df = detalle_apagones_financiero_df(perdidas, beneficio)
        escenarios_df = escenarios_financieros_df(comparacion)

        st.markdown("Detalle por tipo de apagon")
        st.dataframe(detalle_df, width="stretch", hide_index=True)
        st.markdown("Payback y ROI por escenario")
        display_escenarios = escenarios_df.copy()
        display_escenarios["payback"] = display_escenarios["payback_anios"].map(formato_anios)
        display_escenarios["roi_simple_anual_pct"] = display_escenarios["roi_simple_anual"].map(
            lambda value: formato_pct(value) if pd.notna(value) else "No aplica"
        )
        st.dataframe(display_escenarios, width="stretch", hide_index=True)

        fig = make_fig("Inversion vs beneficio anual")
        plot_df = escenarios_df[escenarios_df["escenario"] != "base_sin_fv_sin_bess"]
        fig.add_trace(go.Bar(x=plot_df["nombre"], y=plot_df["inversion_mxn"], name="Inversion"))
        fig.add_trace(go.Bar(x=plot_df["nombre"], y=plot_df["beneficio_anual_mxn"], name="Beneficio anual"))
        fig.update_layout(barmode="group", yaxis_title="MXN")
        st.plotly_chart(fig, width="stretch")

        st.download_button(
            "Descargar resumen financiero JSON",
            data=dict_to_json_bytes(resumen_fin),
            file_name="resumen_financiero_streger.json",
            mime="application/json",
        )
        st.download_button(
            "Descargar escenarios financieros CSV",
            data=dataframe_to_csv_bytes(escenarios_df),
            file_name="escenarios_financieros.csv",
            mime="text/csv",
        )
        st.session_state["resumen_financiero"] = resumen_fin
    else:
        st.info("Ejecuta la simulacion para calcular finanzas desde autoconsumo real.")


with tabs[6]:
    st.subheader("Exportar resultados")
    if simulation:
        primary = simulation["primary"]
        st.download_button(
            "Descargar serie 15 min escenario principal",
            data=dataframe_to_csv_bytes(primary["export_result"]),
            file_name="clima_solar_demanda_15min.csv",
            mime="text/csv",
        )
        st.download_button(
            "Descargar resumen mensual",
            data=dataframe_to_csv_bytes(primary["summary"].reset_index()),
            file_name="resumen_mensual.csv",
            mime="text/csv",
        )
        st.download_button(
            "Descargar resumen anual",
            data=dataframe_to_csv_bytes(primary["annual"]),
            file_name="resumen_anual.csv",
            mime="text/csv",
        )
        st.download_button(
            "Descargar comparacion modulos",
            data=dataframe_to_csv_bytes(simulation["module_comparison"]),
            file_name="comparacion_modulos.csv",
            mime="text/csv",
        )
        metadata = {
            "site": simulation["site_dict"],
            "source_url": simulation["source_url"],
            "weather_start_date": str(simulation["weather_start_date"]),
            "weather_end_date": str(simulation["weather_end_date"]),
            "inputs": st.session_state.get("inputs", {}),
        }
        if "resumen_financiero" in st.session_state:
            metadata["resumen_financiero"] = st.session_state["resumen_financiero"]
        st.download_button(
            "Descargar metadata JSON",
            data=json.dumps(metadata, indent=2, ensure_ascii=False, default=str).encode("utf-8"),
            file_name="metadata.json",
            mime="application/json",
        )
    else:
        st.info("Ejecuta la simulacion para habilitar descargas.")
