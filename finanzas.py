from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


TIMESTEP_HOURS = 0.25


RECIBO_STREGER_MAYO_2026 = {
    "empresa": "STREGER S.A.",
    "ubicacion": "Coatepec, Veracruz",
    "tarifa": "GDMTO",
    "periodo": "28 ABR 26 - 29 MAY 26",
    "dias_facturados": 31,
    "multiplicador": 80,
    "carga_conectada_kw": 99,
    "demanda_contratada_kw": 99,
    "energia_kwh": 9520,
    "demanda_maxima_kw": 80,
    "factor_potencia_pct": 89.89,
    "total_pagar_mxn": 29591.85,
    "subtotal_periodo_mxn": 25509.62,
    "iva_mxn": 4081.54,
}


DESGLOSE_CFE_GDMTO_MAYO_2026 = {
    "suministro_mxn": 648.84,
    "distribucion_mxn": 4869.49,
    "transmision_mxn": 1714.56,
    "cenace_mxn": 72.35,
    "energia_mxn": 10669.15,
    "capacidad_mxn": 6944.37,
    "scnmem_mxn": 65.69,
    "total_mem_mxn": 24984.44,
    "ajuste_baja_tension_mxn": 499.69,
    "cargo_factor_potencia_mxn": 25.48,
    "iva_mxn": 4081.54,
}


FRECUENCIA_APAGONES_STREGER = {
    "largo": {
        "frecuencia_anual": 3,
        "duracion_referencia_horas": 1.5,
        "descripcion": "Apagon mayor a 1 hora",
    },
    "medio": {
        "frecuencia_anual": 12,
        "duracion_referencia_horas": 10 / 60,
        "descripcion": "Apagon menor a 10 minutos",
    },
    "corto": {
        "frecuencia_anual": 24,
        "duracion_referencia_horas": 1 / 60,
        "descripcion": "Apagon menor a 1 minuto",
    },
}


PROTECCION_EXISTENTE_STREGER = {
    "nobreak_estandar_min": 5,
    "nobreak_especial_min": 30,
}


PERDIDA_ANUAL_REPORTADA_REFERENCIA_MXN = 2_000_000


COSTO_EVENTO_APAGON_MXN = {
    "conservador": {"largo": 150_000, "medio": 10_000, "corto": 2_000},
    "medio": {"largo": 350_000, "medio": 35_000, "corto": 8_000},
    "alto": {"largo": 700_000, "medio": 90_000, "corto": 25_000},
}


MITIGACION_BESS_UPS = {
    "conservadora": {"largo": 0.40, "medio": 0.60, "corto": 0.70},
    "media": {"largo": 0.55, "medio": 0.75, "corto": 0.85},
    "alta": {"largo": 0.70, "medio": 0.85, "corto": 0.95},
}


BATERIA_REFERENCIA_SUNGROW = {
    "fabricante": "Sungrow",
    "modelo": "PowerStack 255CS ST255CS-2H",
    "quimica": "LFP",
    "capacidad_nominal_kwh": 257,
    "potencia_nominal_kw": 125,
    "rte": 0.90,
    "ciclos_min": 4000,
    "ciclos_max": 5000,
}


@dataclass
class InputsBESSFinanciero:
    carga_critica_kw: float
    horas_respaldo: float
    factor_seguridad: float = 1.15


def safe_div(numerador: float, denominador: float) -> float:
    if denominador == 0:
        return 0.0
    return numerador / denominador


def formato_mxn(valor: float) -> str:
    return f"${valor:,.2f} MXN"


def formato_pct(valor: float) -> str:
    return f"{valor * 100:.1f}%"


def formato_anios(valor: Optional[float]) -> str:
    if valor is None or pd.isna(valor):
        return "No aplica"
    return f"{valor:.1f} anos"


def diagnostico_recibo_gdmto(
    recibo: Dict = RECIBO_STREGER_MAYO_2026,
    desglose: Dict = DESGLOSE_CFE_GDMTO_MAYO_2026,
) -> Dict:
    energia_kwh = recibo["energia_kwh"]
    total_pagar = recibo["total_pagar_mxn"]
    subtotal = recibo["subtotal_periodo_mxn"]
    distribucion = desglose["distribucion_mxn"]
    capacidad = desglose["capacidad_mxn"]
    total_mem = desglose["total_mem_mxn"]
    cargos_potencia = distribucion + capacidad

    return {
        "empresa": recibo["empresa"],
        "ubicacion": recibo["ubicacion"],
        "tarifa": recibo["tarifa"],
        "periodo": recibo["periodo"],
        "energia_kwh": energia_kwh,
        "demanda_maxima_kw": recibo["demanda_maxima_kw"],
        "demanda_contratada_kw": recibo["demanda_contratada_kw"],
        "carga_conectada_kw": recibo["carga_conectada_kw"],
        "factor_potencia_pct": recibo["factor_potencia_pct"],
        "total_pagar_mxn": total_pagar,
        "subtotal_periodo_mxn": subtotal,
        "precio_medio_total_mxn_kwh": safe_div(total_pagar, energia_kwh),
        "precio_medio_sin_iva_mxn_kwh": safe_div(subtotal, energia_kwh),
        "precio_energia_base_mxn_kwh": safe_div(desglose["energia_mxn"], energia_kwh),
        "cargos_distribucion_capacidad_mxn": cargos_potencia,
        "proporcion_distribucion_capacidad_mem": safe_div(cargos_potencia, total_mem),
        "lectura": (
            "La factura no depende solamente de energia kWh. Una parte importante "
            "esta asociada a demanda, capacidad e infraestructura."
        ),
    }


def calcular_perdidas_por_apagones(
    escenario_costo: str = "medio",
    frecuencia_apagones: Dict = FRECUENCIA_APAGONES_STREGER,
) -> Dict:
    if escenario_costo not in COSTO_EVENTO_APAGON_MXN:
        raise ValueError(f"Escenario no valido: {escenario_costo}")

    costos = COSTO_EVENTO_APAGON_MXN[escenario_costo]
    detalle = {}
    perdida_total = 0.0
    eventos_totales = 0

    for tipo, datos in frecuencia_apagones.items():
        frecuencia = int(datos["frecuencia_anual"])
        costo_evento = float(costos[tipo])
        perdida_tipo = frecuencia * costo_evento
        eventos_totales += frecuencia
        perdida_total += perdida_tipo
        detalle[tipo] = {
            "descripcion": datos["descripcion"],
            "frecuencia_anual": frecuencia,
            "duracion_referencia_horas": datos["duracion_referencia_horas"],
            "costo_promedio_evento_mxn": costo_evento,
            "perdida_anual_mxn": perdida_tipo,
        }

    return {
        "escenario_costo": escenario_costo,
        "eventos_totales_anuales": eventos_totales,
        "perdida_anual_estimada_mxn": perdida_total,
        "perdida_anual_reportada_referencia_mxn": PERDIDA_ANUAL_REPORTADA_REFERENCIA_MXN,
        "diferencia_vs_referencia_mxn": perdida_total - PERDIDA_ANUAL_REPORTADA_REFERENCIA_MXN,
        "proporcion_vs_referencia": safe_div(perdida_total, PERDIDA_ANUAL_REPORTADA_REFERENCIA_MXN),
        "detalle_por_tipo": detalle,
        "lectura": (
            "La perdida se estima desde frecuencia anual y costo promedio por severidad. "
            "El dato reportado de referencia se usa para contraste."
        ),
    }


def calcular_beneficio_evitable_con_respaldo(
    perdidas_apagones: Dict,
    escenario_mitigacion: str = "media",
) -> Dict:
    if escenario_mitigacion not in MITIGACION_BESS_UPS:
        raise ValueError(f"Escenario no valido: {escenario_mitigacion}")

    mitigacion = MITIGACION_BESS_UPS[escenario_mitigacion]
    detalle = {}
    beneficio_total = 0.0

    for tipo, datos in perdidas_apagones["detalle_por_tipo"].items():
        perdida_tipo = datos["perdida_anual_mxn"]
        porcentaje_evitable = mitigacion[tipo]
        beneficio_tipo = perdida_tipo * porcentaje_evitable
        beneficio_total += beneficio_tipo
        detalle[tipo] = {
            **datos,
            "porcentaje_evitable": porcentaje_evitable,
            "beneficio_evitable_mxn": beneficio_tipo,
        }

    perdida_total = perdidas_apagones["perdida_anual_estimada_mxn"]
    return {
        "escenario_mitigacion": escenario_mitigacion,
        "beneficio_anual_evitable_mxn": beneficio_total,
        "porcentaje_total_evitable": safe_div(beneficio_total, perdida_total),
        "detalle_por_tipo": detalle,
        "lectura": (
            "El beneficio evitable representa la fraccion de perdidas que un sistema "
            "UPS/BESS bien integrado podria reducir. No asume eliminacion total del riesgo."
        ),
    }


def dimensionar_bess_respaldo(inputs: InputsBESSFinanciero) -> Dict:
    energia_critica_kwh = inputs.carga_critica_kw * inputs.horas_respaldo
    energia_requerida_kwh = (
        energia_critica_kwh / BATERIA_REFERENCIA_SUNGROW["rte"] * inputs.factor_seguridad
    )
    baterias_equivalentes = safe_div(
        energia_requerida_kwh,
        BATERIA_REFERENCIA_SUNGROW["capacidad_nominal_kwh"],
    )
    return {
        "inputs": asdict(inputs),
        "bateria_referencia": BATERIA_REFERENCIA_SUNGROW,
        "energia_critica_kwh": energia_critica_kwh,
        "capacidad_sugerida_kwh": energia_requerida_kwh,
        "baterias_equivalentes": baterias_equivalentes,
        "potencia_suficiente_una_bateria": (
            inputs.carga_critica_kw <= BATERIA_REFERENCIA_SUNGROW["potencia_nominal_kw"]
        ),
        "lectura": (
            "Predimensionamiento financiero-tecnico. El diseno final requiere matriz "
            "de cargas criticas, autonomia por proceso, selectividad y coordinacion UPS."
        ),
    }


def calcular_ahorro_solar_express(
    energia_autoconsumida_kwh_anual: float,
    precio_medio_mxn_kwh: Optional[float] = None,
) -> Dict:
    diagnostico = diagnostico_recibo_gdmto()
    if precio_medio_mxn_kwh is None:
        precio_medio_mxn_kwh = diagnostico["precio_medio_total_mxn_kwh"]
    ahorro_anual = energia_autoconsumida_kwh_anual * precio_medio_mxn_kwh
    return {
        "energia_autoconsumida_kwh_anual": energia_autoconsumida_kwh_anual,
        "precio_medio_mxn_kwh": precio_medio_mxn_kwh,
        "ahorro_solar_anual_mxn": ahorro_anual,
        "lectura": (
            "Calculo express con precio medio. Debe reemplazarse por calculo "
            "horario/tarifario CFE para estudio definitivo."
        ),
    }


def calcular_payback(inversion_mxn: float, beneficio_anual_mxn: float) -> Optional[float]:
    if beneficio_anual_mxn <= 0:
        return None
    return inversion_mxn / beneficio_anual_mxn


def calcular_roi_simple_anual(
    inversion_mxn: float,
    beneficio_anual_mxn: float,
) -> Optional[float]:
    if inversion_mxn <= 0:
        return None
    return beneficio_anual_mxn / inversion_mxn


def comparar_escenarios_financieros(
    inversion_fv_mxn: float,
    inversion_bess_mxn: float,
    ahorro_solar_anual_mxn: float,
    beneficio_respaldo_anual_mxn: float,
) -> Dict:
    inversion_total = inversion_fv_mxn + inversion_bess_mxn
    escenarios = {
        "base_sin_fv_sin_bess": {
            "inversion_mxn": 0.0,
            "beneficio_anual_mxn": 0.0,
            "payback_anios": None,
            "roi_simple_anual": None,
            "descripcion": "Escenario actual sin inversion.",
        },
        "solo_fv": {
            "inversion_mxn": inversion_fv_mxn,
            "beneficio_anual_mxn": ahorro_solar_anual_mxn,
            "payback_anios": calcular_payback(inversion_fv_mxn, ahorro_solar_anual_mxn),
            "roi_simple_anual": calcular_roi_simple_anual(inversion_fv_mxn, ahorro_solar_anual_mxn),
            "descripcion": "Ahorro por autoconsumo solar.",
        },
        "solo_bess_ups": {
            "inversion_mxn": inversion_bess_mxn,
            "beneficio_anual_mxn": beneficio_respaldo_anual_mxn,
            "payback_anios": calcular_payback(inversion_bess_mxn, beneficio_respaldo_anual_mxn),
            "roi_simple_anual": calcular_roi_simple_anual(inversion_bess_mxn, beneficio_respaldo_anual_mxn),
            "descripcion": "Reduccion de perdidas operativas por apagones.",
        },
        "fv_mas_bess_ups": {
            "inversion_mxn": inversion_total,
            "beneficio_anual_mxn": ahorro_solar_anual_mxn + beneficio_respaldo_anual_mxn,
            "payback_anios": calcular_payback(
                inversion_total,
                ahorro_solar_anual_mxn + beneficio_respaldo_anual_mxn,
            ),
            "roi_simple_anual": calcular_roi_simple_anual(
                inversion_total,
                ahorro_solar_anual_mxn + beneficio_respaldo_anual_mxn,
            ),
            "descripcion": "Ahorro energetico mas continuidad operativa.",
        },
    }
    return {
        "inputs": {
            "inversion_fv_mxn": inversion_fv_mxn,
            "inversion_bess_mxn": inversion_bess_mxn,
            "ahorro_solar_anual_mxn": ahorro_solar_anual_mxn,
            "beneficio_respaldo_anual_mxn": beneficio_respaldo_anual_mxn,
        },
        "escenarios": escenarios,
    }


def _pick_column(df: pd.DataFrame, candidates: list[str]) -> str:
    normalized = {str(col).strip().lower(): str(col) for col in df.columns}
    for candidate in candidates:
        found = normalized.get(candidate.lower())
        if found is not None:
            return found
    raise ValueError(
        "No se encontro ninguna columna esperada. Opciones: "
        + ", ".join(candidates)
    )


def calcular_autoconsumo_desde_simulacion(
    df_simulacion: pd.DataFrame,
    timestep_hours: float = TIMESTEP_HOURS,
) -> float:
    if "Energia_Autoconsumida_kWh" in df_simulacion.columns:
        return float(pd.to_numeric(df_simulacion["Energia_Autoconsumida_kWh"], errors="coerce").sum())

    demand_col = _pick_column(df_simulacion, ["Demanda_kW", "demanda_kw"])
    generation_col = _pick_column(
        df_simulacion,
        ["Generacion_Solar_kW", "Generacion_AC_kW", "generacion_solar_kw"],
    )
    demand_kw = pd.to_numeric(df_simulacion[demand_col], errors="coerce").fillna(0)
    generation_kw = pd.to_numeric(df_simulacion[generation_col], errors="coerce").fillna(0)
    return float(np.minimum(demand_kw, generation_kw).clip(lower=0).sum() * timestep_hours)


def generar_resumen_financiero_streger(
    escenario_costo_apagon: str = "medio",
    escenario_mitigacion: str = "media",
    carga_critica_kw: float = 30,
    horas_respaldo: float = 4,
    inversion_fv_mxn: float = 1_500_000,
    inversion_bess_mxn: float = 1_200_000,
    energia_autoconsumida_kwh_anual: float = 50_000,
) -> Dict:
    diagnostico = diagnostico_recibo_gdmto()
    perdidas_apagones = calcular_perdidas_por_apagones(escenario_costo_apagon)
    beneficio_respaldo = calcular_beneficio_evitable_con_respaldo(
        perdidas_apagones,
        escenario_mitigacion,
    )
    bess_financiero = dimensionar_bess_respaldo(
        InputsBESSFinanciero(
            carga_critica_kw=carga_critica_kw,
            horas_respaldo=horas_respaldo,
        )
    )
    ahorro_solar = calcular_ahorro_solar_express(energia_autoconsumida_kwh_anual)
    comparacion = comparar_escenarios_financieros(
        inversion_fv_mxn=inversion_fv_mxn,
        inversion_bess_mxn=inversion_bess_mxn,
        ahorro_solar_anual_mxn=ahorro_solar["ahorro_solar_anual_mxn"],
        beneficio_respaldo_anual_mxn=beneficio_respaldo["beneficio_anual_evitable_mxn"],
    )
    return {
        "diagnostico_recibo": diagnostico,
        "frecuencia_apagones_oficial": FRECUENCIA_APAGONES_STREGER,
        "proteccion_existente": PROTECCION_EXISTENTE_STREGER,
        "perdidas_apagones": perdidas_apagones,
        "beneficio_respaldo": beneficio_respaldo,
        "dimensionamiento_bess_financiero": bess_financiero,
        "ahorro_solar_express": ahorro_solar,
        "comparacion_escenarios": comparacion,
        "nota": (
            "Analisis preliminar. No sustituye auditoria electrica, cotizacion formal, "
            "estudio tarifario CFE ni ingenieria de detalle."
        ),
    }


def calcular_finanzas_desde_simulacion(
    df_simulacion: pd.DataFrame,
    escenario_costo_apagon: str = "medio",
    escenario_mitigacion: str = "media",
    carga_critica_kw: float = 30,
    horas_respaldo: float = 4,
    inversion_fv_mxn: float = 1_500_000,
    inversion_bess_mxn: float = 1_200_000,
    timestep_hours: float = TIMESTEP_HOURS,
) -> Dict:
    energia_autoconsumida = calcular_autoconsumo_desde_simulacion(
        df_simulacion,
        timestep_hours=timestep_hours,
    )
    resumen = generar_resumen_financiero_streger(
        escenario_costo_apagon=escenario_costo_apagon,
        escenario_mitigacion=escenario_mitigacion,
        carga_critica_kw=carga_critica_kw,
        horas_respaldo=horas_respaldo,
        inversion_fv_mxn=inversion_fv_mxn,
        inversion_bess_mxn=inversion_bess_mxn,
        energia_autoconsumida_kwh_anual=energia_autoconsumida,
    )
    resumen["fuente_autoconsumo"] = {
        "metodo": "Energia_Autoconsumida_kWh si existe; si no, min(Demanda_kW, Generacion_Solar_kW) * 0.25",
        "energia_autoconsumida_kwh_anual": energia_autoconsumida,
    }
    return resumen


def detalle_apagones_financiero_df(
    perdidas_apagones: Dict,
    beneficio_respaldo: Dict,
) -> pd.DataFrame:
    rows = []
    for tipo, datos_perdida in perdidas_apagones["detalle_por_tipo"].items():
        datos_beneficio = beneficio_respaldo["detalle_por_tipo"][tipo]
        rows.append(
            {
                "tipo_apagon": tipo,
                "descripcion": datos_perdida["descripcion"],
                "frecuencia_anual": datos_perdida["frecuencia_anual"],
                "costo_promedio_evento_mxn": datos_perdida["costo_promedio_evento_mxn"],
                "perdida_anual_mxn": datos_perdida["perdida_anual_mxn"],
                "porcentaje_evitable": datos_beneficio["porcentaje_evitable"],
                "beneficio_evitable_mxn": datos_beneficio["beneficio_evitable_mxn"],
            }
        )
    return pd.DataFrame(rows)


def escenarios_financieros_df(comparacion_escenarios: Dict) -> pd.DataFrame:
    nombres = {
        "base_sin_fv_sin_bess": "Base sin FV ni BESS",
        "solo_fv": "Solo FV",
        "solo_bess_ups": "Solo BESS/UPS",
        "fv_mas_bess_ups": "FV + BESS/UPS",
    }
    rows = []
    for key, datos in comparacion_escenarios["escenarios"].items():
        rows.append({"escenario": key, "nombre": nombres.get(key, key), **datos})
    return pd.DataFrame(rows)


def exportar_resumen_json(resumen: Dict, ruta: str | Path) -> Path:
    path = Path(ruta)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
