"""Version Streamlit unificada para prueba provisional de portabilidad."""

from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import math
import traceback
import urllib.parse
import urllib.request
import warnings
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from io import BytesIO, StringIO
from pathlib import Path
from typing import Dict, Optional
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pvlib
import seaborn as sns
import streamlit as st


# =============================================================================
# SECCION MOTOR SOLAR-INDUSTRIAL
# =============================================================================

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FINAL_FREQUENCY = "15min"
TIMESTEP_HOURS = 0.25

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "cloud_cover",
    "wind_speed_10m",
    "shortwave_radiation",
    "direct_normal_irradiance",
    "diffuse_radiation",
    "direct_radiation",
]


@dataclass(frozen=True)
class SiteConfig:
    name: str = "Veracruz"
    latitude: float = 19.48
    longitude: float = -96.95
    timezone: str = "America/Mexico_City"
    altitude_m: float = 500.0
    surface_tilt: float = 19.5
    surface_azimuth: float = 180.0
    albedo: float = 0.2


@dataclass(frozen=True)
class ModuleConfig:
    key: str
    manufacturer: str
    model: str
    module_type: str
    technology: str
    pdc0_w: float
    efficiency: float
    area_m2: float
    mass_kg: float
    gamma_pdc_per_c: float
    voc_v: float
    isc_a: float
    vmp_v: float
    imp_a: float
    default_bifacial_gain: float = 0.0
    bifacial_factor: float | None = None


MODULE_OPTIONS = {
    "monofacial": ModuleConfig(
        key="monofacial",
        manufacturer="Jinko Solar",
        model="Tiger Neo 72HC JKM605N-72HL4",
        module_type="Monofacial",
        technology="n-type TOPCon",
        pdc0_w=605.0,
        efficiency=0.2342,
        area_m2=2.583,
        mass_kg=27.0,
        gamma_pdc_per_c=-0.0029,
        voc_v=53.11,
        isc_a=14.31,
        vmp_v=44.23,
        imp_a=13.68,
    ),
    "bifacial": ModuleConfig(
        key="bifacial",
        manufacturer="Jinko Solar",
        model="Tiger Neo N-type JKM625N-78HL4-BDV",
        module_type="Bifacial doble vidrio",
        technology="n-type",
        pdc0_w=625.0,
        efficiency=0.2236,
        area_m2=2.795,
        mass_kg=34.6,
        gamma_pdc_per_c=-0.0029,
        voc_v=55.72,
        isc_a=14.27,
        vmp_v=46.10,
        imp_a=13.56,
        default_bifacial_gain=0.15,
        bifacial_factor=0.80,
    ),
}


# =============================================================================
# SECCION MOTOR BESS
# =============================================================================

@dataclass(frozen=True)
class EspecificacionesBESS:
    modelo: str = "Sungrow PowerStack 255CS (ST255CS-2H)"
    quimica: str = "LFP (Litio Ferro-Fosfato) - ciclos profundos"
    capacidad_nominal_kwh: float = 257.0
    potencia_nominal_ac_kw: float = 125.0
    rte: float = 0.90
    ciclos_minimos: int = 4_000
    ciclos_maximos: int = 5_000
    vida_util_anos_min: int = 10
    vida_util_anos_max: int = 15


@dataclass
class ResultadoDimensionamiento:
    horas_respaldo: float
    carga_critica_kw: float
    capacidad_requerida_kwh: float
    unidades_bess: int
    potencia_total_kw: float
    capacidad_total_kwh: float
    ciclos_referencia: int
    modelo_referencia: str
    sobredimensionamiento_pct: float


@dataclass
class ResultadoAnalisisCortes:
    cortes_largos_ano: int
    cortes_medios_ano: int
    cortes_cortos_ano: int
    total_eventos_ano: int
    horas_desabasto_estimadas: float
    nivel_riesgo: str
    justificacion_riesgo: str
    autonomia_recomendada_hrs: float
    impacto_anual_estimado: dict = field(default_factory=dict)


class MotorBESS:
    _DURACION_CORTE_LARGO_HRS: float = 2.5
    _DURACION_CORTE_MEDIO_HRS: float = 0.10
    _DURACION_CORTE_CORTO_HRS: float = 0.003

    def __init__(self, especificaciones: Optional[EspecificacionesBESS] = None):
        self.specs = especificaciones if especificaciones else EspecificacionesBESS()

    def calcular_dimensionamiento(
        self,
        horas_respaldo: float,
        carga_critica_kw: float,
    ) -> ResultadoDimensionamiento:
        if horas_respaldo <= 0:
            raise ValueError(f"horas_respaldo debe ser mayor a 0. Recibido: {horas_respaldo}")
        if carga_critica_kw <= 0:
            raise ValueError(f"carga_critica_kw debe ser mayor a 0. Recibido: {carga_critica_kw}")

        capacidad_requerida_kwh = (carga_critica_kw * horas_respaldo) / self.specs.rte
        unidades_bess = math.ceil(capacidad_requerida_kwh / self.specs.capacidad_nominal_kwh)
        potencia_total_kw = self.specs.potencia_nominal_ac_kw * unidades_bess
        capacidad_total_kwh = self.specs.capacidad_nominal_kwh * unidades_bess
        ciclos_referencia = (self.specs.ciclos_minimos + self.specs.ciclos_maximos) // 2
        sobredimensionamiento_pct = (
            (capacidad_total_kwh - capacidad_requerida_kwh) / capacidad_requerida_kwh * 100
        )

        return ResultadoDimensionamiento(
            horas_respaldo=horas_respaldo,
            carga_critica_kw=carga_critica_kw,
            capacidad_requerida_kwh=round(capacidad_requerida_kwh, 2),
            unidades_bess=unidades_bess,
            potencia_total_kw=potencia_total_kw,
            capacidad_total_kwh=capacidad_total_kwh,
            ciclos_referencia=ciclos_referencia,
            modelo_referencia=self.specs.modelo,
            sobredimensionamiento_pct=round(sobredimensionamiento_pct, 1),
        )

    def analizar_historico_cortes(
        self,
        frecuencia_largos: int = 3,
        frecuencia_medios: int = 12,
        frecuencia_cortos: int = 24,
    ) -> ResultadoAnalisisCortes:
        for nombre, valor in [
            ("frecuencia_largos", frecuencia_largos),
            ("frecuencia_medios", frecuencia_medios),
            ("frecuencia_cortos", frecuencia_cortos),
        ]:
            if valor < 0:
                raise ValueError(f"{nombre} no puede ser negativo. Recibido: {valor}")

        total_eventos = frecuencia_largos + frecuencia_medios + frecuencia_cortos
        horas_largos = frecuencia_largos * self._DURACION_CORTE_LARGO_HRS
        horas_medios = frecuencia_medios * self._DURACION_CORTE_MEDIO_HRS
        horas_cortos = frecuencia_cortos * self._DURACION_CORTE_CORTO_HRS
        horas_desabasto_total = round(horas_largos + horas_medios + horas_cortos, 2)
        impacto_anual = {
            "largos": {
                "eventos": frecuencia_largos,
                "duracion_representativa_hrs": self._DURACION_CORTE_LARGO_HRS,
                "horas_desabasto_acumuladas": round(horas_largos, 2),
                "descripcion": "Apagones >1 hr: riesgo de dano en equipos y perdida de proceso",
            },
            "medios": {
                "eventos": frecuencia_medios,
                "duracion_representativa_hrs": self._DURACION_CORTE_MEDIO_HRS,
                "horas_desabasto_acumuladas": round(horas_medios, 2),
                "descripcion": "Apagones 1-10 min: reset de PLC/SCADA y scrap",
            },
            "cortos": {
                "eventos": frecuencia_cortos,
                "duracion_representativa_hrs": self._DURACION_CORTE_CORTO_HRS,
                "horas_desabasto_acumuladas": round(horas_cortos, 2),
                "descripcion": "Microapagones <1 min: reinicios de equipo sensible",
            },
        }

        nivel_riesgo, justificacion, autonomia_recomendada = self._clasificar_riesgo(
            horas_desabasto=horas_desabasto_total,
            frecuencia_largos=frecuencia_largos,
            total_eventos=total_eventos,
        )

        return ResultadoAnalisisCortes(
            cortes_largos_ano=frecuencia_largos,
            cortes_medios_ano=frecuencia_medios,
            cortes_cortos_ano=frecuencia_cortos,
            total_eventos_ano=total_eventos,
            horas_desabasto_estimadas=horas_desabasto_total,
            nivel_riesgo=nivel_riesgo,
            justificacion_riesgo=justificacion,
            autonomia_recomendada_hrs=autonomia_recomendada,
            impacto_anual_estimado=impacto_anual,
        )

    def _clasificar_riesgo(
        self,
        horas_desabasto: float,
        frecuencia_largos: int,
        total_eventos: int,
    ) -> tuple[str, str, float]:
        if horas_desabasto < 5:
            riesgo_horas = "Bajo"
        elif horas_desabasto <= 20:
            riesgo_horas = "Medio"
        else:
            riesgo_horas = "Alto"

        if frecuencia_largos <= 1:
            riesgo_severidad = "Bajo"
        elif frecuencia_largos <= 4:
            riesgo_severidad = "Medio"
        else:
            riesgo_severidad = "Alto"

        niveles = {"Bajo": 1, "Medio": 2, "Alto": 3}
        nivel_final = max(riesgo_horas, riesgo_severidad, key=lambda n: niveles[n])
        autonomia_map = {"Bajo": 2.0, "Medio": 4.0, "Alto": 8.0}
        autonomia = autonomia_map[nivel_final]

        justificacion_map = {
            "Bajo": (
                f"Con {horas_desabasto:.1f} horas anuales de desabasto y "
                f"{frecuencia_largos} apagones largos, el riesgo operativo es manejable. "
                f"Se recomienda un BESS de {autonomia:.0f} horas como proteccion base."
            ),
            "Medio": (
                f"Con {horas_desabasto:.1f} horas anuales de desabasto, {total_eventos} "
                f"eventos totales y {frecuencia_largos} apagones largos, la operacion "
                f"enfrenta un riesgo medio. Un BESS de {autonomia:.0f} horas cubre la "
                "mayoria de eventos historicos y protege continuidad de proceso."
            ),
            "Alto": (
                f"Perfil de riesgo alto: {horas_desabasto:.1f} horas anuales de desabasto, "
                f"{frecuencia_largos} apagones severos y {total_eventos} eventos totales. "
                f"Se justifica evaluar un arreglo BESS de {autonomia:.0f} horas para "
                "continuidad operativa."
            ),
        }
        return nivel_final, justificacion_map[nivel_final], autonomia

    def generar_resumen_ejecutivo(
        self,
        horas_respaldo: float,
        carga_critica_kw: float,
        frecuencia_largos: int = 3,
        frecuencia_medios: int = 12,
        frecuencia_cortos: int = 24,
    ) -> str:
        dim = self.calcular_dimensionamiento(horas_respaldo, carga_critica_kw)
        ana = self.analizar_historico_cortes(
            frecuencia_largos, frecuencia_medios, frecuencia_cortos
        )
        return f"""
RESUMEN EJECUTIVO - SISTEMA BESS DE RESPALDO

EQUIPO DE REFERENCIA: {dim.modelo_referencia}
QUIMICA: {self.specs.quimica}

DIMENSIONAMIENTO FISICO
Carga critica protegida: {dim.carga_critica_kw:,.0f} kW
Autonomia solicitada: {dim.horas_respaldo:.1f} horas
Capacidad requerida: {dim.capacidad_requerida_kwh:,.2f} kWh (incluye RTE {self.specs.rte * 100:.0f}%)
Gabinetes necesarios: {dim.unidades_bess}
Potencia AC instalada: {dim.potencia_total_kw:,.0f} kW
Capacidad instalada: {dim.capacidad_total_kwh:,.0f} kWh
Margen de energia: {dim.sobredimensionamiento_pct:.1f}%
Vida util estimada: {dim.ciclos_referencia:,} ciclos

ANALISIS DE RIESGO
Apagones largos (>1 hr): {ana.cortes_largos_ano} eventos/ano
Apagones medios (1-10 min): {ana.cortes_medios_ano} eventos/ano
Microapagones (<1 min): {ana.cortes_cortos_ano} eventos/ano
Total de eventos: {ana.total_eventos_ano} eventos/ano
Horas de desabasto: {ana.horas_desabasto_estimadas:.2f} hrs/ano
Nivel de riesgo: {ana.nivel_riesgo}
Autonomia recomendada: {ana.autonomia_recomendada_hrs:.0f} horas

JUSTIFICACION
{ana.justificacion_riesgo}
""".strip()


# =============================================================================
# SECCION MOTOR FINANCIERO STREGER
# =============================================================================

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
    "conservador": {
        "largo": 150_000,
        "medio": 10_000,
        "corto": 2_000,
    },
    "medio": {
        "largo": 350_000,
        "medio": 35_000,
        "corto": 8_000,
    },
    "alto": {
        "largo": 700_000,
        "medio": 90_000,
        "corto": 25_000,
    },
}


MITIGACION_BESS_UPS = {
    "conservadora": {
        "largo": 0.40,
        "medio": 0.60,
        "corto": 0.70,
    },
    "media": {
        "largo": 0.55,
        "medio": 0.75,
        "corto": 0.85,
    },
    "alta": {
        "largo": 0.70,
        "medio": 0.85,
        "corto": 0.95,
    },
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


def safe_div(numerador: float, denominador: float) -> float:
    if denominador == 0:
        return 0.0
    return numerador / denominador


def formato_mxn(valor: float) -> str:
    return f"${valor:,.2f} MXN"


def formato_pct(valor: float) -> str:
    return f"{valor * 100:.1f}%"


def formato_anios(valor: Optional[float]) -> str:
    if valor is None:
        return "No aplica"
    return f"{valor:.2f} anos"


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

    precio_medio_total = safe_div(total_pagar, energia_kwh)
    precio_medio_sin_iva = safe_div(subtotal, energia_kwh)
    precio_energia_base = safe_div(desglose["energia_mxn"], energia_kwh)
    proporcion_potencia_mem = safe_div(cargos_potencia, total_mem)

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
        "precio_medio_total_mxn_kwh": precio_medio_total,
        "precio_medio_sin_iva_mxn_kwh": precio_medio_sin_iva,
        "precio_energia_base_mxn_kwh": precio_energia_base,
        "cargos_distribucion_capacidad_mxn": cargos_potencia,
        "proporcion_distribucion_capacidad_mem": proporcion_potencia_mem,
        "lectura": (
            "La factura no depende solamente de la energia consumida en kWh. "
            "Una parte importante esta asociada a demanda, capacidad e infraestructura."
        ),
    }


def calcular_perdidas_por_apagones(
    escenario_costo: str = "medio",
    frecuencia_apagones: Dict = FRECUENCIA_APAGONES_STREGER,
) -> Dict:
    if escenario_costo not in COSTO_EVENTO_APAGON_MXN:
        raise ValueError(
            f"Escenario no valido: {escenario_costo}. "
            f"Opciones: {list(COSTO_EVENTO_APAGON_MXN.keys())}"
        )

    costos = COSTO_EVENTO_APAGON_MXN[escenario_costo]
    detalle = {}
    perdida_total = 0.0
    eventos_totales = 0

    for tipo, datos in frecuencia_apagones.items():
        frecuencia = datos["frecuencia_anual"]
        costo_evento = costos[tipo]
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

    diferencia_vs_referencia = perdida_total - PERDIDA_ANUAL_REPORTADA_REFERENCIA_MXN
    proporcion_vs_referencia = safe_div(
        perdida_total,
        PERDIDA_ANUAL_REPORTADA_REFERENCIA_MXN,
    )

    return {
        "escenario_costo": escenario_costo,
        "eventos_totales_anuales": eventos_totales,
        "perdida_anual_estimada_mxn": perdida_total,
        "perdida_anual_reportada_referencia_mxn": PERDIDA_ANUAL_REPORTADA_REFERENCIA_MXN,
        "diferencia_vs_referencia_mxn": diferencia_vs_referencia,
        "proporcion_vs_referencia": proporcion_vs_referencia,
        "detalle_por_tipo": detalle,
        "lectura": (
            "El dato reportado por Streger se usa como referencia de validacion, "
            "no como centro del calculo. La perdida se estima desde frecuencia "
            "oficial y costo externo por severidad."
        ),
    }


def calcular_beneficio_evitable_con_respaldo(
    perdidas_apagones: Dict,
    escenario_mitigacion: str = "media",
) -> Dict:
    if escenario_mitigacion not in MITIGACION_BESS_UPS:
        raise ValueError(
            f"Escenario no valido: {escenario_mitigacion}. "
            f"Opciones: {list(MITIGACION_BESS_UPS.keys())}"
        )

    mitigacion = MITIGACION_BESS_UPS[escenario_mitigacion]
    detalle = {}
    beneficio_total = 0.0

    for tipo, datos in perdidas_apagones["detalle_por_tipo"].items():
        perdida_tipo = datos["perdida_anual_mxn"]
        porcentaje_evitable = mitigacion[tipo]
        beneficio_tipo = perdida_tipo * porcentaje_evitable
        beneficio_total += beneficio_tipo

        detalle[tipo] = {
            "perdida_anual_mxn": perdida_tipo,
            "porcentaje_evitable": porcentaje_evitable,
            "beneficio_evitable_mxn": beneficio_tipo,
        }

    perdida_total = perdidas_apagones["perdida_anual_estimada_mxn"]
    porcentaje_total_evitable = safe_div(beneficio_total, perdida_total)

    return {
        "escenario_mitigacion": escenario_mitigacion,
        "beneficio_anual_evitable_mxn": beneficio_total,
        "porcentaje_total_evitable": porcentaje_total_evitable,
        "detalle_por_tipo": detalle,
        "lectura": (
            "El beneficio evitable representa la fraccion de perdidas que un sistema "
            "UPS/BESS bien integrado podria reducir. No se asume eliminacion total "
            "del riesgo."
        ),
    }


@dataclass
class InputsBESSFinanciero:
    carga_critica_kw: float
    horas_respaldo: float
    dod: float = 0.80
    eficiencia_sistema: float = 0.90
    margen_seguridad: float = 1.10


def dimensionar_bess_respaldo(inputs: InputsBESSFinanciero) -> Dict:
    energia_critica_kwh = inputs.carga_critica_kw * inputs.horas_respaldo
    capacidad_nominal_minima_kwh = safe_div(
        energia_critica_kwh,
        inputs.dod * inputs.eficiencia_sistema,
    )
    capacidad_sugerida_kwh = capacidad_nominal_minima_kwh * inputs.margen_seguridad
    bateria = BATERIA_REFERENCIA_SUNGROW
    baterias_equivalentes = safe_div(
        capacidad_sugerida_kwh,
        bateria["capacidad_nominal_kwh"],
    )
    potencia_suficiente_una_bateria = (
        inputs.carga_critica_kw <= bateria["potencia_nominal_kw"]
    )

    return {
        "inputs": asdict(inputs),
        "energia_critica_kwh": energia_critica_kwh,
        "capacidad_nominal_minima_kwh": capacidad_nominal_minima_kwh,
        "capacidad_sugerida_kwh": capacidad_sugerida_kwh,
        "bateria_referencia": bateria,
        "baterias_equivalentes": baterias_equivalentes,
        "potencia_suficiente_una_bateria": potencia_suficiente_una_bateria,
        "lectura": (
            "Este es un predimensionamiento financiero-tecnico. El diseno final "
            "requiere revision electrica, transferencia, protecciones, tableros, "
            "UPS para microcortes y seleccion real de cargas criticas."
        ),
    }


def dimensionar_bess_respaldo_financiero(
    carga_critica_kw: float,
    horas_respaldo: float,
) -> Dict:
    return dimensionar_bess_respaldo(
        InputsBESSFinanciero(
            carga_critica_kw=carga_critica_kw,
            horas_respaldo=horas_respaldo,
        )
    )


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
            "Calculo express con precio medio. Despues debe reemplazarse por "
            "un calculo horario/tarifario usando la salida del motor solar."
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
            "payback_anios": calcular_payback(
                inversion_fv_mxn,
                ahorro_solar_anual_mxn,
            ),
            "roi_simple_anual": calcular_roi_simple_anual(
                inversion_fv_mxn,
                ahorro_solar_anual_mxn,
            ),
            "descripcion": "Ahorro por autoconsumo solar.",
        },
        "solo_bess_ups": {
            "inversion_mxn": inversion_bess_mxn,
            "beneficio_anual_mxn": beneficio_respaldo_anual_mxn,
            "payback_anios": calcular_payback(
                inversion_bess_mxn,
                beneficio_respaldo_anual_mxn,
            ),
            "roi_simple_anual": calcular_roi_simple_anual(
                inversion_bess_mxn,
                beneficio_respaldo_anual_mxn,
            ),
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


def generar_resumen_financiero_streger(
    escenario_costo_apagon: str = "medio",
    escenario_mitigacion: str = "media",
    carga_critica_kw: float = 30,
    horas_respaldo: float = 4,
    energia_autoconsumida_kwh_anual: float = 50_000,
    inversion_fv_mxn: float = 1_500_000,
    inversion_bess_mxn: float = 1_200_000,
) -> Dict:
    diagnostico = diagnostico_recibo_gdmto()
    perdidas_apagones = calcular_perdidas_por_apagones(
        escenario_costo=escenario_costo_apagon,
    )
    beneficio_respaldo = calcular_beneficio_evitable_con_respaldo(
        perdidas_apagones=perdidas_apagones,
        escenario_mitigacion=escenario_mitigacion,
    )
    bess = dimensionar_bess_respaldo_financiero(
        carga_critica_kw=carga_critica_kw,
        horas_respaldo=horas_respaldo,
    )
    ahorro_solar = calcular_ahorro_solar_express(
        energia_autoconsumida_kwh_anual=energia_autoconsumida_kwh_anual,
        precio_medio_mxn_kwh=diagnostico["precio_medio_total_mxn_kwh"],
    )
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
        "dimensionamiento_bess": bess,
        "ahorro_solar_express": ahorro_solar,
        "comparacion_escenarios": comparacion,
    }


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
                "frecuencia_anual": datos_perdida["frecuencia_anual"],
                "descripcion": datos_perdida["descripcion"],
                "costo_promedio_evento_mxn": datos_perdida["costo_promedio_evento_mxn"],
                "perdida_anual_mxn": datos_perdida["perdida_anual_mxn"],
                "porcentaje_evitable": datos_beneficio["porcentaje_evitable"],
                "beneficio_evitable_mxn": datos_beneficio["beneficio_evitable_mxn"],
            }
        )
    return pd.DataFrame(rows)


def escenarios_financieros_df(comparacion_escenarios: Dict) -> pd.DataFrame:
    rows = []
    etiquetas = {
        "base_sin_fv_sin_bess": "Base sin FV ni BESS",
        "solo_fv": "Solo FV",
        "solo_bess_ups": "Solo BESS/UPS",
        "fv_mas_bess_ups": "FV + BESS/UPS",
    }
    for clave, datos in comparacion_escenarios["escenarios"].items():
        rows.append(
            {
                "escenario": clave,
                "nombre": etiquetas.get(clave, clave),
                "inversion_mxn": datos["inversion_mxn"],
                "beneficio_anual_mxn": datos["beneficio_anual_mxn"],
                "payback_anios": datos["payback_anios"],
                "roi_simple_anual": datos["roi_simple_anual"],
                "descripcion": datos["descripcion"],
            }
        )
    return pd.DataFrame(rows)


def figura_financiera_base(height: int = 340) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#12171f",
        font=dict(family="IBM Plex Mono", color="#8892a4"),
        height=height,
        margin=dict(l=10, r=10, t=48, b=40),
        legend=dict(orientation="h", y=1.1),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
    return fig


def default_date_range(timezone: str) -> tuple[str, str]:
    """Return the last complete calendar year as ISO dates."""
    today = datetime.now(ZoneInfo(timezone)).date()
    year = today.year - 1
    return f"{year}-01-01", f"{year}-12-31"


def parse_iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Fecha invalida: {value!r}. Usa formato YYYY-MM-DD."
        ) from exc


def validate_date_range(start_date: date, end_date: date, timezone: str) -> None:
    if end_date < start_date:
        raise ValueError("La fecha final no puede ser anterior a la fecha inicial.")

    today = datetime.now(ZoneInfo(timezone)).date()
    if end_date > today:
        raise ValueError(
            "Open-Meteo historico no sirve fechas futuras. "
            f"Fecha final recibida: {end_date}; hoy es {today}."
        )


def build_open_meteo_url(site: SiteConfig, start_date: date, end_date: date) -> str:
    params = {
        "latitude": site.latitude,
        "longitude": site.longitude,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": site.timezone,
        "timeformat": "unixtime",
        "wind_speed_unit": "ms",
    }
    return f"{OPEN_METEO_ARCHIVE_URL}?{urllib.parse.urlencode(params)}"


def fetch_historical_weather(
    site: SiteConfig, start_date: date, end_date: date
) -> tuple[pd.DataFrame, str]:
    """Download hourly historical weather from Open-Meteo."""
    validate_date_range(start_date, end_date, site.timezone)
    url = build_open_meteo_url(site, start_date, end_date)

    with urllib.request.urlopen(url, timeout=90) as response:
        payload = json.loads(response.read().decode("utf-8"))

    if payload.get("error"):
        reason = payload.get("reason", "sin detalle")
        raise RuntimeError(f"Open-Meteo rechazo la solicitud: {reason}")

    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        raise RuntimeError("La respuesta de Open-Meteo no contiene datos horarios.")

    weather = pd.DataFrame(hourly)
    weather["time"] = pd.to_datetime(weather["time"], unit="s", utc=True).dt.tz_convert(
        site.timezone
    )
    weather = weather.set_index("time").sort_index()
    weather = weather[~weather.index.duplicated(keep="first")]

    for column in weather.columns:
        weather[column] = pd.to_numeric(weather[column], errors="coerce")

    return weather, url


def build_target_index(
    start_date: date,
    end_date: date,
    timezone: str,
    freq: str = FINAL_FREQUENCY,
) -> pd.DatetimeIndex:
    """Create the weather master index, including the final interval of each day."""
    requested_offset = pd.tseries.frequencies.to_offset(freq)
    final_offset = pd.tseries.frequencies.to_offset(FINAL_FREQUENCY)
    if requested_offset != final_offset:
        warnings.warn(
            f"La frecuencia solicitada {freq!r} se remuestreara a {FINAL_FREQUENCY!r}; "
            "el motor industrial trabaja en intervalos de 15 minutos.",
            stacklevel=2,
        )

    start = pd.Timestamp(start_date, tz=timezone)
    end = pd.Timestamp(end_date + pd.Timedelta(days=1), tz=timezone) - final_offset
    return pd.date_range(start=start, end=end, freq=FINAL_FREQUENCY)


def _interpolate_to_target(
    series: pd.Series,
    target_index: pd.DatetimeIndex,
    fallback: float,
) -> pd.Series:
    expanded_index = series.index.union(target_index)
    return (
        pd.to_numeric(series, errors="coerce")
        .reindex(expanded_index)
        .interpolate(method="time")
        .ffill()
        .bfill()
        .reindex(target_index)
        .fillna(fallback)
    )


def _clearness_ratio_to_target(
    observed: pd.Series,
    clear_sky_hourly: pd.Series,
    target_index: pd.DatetimeIndex,
    upper_limit: float,
) -> pd.Series:
    ratio = observed.clip(lower=0) / clear_sky_hourly.where(clear_sky_hourly > 10)
    ratio = ratio.replace([np.inf, -np.inf], np.nan).clip(lower=0, upper=upper_limit)
    expanded_index = ratio.index.union(target_index)
    return (
        ratio.reindex(expanded_index)
        .interpolate(method="time")
        .ffill()
        .bfill()
        .reindex(target_index)
        .fillna(0)
    )


def resample_weather_to_15min(
    weather: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    site: SiteConfig,
) -> pd.DataFrame:
    """Convert hourly weather to 15 minutes using clear-sky radiation profiles.

    Scalar weather variables are interpolated in time. For GHI, DNI and DHI,
    the hourly ratio against clear sky is interpolated and applied to a
    15-minute clear-sky curve. This keeps sunrise, sunset and nighttime
    physically consistent instead of drawing straight radiation ramps.
    """
    if target_index.tz is None:
        raise ValueError("El indice maestro climatico debe incluir zona horaria.")

    weather = weather.copy()
    if weather.index.tz is None:
        weather.index = weather.index.tz_localize(site.timezone)
    else:
        weather.index = weather.index.tz_convert(site.timezone)

    location = pvlib.location.Location(
        latitude=site.latitude,
        longitude=site.longitude,
        tz=site.timezone,
        altitude=site.altitude_m,
        name=site.name,
    )
    clear_hourly = location.get_clearsky(weather.index, model="ineichen")
    clear_15min = location.get_clearsky(target_index, model="ineichen")
    solpos_15min = location.get_solarposition(target_index)
    daylight = solpos_15min["apparent_elevation"] > 0

    result = pd.DataFrame(index=target_index)
    scalar_fallbacks = {
        "temperature_2m": 25.0,
        "relative_humidity_2m": 0.0,
        "cloud_cover": 0.0,
        "wind_speed_10m": 1.0,
    }
    for column, fallback in scalar_fallbacks.items():
        if column not in weather.columns:
            warnings.warn(
                f"Open-Meteo no entrego {column!r}; se usara fallback {fallback}.",
                stacklevel=2,
            )
            result[column] = fallback
        else:
            result[column] = _interpolate_to_target(
                weather[column], target_index, fallback=fallback
            )

    if "precipitation" in weather.columns:
        # Open-Meteo reports an hourly accumulation. Distribute it over 4 slots.
        result["precipitation"] = (
            pd.to_numeric(weather["precipitation"], errors="coerce")
            .fillna(0)
            .clip(lower=0)
            .reindex(target_index, method="ffill")
            .fillna(0)
            / 4
        )
    else:
        result["precipitation"] = 0.0

    radiation_config = {
        "shortwave_radiation": ("ghi", 1.5),
        "direct_normal_irradiance": ("dni", 1.5),
        "diffuse_radiation": ("dhi", 3.0),
    }
    for weather_column, (clear_column, upper_limit) in radiation_config.items():
        if weather_column not in weather.columns:
            raise ValueError(
                f"Open-Meteo no entrego la variable de radiacion {weather_column!r}."
            )
        ratio = _clearness_ratio_to_target(
            pd.to_numeric(weather[weather_column], errors="coerce").fillna(0),
            clear_hourly[clear_column],
            target_index,
            upper_limit=upper_limit,
        )
        result[weather_column] = (
            clear_15min[clear_column].mul(ratio).where(daylight, 0).clip(lower=0)
        )

    cosine_zenith = np.cos(np.radians(solpos_15min["apparent_zenith"])).clip(lower=0)
    result["direct_radiation"] = (
        result["direct_normal_irradiance"].mul(cosine_zenith).where(daylight, 0)
    )
    return result


def _localize_datetime_series(
    timestamps: pd.Series,
    timezone: str,
    column_name: str,
) -> pd.Series:
    if timestamps.isna().any():
        invalid_count = int(timestamps.isna().sum())
        raise ValueError(
            f"Hay {invalid_count} fechas invalidas en la columna {column_name!r}."
        )

    if timestamps.dt.tz is None:
        return timestamps.dt.tz_localize(
            timezone, ambiguous="infer", nonexistent="shift_forward"
        )
    return timestamps.dt.tz_convert(timezone)


def _read_demand_file(demand_file: Path) -> pd.DataFrame:
    if not demand_file.exists():
        raise FileNotFoundError(
            f"No se encontro el archivo de demanda: {demand_file.resolve()}"
        )
    suffix = demand_file.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(demand_file)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(demand_file)
    raise ValueError("La demanda debe venir en un archivo .csv, .xlsx o .xls.")


def _find_column(columns: pd.Index, candidates: list[str]) -> str | None:
    normalized = {str(column).strip().lower(): str(column) for column in columns}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    return None


def _resample_demand_to_15min(demand: pd.DataFrame) -> pd.DataFrame:
    if len(demand) < 2:
        return demand
    diffs = demand.index.to_series().diff().dropna()
    expected = pd.Timedelta(FINAL_FREQUENCY)
    if not diffs.eq(expected).all():
        warnings.warn(
            "La demanda no tiene frecuencia exacta de 15 minutos; se remuestreara "
            "con interpolacion temporal.",
            stacklevel=2,
        )
        target = pd.date_range(
            start=demand.index.min(),
            end=demand.index.max(),
            freq=FINAL_FREQUENCY,
        )
        demand = (
            demand.reindex(demand.index.union(target))
            .interpolate(method="time")
            .reindex(target)
        )
    return demand


def load_demand(
    demand_file: Path,
    target_index: pd.DatetimeIndex,
    timezone: str,
    align_by_position: bool = False,
) -> pd.DataFrame:
    """Load industrial demand from disk and align it with weather."""
    raw = _read_demand_file(demand_file)
    return normalize_demand_dataframe(
        raw,
        target_index=target_index,
        timezone=timezone,
        align_by_position=align_by_position,
    )


def normalize_demand_dataframe(
    raw: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    timezone: str,
    align_by_position: bool = False,
) -> pd.DataFrame:
    """Validate and align an in-memory industrial demand DataFrame."""
    raw = raw.copy()
    timestamp_column = _find_column(raw.columns, ["Fecha_Hora", "time", "timestamp"])
    demand_column = _find_column(raw.columns, ["Demanda_kW"])
    power_factor_column = _find_column(raw.columns, ["Factor_Potencia"])

    if timestamp_column is None:
        raise ValueError("Falta columna de fecha: Fecha_Hora, time o timestamp.")
    if demand_column is None:
        raise ValueError("Falta la columna obligatoria 'Demanda_kW'.")

    if timestamp_column == "Fecha_Hora":
        timestamps = pd.to_datetime(
            raw[timestamp_column],
            format="%Y-%m-%d %H:%M:%S",
            errors="coerce",
        )
    else:
        timestamps = pd.to_datetime(raw[timestamp_column], errors="coerce")
    timestamps = _localize_datetime_series(timestamps, timezone, timestamp_column)

    if timestamps.duplicated().any():
        raise ValueError("La curva de demanda contiene fechas duplicadas.")

    demand_values = pd.to_numeric(raw[demand_column], errors="coerce")
    if demand_values.isna().any():
        raise ValueError("La columna 'Demanda_kW' contiene NaN o valores no numericos.")

    if power_factor_column is None:
        warnings.warn(
            "No se encontro 'Factor_Potencia'; se conservara como NaN.",
            stacklevel=2,
        )
        power_factor = pd.Series(np.nan, index=raw.index)
    else:
        power_factor = pd.to_numeric(raw[power_factor_column], errors="coerce")

    demand = pd.DataFrame(
        {
            "Demanda_kW": demand_values.to_numpy(),
            "Factor_Potencia": power_factor.to_numpy(),
        },
        index=pd.DatetimeIndex(timestamps),
    ).sort_index()
    demand = _resample_demand_to_15min(demand)

    if len(target_index) == 35040 and len(demand) != 35040:
        raise ValueError(
            "Un periodo de 365 dias a 15 minutos requiere 35,040 registros de "
            f"demanda; se recibieron {len(demand):,}."
        )

    if align_by_position:
        if len(demand) != len(target_index):
            raise ValueError(
                "No se puede alinear demanda y clima por posicion: "
                f"demanda={len(demand):,}, clima={len(target_index):,} intervalos."
            )
        aligned = demand.copy()
        aligned.insert(
            0,
            "Fecha_Hora_Demanda_Original",
            demand.index.tz_convert(timezone).tz_localize(None),
        )
        aligned.index = target_index
        return aligned

    aligned = demand.reindex(target_index)
    if aligned["Demanda_kW"].isna().any():
        raise ValueError(
            "Las fechas de demanda y clima no coinciden. Usa "
            "--align-demand-by-position para aplicar un ano climatico analogo."
        )
    return aligned


def generate_demand(target_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Generate a deterministic industrial profile only when explicitly requested."""
    hour = target_index.hour + target_index.minute / 60
    weekday = target_index.dayofweek < 5
    work_shift = (hour >= 7) & (hour < 19)
    demand_kw = 22 + 22 * work_shift.astype(float) + 5 * weekday.astype(float)
    generated = pd.DataFrame(index=target_index)
    generated["Demanda_kW"] = demand_kw
    generated["Factor_Potencia"] = 0.90
    return generated


def generar_demanda_sintetica_15min(
    target_index: pd.DatetimeIndex,
    consumo_anual_kwh: float | None = None,
    consumos_mensuales_kwh: list[float] | None = None,
    factor_potencia: float = 0.90,
    tipo_perfil: str = "industrial",
) -> pd.DataFrame:
    """Build a deterministic 15-minute industrial demand curve for the UI."""
    if len(target_index) == 0:
        raise ValueError("El indice objetivo de demanda esta vacio.")
    if target_index.tz is None:
        local_index = target_index
    else:
        local_index = target_index.tz_localize(None)

    if tipo_perfil != "industrial":
        raise ValueError("Solo esta disponible tipo_perfil='industrial'.")
    if consumo_anual_kwh is None and consumos_mensuales_kwh is None:
        raise ValueError("Indica consumo_anual_kwh o consumos_mensuales_kwh.")
    if consumo_anual_kwh is not None and consumos_mensuales_kwh is not None:
        raise ValueError("Usa consumo anual o consumos mensuales, no ambos.")

    hour = local_index.hour + local_index.minute / 60
    weekday = local_index.dayofweek < 5
    work_shift = (hour >= 7) & (hour < 19)
    shoulder_shift = ((hour >= 6) & (hour < 7)) | ((hour >= 19) & (hour < 22))
    day_of_year = local_index.dayofyear.to_numpy()

    base_profile = np.full(len(local_index), 0.46, dtype=float)
    base_profile += np.where(shoulder_shift, 0.18, 0.0)
    base_profile += np.where(work_shift, 0.62, 0.0)
    base_profile *= np.where(weekday, 1.10, 0.72)
    base_profile *= 1.0 + 0.045 * np.sin(2 * np.pi * day_of_year / 14.0)
    base_profile *= 1.0 + 0.025 * np.sin(2 * np.pi * hour / 24.0)
    base_profile = np.clip(base_profile, 0.05, None)

    timestep_hours = infer_timestep_hours(target_index)
    demand_kw = pd.Series(base_profile, index=local_index, dtype=float)

    if consumo_anual_kwh is not None:
        if consumo_anual_kwh <= 0:
            raise ValueError("El consumo anual debe ser mayor a 0.")
        current_energy = float(demand_kw.sum() * timestep_hours)
        if current_energy <= 0:
            raise ValueError("El perfil sintetico no tiene energia escalable.")
        demand_kw *= consumo_anual_kwh / current_energy
    else:
        if len(consumos_mensuales_kwh) != 12:
            raise ValueError("Debes indicar 12 consumos mensuales.")
        if any(value < 0 for value in consumos_mensuales_kwh):
            raise ValueError("Los consumos mensuales deben ser mayores o iguales a 0.")
        if sum(consumos_mensuales_kwh) <= 0:
            raise ValueError("La suma de consumos mensuales debe ser mayor a 0.")

        scaled = demand_kw.copy()
        months = pd.Index(local_index.month)
        for month, target_kwh in enumerate(consumos_mensuales_kwh, start=1):
            mask = months == month
            if not mask.any():
                continue
            if target_kwh == 0:
                scaled.iloc[np.where(mask)[0]] = 0.0
                continue
            current_energy = float(scaled.iloc[np.where(mask)[0]].sum() * timestep_hours)
            if current_energy <= 0:
                raise ValueError(f"No hay energia escalable para el mes {month}.")
            scaled.iloc[np.where(mask)[0]] *= target_kwh / current_energy
        demand_kw = scaled

    return pd.DataFrame(
        {
            "Fecha_Hora": local_index.strftime("%Y-%m-%d %H:%M:%S"),
            "Demanda_kW": demand_kw.to_numpy(),
            "Factor_Potencia": float(factor_potencia),
        }
    )


def infer_timestep_hours(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 1.0
    seconds = index.to_series().diff().dt.total_seconds().dropna().median()
    return float(seconds / 3600) if pd.notna(seconds) and seconds > 0 else 1.0


def module_gain(module: ModuleConfig, bifacial_gain: float) -> float:
    if module.bifacial_factor is None:
        return 0.0
    # Rear-side gain is an irradiance contribution. The module only converts
    # its bifacial fraction, so 15% rear gain * 0.80 factor becomes 12%.
    return bifacial_gain * module.bifacial_factor


def resolve_system_capacity(
    module: ModuleConfig,
    system_dc_kwp: float,
    num_modules: int | None,
) -> tuple[float, float]:
    if num_modules is not None:
        if num_modules <= 0:
            raise ValueError("--num-modules debe ser mayor que cero.")
        resolved_dc_kwp = num_modules * module.pdc0_w / 1000
        return resolved_dc_kwp, float(num_modules)
    if system_dc_kwp <= 0:
        raise ValueError("--system-dc-kwp debe ser mayor que cero.")
    return system_dc_kwp, system_dc_kwp * 1000 / module.pdc0_w


def add_module_calculations(
    result: pd.DataFrame,
    module: ModuleConfig,
    bifacial_gain: float,
    system_dc_kwp: float,
    num_modules: int | None,
    inverter_efficiency: float,
    system_losses: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Estimate module and complete-system generation with PVWatts."""
    result = result.copy()
    timestep_hours = infer_timestep_hours(result.index)
    if not np.isclose(timestep_hours, TIMESTEP_HOURS):
        raise ValueError(
            f"El calculo FV requiere intervalos de 15 minutos; se detectaron "
            f"{timestep_hours:.4f} horas."
        )

    if "temperature_2m" not in result:
        warnings.warn(
            "Falta temperatura historica; Faiman usara fallback explicito de 25 C.",
            stacklevel=2,
        )
        temp_air = pd.Series(25.0, index=result.index)
    else:
        temp_air = (
            result["temperature_2m"].interpolate(limit_direction="both").fillna(25.0)
        )

    if "wind_speed_10m" not in result:
        warnings.warn(
            "Falta viento historico; Faiman usara fallback explicito de 1 m/s.",
            stacklevel=2,
        )
        wind_speed = pd.Series(1.0, index=result.index)
    else:
        wind_speed = (
            result["wind_speed_10m"]
            .interpolate(limit_direction="both")
            .fillna(1.0)
            .clip(lower=0.0)
        )

    resolved_dc_kwp, equivalent_modules = resolve_system_capacity(
        module, system_dc_kwp, num_modules
    )
    effective_gain = module_gain(module, bifacial_gain)
    effective_poa = result["poa_global"] * (1 + effective_gain)
    cell_temperature = pvlib.temperature.faiman(
        effective_poa,
        temp_air=temp_air,
        wind_speed=wind_speed,
    )
    module_dc_power_w = pvlib.pvsystem.pvwatts_dc(
        effective_irradiance=effective_poa,
        temp_cell=cell_temperature,
        pdc0=module.pdc0_w,
        gamma_pdc=module.gamma_pdc_per_c,
    ).fillna(0).clip(lower=0)
    system_dc_power_kw = (
        pvlib.pvsystem.pvwatts_dc(
            effective_irradiance=effective_poa,
            temp_cell=cell_temperature,
            pdc0=resolved_dc_kwp * 1000,
            gamma_pdc=module.gamma_pdc_per_c,
        )
        .fillna(0)
        .clip(lower=0)
        / 1000
    )
    system_ac_power_kw = (
        system_dc_power_kw * (1 - system_losses) * inverter_efficiency
    )
    system_dc_power_ref_kw = (
        pvlib.pvsystem.pvwatts_dc(
            effective_irradiance=effective_poa,
            temp_cell=25.0,
            pdc0=resolved_dc_kwp * 1000,
            gamma_pdc=module.gamma_pdc_per_c,
        )
        .fillna(0)
        .clip(lower=0)
        / 1000
    )
    system_ac_power_ref_kw = (
        system_dc_power_ref_kw * (1 - system_losses) * inverter_efficiency
    )
    thermal_penalty_kw = system_ac_power_ref_kw - system_ac_power_kw

    result["Ganancia_Bifacial_Efectiva"] = effective_gain
    result["POA_Efectiva_Wm2"] = effective_poa.fillna(0).clip(lower=0)
    result["Temperatura_Celda_C"] = cell_temperature
    result["Generacion_DC_Modulo_W"] = module_dc_power_w
    result["Energia_DC_Modulo_kWh"] = module_dc_power_w * timestep_hours / 1000
    result["Generacion_DC_kW"] = system_dc_power_kw
    result["Generacion_AC_kW"] = system_ac_power_kw
    result["Energia_Solar_DC_kWh"] = system_dc_power_kw * timestep_hours
    result["Energia_Solar_AC_kWh"] = system_ac_power_kw * timestep_hours
    result["Generacion_AC_Sin_Temp_kW"] = system_ac_power_ref_kw
    result["Energia_Solar_AC_Sin_Temp_kWh"] = (
        system_ac_power_ref_kw * timestep_hours
    )
    result["Generacion_Solar_Sin_Temp_kW"] = result["Generacion_AC_Sin_Temp_kW"]
    result["Generacion_Solar_kW"] = result["Generacion_AC_kW"]
    result["Penalizacion_Temperatura_kW"] = thermal_penalty_kw
    result["Energia_Solar_Sin_Temp_kWh"] = result[
        "Energia_Solar_AC_Sin_Temp_kWh"
    ]
    result["Energia_Solar_kWh"] = result["Energia_Solar_AC_kWh"]
    result["Perdida_Temperatura_kWh"] = (
        result["Penalizacion_Temperatura_kW"] * timestep_hours
    )
    result["Penalizacion_Temperatura_pct_inst"] = np.where(
        result["Generacion_AC_Sin_Temp_kW"] > 0,
        result["Penalizacion_Temperatura_kW"]
        / result["Generacion_AC_Sin_Temp_kW"]
        * 100.0,
        0.0,
    )
    result["Modulo"] = module.model
    result["Tipo_Modulo"] = module.module_type
    result["Potencia_DC_Sistema_kWp"] = resolved_dc_kwp
    result["Num_Modulos_Equivalente"] = equivalent_modules

    scenario = {
        "system_dc_kwp": resolved_dc_kwp,
        "num_modules_equivalent": equivalent_modules,
        "effective_bifacial_gain": effective_gain,
    }
    return result, scenario


def add_demand_balance(result: pd.DataFrame, demand: pd.DataFrame) -> pd.DataFrame:
    """Join demand and compute the AC-side post-injection energy balance."""
    result = result.join(demand, how="left")
    if result["Demanda_kW"].isna().any():
        raise ValueError("La demanda alineada contiene intervalos vacios.")

    result["Demanda_Post_Inyeccion_Solar_kW"] = (
        result["Demanda_kW"] - result["Generacion_AC_kW"]
    ).clip(lower=0)
    result["Energia_Demanda_kWh"] = result["Demanda_kW"] * TIMESTEP_HOURS
    result["Energia_Red_kWh"] = (
        result["Demanda_Post_Inyeccion_Solar_kW"] * TIMESTEP_HOURS
    )
    result["Energia_Autoconsumida_kWh"] = (
        np.minimum(result["Demanda_kW"], result["Generacion_AC_kW"]) * TIMESTEP_HOURS
    )
    result["Energia_Excedente_kWh"] = (
        (result["Generacion_AC_kW"] - result["Demanda_kW"]).clip(lower=0)
        * TIMESTEP_HOURS
    )
    return result


def add_pv_calculations(weather: pd.DataFrame, site: SiteConfig) -> pd.DataFrame:
    """Calculate POA irradiance and clear-sky baseline."""
    result = weather.copy()
    solpos = pvlib.solarposition.get_solarposition(
        result.index,
        latitude=site.latitude,
        longitude=site.longitude,
        altitude=site.altitude_m,
    )

    radiation_columns = [
        "shortwave_radiation",
        "direct_normal_irradiance",
        "diffuse_radiation",
        "direct_radiation",
    ]
    for column in radiation_columns:
        result[column] = result[column].fillna(0).clip(lower=0)

    daylight = solpos["apparent_zenith"] < 90
    ghi = result["shortwave_radiation"].where(daylight, 0)
    dni = result["direct_normal_irradiance"].where(daylight, 0)
    dhi = result["diffuse_radiation"].where(daylight, 0)

    dni_extra = pvlib.irradiance.get_extra_radiation(result.index)
    airmass = pvlib.atmosphere.get_relative_airmass(solpos["apparent_zenith"])

    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=site.surface_tilt,
        surface_azimuth=site.surface_azimuth,
        solar_zenith=solpos["apparent_zenith"],
        solar_azimuth=solpos["azimuth"],
        dni=dni,
        ghi=ghi,
        dhi=dhi,
        dni_extra=dni_extra,
        airmass=airmass,
        albedo=site.albedo,
        model="haydavies",
    )

    location = pvlib.location.Location(
        latitude=site.latitude,
        longitude=site.longitude,
        tz=site.timezone,
        altitude=site.altitude_m,
        name=site.name,
    )
    clearsky = location.get_clearsky(result.index, model="ineichen")
    poa_clearsky = pvlib.irradiance.get_total_irradiance(
        surface_tilt=site.surface_tilt,
        surface_azimuth=site.surface_azimuth,
        solar_zenith=solpos["apparent_zenith"],
        solar_azimuth=solpos["azimuth"],
        dni=clearsky["dni"].where(daylight, 0),
        ghi=clearsky["ghi"].where(daylight, 0),
        dhi=clearsky["dhi"].where(daylight, 0),
        dni_extra=dni_extra,
        airmass=airmass,
        albedo=site.albedo,
        model="haydavies",
    )

    timestep_hours = infer_timestep_hours(result.index)
    result["ghi"] = ghi
    result["dni"] = dni
    result["dhi"] = dhi
    result["solar_zenith"] = solpos["apparent_zenith"]
    result["solar_elevation"] = solpos["apparent_elevation"]
    result["solar_azimuth"] = solpos["azimuth"]
    result["poa_global"] = poa["poa_global"].fillna(0).clip(lower=0)
    result["poa_direct"] = poa["poa_direct"].fillna(0).clip(lower=0)
    result["poa_diffuse"] = poa["poa_diffuse"].fillna(0).clip(lower=0)
    result["ghi_clear_sky"] = clearsky["ghi"].where(daylight, 0).clip(lower=0)
    result["poa_clear_sky"] = poa_clearsky["poa_global"].fillna(0).clip(lower=0)
    result["ghi_energy_kwh_m2"] = result["ghi"] * timestep_hours / 1000
    result["poa_energy_kwh_m2"] = result["poa_global"] * timestep_hours / 1000
    result["poa_clear_sky_energy_kwh_m2"] = (
        result["poa_clear_sky"] * timestep_hours / 1000
    )
    result["clearness_index"] = (
        result["ghi"] / result["ghi_clear_sky"].replace(0, np.nan)
    ).clip(lower=0, upper=1.3)

    return result


def _reporting_month(result: pd.DataFrame) -> pd.Series:
    if "Fecha_Hora_Demanda_Original" in result:
        timestamps = pd.to_datetime(result["Fecha_Hora_Demanda_Original"])
    else:
        timestamps = pd.Series(result.index.tz_localize(None), index=result.index)
    return timestamps.dt.strftime("%Y-%m")


def monthly_summary(result: pd.DataFrame) -> pd.DataFrame:
    """Summarize demand, generation and climate for the technical report."""
    grouped = result.groupby(_reporting_month(result))
    summary = grouped.agg(
        Energia_Demandada_kWh=("Energia_Demanda_kWh", "sum"),
        Energia_Solar_DC_kWh=("Energia_Solar_DC_kWh", "sum"),
        Energia_Solar_AC_kWh=("Energia_Solar_AC_kWh", "sum"),
        Energia_Solar_AC_Sin_Temp_kWh=("Energia_Solar_AC_Sin_Temp_kWh", "sum"),
        Perdida_Temperatura_kWh=("Perdida_Temperatura_kWh", "sum"),
        Energia_Autoconsumida_kWh=("Energia_Autoconsumida_kWh", "sum"),
        Energia_Red_kWh=("Energia_Red_kWh", "sum"),
        Energia_Excedente_kWh=("Energia_Excedente_kWh", "sum"),
        Demanda_Maxima_Original_kW=("Demanda_kW", "max"),
        Demanda_Maxima_Post_Solar_kW=("Demanda_Post_Inyeccion_Solar_kW", "max"),
        POA_Mensual_kWh_m2=("poa_energy_kwh_m2", "sum"),
        POA_Cielo_Despejado_Mensual_kWh_m2=("poa_clear_sky_energy_kwh_m2", "sum"),
        GHI_Mensual_kWh_m2=("ghi_energy_kwh_m2", "sum"),
        Energia_DC_Modulo_kWh=("Energia_DC_Modulo_kWh", "sum"),
        Temperatura_Media_C=("temperature_2m", "mean"),
        Temperatura_Celda_Prom_C=("Temperatura_Celda_C", "mean"),
        Temperatura_Celda_Max_C=("Temperatura_Celda_C", "max"),
        Nubosidad_Media_pct=("cloud_cover", "mean"),
        Precipitacion_mm=("precipitation", "sum"),
    )
    summary.index.name = "Mes"
    summary["Reduccion_Demanda_Maxima_kW"] = (
        summary["Demanda_Maxima_Original_kW"]
        - summary["Demanda_Maxima_Post_Solar_kW"]
    )
    summary["Relacion_POA_Historico_vs_Cielo_Despejado_pct"] = (
        100
        * summary["POA_Mensual_kWh_m2"]
        / summary["POA_Cielo_Despejado_Mensual_kWh_m2"].replace(0, np.nan)
    )
    summary["Penalizacion_Temperatura_pct"] = (
        100
        * summary["Perdida_Temperatura_kWh"]
        / summary["Energia_Solar_AC_Sin_Temp_kWh"].replace(0, np.nan)
    )
    return summary.round(3)


def resumen_penalizacion_temperatura_mensual(result: pd.DataFrame) -> pd.DataFrame:
    """Resume thermal penalty using the same monthly alignment as the report."""
    grouped = result.groupby(_reporting_month(result))
    summary = grouped.agg(
        Generacion_Sin_Temp_kWh=("Energia_Solar_AC_Sin_Temp_kWh", "sum"),
        Generacion_Con_Temp_kWh=("Energia_Solar_AC_kWh", "sum"),
        Perdida_Temperatura_kWh=("Perdida_Temperatura_kWh", "sum"),
        Temperatura_Ambiente_Prom_C=("temperature_2m", "mean"),
        Temperatura_Celda_Prom_C=("Temperatura_Celda_C", "mean"),
        Temperatura_Celda_Max_C=("Temperatura_Celda_C", "max"),
    )
    summary.index.name = "Mes"
    summary["Penalizacion_Temperatura_pct"] = np.where(
        summary["Generacion_Sin_Temp_kWh"] > 0,
        summary["Perdida_Temperatura_kWh"]
        / summary["Generacion_Sin_Temp_kWh"]
        * 100.0,
        0.0,
    )
    return summary.reset_index().round(3)


def annual_summary(
    result: pd.DataFrame,
    module: ModuleConfig,
    scenario: dict[str, float],
) -> pd.DataFrame:
    """Build one annual row per module scenario."""
    hours = len(result) * TIMESTEP_HOURS
    demand_energy = result["Energia_Demanda_kWh"].sum()
    solar_dc_energy = result["Energia_Solar_DC_kWh"].sum()
    solar_ac_energy = result["Energia_Solar_AC_kWh"].sum()
    solar_ac_no_temp_energy = result["Energia_Solar_AC_Sin_Temp_kWh"].sum()
    thermal_penalty_energy = result["Perdida_Temperatura_kWh"].sum()
    self_consumed = result["Energia_Autoconsumida_kWh"].sum()
    grid_energy = result["Energia_Red_kWh"].sum()
    surplus = result["Energia_Excedente_kWh"].sum()
    peak_original = result["Demanda_kW"].max()
    peak_post_solar = result["Demanda_Post_Inyeccion_Solar_kW"].max()
    peak_reduction = peak_original - peak_post_solar
    dc_kwp = scenario["system_dc_kwp"]

    row = {
        "Escenario": module.key,
        "Modulo": module.model,
        "Tipo_Modulo": module.module_type,
        "Potencia_DC_Sistema_kWp": dc_kwp,
        "Num_Modulos_Equivalente": scenario["num_modules_equivalent"],
        "Ganancia_Bifacial_Efectiva_pct": 100
        * scenario["effective_bifacial_gain"],
        "Energia_Demandada_Anual_kWh": demand_energy,
        "Energia_Solar_DC_Anual_kWh": solar_dc_energy,
        "Energia_Solar_AC_Anual_kWh": solar_ac_energy,
        "Energia_Solar_AC_Sin_Temp_Anual_kWh": solar_ac_no_temp_energy,
        "Perdida_Temperatura_Anual_kWh": thermal_penalty_energy,
        "Penalizacion_Temperatura_Anual_pct": 100
        * thermal_penalty_energy
        / solar_ac_no_temp_energy
        if solar_ac_no_temp_energy
        else np.nan,
        "Energia_Autoconsumida_Anual_kWh": self_consumed,
        "Energia_Red_Anual_kWh": grid_energy,
        "Energia_Excedente_Anual_kWh": surplus,
        "Autoconsumo_pct": 100 * self_consumed / solar_ac_energy
        if solar_ac_energy
        else np.nan,
        "Cobertura_Solar_pct": 100 * self_consumed / demand_energy
        if demand_energy
        else np.nan,
        "Demanda_Maxima_Original_kW": peak_original,
        "Demanda_Maxima_Post_Solar_kW": peak_post_solar,
        "Reduccion_Demanda_Maxima_kW": peak_reduction,
        "Reduccion_Demanda_Maxima_pct": 100 * peak_reduction / peak_original
        if peak_original
        else np.nan,
        "Factor_Planta_Demanda_pct": 100 * demand_energy / (peak_original * hours)
        if peak_original and hours
        else np.nan,
        "Capacity_Factor_FV_pct": 100 * solar_ac_energy / (dc_kwp * hours)
        if dc_kwp and hours
        else np.nan,
        "Generacion_Especifica_AC_kWh_kWp": solar_ac_energy / dc_kwp
        if dc_kwp
        else np.nan,
    }
    return pd.DataFrame([row]).round(3)


def module_comparison_summary(annual_summaries: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(annual_summaries, ignore_index=True)


def _representative_week(result: pd.DataFrame) -> pd.DataFrame:
    weekly_energy = result["Energia_Solar_AC_kWh"].resample("7D").sum()
    if weekly_energy.empty:
        return result
    median_energy = weekly_energy.median()
    week_start = (weekly_energy - median_energy).abs().idxmin()
    return result.loc[week_start : week_start + pd.Timedelta(days=7) - pd.Timedelta(minutes=15)]


def save_plots(
    result: pd.DataFrame,
    summary: pd.DataFrame,
    figures_dir: Path,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    daily = result.resample("D").agg(
        poa_energy_kwh_m2=("poa_energy_kwh_m2", "sum"),
        poa_clear_sky_energy_kwh_m2=("poa_clear_sky_energy_kwh_m2", "sum"),
        temperature_mean_c=("temperature_2m", "mean"),
        cloud_cover_mean_pct=("cloud_cover", "mean"),
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    daily[["poa_energy_kwh_m2", "poa_clear_sky_energy_kwh_m2"]].plot(ax=ax)
    ax.set_title("Energia diaria en el plano del panel")
    ax.set_xlabel("Fecha climatica analoga")
    ax.set_ylabel("kWh/m^2 por dia")
    ax.legend(["Historico", "Cielo despejado"])
    fig.tight_layout()
    fig.savefig(figures_dir / "energia_diaria_poa.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        summary.index,
        summary["POA_Mensual_kWh_m2"],
        color="#d97706",
        label="Historico",
    )
    ax.plot(
        summary.index,
        summary["POA_Cielo_Despejado_Mensual_kWh_m2"],
        color="black",
        marker="o",
        label="Cielo despejado",
    )
    ax.set_title("Irradiancia POA mensual historica")
    ax.set_xlabel("Mes")
    ax.set_ylabel("kWh/m^2")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "energia_mensual.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(summary.index, summary["Energia_DC_Modulo_kWh"], color="#0f766e")
    ax.set_title("Energia mensual DC por modulo")
    ax.set_xlabel("Mes")
    ax.set_ylabel("kWh por modulo")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(figures_dir / "energia_mensual_modulos.png", dpi=160)
    plt.close(fig)

    heatmap_source = result.copy()
    heatmap_source["month"] = heatmap_source.index.month
    heatmap_source["clock_time"] = heatmap_source.index.strftime("%H:%M")
    heatmap = heatmap_source.pivot_table(
        index="clock_time", columns="month", values="poa_global", aggfunc="mean"
    )
    heatmap = heatmap.reindex(sorted(heatmap.index))
    heatmap.columns = [calendar.month_abbr[i] for i in heatmap.columns]

    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(
        heatmap,
        ax=ax,
        cmap="inferno",
        cbar_kws={"label": "Irradiancia POA historica (W/m^2)"},
    )
    ax.set_title("Irradiancia promedio por hora y mes")
    ax.set_xlabel("Mes")
    ax.set_ylabel("Hora local")
    fig.tight_layout()
    fig.savefig(figures_dir / "heatmap_poa_historica.png", dpi=160)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(daily.index, daily["temperature_mean_c"], color="#b23a48", label="Temp.")
    ax1.set_ylabel("Temperatura media (C)")
    ax2 = ax1.twinx()
    ax2.plot(
        daily.index,
        daily["cloud_cover_mean_pct"],
        color="#3a6ea5",
        alpha=0.75,
        label="Nubosidad",
    )
    ax2.set_ylabel("Nubosidad media (%)")
    ax1.set_title("Clima diario historico")
    ax1.set_xlabel("Fecha climatica analoga")
    fig.tight_layout()
    fig.savefig(figures_dir / "clima_diario.png", dpi=160)
    plt.close(fig)

    week = _representative_week(result)
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(week.index, week["Demanda_kW"], label="Demanda", color="#1f2937")
    ax.plot(week.index, week["Generacion_AC_kW"], label="Generacion solar AC", color="#d97706")
    ax.set_title("Demanda industrial vs generacion solar AC: semana representativa")
    ax.set_xlabel("Fecha climatica analoga")
    ax.set_ylabel("Potencia (kW)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "demanda_vs_solar.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(week.index, week["Demanda_kW"], label="Demanda original", color="#1f2937")
    ax.plot(
        week.index,
        week["Demanda_Post_Inyeccion_Solar_kW"],
        label="Demanda post-solar",
        color="#2563eb",
    )
    ax.set_title("Demanda original vs demanda post-inyeccion solar")
    ax.set_xlabel("Fecha climatica analoga")
    ax.set_ylabel("Potencia (kW)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "demanda_original_vs_post_solar.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    summary[
        [
            "Energia_Demandada_kWh",
            "Energia_Solar_AC_kWh",
            "Energia_Autoconsumida_kWh",
            "Energia_Red_kWh",
        ]
    ].plot(kind="bar", ax=ax)
    ax.set_title("Energia mensual: demanda, solar, autoconsumo y red")
    ax.set_xlabel("Mes")
    ax.set_ylabel("Energia (kWh)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(figures_dir / "energia_mensual_demanda_solar.png", dpi=160)
    plt.close(fig)

    typical_day = result.groupby(result.index.strftime("%H:%M"))[
        ["Demanda_kW", "Generacion_AC_kW", "Demanda_Post_Inyeccion_Solar_kW"]
    ].mean()
    fig, ax = plt.subplots(figsize=(12, 5))
    typical_day.plot(ax=ax)
    ax.set_title("Perfil promedio diario")
    ax.set_xlabel("Hora local")
    ax.set_ylabel("Potencia media (kW)")
    ax.set_xticks(np.arange(0, len(typical_day), 8))
    ax.set_xticklabels(typical_day.index[::8], rotation=45)
    fig.tight_layout()
    fig.savefig(figures_dir / "perfil_dia_tipico.png", dpi=160)
    plt.close(fig)


def write_outputs(
    result: pd.DataFrame,
    summary: pd.DataFrame,
    annual: pd.DataFrame,
    module: ModuleConfig,
    scenario: dict[str, float],
    site: SiteConfig,
    source_url: str,
    output_dir: Path,
    start_date: date,
    end_date: date,
    bifacial_gain: float,
    inverter_efficiency: float,
    system_losses: float,
    demand_file: Path | None,
    align_demand_by_position: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figuras"

    export = result.rename(
        columns={
            "ghi": "GHI_Wm2",
            "dni": "DNI_Wm2",
            "dhi": "DHI_Wm2",
            "poa_global": "Gtot_POA_Wm2",
            "temperature_2m": "Temperatura_Ambiente_C",
            "wind_speed_10m": "Velocidad_Viento_ms",
        }
    ).copy()
    ordered_columns = [
        "Fecha_Hora_Demanda_Original",
        "Demanda_kW",
        "Factor_Potencia",
        "GHI_Wm2",
        "DNI_Wm2",
        "DHI_Wm2",
        "Gtot_POA_Wm2",
        "Temperatura_Ambiente_C",
        "Velocidad_Viento_ms",
        "Generacion_DC_kW",
        "Generacion_AC_kW",
        "Energia_Solar_DC_kWh",
        "Energia_Solar_AC_kWh",
        "Demanda_Post_Inyeccion_Solar_kW",
        "Energia_Demanda_kWh",
        "Energia_Red_kWh",
        "Energia_Autoconsumida_kWh",
        "Energia_Excedente_kWh",
        "Modulo",
        "Tipo_Modulo",
        "Potencia_DC_Sistema_kWp",
        "Num_Modulos_Equivalente",
        "Ganancia_Bifacial_Efectiva",
        "POA_Efectiva_Wm2",
        "Temperatura_Celda_C",
        "Generacion_DC_Modulo_W",
        "Energia_DC_Modulo_kWh",
        "relative_humidity_2m",
        "cloud_cover",
        "precipitation",
        "ghi_clear_sky",
        "poa_clear_sky",
        "clearness_index",
    ]
    export = export[[column for column in ordered_columns if column in export.columns]]
    export.index = export.index.tz_localize(None)
    export.index.name = "Fecha_Hora"
    export.to_csv(output_dir / "clima_solar_demanda_15min.csv")
    summary.to_csv(output_dir / "resumen_mensual.csv")
    annual.to_csv(output_dir / "resumen_anual.csv", index=False)
    save_plots(result, summary, figures_dir)

    metadata = {
        "generated_at": datetime.now(ZoneInfo(site.timezone)).isoformat(),
        "weather_date_range": {
            "weather_start_date": start_date.isoformat(),
            "weather_end_date": end_date.isoformat(),
        },
        "site": asdict(site),
        "module": asdict(module),
        "system": {
            **scenario,
            "requested_bifacial_gain": bifacial_gain,
            "inverter_efficiency": inverter_efficiency,
            "system_losses": system_losses,
        },
        "demand": {
            "file": str(demand_file.resolve()) if demand_file else None,
            "align_by_position": align_demand_by_position,
            "intervals": len(result),
            "frequency": FINAL_FREQUENCY,
        },
        "source": {
            "name": "Open-Meteo Historical Weather API",
            "url": source_url,
        },
        "notes": [
            "GHI, DNI y DHI historicos horarios se remuestrean a 15 minutos.",
            "El remuestreo aplica indices de claridad horarios a curvas de cielo despejado de 15 minutos.",
            "POA se calcula con pvlib.irradiance.get_total_irradiance.",
            "La generacion DC del modulo y sistema se calcula con pvlib.pvsystem.pvwatts_dc.",
            "La temperatura de celda se estima con el modelo Faiman.",
            "La demanda industrial se descuenta usando generacion AC despues de inversor y perdidas.",
            "La bifacialidad escala irradiancia POA, no la potencia nominal STC.",
        ],
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8"
    )


def run_solar_demand_simulation(
    site: SiteConfig,
    demand_df: pd.DataFrame | None = None,
    demand_file: Path | None = None,
    weather_start_date: date | None = None,
    weather_end_date: date | None = None,
    module_key: str = "bifacial",
    bifacial_gain: float = 0.15,
    system_dc_kwp: float = 399.3,
    num_modules: int | None = None,
    inverter_efficiency: float = 0.96,
    system_losses: float = 0.0,
    align_demand_by_position: bool = True,
    output_dir: Path | None = None,
    write_files: bool = False,
) -> dict[str, object]:
    """Run one solar-industrial scenario from Python without CLI arguments."""
    if module_key not in MODULE_OPTIONS:
        raise ValueError(
            f"module_key={module_key!r} no reconocido. "
            f"Opciones: {list(MODULE_OPTIONS)}."
        )
    if demand_df is not None and demand_file is not None:
        raise ValueError("Usa demand_df o demand_file, no ambos al mismo tiempo.")
    if demand_df is None and demand_file is None:
        raise ValueError("La simulacion requiere demand_df o demand_file.")
    if bifacial_gain < 0:
        raise ValueError("bifacial_gain no puede ser negativo.")
    if not 0 < inverter_efficiency <= 1:
        raise ValueError("inverter_efficiency debe estar entre 0 y 1.")
    if not 0 <= system_losses < 1:
        raise ValueError("system_losses debe estar entre 0 y 1.")

    if weather_start_date is None or weather_end_date is None:
        default_start, default_end = default_date_range(site.timezone)
        weather_start_date = weather_start_date or date.fromisoformat(default_start)
        weather_end_date = weather_end_date or date.fromisoformat(default_end)

    target_index = build_target_index(
        weather_start_date,
        weather_end_date,
        timezone=site.timezone,
    )
    hourly_weather, source_url = fetch_historical_weather(
        site,
        weather_start_date,
        weather_end_date,
    )
    weather = resample_weather_to_15min(hourly_weather, target_index, site)
    solar_resource = add_pv_calculations(weather, site)

    if demand_df is not None:
        demand = normalize_demand_dataframe(
            demand_df,
            target_index=target_index,
            timezone=site.timezone,
            align_by_position=align_demand_by_position,
        )
    else:
        demand = load_demand(
            Path(demand_file),
            target_index=target_index,
            timezone=site.timezone,
            align_by_position=align_demand_by_position,
        )

    module = MODULE_OPTIONS[module_key]
    result, scenario = add_module_calculations(
        solar_resource,
        module=module,
        bifacial_gain=bifacial_gain,
        system_dc_kwp=system_dc_kwp,
        num_modules=num_modules,
        inverter_efficiency=inverter_efficiency,
        system_losses=system_losses,
    )
    result = add_demand_balance(result, demand)
    summary = monthly_summary(result)
    annual = annual_summary(result, module, scenario)
    module_comparison = module_comparison_summary([annual])

    if write_files:
        resolved_output_dir = output_dir or Path("outputs")
        write_outputs(
            result=result,
            summary=summary,
            annual=annual,
            module=module,
            scenario=scenario,
            site=site,
            source_url=source_url,
            output_dir=resolved_output_dir,
            start_date=weather_start_date,
            end_date=weather_end_date,
            bifacial_gain=bifacial_gain,
            inverter_efficiency=inverter_efficiency,
            system_losses=system_losses,
            demand_file=demand_file,
            align_demand_by_position=align_demand_by_position,
        )

    return {
        "result": result,
        "summary": summary,
        "annual": annual,
        "module_comparison": module_comparison,
        "site": site,
        "module": module,
        "scenario": scenario,
        "source_url": source_url,
    }


def selected_modules(selection: str) -> list[ModuleConfig]:
    if selection == "all":
        return list(MODULE_OPTIONS.values())
    return [MODULE_OPTIONS[selection]]


def parse_args() -> argparse.Namespace:
    default_start, default_end = default_date_range(SiteConfig.timezone)
    parser = argparse.ArgumentParser(
        description="Motor solar-industrial con clima historico y demanda a 15 minutos."
    )
    parser.add_argument("--name", default=SiteConfig.name, help="Nombre del sitio.")
    parser.add_argument("--lat", type=float, default=SiteConfig.latitude, help="Latitud.")
    parser.add_argument(
        "--lon",
        type=float,
        default=SiteConfig.longitude,
        help="Longitud. Para Veracruz debe ser negativa.",
    )
    parser.add_argument("--tz", default=SiteConfig.timezone, help="Zona horaria IANA.")
    parser.add_argument(
        "--altitude",
        type=float,
        default=SiteConfig.altitude_m,
        help="Altitud en metros.",
    )
    parser.add_argument(
        "--tilt",
        type=float,
        default=SiteConfig.surface_tilt,
        help="Inclinacion del panel en grados. 0 es horizontal.",
    )
    parser.add_argument(
        "--azimuth",
        type=float,
        default=SiteConfig.surface_azimuth,
        help="Azimut pvlib del panel. 180 es sur, 90 este, 270 oeste.",
    )
    parser.add_argument(
        "--albedo",
        type=float,
        default=SiteConfig.albedo,
        help="Reflectancia del suelo.",
    )
    demand_group = parser.add_mutually_exclusive_group()
    demand_group.add_argument(
        "--demand-file",
        type=Path,
        help="Curva industrial .csv, .xlsx o .xls.",
    )
    demand_group.add_argument(
        "--generate-demand",
        action="store_true",
        help="Genera un perfil industrial sintetico explicito para pruebas.",
    )
    parser.add_argument(
        "--align-demand-by-position",
        action="store_true",
        help="Alinea demanda y clima por posicion aunque pertenezcan a anos distintos.",
    )
    parser.add_argument(
        "--module",
        choices=["all", *MODULE_OPTIONS.keys()],
        default="all",
        help="Modulo a evaluar. Default: all.",
    )
    parser.add_argument(
        "--bifacial-gain",
        type=float,
        default=MODULE_OPTIONS["bifacial"].default_bifacial_gain,
        help="Ganancia bifacial como fraccion. Ejemplo: 0.15 para 15%%.",
    )
    parser.add_argument(
        "--system-dc-kwp",
        type=float,
        default=399.3,
        help="Potencia DC total para comparacion justa entre escenarios.",
    )
    parser.add_argument(
        "--num-modules",
        type=int,
        help="Numero fisico de modulos. Si se define, recalcula la potencia DC total.",
    )
    parser.add_argument(
        "--inverter-efficiency",
        type=float,
        default=0.96,
        help="Eficiencia del inversor como fraccion. Default: 0.96.",
    )
    parser.add_argument(
        "--system-losses",
        type=float,
        default=0.0,
        help="Perdidas adicionales del sistema como fraccion. Default: 0.",
    )
    parser.add_argument(
        "--weather-start-date",
        "--start-date",
        dest="weather_start_date",
        type=parse_iso_date,
        default=parse_iso_date(default_start),
        help=f"Fecha climatica inicial YYYY-MM-DD. Default: {default_start}.",
    )
    parser.add_argument(
        "--weather-end-date",
        "--end-date",
        dest="weather_end_date",
        type=parse_iso_date,
        default=parse_iso_date(default_end),
        help=f"Fecha climatica final YYYY-MM-DD. Default: {default_end}.",
    )
    parser.add_argument(
        "--freq",
        default=FINAL_FREQUENCY,
        help="Frecuencia solicitada. El motor final remuestrea a 15min.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Carpeta de salida.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bifacial_gain < 0:
        raise ValueError("--bifacial-gain no puede ser negativo.")
    if not 0 < args.inverter_efficiency <= 1:
        raise ValueError("--inverter-efficiency debe estar entre 0 y 1.")
    if not 0 <= args.system_losses < 1:
        raise ValueError("--system-losses debe estar entre 0 y 1.")
    if not args.demand_file and not args.generate_demand:
        raise ValueError(
            "Indica --demand-file o activa explicitamente --generate-demand."
        )

    site = SiteConfig(
        name=args.name,
        latitude=args.lat,
        longitude=args.lon,
        timezone=args.tz,
        altitude_m=args.altitude,
        surface_tilt=args.tilt,
        surface_azimuth=args.azimuth,
        albedo=args.albedo,
    )
    modules = selected_modules(args.module)

    target_index = build_target_index(
        args.weather_start_date,
        args.weather_end_date,
        site.timezone,
        freq=args.freq,
    )
    hourly_weather, source_url = fetch_historical_weather(
        site, args.weather_start_date, args.weather_end_date
    )
    weather = resample_weather_to_15min(hourly_weather, target_index, site)
    solar_resource = add_pv_calculations(weather, site)

    if args.demand_file:
        demand = load_demand(
            args.demand_file,
            target_index,
            timezone=site.timezone,
            align_by_position=args.align_demand_by_position,
        )
    else:
        demand = generate_demand(target_index)

    annual_summaries = []
    for module in modules:
        result, scenario = add_module_calculations(
            solar_resource,
            module=module,
            bifacial_gain=args.bifacial_gain,
            system_dc_kwp=args.system_dc_kwp,
            num_modules=args.num_modules,
            inverter_efficiency=args.inverter_efficiency,
            system_losses=args.system_losses,
        )
        result = add_demand_balance(result, demand)
        summary = monthly_summary(result)
        annual = annual_summary(result, module, scenario)
        annual_summaries.append(annual)

        scenario_output_dir = (
            args.output_dir / module.key if len(modules) > 1 else args.output_dir
        )
        write_outputs(
            result=result,
            summary=summary,
            annual=annual,
            module=module,
            scenario=scenario,
            site=site,
            source_url=source_url,
            output_dir=scenario_output_dir,
            start_date=args.weather_start_date,
            end_date=args.weather_end_date,
            bifacial_gain=args.bifacial_gain,
            inverter_efficiency=args.inverter_efficiency,
            system_losses=args.system_losses,
            demand_file=args.demand_file,
            align_demand_by_position=args.align_demand_by_position,
        )

    module_comparison = module_comparison_summary(annual_summaries)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    module_comparison.to_csv(args.output_dir / "comparacion_modulos.csv", index=False)

    total_energy = solar_resource["poa_energy_kwh_m2"].sum()
    clear_sky_energy = solar_resource["poa_clear_sky_energy_kwh_m2"].sum()
    ratio = 100 * total_energy / clear_sky_energy if clear_sky_energy else np.nan

    print("Analisis terminado")
    print(f"Sitio: {site.name} ({site.latitude}, {site.longitude})")
    print(
        f"Periodo climatico: {args.weather_start_date} a {args.weather_end_date} "
        f"({len(target_index):,} intervalos)"
    )
    print(f"Inclinacion / azimut: {site.surface_tilt} / {site.surface_azimuth}")
    print(f"Energia POA historica: {total_energy:.1f} kWh/m^2")
    print(f"Equivalente vs cielo despejado: {ratio:.1f}%")
    for _, row in module_comparison.iterrows():
        print(
            f"{row['Escenario']}: {row['Energia_Solar_AC_Anual_kWh']:.1f} kWh AC, "
            f"autoconsumo={row['Autoconsumo_pct']:.1f}%, "
            f"cobertura={row['Cobertura_Solar_pct']:.1f}%"
        )
    print(f"Salidas: {args.output_dir.resolve()}")


# main() del motor se conserva como funcion, pero no se invoca automaticamente
# en esta version Streamlit unificada para no interferir con la interfaz.


# =============================================================================
# SECCION INTERFAZ STREAMLIT
# =============================================================================

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



# =============================================================================
# SECCION CONFIGURACION STREAMLIT
# =============================================================================
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
    demand_mode = st.selectbox(
        "Modo de demanda",
        options=[
            "Consumo anual estimado",
            "Consumo mensual",
            "Curva detallada CSV/XLSX",
        ],
        key="demand_mode",
    )
    demand_df = None
    demand_validation_error = None
    uploaded_file = None
    target_index_demanda = None

    if weather_start_date <= weather_end_date:
        try:
            target_index_demanda = build_target_index(
                weather_start_date,
                weather_end_date,
                timezone,
            )
        except Exception as exc:
            demand_validation_error = f"No se pudo construir el indice de demanda: {exc}"

    if demand_mode == "Consumo anual estimado":
        consumo_anual_kwh = st.number_input(
            "Consumo anual total (kWh)",
            min_value=0.0,
            value=100_000.0,
            step=5_000.0,
            key="manual_consumo_anual_kwh",
        )
        factor_potencia_manual = st.number_input(
            "Factor de potencia estimado",
            min_value=0.01,
            max_value=1.0,
            value=0.90,
            step=0.01,
            key="manual_pf_anual",
        )
        if consumo_anual_kwh <= 0:
            demand_validation_error = "El consumo anual debe ser mayor a 0."
        elif target_index_demanda is not None:
            try:
                demand_df = generar_demanda_sintetica_15min(
                    target_index=target_index_demanda,
                    consumo_anual_kwh=float(consumo_anual_kwh),
                    factor_potencia=float(factor_potencia_manual),
                )
                energia_generada = (
                    pd.to_numeric(demand_df["Demanda_kW"], errors="coerce").sum()
                    * TIMESTEP_HOURS
                )
                st.info(
                    "La curva de demanda fue generada de forma sintetica a partir "
                    "del consumo ingresado. Para mayor precision, use una curva "
                    "medida en CSV/XLSX."
                )
                st.caption(
                    f"Energia sintetica generada: {energia_generada:,.1f} kWh "
                    f"en {len(demand_df):,} intervalos de 15 minutos."
                )
            except Exception as exc:
                demand_validation_error = str(exc)
    elif demand_mode == "Consumo mensual":
        st.caption("Consumos mensuales estimados (kWh)")
        default_monthly = [8_500.0] * 12
        consumos_mensuales_kwh = []
        month_labels = [
            "Enero",
            "Febrero",
            "Marzo",
            "Abril",
            "Mayo",
            "Junio",
            "Julio",
            "Agosto",
            "Septiembre",
            "Octubre",
            "Noviembre",
            "Diciembre",
        ]
        month_cols = st.columns(3)
        for idx, label in enumerate(month_labels):
            with month_cols[idx % 3]:
                consumos_mensuales_kwh.append(
                    st.number_input(
                        label,
                        min_value=0.0,
                        value=default_monthly[idx],
                        step=500.0,
                        key=f"manual_consumo_mes_{idx + 1}",
                    )
                )
        factor_potencia_manual = st.number_input(
            "Factor de potencia estimado",
            min_value=0.01,
            max_value=1.0,
            value=0.90,
            step=0.01,
            key="manual_pf_mensual",
        )
        if any(value < 0 for value in consumos_mensuales_kwh):
            demand_validation_error = "Los consumos mensuales deben ser mayores o iguales a 0."
        elif sum(consumos_mensuales_kwh) <= 0:
            demand_validation_error = "La suma mensual debe ser mayor a 0."
        elif target_index_demanda is not None:
            try:
                demand_df = generar_demanda_sintetica_15min(
                    target_index=target_index_demanda,
                    consumos_mensuales_kwh=[float(v) for v in consumos_mensuales_kwh],
                    factor_potencia=float(factor_potencia_manual),
                )
                demand_check = demand_df.copy()
                demand_check["Fecha_Hora"] = pd.to_datetime(demand_check["Fecha_Hora"])
                demand_check["Mes"] = demand_check["Fecha_Hora"].dt.month
                energia_por_mes = (
                    demand_check.groupby("Mes")["Demanda_kW"].sum() * TIMESTEP_HOURS
                )
                st.info(
                    "La curva de demanda fue generada de forma sintetica a partir "
                    "del consumo ingresado. Para mayor precision, use una curva "
                    "medida en CSV/XLSX."
                )
                st.caption(
                    f"Energia sintetica generada: {energia_por_mes.sum():,.1f} kWh "
                    f"en {len(demand_df):,} intervalos de 15 minutos."
                )
            except Exception as exc:
                demand_validation_error = str(exc)
    if demand_mode == "Curva detallada CSV/XLSX":
        uploaded_file = st.file_uploader(
            "Sube CSV o XLSX de demanda",
            type=["csv", "xlsx", "xls"],
            key="demand_uploader",
            help="Debe incluir Fecha_Hora y Demanda_kW. Factor_Potencia se conserva si existe.",
        )

    if uploaded_file is not None:
        try:
            demand_df = _read_uploaded_demand(uploaded_file)
            st.success(
                f"{uploaded_file.name} · {demand_df.shape[0]:,} filas x {demand_df.shape[1]} columnas"
            )
        except Exception as exc:
            st.error(f"No se pudo leer la demanda: {exc}")

    if demand_validation_error:
        st.warning(demand_validation_error)

    st.markdown("---")
    run_button = st.button("Ejecutar simulacion", width="stretch")

if run_button:
    if demand_validation_error:
        st.warning(demand_validation_error)
    elif weather_start_date > weather_end_date:
        st.error("La fecha climatica inicial no puede ser posterior a la final.")
    elif demand_df is None:
        if demand_mode == "Curva detallada CSV/XLSX":
            st.warning("Sube una curva de demanda antes de ejecutar la simulacion.")
        else:
            st.warning("No se pudo generar la curva de demanda sintetica.")
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
    df_temp_mensual = resumen_penalizacion_temperatura_mensual(
        simulation["result"]
    )
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
    st.markdown("### Penalizacion por temperatura")
    perdida_anual_temp = float(annual["Perdida_Temperatura_Anual_kWh"])
    penalizacion_anual_pct = float(annual["Penalizacion_Temperatura_Anual_pct"])
    temp_celda_prom = float(df_motor["Temperatura_Celda_C"].mean())
    temp_celda_max = float(df_motor["Temperatura_Celda_C"].max())

    mt1, mt2, mt3, mt4 = st.columns(4)
    mt1.metric("Perdida anual por temperatura", f"{perdida_anual_temp:,.0f} kWh")
    mt2.metric("Penalizacion termica anual", f"{penalizacion_anual_pct:.2f}%")
    mt3.metric("Temperatura celda promedio", f"{temp_celda_prom:.1f} C")
    mt4.metric("Temperatura celda maxima", f"{temp_celda_max:.1f} C")

    df_temp_plot = df_temp_mensual.copy()
    df_temp_plot["Mes_Str"] = df_temp_plot["Mes"].astype(str)

    fig_temp_bar = go.Figure()
    fig_temp_bar.add_trace(
        go.Bar(
            x=df_temp_plot["Mes_Str"],
            y=df_temp_plot["Generacion_Sin_Temp_kWh"],
            name="Generacion sin temp. (25 C)",
            marker_color="#4ecdc4",
        )
    )
    fig_temp_bar.add_trace(
        go.Bar(
            x=df_temp_plot["Mes_Str"],
            y=df_temp_plot["Generacion_Con_Temp_kWh"],
            name="Generacion con temp.",
            marker_color="#f5a623",
        )
    )
    fig_temp_bar.add_trace(
        go.Bar(
            x=df_temp_plot["Mes_Str"],
            y=df_temp_plot["Perdida_Temperatura_kWh"],
            name="Perdida termica",
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
    st.plotly_chart(fig_temp_bar, width="stretch", config={"displayModeBar": False})

    fig_temp_pct = _make_fig("Penalizacion termica mensual (%)")
    fig_temp_pct.add_trace(
        go.Scatter(
            x=df_temp_plot["Mes_Str"],
            y=df_temp_plot["Penalizacion_Temperatura_pct"],
            name="Penalizacion termica",
            mode="lines+markers",
            line=dict(color="#ffe033", width=2),
            marker=dict(size=7),
        )
    )
    fig_temp_pct.update_layout(yaxis_title="%", height=300)
    st.plotly_chart(fig_temp_pct, width="stretch", config={"displayModeBar": False})

    fig_temp_un_eje = _make_fig("Temperatura ambiente y temperatura de celda mensual")
    fig_temp_un_eje.add_trace(
        go.Scatter(
            x=df_temp_plot["Mes_Str"],
            y=df_temp_plot["Temperatura_Ambiente_Prom_C"],
            name="Temp. ambiente prom.",
            mode="lines+markers",
            line=dict(color="#4ecdc4", width=2),
            marker=dict(size=7),
            hovertemplate="<b>%{x}</b><br>Temp. ambiente prom.: %{y:.2f} C<extra></extra>",
        )
    )
    fig_temp_un_eje.add_trace(
        go.Scatter(
            x=df_temp_plot["Mes_Str"],
            y=df_temp_plot["Temperatura_Celda_Prom_C"],
            name="Temp. celda prom.",
            mode="lines+markers",
            line=dict(color="#f5a623", width=2),
            marker=dict(size=7),
            hovertemplate="<b>%{x}</b><br>Temp. celda prom.: %{y:.2f} C<extra></extra>",
        )
    )
    fig_temp_un_eje.add_trace(
        go.Scatter(
            x=df_temp_plot["Mes_Str"],
            y=df_temp_plot["Temperatura_Celda_Max_C"],
            name="Temp. celda max.",
            mode="lines+markers",
            line=dict(color="#ff6b6b", width=2, dash="dash"),
            marker=dict(size=7),
            hovertemplate="<b>%{x}</b><br>Temp. celda max.: %{y:.2f} C<extra></extra>",
        )
    )
    fig_temp_un_eje.update_layout(
        yaxis_title="Temperatura (C)",
        height=320,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig_temp_un_eje.update_yaxes(title_text="Temperatura (C)", gridcolor="#1e2535")
    st.plotly_chart(fig_temp_un_eje, width="stretch", config={"displayModeBar": False})

    with st.expander("Ver resumen mensual de penalizacion por temperatura"):
        st.dataframe(df_temp_mensual, width="stretch", height=280)
        st.download_button(
            "Descargar resumen mensual de temperatura",
            data=df_temp_mensual.to_csv(index=False).encode("utf-8"),
            file_name="resumen_penalizacion_temperatura_mensual.csv",
            mime="text/csv",
        )

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

st.markdown("---")
st.markdown("### BESS de respaldo")
with st.expander("Dimensionamiento y analisis de cortes", expanded=True):
    motor_bess = MotorBESS()
    default_carga_critica = 300.0
    if st.session_state.get("simulation") is not None:
        try:
            sim_result = st.session_state["simulation"]["result"]
            default_carga_critica = float(
                sim_result["Demanda_Post_Inyeccion_Solar_kW"].quantile(0.95)
            )
        except Exception:
            default_carga_critica = 300.0

    st.caption(
        "Dimensionamiento basado en gabinete Sungrow PowerStack 255CS "
        "(257 kWh, 125 kW AC, RTE 90%)."
    )
    b1, b2 = st.columns(2)
    with b1:
        horas_respaldo = st.number_input(
            "Autonomia deseada (horas)",
            min_value=0.25,
            max_value=24.0,
            value=4.0,
            step=0.25,
            key="bess_horas_respaldo",
        )
    with b2:
        carga_critica_kw = st.number_input(
            "Carga critica protegida (kW)",
            min_value=1.0,
            value=max(1.0, round(default_carga_critica, 1)),
            step=10.0,
            key="bess_carga_critica",
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        frecuencia_largos = st.number_input(
            "Apagones largos >1 hr / ano",
            min_value=0,
            value=3,
            step=1,
            key="bess_cortes_largos",
        )
    with c2:
        frecuencia_medios = st.number_input(
            "Apagones medios 1-10 min / ano",
            min_value=0,
            value=12,
            step=1,
            key="bess_cortes_medios",
        )
    with c3:
        frecuencia_cortos = st.number_input(
            "Microapagones <1 min / ano",
            min_value=0,
            value=24,
            step=1,
            key="bess_cortes_cortos",
        )

    try:
        dim_bess = motor_bess.calcular_dimensionamiento(
            horas_respaldo=float(horas_respaldo),
            carga_critica_kw=float(carga_critica_kw),
        )
        ana_bess = motor_bess.analizar_historico_cortes(
            frecuencia_largos=int(frecuencia_largos),
            frecuencia_medios=int(frecuencia_medios),
            frecuencia_cortos=int(frecuencia_cortos),
        )
        resumen_bess = motor_bess.generar_resumen_ejecutivo(
            horas_respaldo=float(horas_respaldo),
            carga_critica_kw=float(carga_critica_kw),
            frecuencia_largos=int(frecuencia_largos),
            frecuencia_medios=int(frecuencia_medios),
            frecuencia_cortos=int(frecuencia_cortos),
        )

        bm1, bm2, bm3, bm4 = st.columns(4)
        bm1.metric("Gabinetes BESS", f"{dim_bess.unidades_bess}")
        bm2.metric("Capacidad instalada", f"{dim_bess.capacidad_total_kwh:,.0f} kWh")
        bm3.metric("Potencia AC instalada", f"{dim_bess.potencia_total_kw:,.0f} kW")
        bm4.metric("Nivel de riesgo", ana_bess.nivel_riesgo)

        bm5, bm6, bm7, bm8 = st.columns(4)
        bm5.metric("Capacidad requerida", f"{dim_bess.capacidad_requerida_kwh:,.0f} kWh")
        bm6.metric("Margen energia", f"{dim_bess.sobredimensionamiento_pct:.1f}%")
        bm7.metric("Horas desabasto", f"{ana_bess.horas_desabasto_estimadas:.2f} h/ano")
        bm8.metric("Autonomia recomendada", f"{ana_bess.autonomia_recomendada_hrs:.0f} h")

        fig_bess_cap = go.Figure()
        fig_bess_cap.add_trace(
            go.Bar(
                x=["Capacidad requerida", "Capacidad instalada"],
                y=[dim_bess.capacidad_requerida_kwh, dim_bess.capacidad_total_kwh],
                marker_color=["#ff6b6b", "#4ecdc4"],
                name="kWh",
            )
        )
        fig_bess_cap.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#12171f",
            font=dict(family="IBM Plex Mono", color="#8892a4"),
            yaxis_title="kWh",
            height=320,
            showlegend=False,
        )

        fig_bess_events = go.Figure()
        fig_bess_events.add_trace(
            go.Bar(
                x=["Largos", "Medios", "Cortos"],
                y=[
                    ana_bess.cortes_largos_ano,
                    ana_bess.cortes_medios_ano,
                    ana_bess.cortes_cortos_ano,
                ],
                marker_color=["#ff6b6b", "#f5a623", "#4ecdc4"],
                name="Eventos/ano",
            )
        )
        fig_bess_events.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#12171f",
            font=dict(family="IBM Plex Mono", color="#8892a4"),
            yaxis_title="eventos/ano",
            height=320,
            showlegend=False,
        )

        g1, g2 = st.columns(2)
        with g1:
            st.plotly_chart(fig_bess_cap, width="stretch", config={"displayModeBar": False})
        with g2:
            st.plotly_chart(fig_bess_events, width="stretch", config={"displayModeBar": False})

        st.markdown("**Justificacion tecnica**")
        st.write(ana_bess.justificacion_riesgo)
        st.download_button(
            "Descargar resumen BESS",
            data=resumen_bess.encode("utf-8"),
            file_name="resumen_bess.txt",
            mime="text/plain",
        )
    except Exception as exc:
        st.error(f"No se pudo calcular BESS: {exc}")

st.markdown("---")
st.markdown("### Análisis financiero y continuidad operativa")
with st.expander("Diagnostico CFE GDMTO, riesgo por apagones y escenarios", expanded=True):
    diagnostico_fin = diagnostico_recibo_gdmto()

    st.markdown("#### Diagnostico CFE GDMTO")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Empresa", diagnostico_fin["empresa"])
    d2.metric("Tarifa", diagnostico_fin["tarifa"])
    d3.metric("Periodo", diagnostico_fin["periodo"])
    d4.metric("Energia mensual", f"{diagnostico_fin['energia_kwh']:,.0f} kWh")

    d5, d6, d7, d8 = st.columns(4)
    d5.metric("Demanda maxima", f"{diagnostico_fin['demanda_maxima_kw']:,.1f} kW")
    d6.metric("Factor de potencia", f"{diagnostico_fin['factor_potencia_pct']:.2f}%")
    d7.metric("Total pagado", formato_mxn(diagnostico_fin["total_pagar_mxn"]))
    d8.metric(
        "Precio medio total",
        f"${diagnostico_fin['precio_medio_total_mxn_kwh']:.2f}/kWh",
    )

    d9, d10, d11, d12 = st.columns(4)
    d9.metric(
        "Precio medio sin IVA",
        f"${diagnostico_fin['precio_medio_sin_iva_mxn_kwh']:.2f}/kWh",
    )
    d10.metric(
        "Precio base energia",
        f"${diagnostico_fin['precio_energia_base_mxn_kwh']:.2f}/kWh",
    )
    d11.metric(
        "Distribucion + Capacidad",
        formato_mxn(diagnostico_fin["cargos_distribucion_capacidad_mxn"]),
    )
    d12.metric(
        "Dist. + Cap. / MEM",
        formato_pct(diagnostico_fin["proporcion_distribucion_capacidad_mem"]),
    )

    st.markdown("#### Riesgo por apagones")
    r1, r2 = st.columns(2)
    with r1:
        escenario_costo_fin = st.selectbox(
            "Escenario de costo de apagon",
            ["conservador", "medio", "alto"],
            index=1,
            key="fin_escenario_costo",
        )
    with r2:
        escenario_mitigacion_fin = st.selectbox(
            "Escenario de mitigacion",
            ["conservadora", "media", "alta"],
            index=1,
            key="fin_escenario_mitigacion",
        )

    st.markdown("#### Inversiones y escenarios")
    default_autoconsumo_fin = 50_000.0
    if st.session_state.get("simulation") is not None:
        try:
            annual_fin = st.session_state["simulation"]["annual"].iloc[0]
            default_autoconsumo_fin = float(
                annual_fin["Energia_Autoconsumida_Anual_kWh"]
            )
        except Exception:
            default_autoconsumo_fin = 50_000.0

    i1, i2 = st.columns(2)
    with i1:
        inversion_fv_fin = st.number_input(
            "Inversion FV estimada MXN",
            min_value=0.0,
            value=1_500_000.0,
            step=50_000.0,
            key="fin_inversion_fv",
        )
        carga_critica_fin = st.number_input(
            "Carga critica kW",
            min_value=1.0,
            value=float(st.session_state.get("bess_carga_critica", 30.0)),
            step=5.0,
            key="fin_carga_critica_kw",
        )
        energia_autoconsumida_fin = st.number_input(
            "Energia autoconsumida anual kWh",
            min_value=0.0,
            value=max(0.0, round(default_autoconsumo_fin, 2)),
            step=1000.0,
            key="fin_autoconsumo_anual_kwh",
        )
    with i2:
        inversion_bess_fin = st.number_input(
            "Inversion BESS/UPS estimada MXN",
            min_value=0.0,
            value=1_200_000.0,
            step=50_000.0,
            key="fin_inversion_bess",
        )
        horas_respaldo_fin = st.number_input(
            "Horas de respaldo",
            min_value=0.25,
            max_value=24.0,
            value=float(st.session_state.get("bess_horas_respaldo", 4.0)),
            step=0.25,
            key="fin_horas_respaldo",
        )
        st.caption(
            "El autoconsumo se toma de la simulacion si ya existe; si no, queda editable."
        )

    try:
        resumen_financiero = generar_resumen_financiero_streger(
            escenario_costo_apagon=escenario_costo_fin,
            escenario_mitigacion=escenario_mitigacion_fin,
            carga_critica_kw=float(carga_critica_fin),
            horas_respaldo=float(horas_respaldo_fin),
            energia_autoconsumida_kwh_anual=float(energia_autoconsumida_fin),
            inversion_fv_mxn=float(inversion_fv_fin),
            inversion_bess_mxn=float(inversion_bess_fin),
        )

        perdidas_fin = resumen_financiero["perdidas_apagones"]
        beneficio_fin = resumen_financiero["beneficio_respaldo"]
        bess_fin = resumen_financiero["dimensionamiento_bess"]
        ahorro_fin = resumen_financiero["ahorro_solar_express"]
        comparacion_fin = resumen_financiero["comparacion_escenarios"]

        rm1, rm2, rm3, rm4 = st.columns(4)
        rm1.metric(
            "Eventos electricos anuales",
            f"{perdidas_fin['eventos_totales_anuales']:,.0f}",
        )
        rm2.metric(
            "Perdida anual estimada",
            formato_mxn(perdidas_fin["perdida_anual_estimada_mxn"]),
        )
        rm3.metric(
            "Referencia Streger",
            formato_mxn(perdidas_fin["perdida_anual_reportada_referencia_mxn"]),
        )
        rm4.metric(
            "Diferencia vs referencia",
            formato_mxn(perdidas_fin["diferencia_vs_referencia_mxn"]),
        )

        rm5, rm6, rm7, rm8 = st.columns(4)
        rm5.metric(
            "Proporcion vs referencia",
            f"{perdidas_fin['proporcion_vs_referencia']:.2f}x",
        )
        rm6.metric(
            "Beneficio evitable anual",
            formato_mxn(beneficio_fin["beneficio_anual_evitable_mxn"]),
        )
        rm7.metric(
            "Porcentaje total evitable",
            formato_pct(beneficio_fin["porcentaje_total_evitable"]),
        )
        rm8.metric(
            "Ahorro solar anual express",
            formato_mxn(ahorro_fin["ahorro_solar_anual_mxn"]),
        )

        fm1, fm2, fm3 = st.columns(3)
        fm1.metric(
            "Capacidad BESS sugerida",
            f"{bess_fin['capacidad_sugerida_kwh']:,.1f} kWh",
        )
        fm2.metric(
            "Baterias Sungrow equivalentes",
            f"{bess_fin['baterias_equivalentes']:.2f}",
        )
        fm3.metric(
            "Potencia suficiente con una bateria",
            "Si" if bess_fin["potencia_suficiente_una_bateria"] else "No",
        )

        detalle_fin_df = detalle_apagones_financiero_df(perdidas_fin, beneficio_fin)
        detalle_fin_display = detalle_fin_df.copy()
        detalle_fin_display["costo_promedio_evento_mxn"] = detalle_fin_display[
            "costo_promedio_evento_mxn"
        ].map(formato_mxn)
        detalle_fin_display["perdida_anual_mxn"] = detalle_fin_display[
            "perdida_anual_mxn"
        ].map(formato_mxn)
        detalle_fin_display["porcentaje_evitable"] = detalle_fin_display[
            "porcentaje_evitable"
        ].map(formato_pct)
        detalle_fin_display["beneficio_evitable_mxn"] = detalle_fin_display[
            "beneficio_evitable_mxn"
        ].map(formato_mxn)

        st.markdown("**Detalle por tipo de apagon**")
        st.dataframe(detalle_fin_display, width="stretch", hide_index=True)

        escenarios_fin_df = escenarios_financieros_df(comparacion_fin)
        escenarios_display = escenarios_fin_df.copy()
        escenarios_display["inversion_mxn"] = escenarios_display["inversion_mxn"].map(
            formato_mxn
        )
        escenarios_display["beneficio_anual_mxn"] = escenarios_display[
            "beneficio_anual_mxn"
        ].map(formato_mxn)
        escenarios_display["payback_anios"] = escenarios_display["payback_anios"].map(
            lambda valor: formato_anios(valor) if pd.notna(valor) else "No aplica"
        )
        escenarios_display["roi_simple_anual"] = escenarios_display[
            "roi_simple_anual"
        ].map(lambda valor: formato_pct(valor) if pd.notna(valor) else "No aplica")

        st.markdown("**Payback y ROI por escenario**")
        st.dataframe(
            escenarios_display[
                [
                    "nombre",
                    "inversion_mxn",
                    "beneficio_anual_mxn",
                    "payback_anios",
                    "roi_simple_anual",
                    "descripcion",
                ]
            ],
            width="stretch",
            hide_index=True,
        )

        st.markdown("#### Visualizaciones financieras")
        fig_loss_type = figura_financiera_base()
        fig_loss_type.add_trace(
            go.Bar(
                x=detalle_fin_df["tipo_apagon"].str.capitalize(),
                y=detalle_fin_df["perdida_anual_mxn"],
                marker_color="#ff6b6b",
                name="Perdida anual",
            )
        )
        fig_loss_type.update_layout(
            title="Perdida anual por tipo de apagon",
            yaxis_title="MXN/ano",
            showlegend=False,
        )

        fig_reference = figura_financiera_base()
        fig_reference.add_trace(
            go.Bar(
                x=["Estimacion externa", "Referencia Streger"],
                y=[
                    perdidas_fin["perdida_anual_estimada_mxn"],
                    perdidas_fin["perdida_anual_reportada_referencia_mxn"],
                ],
                marker_color=["#ffe033", "#5b8def"],
                name="MXN",
            )
        )
        fig_reference.update_layout(
            title="Perdida estimada vs referencia Streger",
            yaxis_title="MXN/ano",
            showlegend=False,
        )

        fig_avoidable = figura_financiera_base()
        fig_avoidable.add_trace(
            go.Bar(
                x=detalle_fin_df["tipo_apagon"].str.capitalize(),
                y=detalle_fin_df["perdida_anual_mxn"],
                marker_color="#ff6b6b",
                name="Perdida estimada",
            )
        )
        fig_avoidable.add_trace(
            go.Bar(
                x=detalle_fin_df["tipo_apagon"].str.capitalize(),
                y=detalle_fin_df["beneficio_evitable_mxn"],
                marker_color="#4ecdc4",
                name="Beneficio evitable",
            )
        )
        fig_avoidable.update_layout(
            title="Perdida estimada vs beneficio evitable",
            yaxis_title="MXN/ano",
            barmode="group",
        )

        escenarios_graf = escenarios_fin_df[
            escenarios_fin_df["escenario"] != "base_sin_fv_sin_bess"
        ]
        fig_invest = figura_financiera_base()
        fig_invest.add_trace(
            go.Bar(
                x=escenarios_graf["nombre"],
                y=escenarios_graf["inversion_mxn"],
                marker_color="#5b8def",
                name="Inversion",
            )
        )
        fig_invest.add_trace(
            go.Bar(
                x=escenarios_graf["nombre"],
                y=escenarios_graf["beneficio_anual_mxn"],
                marker_color="#4ecdc4",
                name="Beneficio anual",
            )
        )
        fig_invest.update_layout(
            title="Inversion vs beneficio anual por escenario",
            yaxis_title="MXN",
            barmode="group",
        )

        payback_graf = escenarios_graf[pd.notna(escenarios_graf["payback_anios"])]
        fig_payback = figura_financiera_base()
        fig_payback.add_trace(
            go.Bar(
                x=payback_graf["nombre"],
                y=payback_graf["payback_anios"],
                marker_color="#f5a623",
                name="Payback",
            )
        )
        fig_payback.update_layout(
            title="Payback por escenario",
            yaxis_title="anos",
            showlegend=False,
        )

        fg1, fg2 = st.columns(2)
        with fg1:
            st.plotly_chart(fig_loss_type, width="stretch", config={"displayModeBar": False})
            st.plotly_chart(fig_avoidable, width="stretch", config={"displayModeBar": False})
            st.plotly_chart(fig_payback, width="stretch", config={"displayModeBar": False})
        with fg2:
            st.plotly_chart(fig_reference, width="stretch", config={"displayModeBar": False})
            st.plotly_chart(fig_invest, width="stretch", config={"displayModeBar": False})

        dl1, dl2, dl3 = st.columns(3)
        with dl1:
            st.download_button(
                "Descargar resumen financiero JSON",
                data=json.dumps(
                    resumen_financiero,
                    ensure_ascii=False,
                    indent=2,
                ).encode("utf-8"),
                file_name="resumen_financiero_streger.json",
                mime="application/json",
            )
        with dl2:
            st.download_button(
                "Descargar escenarios financieros",
                data=escenarios_fin_df.to_csv(index=False).encode("utf-8"),
                file_name="escenarios_financieros.csv",
                mime="text/csv",
            )
        with dl3:
            st.download_button(
                "Descargar detalle apagones",
                data=detalle_fin_df.to_csv(index=False).encode("utf-8"),
                file_name="detalle_apagones_financiero.csv",
                mime="text/csv",
            )

        st.caption(
            "Analisis preliminar: no sustituye auditoria electrica, cotizacion formal "
            "ni diseno ejecutivo."
        )
    except Exception as exc:
        st.error(f"No se pudo calcular el analisis financiero: {exc}")
