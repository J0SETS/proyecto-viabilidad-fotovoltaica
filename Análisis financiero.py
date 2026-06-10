"""
finanzas_streger_v2.py

Módulo financiero preliminar para Streger S.A.

Objetivo:
Analizar el impacto financiero de:
1. Recibo CFE GDMTO.
2. Riesgo operativo por apagones en producción farmacéutica.
3. Pérdidas potenciales por eventos largos, medios y cortos.
4. Beneficio evitable con BESS / UPS / sistema de respaldo.
5. Payback y ROI preliminar.

Este archivo está hecho para poder ejecutarse solo.
Después puede integrarse al motor solar y a la interfaz Streamlit.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Optional
import json
from pathlib import Path
import matplotlib.pyplot as plt


# ============================================================
# 1. DATOS BASE DEL RECIBO CFE DE STREGER
# ============================================================

RECIBO_STREGER_MAYO_2026 = {
    "empresa": "STREGER S.A.",
    "ubicacion": "Coatepec, Veracruz",
    "tarifa": "GDMTO",
    "periodo": "28 ABR 26 - 29 MAY 26",
    "dias_facturados": 31,
    "multiplicador": 80,

    # Datos eléctricos
    "carga_conectada_kw": 99,
    "demanda_contratada_kw": 99,
    "energia_kwh": 9520,
    "demanda_maxima_kw": 80,
    "factor_potencia_pct": 89.89,

    # Datos económicos
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


# ============================================================
# 2. DATOS OFICIALES DE APAGONES REPORTADOS POR STREGER
# ============================================================

"""
Información tomada de la captura oficial compartida por Streger:

Análisis de apagones del departamento de sistemas:
- Largo: 3 al año, duración > 1 hr
- Medio: 1 al mes, duración < 10 min
- Corto: 2 al mes, duración < 1 min

Tiempo de no-break protector:
- Estándar: 5 min
- Especiales: 30 min

Cortes programados CFE observados:
- sep-21, 6 hr, 9am
- ene-23, 5 hr, 10am
- 05-ene-23, 5 hr, 10am
"""

FRECUENCIA_APAGONES_STREGER = {
    "largo": {
        "frecuencia_anual": 3,
        "duracion_referencia_horas": 1.5,
        "descripcion": "Apagón mayor a 1 hora",
    },
    "medio": {
        "frecuencia_anual": 12,
        "duracion_referencia_horas": 10 / 60,
        "descripcion": "Apagón menor a 10 minutos",
    },
    "corto": {
        "frecuencia_anual": 24,
        "duracion_referencia_horas": 1 / 60,
        "descripcion": "Apagón menor a 1 minuto",
    },
}


PROTECCION_EXISTENTE_STREGER = {
    "nobreak_estandar_min": 5,
    "nobreak_especial_min": 30,
}


PERDIDA_ANUAL_REPORTADA_REFERENCIA_MXN = 2_000_000


# ============================================================
# 3. ESCENARIOS EXTERNOS DE COSTO POR TIPO DE APAGÓN
# ============================================================

"""
Estos valores NO fuerzan el dato de $2,000,000 MXN.
Son una estimación externa por severidad del evento.

La idea:
- Cortes largos: pueden detener producción, afectar cultivos, refrigeración,
  HVAC, incubación, limpieza, validación y liberación de lotes.
- Cortes medios: pueden causar reinicios, alarmas, pérdidas parciales,
  desviaciones de proceso o fallos de sistemas críticos.
- Cortes cortos: pueden causar resets, descalibración, paro de PLC,
  fallas en variadores, pérdida de datos o microdesviaciones.
"""

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


# ============================================================
# 4. MITIGACIÓN CON BESS / UPS
# ============================================================

"""
El sistema de respaldo no debe asumirse como una solución perfecta.
Por eso se modela un porcentaje evitable por tipo de evento.

Los eventos cortos suelen ser más evitables si hay UPS/no-break bien integrado.
Los eventos largos dependen más del tamaño real del BESS y de la carga crítica.
"""

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


# ============================================================
# 5. BATERÍA DE REFERENCIA
# ============================================================

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


# ============================================================
# 6. UTILIDADES
# ============================================================

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
    return f"{valor:.2f} años"


# ============================================================
# 7. DIAGNÓSTICO DEL RECIBO
# ============================================================

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
            "La factura no depende solamente de la energía consumida en kWh. "
            "Una parte importante está asociada a demanda, capacidad e infraestructura."
        ),
    }


# ============================================================
# 8. MODELO DE PÉRDIDAS POR APAGONES
# ============================================================

def calcular_perdidas_por_apagones(
    escenario_costo: str = "medio",
    frecuencia_apagones: Dict = FRECUENCIA_APAGONES_STREGER,
) -> Dict:
    if escenario_costo not in COSTO_EVENTO_APAGON_MXN:
        raise ValueError(
            f"Escenario no válido: {escenario_costo}. "
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

    diferencia_vs_referencia = (
        perdida_total - PERDIDA_ANUAL_REPORTADA_REFERENCIA_MXN
    )

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
            "El dato reportado por Streger se usa como referencia de validación, "
            "no como centro del cálculo. La pérdida se estima desde frecuencia "
            "oficial y costo externo por severidad."
        ),
    }


def calcular_beneficio_evitable_con_respaldo(
    perdidas_apagones: Dict,
    escenario_mitigacion: str = "media",
) -> Dict:
    if escenario_mitigacion not in MITIGACION_BESS_UPS:
        raise ValueError(
            f"Escenario no válido: {escenario_mitigacion}. "
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
            "El beneficio evitable representa la fracción de pérdidas que un sistema "
            "UPS/BESS bien integrado podría reducir. No se asume eliminación total "
            "del riesgo."
        ),
    }


# ============================================================
# 9. DIMENSIONAMIENTO PRELIMINAR BESS
# ============================================================

@dataclass
class InputsBESS:
    carga_critica_kw: float
    horas_respaldo: float
    dod: float = 0.80
    eficiencia_sistema: float = 0.90
    margen_seguridad: float = 1.10


def dimensionar_bess_respaldo(inputs: InputsBESS) -> Dict:
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
            "Este es un predimensionamiento financiero-técnico. El diseño final "
            "requiere revisión eléctrica, transferencia, protecciones, tableros, "
            "UPS para microcortes y selección real de cargas críticas."
        ),
    }


# ============================================================
# 10. AHORRO SOLAR Y MÉTRICAS FINANCIERAS
# ============================================================

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
            "Cálculo express con precio medio. Después debe reemplazarse por "
            "un cálculo horario/tarifario usando la salida del motor solar."
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
            "descripcion": "Escenario actual sin inversión.",
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
            "descripcion": "Reducción de pérdidas operativas por apagones.",
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
            "descripcion": "Ahorro energético más continuidad operativa.",
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


# ============================================================
# 11. RESUMEN INTEGRADO
# ============================================================

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

    bess = dimensionar_bess_respaldo(
        InputsBESS(
            carga_critica_kw=carga_critica_kw,
            horas_respaldo=horas_respaldo,
        )
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


# ============================================================
# 12. EXPORTAR RESULTADOS
# ============================================================

def exportar_resumen_json(
    resumen: Dict,
    carpeta_salida: str = "salidas_financieras_streger",
    nombre_archivo: str = "resumen_financiero_streger.json",
) -> Path:
    carpeta = Path(carpeta_salida)
    carpeta.mkdir(parents=True, exist_ok=True)

    ruta = carpeta / nombre_archivo

    with open(ruta, "w", encoding="utf-8") as archivo:
        json.dump(resumen, archivo, indent=4, ensure_ascii=False)

    return ruta


# ============================================================
# 13. IMPRESIÓN EN CONSOLA
# ============================================================

def imprimir_resumen(resumen: Dict) -> None:
    diagnostico = resumen["diagnostico_recibo"]
    perdidas = resumen["perdidas_apagones"]
    beneficio = resumen["beneficio_respaldo"]
    bess = resumen["dimensionamiento_bess"]
    ahorro = resumen["ahorro_solar_express"]
    escenarios = resumen["comparacion_escenarios"]["escenarios"]

    print("\n" + "=" * 70)
    print("ANÁLISIS FINANCIERO STREGER - V2")
    print("=" * 70)

    print("\n1. DIAGNÓSTICO RECIBO CFE")
    print("-" * 70)
    print(f"Empresa: {diagnostico['empresa']}")
    print(f"Ubicación: {diagnostico['ubicacion']}")
    print(f"Tarifa: {diagnostico['tarifa']}")
    print(f"Periodo: {diagnostico['periodo']}")
    print(f"Energía mensual: {diagnostico['energia_kwh']:,.0f} kWh")
    print(f"Demanda máxima: {diagnostico['demanda_maxima_kw']:,.1f} kW")
    print(f"Factor de potencia: {diagnostico['factor_potencia_pct']:.2f}%")
    print(f"Total pagado: {formato_mxn(diagnostico['total_pagar_mxn'])}")
    print(
        "Precio medio total: "
        f"${diagnostico['precio_medio_total_mxn_kwh']:.2f} MXN/kWh"
    )
    print(
        "Distribución + Capacidad: "
        f"{formato_mxn(diagnostico['cargos_distribucion_capacidad_mxn'])}"
    )
    print(
        "Proporción Distribución + Capacidad sobre MEM: "
        f"{formato_pct(diagnostico['proporcion_distribucion_capacidad_mem'])}"
    )

    print("\n2. FRECUENCIA OFICIAL DE APAGONES")
    print("-" * 70)
    print(f"Eventos eléctricos estimados al año: {perdidas['eventos_totales_anuales']}")

    for tipo, datos in perdidas["detalle_por_tipo"].items():
        print(
            f"{tipo.capitalize()}: "
            f"{datos['frecuencia_anual']} eventos/año | "
            f"{datos['descripcion']} | "
            f"Costo/evento: {formato_mxn(datos['costo_promedio_evento_mxn'])}"
        )

    print("\n3. PÉRDIDA ANUAL ESTIMADA POR APAGONES")
    print("-" * 70)
    print(f"Escenario de costo: {perdidas['escenario_costo']}")
    print(
        "Pérdida anual estimada desde análisis externo: "
        f"{formato_mxn(perdidas['perdida_anual_estimada_mxn'])}"
    )
    print(
        "Referencia reportada por Streger: "
        f"{formato_mxn(perdidas['perdida_anual_reportada_referencia_mxn'])}"
    )
    print(
        "Diferencia vs referencia: "
        f"{formato_mxn(perdidas['diferencia_vs_referencia_mxn'])}"
    )
    print(
        "Proporción vs referencia: "
        f"{perdidas['proporcion_vs_referencia']:.2f}x"
    )

    print("\n4. BENEFICIO EVITABLE CON BESS / UPS")
    print("-" * 70)
    print(f"Escenario de mitigación: {beneficio['escenario_mitigacion']}")
    print(
        "Beneficio anual evitable: "
        f"{formato_mxn(beneficio['beneficio_anual_evitable_mxn'])}"
    )
    print(
        "Porcentaje total evitable: "
        f"{formato_pct(beneficio['porcentaje_total_evitable'])}"
    )

    for tipo, datos in beneficio["detalle_por_tipo"].items():
        print(
            f"{tipo.capitalize()}: "
            f"evitable {formato_pct(datos['porcentaje_evitable'])} | "
            f"beneficio {formato_mxn(datos['beneficio_evitable_mxn'])}"
        )

    print("\n5. PREDIMENSIONAMIENTO BESS")
    print("-" * 70)
    print(
        "Carga crítica considerada: "
        f"{bess['inputs']['carga_critica_kw']:.1f} kW"
    )
    print(
        "Horas de respaldo: "
        f"{bess['inputs']['horas_respaldo']:.1f} h"
    )
    print(
        "Energía crítica: "
        f"{bess['energia_critica_kwh']:.1f} kWh"
    )
    print(
        "Capacidad sugerida: "
        f"{bess['capacidad_sugerida_kwh']:.1f} kWh"
    )
    print(
        "Baterías Sungrow equivalentes: "
        f"{bess['baterias_equivalentes']:.2f}"
    )

    print("\n6. AHORRO SOLAR EXPRESS")
    print("-" * 70)
    print(
        "Energía autoconsumida anual: "
        f"{ahorro['energia_autoconsumida_kwh_anual']:,.0f} kWh"
    )
    print(
        "Ahorro solar anual estimado: "
        f"{formato_mxn(ahorro['ahorro_solar_anual_mxn'])}"
    )

    print("\n7. COMPARACIÓN DE ESCENARIOS")
    print("-" * 70)

    for nombre, datos in escenarios.items():
        print(f"\n{nombre}")
        print(f"  Descripción: {datos['descripcion']}")
        print(f"  Inversión: {formato_mxn(datos['inversion_mxn'])}")
        print(f"  Beneficio anual: {formato_mxn(datos['beneficio_anual_mxn'])}")
        print(f"  Payback: {formato_anios(datos['payback_anios'])}")

        roi = datos["roi_simple_anual"]
        if roi is None:
            print("  ROI simple anual: No aplica")
        else:
            print(f"  ROI simple anual: {formato_pct(roi)}")


# ============================================================
# 14. VISUALIZACIONES SIMPLES EN VENTANAS
# ============================================================

def mostrar_perdidas_por_tipo(resumen: Dict) -> None:
    detalle = resumen["perdidas_apagones"]["detalle_por_tipo"]

    tipos = []
    perdidas = []

    for tipo, datos in detalle.items():
        tipos.append(tipo.capitalize())
        perdidas.append(datos["perdida_anual_mxn"])

    plt.figure("Pérdida anual por tipo de apagón", figsize=(8, 5))
    plt.bar(tipos, perdidas)
    plt.title("Pérdida anual estimada por tipo de apagón")
    plt.xlabel("Tipo de apagón")
    plt.ylabel("Pérdida anual estimada (MXN)")
    plt.grid(axis="y", alpha=0.3)

    for i, valor in enumerate(perdidas):
        plt.text(i, valor, f"${valor:,.0f}", ha="center", va="bottom", fontsize=9)


def mostrar_perdida_vs_referencia(resumen: Dict) -> None:
    perdidas = resumen["perdidas_apagones"]

    categorias = ["Estimación externa", "Referencia Streger"]
    valores = [
        perdidas["perdida_anual_estimada_mxn"],
        perdidas["perdida_anual_reportada_referencia_mxn"],
    ]

    plt.figure("Pérdida estimada vs referencia", figsize=(8, 5))
    plt.bar(categorias, valores)
    plt.title("Pérdida anual por apagones: estimación vs referencia")
    plt.ylabel("Pérdida anual (MXN)")
    plt.grid(axis="y", alpha=0.3)

    for i, valor in enumerate(valores):
        plt.text(i, valor, f"${valor:,.0f}", ha="center", va="bottom", fontsize=9)


def mostrar_perdidas_y_beneficio_evitable(resumen: Dict) -> None:
    detalle = resumen["beneficio_respaldo"]["detalle_por_tipo"]

    tipos = []
    perdidas = []
    beneficios = []

    for tipo, datos in detalle.items():
        tipos.append(tipo.capitalize())
        perdidas.append(datos["perdida_anual_mxn"])
        beneficios.append(datos["beneficio_evitable_mxn"])

    x = range(len(tipos))
    ancho = 0.35

    plt.figure("Pérdida vs beneficio evitable", figsize=(9, 5))
    plt.bar([i - ancho / 2 for i in x], perdidas, width=ancho, label="Pérdida estimada")
    plt.bar([i + ancho / 2 for i in x], beneficios, width=ancho, label="Beneficio evitable")

    plt.title("Pérdida estimada vs beneficio evitable con respaldo")
    plt.xlabel("Tipo de apagón")
    plt.ylabel("Monto anual (MXN)")
    plt.xticks(list(x), tipos)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)


def mostrar_comparacion_escenarios(resumen: Dict) -> None:
    escenarios = resumen["comparacion_escenarios"]["escenarios"]

    nombres = []
    inversiones = []
    beneficios = []

    for nombre, datos in escenarios.items():
        if nombre == "base_sin_fv_sin_bess":
            continue

        nombre_limpio = (
            nombre
            .replace("_", " ")
            .replace("fv", "FV")
            .replace("bess", "BESS")
            .replace("ups", "UPS")
            .capitalize()
        )

        nombres.append(nombre_limpio)
        inversiones.append(datos["inversion_mxn"])
        beneficios.append(datos["beneficio_anual_mxn"])

    x = range(len(nombres))
    ancho = 0.35

    plt.figure("Comparación de escenarios", figsize=(10, 5))
    plt.bar([i - ancho / 2 for i in x], inversiones, width=ancho, label="Inversión")
    plt.bar([i + ancho / 2 for i in x], beneficios, width=ancho, label="Beneficio anual")

    plt.title("Comparación financiera de escenarios")
    plt.xlabel("Escenario")
    plt.ylabel("Monto (MXN)")
    plt.xticks(list(x), nombres, rotation=15, ha="right")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)


def mostrar_payback_escenarios(resumen: Dict) -> None:
    escenarios = resumen["comparacion_escenarios"]["escenarios"]

    nombres = []
    paybacks = []

    for nombre, datos in escenarios.items():
        payback = datos["payback_anios"]
        if payback is None:
            continue

        nombre_limpio = (
            nombre
            .replace("_", " ")
            .replace("fv", "FV")
            .replace("bess", "BESS")
            .replace("ups", "UPS")
            .capitalize()
        )

        nombres.append(nombre_limpio)
        paybacks.append(payback)

    plt.figure("Payback por escenario", figsize=(9, 5))
    plt.bar(nombres, paybacks)
    plt.title("Payback simple por escenario")
    plt.xlabel("Escenario")
    plt.ylabel("Payback simple (años)")
    plt.xticks(rotation=15, ha="right")
    plt.grid(axis="y", alpha=0.3)

    for i, valor in enumerate(paybacks):
        plt.text(i, valor, f"{valor:.2f} años", ha="center", va="bottom", fontsize=9)


def mostrar_visualizaciones_financieras(resumen: Dict) -> None:
    """
    Abre todas las gráficas en ventanas, estilo MATLAB.
    """
    mostrar_perdidas_por_tipo(resumen)
    mostrar_perdida_vs_referencia(resumen)
    mostrar_perdidas_y_beneficio_evitable(resumen)
    mostrar_comparacion_escenarios(resumen)
    mostrar_payback_escenarios(resumen)

    plt.show()


# ============================================================
# 15. DEMO EJECUTABLE
# ============================================================

if __name__ == "__main__":

    resumen = generar_resumen_financiero_streger(
        escenario_costo_apagon="medio",
        escenario_mitigacion="media",
        carga_critica_kw=30,
        horas_respaldo=4,
        energia_autoconsumida_kwh_anual=50_000,
        inversion_fv_mxn=1_500_000,
        inversion_bess_mxn=1_200_000,
    )

    imprimir_resumen(resumen)

    mostrar_visualizaciones_financieras(resumen)

