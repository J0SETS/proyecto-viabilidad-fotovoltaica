from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional


@dataclass(frozen=True)
class EspecificacionesBESS:
    modelo: str = "Sungrow PowerStack 255CS (ST255CS-2H)"
    quimica: str = "LFP"
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

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ResultadoAnalisisCortes:
    cortes_largos_ano: int
    cortes_medios_ano: int
    cortes_cortos_ano: int
    total_eventos_ano: int
    horas_desabasto_estimadas: float
    nivel_riesgo: str
    autonomia_recomendada_hrs: float
    justificacion_riesgo: str

    def to_dict(self) -> dict:
        return asdict(self)


class MotorBESS:
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
        sobredimensionamiento_pct = (
            (capacidad_total_kwh - capacidad_requerida_kwh) / capacidad_requerida_kwh * 100
            if capacidad_requerida_kwh
            else 0.0
        )

        return ResultadoDimensionamiento(
            horas_respaldo=horas_respaldo,
            carga_critica_kw=carga_critica_kw,
            capacidad_requerida_kwh=capacidad_requerida_kwh,
            unidades_bess=unidades_bess,
            potencia_total_kw=potencia_total_kw,
            capacidad_total_kwh=capacidad_total_kwh,
            ciclos_referencia=round((self.specs.ciclos_minimos + self.specs.ciclos_maximos) / 2),
            modelo_referencia=self.specs.modelo,
            sobredimensionamiento_pct=sobredimensionamiento_pct,
        )

    def analizar_historico_cortes(
        self,
        frecuencia_largos: int = 3,
        frecuencia_medios: int = 12,
        frecuencia_cortos: int = 24,
    ) -> ResultadoAnalisisCortes:
        for nombre, valor in {
            "frecuencia_largos": frecuencia_largos,
            "frecuencia_medios": frecuencia_medios,
            "frecuencia_cortos": frecuencia_cortos,
        }.items():
            if valor < 0:
                raise ValueError(f"{nombre} no puede ser negativo. Recibido: {valor}")

        total_eventos = frecuencia_largos + frecuencia_medios + frecuencia_cortos
        horas_desabasto = frecuencia_largos * 1.5 + frecuencia_medios * (10 / 60) + frecuencia_cortos * (1 / 60)

        if total_eventos <= 10 and frecuencia_largos <= 1:
            nivel = "Bajo"
            autonomia = 2.0
            justificacion = (
                "El historial tiene pocos eventos severos. Un respaldo base puede "
                "cubrir microcortes y proteger cargas sensibles."
            )
        elif total_eventos <= 35 and frecuencia_largos <= 3:
            nivel = "Medio"
            autonomia = 4.0
            justificacion = (
                "El historial combina microcortes y eventos largos. Un BESS de "
                "autonomia media reduce riesgo operativo y reinicios de proceso."
            )
        else:
            nivel = "Alto"
            autonomia = 8.0
            justificacion = (
                "La frecuencia de eventos justifica evaluar respaldo extendido para "
                "continuidad operativa de cargas criticas."
            )

        return ResultadoAnalisisCortes(
            cortes_largos_ano=int(frecuencia_largos),
            cortes_medios_ano=int(frecuencia_medios),
            cortes_cortos_ano=int(frecuencia_cortos),
            total_eventos_ano=int(total_eventos),
            horas_desabasto_estimadas=float(horas_desabasto),
            nivel_riesgo=nivel,
            autonomia_recomendada_hrs=autonomia,
            justificacion_riesgo=justificacion,
        )

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
            frecuencia_largos,
            frecuencia_medios,
            frecuencia_cortos,
        )
        return f"""RESUMEN EJECUTIVO - SISTEMA BESS DE RESPALDO

Modelo: {dim.modelo_referencia}
Carga critica: {dim.carga_critica_kw:,.1f} kW
Autonomia solicitada: {dim.horas_respaldo:.1f} h
Capacidad requerida: {dim.capacidad_requerida_kwh:,.1f} kWh
Gabinetes requeridos: {dim.unidades_bess}
Capacidad instalada: {dim.capacidad_total_kwh:,.1f} kWh
Potencia AC instalada: {dim.potencia_total_kw:,.1f} kW
Margen de energia: {dim.sobredimensionamiento_pct:.1f}%

Eventos anuales estimados: {ana.total_eventos_ano}
Horas de desabasto estimadas: {ana.horas_desabasto_estimadas:.2f} h/ano
Nivel de riesgo: {ana.nivel_riesgo}
Autonomia recomendada: {ana.autonomia_recomendada_hrs:.1f} h

Lectura tecnica:
{ana.justificacion_riesgo}
""".strip()


def dimensionar_bess(
    horas_respaldo: float,
    carga_critica_kw: float,
    frecuencia_largos: int = 3,
    frecuencia_medios: int = 12,
    frecuencia_cortos: int = 24,
) -> dict:
    motor = MotorBESS()
    dim = motor.calcular_dimensionamiento(horas_respaldo, carga_critica_kw)
    ana = motor.analizar_historico_cortes(
        frecuencia_largos,
        frecuencia_medios,
        frecuencia_cortos,
    )
    return {
        "dimensionamiento": dim.to_dict(),
        "analisis_cortes": ana.to_dict(),
        "especificaciones": asdict(motor.specs),
    }
