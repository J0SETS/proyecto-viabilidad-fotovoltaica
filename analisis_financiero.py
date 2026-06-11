# =============================================================================
# analisis_financiero.py — Flujo de Caja 20 Años y ROI para Sistema FV + BESS
# =============================================================================

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# TIPOS
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class InputsFinancieros:
    capex_mxn: float                   # Inversión inicial total (MXN)
    opex_anual_mxn: float              # Costo O&M anual (MXN)
    perdidas_apagon_mxn: float         # Pérdidas económicas anuales por apagones (MXN)
    tarifa_cfe_mxn_kwh: float          # Tarifa CFE actual (MXN/kWh)
    consumo_cubierto_kwh: float        # kWh anuales cubiertos por el sistema FV
    inflacion_tarifa_pct: float        # Inflación anual de tarifa eléctrica (fracción, ej. 0.07)
    degradacion_panel_pct: float = 0.005  # Degradación anual de paneles (fracción, ej. 0.005 = 0.5 %)
    horizonte_anios: int = 20


# ─────────────────────────────────────────────────────────────────────────────
# MOTOR DE CÁLCULO
# ─────────────────────────────────────────────────────────────────────────────
def calcular_cashflow_20_anios(
    inputs: InputsFinancieros,
) -> tuple[pd.DataFrame, float]:
    """
    Calcula el flujo de caja neto anual y el payback (ROI) del sistema FV+BESS.

    Returns
    -------
    df : pd.DataFrame
        Columnas: Año, Ahorro_Tarifa_MXN, Ahorro_Apagones_MXN, OPEX_MXN,
                  Flujo_Neto_MXN, Flujo_Acumulado_MXN
    roi_anios : float
        Años hasta recuperar la inversión (interpolado). `inf` si no se recupera
        dentro del horizonte.
    """
    h = inputs.horizonte_anios
    registros: list[dict] = []
    acumulado = -inputs.capex_mxn

    roi_anios = float("inf")

    for anio in range(1, h + 1):
        factor_tarifa     = (1 + inputs.inflacion_tarifa_pct) ** (anio - 1)
        factor_panel      = (1 - inputs.degradacion_panel_pct) ** (anio - 1)

        tarifa_anio       = inputs.tarifa_cfe_mxn_kwh * factor_tarifa
        kwh_generados     = inputs.consumo_cubierto_kwh * factor_panel

        ahorro_tarifa     = tarifa_anio * kwh_generados
        ahorro_apagones   = inputs.perdidas_apagon_mxn          # conservador: constante
        opex              = inputs.opex_anual_mxn

        flujo_neto        = ahorro_tarifa + ahorro_apagones - opex
        acumulado_prev    = acumulado
        acumulado        += flujo_neto

        # Interpolación lineal del payback
        if roi_anios == float("inf") and acumulado >= 0 and acumulado_prev < 0:
            fraccion   = -acumulado_prev / flujo_neto
            roi_anios  = (anio - 1) + fraccion

        registros.append({
            "Año":                  anio,
            "Ahorro_Tarifa_MXN":    ahorro_tarifa,
            "Ahorro_Apagones_MXN":  ahorro_apagones,
            "OPEX_MXN":             opex,
            "Flujo_Neto_MXN":       flujo_neto,
            "Flujo_Acumulado_MXN":  acumulado,
        })

    df = pd.DataFrame(registros)
    return df, roi_anios


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS DE FORMATO (reutilizables en la UI)
# ─────────────────────────────────────────────────────────────────────────────
def fmt_mxn(valor: float, decimales: int = 0) -> str:
    signo = "-" if valor < 0 else ""
    return f"{signo}${abs(valor):,.{decimales}f} MXN"


def fmt_anios(roi: float) -> str:
    if roi == float("inf"):
        return "No recuperado en el horizonte"
    anios_int = int(roi)
    meses     = round((roi - anios_int) * 12)
    return f"{anios_int} años {meses} meses"