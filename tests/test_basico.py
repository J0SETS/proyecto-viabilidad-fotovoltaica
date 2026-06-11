from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from demanda import read_demand_file
from finanzas import (
    calcular_autoconsumo_desde_simulacion,
    calcular_payback,
    calcular_roi_simple_anual,
)


class TestBasico(unittest.TestCase):
    def test_autoconsumo_desde_columnas_kw(self):
        df = pd.DataFrame(
            {
                "Demanda_kW": [10, 20, 5, 0],
                "Generacion_Solar_kW": [5, 30, 5, 10],
            }
        )
        self.assertEqual(calcular_autoconsumo_desde_simulacion(df), 7.5)

    def test_payback(self):
        self.assertEqual(calcular_payback(1000, 250), 4)
        self.assertIsNone(calcular_payback(1000, 0))

    def test_roi_simple_anual(self):
        self.assertEqual(calcular_roi_simple_anual(1000, 250), 0.25)
        self.assertIsNone(calcular_roi_simple_anual(0, 250))

    def test_lectura_demanda_ejemplo(self):
        demand = read_demand_file(ROOT / "data" / "demanda_ejemplo.csv")
        self.assertTrue({"Fecha_Hora", "Demanda_kW"}.issubset(demand.columns))
        self.assertEqual(len(demand), 35040)

    def test_dataframe_final_columnas_esperadas(self):
        df = pd.DataFrame(
            {
                "Demanda_kW": [10],
                "Generacion_Solar_kW": [4],
                "Energia_Autoconsumida_kWh": [1],
            }
        )
        expected = {"Demanda_kW", "Generacion_Solar_kW", "Energia_Autoconsumida_kWh"}
        self.assertTrue(expected.issubset(df.columns))


if __name__ == "__main__":
    unittest.main()
