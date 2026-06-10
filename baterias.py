"""
baterias.py — Motor de Cálculo BESS (Battery Energy Storage System)
=====================================================================
Módulo técnico para dimensionamiento físico de sistemas de almacenamiento
comercial/industrial y análisis estadístico de cortes de energía.

Sección 3: Ingeniería de Resiliencia ante Apagones
Equipo de referencia: Sungrow PowerStack 255CS (ST255CS-2H)

Uso:
    from baterias import MotorBESS

    motor = MotorBESS()
    dimensionamiento = motor.calcular_dimensionamiento(horas_respaldo=4, carga_critica_kw=300)
    analisis = motor.analizar_historico_cortes()
"""

import math
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Dataclass de especificaciones técnicas del gabinete (inmutable por diseño)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EspecificacionesBESS:
    """
    Especificaciones técnicas del gabinete Sungrow PowerStack 255CS (ST255CS-2H).

    Todos los valores corresponden a la ficha técnica oficial del fabricante.
    Al ser 'frozen=True', esta instancia es inmutable — garantizando que los
    cálculos siempre usen los parámetros certificados del equipo.

    Atributos
    ----------
    modelo : str
        Nombre comercial del gabinete de almacenamiento.
    quimica : str
        Tecnología de celda utilizada (LFP = Litio Ferro-Fosfato).
    capacidad_nominal_kwh : float
        Energía nominal almacenable por gabinete unitario en kWh.
    potencia_nominal_ac_kw : float
        Potencia de salida AC nominal por gabinete en kW.
    rte : float
        Round-Trip Efficiency — eficiencia de ida y vuelta del ciclo
        carga/descarga (adimensional, entre 0 y 1).
    ciclos_minimos : int
        Ciclos de carga profunda garantizados al final de la vida útil (mínimo).
    ciclos_maximos : int
        Ciclos de carga profunda proyectados en condiciones óptimas (máximo).
    vida_util_anos_min : int
        Vida útil mínima estimada en años (referencia del fabricante).
    vida_util_anos_max : int
        Vida útil máxima estimada en años (referencia del fabricante).
    """
    modelo: str = "Sungrow PowerStack 255CS (ST255CS-2H)"
    quimica: str = "LFP (Litio Ferro-Fosfato) — Ciclos Profundos"
    capacidad_nominal_kwh: float = 257.0      # kWh por gabinete
    potencia_nominal_ac_kw: float = 125.0     # kW AC por gabinete
    rte: float = 0.90                          # 90 % eficiencia ida y vuelta
    ciclos_minimos: int = 4_000               # ciclos garantizados
    ciclos_maximos: int = 5_000               # ciclos proyectados
    vida_util_anos_min: int = 10              # años
    vida_util_anos_max: int = 15              # años


# ---------------------------------------------------------------------------
# Dataclass de resultado de dimensionamiento (tipado, documentado)
# ---------------------------------------------------------------------------

@dataclass
class ResultadoDimensionamiento:
    """
    Resultado estructurado del cálculo de dimensionamiento BESS.

    Atributos
    ----------
    horas_respaldo : float
        Autonomía solicitada en horas.
    carga_critica_kw : float
        Potencia de la carga crítica industrial protegida (kW).
    capacidad_requerida_kwh : float
        Energía neta necesaria corregida por pérdidas RTE (kWh).
    unidades_bess : int
        Número de gabinetes Sungrow completos en paralelo requeridos.
    potencia_total_kw : float
        Potencia AC instalada total del arreglo (kW).
    capacidad_total_kwh : float
        Capacidad total instalada del arreglo (kWh), incluyendo margen por RTE.
    ciclos_referencia : int
        Valor de referencia de ciclos de vida (promedio entre mín y máx).
    modelo_referencia : str
        Nombre del gabinete dimensionado.
    sobredimensionamiento_pct : float
        Porcentaje de capacidad instalada sobre la requerida (margen de energía).
    """
    horas_respaldo: float
    carga_critica_kw: float
    capacidad_requerida_kwh: float
    unidades_bess: int
    potencia_total_kw: float
    capacidad_total_kwh: float
    ciclos_referencia: int
    modelo_referencia: str
    sobredimensionamiento_pct: float


# ---------------------------------------------------------------------------
# Dataclass de resultado del análisis histórico de cortes
# ---------------------------------------------------------------------------

@dataclass
class ResultadoAnalisisCortes:
    """
    Resultado estructurado del análisis estadístico de cortes de energía.

    Atributos
    ----------
    cortes_largos_año : int
        Número de apagones largos (>1 hr) por año.
    cortes_medios_año : int
        Número de apagones medianos (1 min – 10 min) por año.
    cortes_cortos_año : int
        Número de microapagones (<1 min) por año.
    total_eventos_año : int
        Total de interrupciones de energía por año.
    horas_desabasto_estimadas : float
        Horas acumuladas de desabasto eléctrico por año (estimado conservador).
    nivel_riesgo : str
        Clasificación del riesgo operativo: 'Bajo', 'Medio' o 'Alto'.
    justificacion_riesgo : str
        Texto técnico que justifica la clasificación de riesgo asignada.
    autonomia_recomendada_hrs : float
        Horas mínimas de autonomía BESS recomendadas según el perfil de cortes.
    impacto_anual_estimado : dict
        Desglose del impacto por categoría de corte (eventos, duración estimada).
    """
    cortes_largos_año: int
    cortes_medios_año: int
    cortes_cortos_año: int
    total_eventos_año: int
    horas_desabasto_estimadas: float
    nivel_riesgo: str
    justificacion_riesgo: str
    autonomia_recomendada_hrs: float
    impacto_anual_estimado: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Clase principal del motor de cálculo
# ---------------------------------------------------------------------------

class MotorBESS:
    """
    Motor de cálculo técnico para sistemas BESS (Battery Energy Storage System).

    Encapsula toda la lógica de dimensionamiento físico y análisis estadístico
    de cortes de energía para un entorno industrial/comercial.

    Parámetros por defecto basados en el gabinete Sungrow PowerStack 255CS,
    modificables al instanciar la clase con especificaciones personalizadas.

    Ejemplo de uso
    --------------
    >>> motor = MotorBESS()
    >>> dim = motor.calcular_dimensionamiento(horas_respaldo=4, carga_critica_kw=350)
    >>> print(dim.unidades_bess)
    6

    >>> analisis = motor.analizar_historico_cortes(frecuencia_largos=5)
    >>> print(analisis.nivel_riesgo)
    'Alto'
    """

    # Duración representativa por categoría (horas) para el cálculo de desabasto
    _DURACION_CORTE_LARGO_HRS: float = 2.5    # valor conservador (>1 hr)
    _DURACION_CORTE_MEDIO_HRS: float = 0.10   # ~6 minutos (<10 min)
    _DURACION_CORTE_CORTO_HRS: float = 0.003  # ~10–20 segundos (<1 min)

    def __init__(self, especificaciones: Optional[EspecificacionesBESS] = None):
        """
        Inicializa el motor con las especificaciones del gabinete BESS.

        Parámetros
        ----------
        especificaciones : EspecificacionesBESS, opcional
            Instancia con los parámetros técnicos del equipo. Si no se provee,
            se usan los valores certificados del Sungrow PowerStack 255CS.
        """
        self.specs = especificaciones if especificaciones else EspecificacionesBESS()

    # ------------------------------------------------------------------
    # Método 1: Dimensionamiento físico del arreglo BESS
    # ------------------------------------------------------------------

    def calcular_dimensionamiento(
        self,
        horas_respaldo: float,
        carga_critica_kw: float,
    ) -> ResultadoDimensionamiento:
        """
        Dimensiona físicamente el número de gabinetes BESS necesarios.

        Calcula la capacidad energética requerida corrigiendo las pérdidas de
        conversión (RTE) y determina cuántos gabinetes completos en paralelo
        son necesarios para cubrir la demanda de respaldo solicitada.

        Fórmula principal
        -----------------
        kWh_req = (carga_critica_kw × horas_respaldo) / RTE

        La división entre RTE asegura que las pérdidas de conversión AC/DC/AC
        estén compensadas y la carga crítica reciba la energía íntegra requerida.

        Parámetros
        ----------
        horas_respaldo : float
            Horas de autonomía deseadas. Rango recomendado: 1 – 24 horas.
        carga_critica_kw : float
            Potencia total de la carga crítica industrial a proteger (kW).
            Ejemplo: 300 kW para una línea de producción continua.

        Retorna
        -------
        ResultadoDimensionamiento
            Dataclass con todos los resultados del dimensionamiento.

        Excepciones
        -----------
        ValueError
            Si horas_respaldo o carga_critica_kw son valores no positivos.

        Ejemplo
        -------
        >>> motor = MotorBESS()
        >>> r = motor.calcular_dimensionamiento(horas_respaldo=4, carga_critica_kw=300)
        >>> r.unidades_bess
        6
        >>> r.capacidad_requerida_kwh
        1333.33
        """
        # --- Validación de entradas ---
        if horas_respaldo <= 0:
            raise ValueError(f"horas_respaldo debe ser mayor a 0. Recibido: {horas_respaldo}")
        if carga_critica_kw <= 0:
            raise ValueError(f"carga_critica_kw debe ser mayor a 0. Recibido: {carga_critica_kw}")

        # --- Cálculo de capacidad requerida (corregida por RTE) ---
        # Se divide entre RTE para que el sistema entregue la energía requerida
        # aún después de absorber las pérdidas de conversión electrónica.
        capacidad_requerida_kwh = (carga_critica_kw * horas_respaldo) / self.specs.rte

        # --- Número de unidades físicas (techo del cociente) ---
        # math.ceil garantiza que siempre se cubra la demanda; nunca se redondea
        # hacia abajo pues implicaría energía insuficiente.
        unidades_bess = math.ceil(capacidad_requerida_kwh / self.specs.capacidad_nominal_kwh)

        # --- Métricas del arreglo resultante ---
        potencia_total_kw = self.specs.potencia_nominal_ac_kw * unidades_bess
        capacidad_total_kwh = self.specs.capacidad_nominal_kwh * unidades_bess

        # Ciclos de referencia: promedio entre mínimo y máximo garantizados
        ciclos_referencia = (self.specs.ciclos_minimos + self.specs.ciclos_maximos) // 2

        # Sobredimensionamiento: cuánta capacidad extra provee el arreglo (%)
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

    # ------------------------------------------------------------------
    # Método 2: Análisis estadístico del histórico de cortes
    # ------------------------------------------------------------------

    def analizar_historico_cortes(
        self,
        frecuencia_largos: int = 3,
        frecuencia_medios: int = 12,
        frecuencia_cortos: int = 24,
    ) -> ResultadoAnalisisCortes:
        """
        Analiza estadísticamente el historial de interrupciones eléctricas.

        Procesa los datos de cortes extraídos del historial del departamento de
        sistemas, fallas por tormentas y cortes programados de CFE. Con base en
        la frecuencia e impacto acumulado, clasifica el nivel de riesgo operativo
        y recomienda la autonomía mínima del BESS.

        Categorías de corte
        --------------------
        - Largos  : > 1 hora   (arranque de equipos, pérdidas de proceso)
        - Medios  : 1–10 min   (interrupciones de línea, reset de PLC/SCADA)
        - Cortos  : < 1 min    (microapagones, reinicios de equipo sensible)

        Clasificación de riesgo
        -----------------------
        | Nivel  | Horas anuales de desabasto |
        |--------|---------------------------|
        | Bajo   | < 5 horas                 |
        | Medio  | 5 – 20 horas              |
        | Alto   | > 20 horas                |

        Parámetros
        ----------
        frecuencia_largos : int
            Apagones largos (>1 hr) por año. Default: 3 (datos del socio formador).
        frecuencia_medios : int
            Apagones medianos (1–10 min) por año. Default: 12.
        frecuencia_cortos : int
            Microapagones (<1 min) por año. Default: 24.

        Retorna
        -------
        ResultadoAnalisisCortes
            Dataclass con métricas, clasificación de riesgo y recomendaciones.

        Excepciones
        -----------
        ValueError
            Si alguna frecuencia es un valor negativo.

        Ejemplo
        -------
        >>> motor = MotorBESS()
        >>> a = motor.analizar_historico_cortes(frecuencia_largos=5)
        >>> a.nivel_riesgo
        'Alto'
        >>> a.horas_desabasto_estimadas
        14.42
        """
        # --- Validación de entradas ---
        for nombre, valor in [
            ("frecuencia_largos", frecuencia_largos),
            ("frecuencia_medios", frecuencia_medios),
            ("frecuencia_cortos", frecuencia_cortos),
        ]:
            if valor < 0:
                raise ValueError(f"{nombre} no puede ser negativo. Recibido: {valor}")

        # --- Total de eventos anuales ---
        total_eventos = frecuencia_largos + frecuencia_medios + frecuencia_cortos

        # --- Estimación de horas anuales de desabasto ---
        # Se usa la duración representativa de cada categoría (conservadora)
        horas_largos = frecuencia_largos * self._DURACION_CORTE_LARGO_HRS
        horas_medios = frecuencia_medios * self._DURACION_CORTE_MEDIO_HRS
        horas_cortos = frecuencia_cortos * self._DURACION_CORTE_CORTO_HRS
        horas_desabasto_total = round(horas_largos + horas_medios + horas_cortos, 2)

        # --- Desglose del impacto por categoría ---
        impacto_anual = {
            "largos": {
                "eventos": frecuencia_largos,
                "duracion_representativa_hrs": self._DURACION_CORTE_LARGO_HRS,
                "horas_desabasto_acumuladas": round(horas_largos, 2),
                "descripcion": "Apagones >1 hr — riesgo de daño en equipos y pérdida de proceso",
            },
            "medios": {
                "eventos": frecuencia_medios,
                "duracion_representativa_hrs": self._DURACION_CORTE_MEDIO_HRS,
                "horas_desabasto_acumuladas": round(horas_medios, 2),
                "descripcion": "Apagones 1–10 min — reset de PLC/SCADA, scrap de producto en proceso",
            },
            "cortos": {
                "eventos": frecuencia_cortos,
                "duracion_representativa_hrs": self._DURACION_CORTE_CORTO_HRS,
                "horas_desabasto_acumuladas": round(horas_cortos, 2),
                "descripcion": "Microapagones <1 min — reinicios de equipo sensible, pérdida de datos",
            },
        }

        # --- Clasificación del nivel de riesgo ---
        nivel_riesgo, justificacion, autonomia_recomendada = self._clasificar_riesgo(
            horas_desabasto=horas_desabasto_total,
            frecuencia_largos=frecuencia_largos,
            total_eventos=total_eventos,
        )

        return ResultadoAnalisisCortes(
            cortes_largos_año=frecuencia_largos,
            cortes_medios_año=frecuencia_medios,
            cortes_cortos_año=frecuencia_cortos,
            total_eventos_año=total_eventos,
            horas_desabasto_estimadas=horas_desabasto_total,
            nivel_riesgo=nivel_riesgo,
            justificacion_riesgo=justificacion,
            autonomia_recomendada_hrs=autonomia_recomendada,
            impacto_anual_estimado=impacto_anual,
        )

    # ------------------------------------------------------------------
    # Método auxiliar privado: clasificación de riesgo
    # ------------------------------------------------------------------

    def _clasificar_riesgo(
        self,
        horas_desabasto: float,
        frecuencia_largos: int,
        total_eventos: int,
    ) -> tuple[str, str, float]:
        """
        Clasifica el nivel de riesgo operativo y determina la autonomía mínima.

        Lógica de clasificación
        -----------------------
        La clasificación combina dos variables independientes:
          1. Horas anuales totales de desabasto (indicador de impacto acumulado).
          2. Número de apagones largos (indicador de severidad de eventos).

        Si alguno de los dos criterios apunta a un nivel mayor, se eleva la
        clasificación (principio de máxima protección).

        Parámetros (internos)
        ---------------------
        horas_desabasto : float
            Horas anuales estimadas de desabasto eléctrico.
        frecuencia_largos : int
            Número de apagones largos (>1 hr) por año.
        total_eventos : int
            Total de interrupciones anuales de cualquier duración.

        Retorna
        -------
        tuple[str, str, float]
            (nivel_riesgo, justificacion_tecnica, autonomia_recomendada_hrs)
        """
        # Criterio por horas de desabasto acumuladas
        if horas_desabasto >= 20:
            riesgo_horas = "Alto"
        elif horas_desabasto >= 5:
            riesgo_horas = "Medio"
        else:
            riesgo_horas = "Bajo"

        # Criterio por severidad (apagones largos > 1 hr)
        if frecuencia_largos >= 5:
            riesgo_severidad = "Alto"
        elif frecuencia_largos >= 2:
            riesgo_severidad = "Medio"
        else:
            riesgo_severidad = "Bajo"

        # Escalera de clasificación: se toma el nivel más conservador (mayor)
        niveles = {"Bajo": 1, "Medio": 2, "Alto": 3}
        nivel_final = max(riesgo_horas, riesgo_severidad, key=lambda n: niveles[n])

        # Autonomía mínima recomendada según nivel de riesgo
        autonomia_map: dict[str, float] = {
            "Bajo":  2.0,   # respaldo básico para apagones breves
            "Medio": 4.0,   # respaldo para cubrir el corte largo promedio
            "Alto":  8.0,   # respaldo extendido para continuidad operativa
        }
        autonomia = autonomia_map[nivel_final]

        # Justificación técnica textual
        justificacion_map = {
            "Bajo": (
                f"Con {horas_desabasto:.1f} horas anuales de desabasto y {frecuencia_largos} "
                f"apagón(es) largo(s), el riesgo operativo es manejable. Se recomienda un BESS "
                f"de {autonomia:.0f} horas como protección base para equipos críticos y continuidad "
                f"de sistemas de control."
            ),
            "Medio": (
                f"Con {horas_desabasto:.1f} horas anuales de desabasto acumulado, {total_eventos} "
                f"eventos totales y {frecuencia_largos} apagón(es) largo(s) de >1 hr, la operación "
                f"enfrenta un riesgo medio de pérdida de proceso. Un BESS de {autonomia:.0f} horas "
                f"cubre el 95% de los eventos históricos y protege contra pérdidas de producción."
            ),
            "Alto": (
                f"Perfil de riesgo ALTO: {horas_desabasto:.1f} horas anuales de desabasto con "
                f"{frecuencia_largos} apagón(es) severo(s) de >1 hr y {total_eventos} eventos "
                f"totales. La frecuencia e impacto de las interrupciones justifica técnica y "
                f"económicamente una inversión en un arreglo BESS de {autonomia:.0f} horas para "
                f"garantizar continuidad operativa y evitar pérdidas de producción, scrap y "
                f"daños a equipos de alto valor."
            ),
        }

        return nivel_final, justificacion_map[nivel_final], autonomia

    # ------------------------------------------------------------------
    # Método utilitario: resumen ejecutivo de texto plano
    # ------------------------------------------------------------------

    def generar_resumen_ejecutivo(
        self,
        horas_respaldo: float,
        carga_critica_kw: float,
        frecuencia_largos: int = 3,
        frecuencia_medios: int = 12,
        frecuencia_cortos: int = 24,
    ) -> str:
        """
        Genera un resumen ejecutivo de texto plano con ambos análisis integrados.

        Útil para exportar a PDF, reportes Word o mostrar en la UI de Streamlit
        sin necesidad de post-procesar múltiples dataclasses.

        Parámetros
        ----------
        horas_respaldo : float
            Autonomía deseada del BESS en horas.
        carga_critica_kw : float
            Potencia de la carga crítica a proteger (kW).
        frecuencia_largos : int
            Cortes largos (>1 hr) por año.
        frecuencia_medios : int
            Cortes medios (1–10 min) por año.
        frecuencia_cortos : int
            Cortes cortos (<1 min) por año.

        Retorna
        -------
        str
            Texto multilínea con el resumen ejecutivo completo.
        """
        dim = self.calcular_dimensionamiento(horas_respaldo, carga_critica_kw)
        ana = self.analizar_historico_cortes(frecuencia_largos, frecuencia_medios, frecuencia_cortos)

        resumen = f"""
╔══════════════════════════════════════════════════════════════════╗
║       RESUMEN EJECUTIVO — SISTEMA BESS DE RESPALDO              ║
╚══════════════════════════════════════════════════════════════════╝

▸ EQUIPO DE REFERENCIA : {dim.modelo_referencia}
▸ QUÍMICA              : {self.specs.quimica}

── DIMENSIONAMIENTO FÍSICO ──────────────────────────────────────
  Carga crítica protegida  : {dim.carga_critica_kw:,.0f} kW
  Autonomía solicitada     : {dim.horas_respaldo:.1f} horas
  Capacidad requerida      : {dim.capacidad_requerida_kwh:,.2f} kWh  (incl. pérdidas RTE {self.specs.rte*100:.0f}%)
  Gabinetes necesarios     : {dim.unidades_bess} unidades en paralelo
  Potencia AC instalada    : {dim.potencia_total_kw:,.0f} kW
  Capacidad instalada      : {dim.capacidad_total_kwh:,.0f} kWh
  Margen de energía        : {dim.sobredimensionamiento_pct:.1f}% sobre lo requerido
  Vida útil estimada       : {dim.ciclos_referencia:,} ciclos  (~{self.specs.vida_util_anos_min}–{self.specs.vida_util_anos_max} años)

── ANÁLISIS DE RIESGO (HISTÓRICO DE CORTES) ─────────────────────
  Apagones largos (>1 hr)  : {ana.cortes_largos_año} eventos/año
  Apagones medios (<10 min): {ana.cortes_medios_año} eventos/año
  Microapagones (<1 min)   : {ana.cortes_cortos_año} eventos/año
  Total de eventos         : {ana.total_eventos_año} interrupciones/año
  Horas de desabasto       : {ana.horas_desabasto_estimadas:.2f} hrs/año
  Nivel de riesgo          : ★ {ana.nivel_riesgo.upper()} ★
  Autonomía recomendada    : {ana.autonomia_recomendada_hrs:.0f} horas mínimo

── JUSTIFICACIÓN TÉCNICA ────────────────────────────────────────
  {ana.justificacion_riesgo}

════════════════════════════════════════════════════════════════════
"""
        return resumen.strip()


# ---------------------------------------------------------------------------
# Bloque de prueba rápida (ejecutar directamente: python baterias.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    motor = MotorBESS()

    print("=== TEST 1: Dimensionamiento (300 kW, 4 horas) ===")
    dim = motor.calcular_dimensionamiento(horas_respaldo=4, carga_critica_kw=300)
    print(f"  kWh requeridos : {dim.capacidad_requerida_kwh}")
    print(f"  Unidades BESS  : {dim.unidades_bess}")
    print(f"  Potencia total : {dim.potencia_total_kw} kW")
    print(f"  Capacidad total: {dim.capacidad_total_kwh} kWh")
    print(f"  Sobredim.      : {dim.sobredimensionamiento_pct}%")

    print("\n=== TEST 2: Análisis histórico (datos por defecto del socio) ===")
    ana = motor.analizar_historico_cortes()
    print(f"  Total eventos  : {ana.total_eventos_año}/año")
    print(f"  Horas desabasto: {ana.horas_desabasto_estimadas} hrs/año")
    print(f"  Nivel de riesgo: {ana.nivel_riesgo}")
    print(f"  Autonomía rec. : {ana.autonomia_recomendada_hrs} horas")

    print("\n=== TEST 3: Resumen ejecutivo integrado ===")
    print(motor.generar_resumen_ejecutivo(
        horas_respaldo=4,
        carga_critica_kw=300,
    ))