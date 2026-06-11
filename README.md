# App final Streger - FV, BESS y finanzas

Proyecto Streamlit modular para evaluar viabilidad fotovoltaica, respaldo BESS y resultados financieros preliminares para Streger.

## Que hace

- Descarga clima historico de Open-Meteo.
- Remuestrea clima horario a 15 minutos.
- Usa GHI, DNI y DHI historicos para calcular POA con `pvlib`.
- Modela temperatura de celda con temperatura ambiente y viento.
- Compara modulo monofacial y bifacial.
- Cruza generacion FV contra demanda electrica.
- Dimensiona BESS de respaldo.
- Estima perdidas por apagones y beneficio evitable.
- Calcula ahorro solar por autoconsumo, payback y ROI simple anual.
- Exporta resultados en CSV y JSON.

## Estructura

```text
app final/
  app.py
  motor_solar.py
  bess.py
  finanzas.py
  demanda.py
  exportaciones.py
  requirements.txt
  ejecutar_app.bat
  README.md
  .gitignore
  data/
    demanda_ejemplo.csv
  tests/
    test_basico.py
```

## Ejecutar con doble clic

Haz doble clic en:

```text
ejecutar_app.bat
```

El archivo crea `.venv`, instala dependencias y abre Streamlit con:

```bat
streamlit run app.py
```

## Ejecutar manualmente

```powershell
cd "C:\Users\AxelL\OneDrive\Documentos\app final"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
streamlit run app.py
```

Si `py` no funciona, usa `python`.

## Supuestos solares

- Resolucion final: 15 minutos.
- Fuente climatica: Open-Meteo Archive.
- Variables usadas: temperatura, humedad, precipitacion, nubosidad, viento, GHI, DNI y DHI.
- POA calculado con `pvlib.irradiance.get_total_irradiance` y modelo Hay-Davies.
- Temperatura de celda con modelo Faiman.
- Bifacial: ganancia trasera multiplicada por factor bifacial del modulo.
- La demanda puede alinearse por posicion para usar demanda futura con un ano climatico historico analogo.

## Supuestos financieros

- Recibo base: CFE GDMTO Streger mayo 2026.
- Ahorro solar: autoconsumo anual calculado desde la simulacion por `Energia_Autoconsumida_kWh`; si no existe, usa `min(Demanda_kW, Generacion_Solar_kW) * 0.25`.
- Precio de energia: precio medio del recibo.
- Perdidas por apagones: escenarios conservador, medio y alto.
- Beneficio BESS/UPS: porcentaje evitable por severidad de evento.
- Payback: inversion / beneficio anual.
- ROI simple anual: beneficio anual / inversion.

## Limitaciones

Este analisis es preliminar. No sustituye auditoria electrica, cotizacion formal, estudio tarifario CFE, estudio de interconexion, diseno ejecutivo ni ingenieria de detalle.

Para un estudio definitivo se debe incorporar tarifa horaria completa, demanda facturable, cargos CFE vigentes, degradacion FV, O&M, reemplazos, financiamiento, impuestos y cotizaciones reales.
