# Motor solar-industrial con clima historico

`analisis_solar_clima.py` es el motor principal del proyecto fotovoltaico para
la planta industrial en Veracruz/Streger. Cruza clima historico, recurso solar,
generacion FV y demanda industrial en resolucion de 15 minutos.

Todavia no incluye tarifa GDMTH ni analisis economico.

## Flujo del motor

1. Descarga clima historico horario desde Open-Meteo.
2. Remuestrea el clima a `15min`.
3. Para GHI, DNI y DHI interpola indices de claridad horarios y los aplica a
   curvas de cielo despejado de 15 minutos.
4. Calcula irradiancia POA con `pvlib`.
5. Estima temperatura de celda con Faiman usando temperatura ambiente y viento
   historicos.
6. Calcula generacion DC del modulo y del sistema completo.
7. Aplica perdidas e inversor para obtener generacion AC.
8. Cruza generacion AC contra demanda industrial y calcula red, autoconsumo y
   excedentes.

## Preparar entorno

En PowerShell, desde esta carpeta:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Interfaz Streamlit

La app `app_mono_bi.py` usa el mismo motor importable y trabaja con la demanda
subida desde el navegador, sin archivos temporales obligatorios.

```powershell
streamlit run .\app_mono_bi.py
```

Desde la barra lateral se puede:

- elegir una ubicacion predefinida de Mexico o capturar coordenadas;
- subir demanda CSV o XLSX;
- elegir modulo monofacial o bifacial;
- ajustar ganancia trasera, potencia DC, inversor, perdidas, tilt, azimut y albedo;
- usar el ano climatico historico analogo;
- descargar serie de 15 minutos y resumenes mensual/anual.

## Demanda industrial y clima analogo

La curva `demanda_ejemplo.csv` contiene 35,040 registros de 15 minutos:

```text
2025-12-21 00:00:00 a 2026-12-20 23:45:00
```

Ese periodo incluye fechas futuras para la API historica. Por eso el caso
recomendado usa clima analogo:

```text
Clima:   2024-12-21 a 2025-12-20
Demanda: 2025-12-21 a 2026-12-20
```

Con `--align-demand-by-position`, el motor valida que ambos calendarios tengan
el mismo numero de intervalos y asigna cada dato de demanda al intervalo
climatico equivalente. El CSV conserva `Fecha_Hora_Demanda_Original` para
trazabilidad.

No se genera demanda aleatoria silenciosamente. Para ejecutar sin archivo se
debe activar de forma explicita `--generate-demand`.

## Ejecucion bifacial recomendada

```powershell
python .\analisis_solar_clima.py `
  --demand-file "C:\Users\AxelL\Downloads\demanda_ejemplo.csv" `
  --weather-start-date 2024-12-21 `
  --weather-end-date 2025-12-20 `
  --module bifacial `
  --system-dc-kwp 399.3 `
  --bifacial-gain 0.15 `
  --align-demand-by-position `
  --output-dir .\outputs_streger_bifacial
```

## Ejecucion monofacial

```powershell
python .\analisis_solar_clima.py `
  --demand-file "C:\Users\AxelL\Downloads\demanda_ejemplo.csv" `
  --weather-start-date 2024-12-21 `
  --weather-end-date 2025-12-20 `
  --module monofacial `
  --system-dc-kwp 399.3 `
  --align-demand-by-position `
  --output-dir .\outputs_streger_monofacial
```

## Comparar ambos modulos

```powershell
python .\analisis_solar_clima.py `
  --demand-file "C:\Users\AxelL\Downloads\demanda_ejemplo.csv" `
  --weather-start-date 2024-12-21 `
  --weather-end-date 2025-12-20 `
  --module all `
  --system-dc-kwp 399.3 `
  --bifacial-gain 0.15 `
  --align-demand-by-position `
  --output-dir .\outputs_streger_comparacion
```

Con `--module all`, cada escenario se exporta en una subcarpeta independiente:

```text
outputs_streger_comparacion/
  comparacion_modulos.csv
  monofacial/
  bifacial/
```

## Modulos incluidos

### Monofacial

Jinko Solar Tiger Neo 72HC `JKM605N-72HL4`.

- Potencia STC: `605 W`
- Eficiencia: `23.42%`
- Area: `2.583 m2`
- Coeficiente termico: `-0.29%/C`

### Bifacial

Jinko Solar Tiger Neo N-type `JKM625N-78HL4-BDV`.

- Potencia frontal STC: `625 W`
- Eficiencia frontal: `22.36%`
- Area: `2.795 m2`
- Coeficiente termico: `-0.29%/C`
- Factor bifacial: `80%`

La bifacialidad se aplica sobre irradiancia efectiva:

```text
ganancia_efectiva = ganancia_trasera * factor_bifacial
POA_efectiva = POA_frontal * (1 + ganancia_efectiva)
```

Por ejemplo, `--bifacial-gain 0.15` produce una ganancia efectiva de `12%`.
No se modifica artificialmente la potencia nominal STC.

## Potencia instalada

La comparacion principal usa la misma capacidad DC para ambos modulos:

```text
--system-dc-kwp 399.3
```

El motor reporta el numero equivalente de modulos. Tambien se puede fijar una
cantidad fisica y dejar que el motor recalcule la capacidad:

```powershell
python .\analisis_solar_clima.py --generate-demand --module monofacial --num-modules 660
```

## Archivos de salida

Cada escenario genera:

- `clima_solar_demanda_15min.csv`: serie completa quinceminutal.
- `resumen_anual.csv`: metricas tecnicas anuales.
- `resumen_mensual.csv`: demanda, generacion, red, autoconsumo, excedentes y clima.
- `metadata.json`: parametros, fuente y supuestos.
- `figuras/energia_diaria_poa.png`
- `figuras/energia_mensual.png`
- `figuras/energia_mensual_modulos.png`
- `figuras/heatmap_poa_historica.png`
- `figuras/clima_diario.png`
- `figuras/demanda_vs_solar.png`
- `figuras/demanda_original_vs_post_solar.png`
- `figuras/energia_mensual_demanda_solar.png`
- `figuras/perfil_dia_tipico.png`

## Columnas principales

El CSV quinceminutal incluye:

- `Fecha_Hora` y, al alinear por posicion, `Fecha_Hora_Demanda_Original`.
- `Demanda_kW`, `Factor_Potencia`.
- `GHI_Wm2`, `DNI_Wm2`, `DHI_Wm2`, `Gtot_POA_Wm2`.
- `Temperatura_Ambiente_C`, `Velocidad_Viento_ms`, `Temperatura_Celda_C`.
- `Generacion_DC_kW`, `Generacion_AC_kW`.
- `Energia_Solar_DC_kWh`, `Energia_Solar_AC_kWh`.
- `Demanda_Post_Inyeccion_Solar_kW`.
- `Energia_Demanda_kWh`, `Energia_Red_kWh`.
- `Energia_Autoconsumida_kWh`, `Energia_Excedente_kWh`.
- `Modulo`, `Tipo_Modulo`, `Potencia_DC_Sistema_kWp`.
- `Num_Modulos_Equivalente`.
