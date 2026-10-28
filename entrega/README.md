# Miniproyecto – Control térmico MIMO 2×2 con desacoplo estático y PI

Sistemas de Control 2026-2 – Universidad de Antioquia.

## Cómo ejecutar
Requisitos: Python 3 con `numpy`, `pandas`, `scipy`, `matplotlib` y `tclab` (solo para el laboratorio).

1. **Análisis completo:** abrir `Miniproyecto_Control_Termico_MIMO.ipynb` desde esta carpeta y ejecutar *Run All*.
   Reconstruye todas las figuras (`figuras/`) y tablas (`tablas/`) a partir de los datos crudos de `datos/`,
   sin conectarse al equipo. El cuaderno sigue el orden de las fases del enunciado.
2. **Ensayos en el equipo (opcional):** el mismo cuaderno contiene el código de adquisición en la fase que le
   corresponde: diagnóstico (sección 1.4), identificación (2.0) y ensayos A/B en lazo cerrado (6.0). Para repetir
   un ensayo, poner `EJECUTAR_EQUIPO = True` (sección 1.4), elegir el ensayo en la celda correspondiente, con el
   equipo frío y supervisado, y volver a ejecutar el análisis: toma el ensayo COMPLETO más reciente de cada caso.

## Estructura
| Ruta | Contenido |
|---|---|
| `Miniproyecto_Control_Termico_MIMO.ipynb` | cuaderno único: código de los ensayos, análisis por fases del enunciado, preguntas de discusión y conclusiones |
| `codigo/mimo_control.py` | planta FOPDT discreta, PI con anti-windup, simulación, indicadores y protecciones |
| `codigo/adquisicion_identificacion.py` | programa de adquisición usado para los ensayos E1 y E2 |
| `datos/identificacion/` | ensayos de identificación E1 y E2 del 23-oct (CSV crudos, metadatos y figuras originales) |
| `datos/lazo_cerrado/` | ensayos A/B en lazo cerrado S1-A, S1-B, S2-A, S2-B del 28-oct |
| `datos/punto_operacion/` | aproximación a 55 °C del 23-oct de la que se obtuvo Q_OP |
| `datos/diagnostico/` | prueba de sensores del 27-oct |
| `figuras/`, `tablas/` | resultados generados por el cuaderno de análisis |
| `anexos/` | gráficas de los intentos fallidos (22, 23 y 27-oct) nombradas por fecha y hora, con índice `A00_indice_anexos.csv` |
| `informe/` | informe IEEE en LaTeX (`main.tex`, `figuras/`) e `informe_overleaf.zip` para subir a Overleaf |

Los archivos de `datos/` son copias sin modificar de las mediciones originales. Solo se incluyen los ensayos
usados; los intentos fallidos se listan con su fecha en la sección 1.2 del cuaderno (gráficas en `anexos/`). El diccionario de variables
está en la sección 10 del cuaderno de análisis.

## Parámetros principales
- Muestreo `Ts = 1 s`; punto de operación `T_OP = 55 °C` en ambas zonas; escalones de +5 °C.
- Saturación de potencia 0–30 %; corte de seguridad `T_MAX = 110 °C`.
- PI por IMC (λ = 1.5·τ), los mismos en los casos A (sin desacoplo) y B (con desacoplo).
