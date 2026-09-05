# Bitácora de prompts y decisiones

Este archivo registra el intercambio que llevó del modelo existente al
paquete reproducible. Se omiten detalles de interfaz que no afectan el
experimento.

## P0 — Requerimiento inicial

> Construir un paper LaTeX reproducible sobre SHAP a partir de un clasificador
> de imágenes, con 1–3 imágenes de prueba, una pregunta explicativa concreta,
> código/notebook SHAP, figuras, referencias, limitaciones y el URL del repo en
> el `.tex`.

## P1 — Pregunta operacional

> ¿Qué regiones de cada imagen aumentan o disminuyen la probabilidad de la
> clase predicha, y el modelo se apoya más en el vehículo o en el contexto?

Decisión: en imágenes SHAP explica píxeles/regiones enmascaradas; por eso no se
presentan porcentajes semánticos inventados para “orejas”, “patas” o “ruedas”.
La agregación 4×4 se reporta como proporción de `|SHAP|` de una región, no como
confianza del modelo.

## P2 — Método SHAP

Decisión: utilizar `shap.Explainer` con `shap.maskers.Image("blur(32,32)")`,
Partition explainer, la salida superior por imagen y 300 evaluaciones por
muestra. El patrón sigue la documentación de SHAP para clasificación de
imágenes y mantiene una comprobación de aditividad.

## P3 — Compatibilidad del modelo

El archivo `vehicle_classifier.keras` registra Keras 3.15.1, mientras el
entorno reproducible usado para generar las figuras contiene Keras 3.12.4.
Como `load_model` rechaza los campos nuevos `input_axes`/`output_axes` de
`GlorotUniform`, `shap_analysis.py` intenta primero la carga directa y, si
falla, reconstruye exactamente la arquitectura del notebook y carga
`model.weights.h5` desde el ZIP `.keras`. El archivo original no se modifica.

## P4 — Verificación

Se ejecutó:

```bash
.venv/bin/python shap_analysis.py
```

La salida reportó predicciones `Cars` (22.29 %) para el tren, `Planes`
(52.20 %) para el barco y `Bikes` (64.26 %) para la bicicleta. La diferencia
aditiva de SHAP fue `0.00e+00` en las tres muestras.
