# REI_Tarea_AsistDev

Entrega reproducible de la explicación SHAP para el clasificador CNN de
vehículos.

## Contenido

- `vehicle-image-classification-using-cnn.ipynb`: entrenamiento y evaluación
  del modelo original.
- `vehicle_classifier.keras`: pesos/modelo entrenado.
- `shap_analysis.py`: genera las explicaciones, CSV/JSON/NPZ y figuras.
- `paper.tex` / `paper.pdf`: paper reproducible en español.
- `figures/inputs/`: imágenes de prueba exportadas del notebook; tres se usan
  en el paper.
- `figures/shap/`: visualizaciones SHAP y resultados numéricos.
- `references.bib`, `requirements-shap.txt`, `prompts_log.md`: referencias,
  dependencias y bitácora.

## Ejecución

El dataset `Vehicles/` se usa para reentrenar el notebook, pero no se incluye
por su tamaño. Las tres imágenes de prueba y el modelo sí están versionados,
por lo que el análisis SHAP se puede repetir directamente:

```bash
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -r requirements-shap.txt
.venv/bin/python shap_analysis.py
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

En el documento se distingue entre importancia de regiones para esta pequeña
muestra y una importancia global del modelo. SHAP no convierte una imagen en
partes semánticas automáticamente: para afirmar “las ruedas pesan 30 %” haría
falta una segmentación y una agregación definida previamente.
