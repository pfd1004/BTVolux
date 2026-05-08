# BTVolux

BTVolux es una aplicación local para la visualización, segmentación automática, validación y cálculo volumétrico de meningiomas en imágenes de resonancia magnética.

## Funcionalidades

- Importación de estudios DICOM.
- Conversión DICOM a NIfTI.
- Visualización multiplanar axial, coronal y sagital.
- Ejecución de varios modelos de segmentación.
- Cálculo de volumen tumoral en mL.
- Validación mediante coeficiente DICE frente a segmentación manual.
- Exportación de resultados a CSV.
- Análisis anatómico exploratorio mediante atlas.
- Mapa de consenso ponderado entre modelos.

## Modelos soportados

- Radionics/Raidionics
- nnU-Net Task501
- AGUNet
- DAGUNet
- PLS-Net
- UNet-FV
- UNet-Slabs
- Media ponderada / consenso

## Aviso sobre datos y modelos

Este repositorio no incluye datos clínicos, imágenes de pacientes ni pesos de modelos.  
Los directorios `Pacientes_nifti`, `runtime` y modelos pesados deben configurarse localmente.

## Ejecución en desarrollo

```bash
python -m shiny run app.py --host 127.0.0.1 --port 8000