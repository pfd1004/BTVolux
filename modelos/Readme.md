# Modelos utilizados en BTVolux

Esta carpeta contiene, o debe contener localmente, los modelos utilizados por BTVolux para la segmentación automática y análisis volumétrico de meningiomas en imágenes de resonancia magnética.

El repositorio de código **no incluye pesos de modelos pesados, datos clínicos ni entornos portables**, por motivos de tamaño, licencias y protección de datos. Cada usuario debe colocar localmente los modelos en las rutas esperadas.

---

## Estructura esperada de la carpeta `modelos/`

```text
modelos/
├── radionics/
│   └── MRI_Meningioma/
│       └── model.onnx
├── nnunet/
│   ├── nnUNet_raw_data_base/
│   ├── nnUNet_preprocessed/
│   └── nnUNet_results/
│       └── nnUNet/
│           └── 3d_fullres/
│               └── Task501_t1c_enhancement/
│                   └── nnUNetTrainerV2__nnUNetPlansv2.1/
├── mri_brain_tumor_segmentation/
│   ├── main.py
│   ├── src/
│   └── resources/
│       └── models/
└── synthstrip/
    ├── synthstrip.1.pt
    └── synthstrip.nocsf.1.pt
```

---

# 1. Radionics / Raidionics

## Nombre en BTVolux

```text
Radionics
```

## Ruta esperada

```text
modelos/radionics/MRI_Meningioma/model.onnx
```

## Fuente

Raidionics es un software abierto para segmentación preoperatoria y postoperatoria de tumores del sistema nervioso central y generación de informes estandarizados.

Repositorio oficial:

```text
https://github.com/raidionics/Raidionics
```

Artículo asociado:

```text
Bouget, D. et al. Raidionics: an open software for pre- and postoperative central nervous system tumor segmentation and standardized reporting.
Scientific Reports, 2023.
DOI: 10.1038/s41598-023-42048-7
```

---

# 2. nnU-Net Task501

## Nombre en BTVolux

```text
nnU-Net (Task501)
```

## Identificador del modelo

```text
Task501_t1c_enhancement
```

## Rutas esperadas

```text
modelos/nnunet/nnUNet_raw_data_base/
modelos/nnunet/nnUNet_preprocessed/
modelos/nnunet/nnUNet_results/
```

El modelo debe encontrarse dentro de:

```text
modelos/nnunet/nnUNet_results/nnUNet/3d_fullres/Task501_t1c_enhancement/nnUNetTrainerV2__nnUNetPlansv2.1/
```

## Fuente

Framework nnU-Net:

```text
https://github.com/MIC-DKFZ/nnUNet
```

Artículo principal:

```text
Isensee, F. et al. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation.
Nature Methods, 2021.
DOI: 10.1038/s41592-020-01008-z
```

Repositorio de modelos preentrenados utilizado como referencia:

```text
https://github.com/ecalabr/nnUNet_models
```

Modelo concreto:

```text
Task501_t1c_enhancement.zip
```

---

# 3. Modelos de `mri_brain_tumor_segmentation`

## Ruta esperada

```text
modelos/mri_brain_tumor_segmentation/
```

El archivo principal esperado es:

```text
modelos/mri_brain_tumor_segmentation/main.py
```

Los pesos de los modelos deben estar en:

```text
modelos/mri_brain_tumor_segmentation/resources/models/
```

## Fuente

Repositorio original:

```text
https://github.com/dbouget/mri_brain_tumor_segmentation
```

El repositorio contiene arquitecturas, código de inferencia y modelos entrenados para segmentación de meningiomas en volúmenes de RM T1 ponderada.

Artículo asociado:

```text
Bouget, D., Pedersen, A., Hosainey, S. A. M., Solheim, O., Reinertsen, I.
Meningioma Segmentation in T1-Weighted MRI Leveraging Global Context and Attention Mechanisms.
Frontiers in Radiology, 2021.
DOI: 10.3389/fradi.2021.711514
```

## Modelos usados desde este repositorio

BTVolux utiliza los siguientes modelos del repositorio:

```text
AGUNet
DAGUNet
PLS-Net
UNet-FV
UNet-Slabs
```

---

## 3.1. AGUNet

### Nombre en BTVolux

```text
AGUNet
```

### Ejecución esperada

```text
python main.py -i <imagen.nii.gz> -o <salida> -m AGUNet -g 0
```

### Entorno en BTVolux portable

```text
runtime/env_agunet/python.exe
```

---

## 3.2. DAGUNet

### Nombre en BTVolux

```text
DAGUNet
```

### Ejecución esperada

```text
python main.py -i <imagen.nii.gz> -o <salida> -m DAGUNet -g 0
```

### Entorno en BTVolux portable

```text
runtime/env_agunet/python.exe
```

---

## 3.3. PLS-Net

### Nombre en BTVolux

```text
PLS-Net
```

### Ejecución esperada

```text
python main.py -i <imagen.nii.gz> -o <salida> -m PLS-Net -g 0
```

### Entorno en BTVolux portable

```text
runtime/env_plsnet/python.exe
```

---

## 3.4. UNet-FV

### Nombre en BTVolux

```text
UNet-FV
```

### Ejecución esperada

```text
python main.py -i <imagen.nii.gz> -o <salida> -m UNet-FV -g 0
```

### Entorno en BTVolux portable

```text
runtime/env_agunet/python.exe
```

---

## 3.5. UNet-Slabs

### Nombre en BTVolux

```text
UNet-Slabs
```

### Ejecución esperada

```text
python main.py -i <imagen.nii.gz> -o <salida> -m UNet-Slabs -g 0
```

### Entorno en BTVolux portable

```text
runtime/env_agunet/python.exe
```

---

# 4. Media ponderada

## Nombre en BTVolux

```text
Media ponderada
```

## Descripción

La media ponderada no es un modelo entrenado independiente. Es una estrategia de consenso implementada en BTVolux.

Utiliza las segmentaciones disponibles de los modelos individuales y las combina con pesos basados en el rendimiento medio de cada modelo. La aplicación la usa de dos formas:

1. **Máscara binaria de consenso**, para cálculo de volumen y DICE.
2. **Mapa de calor**, para visualizar el grado de acuerdo ponderado entre modelos.

En el visor, el mapa de calor representa:

```text
valor bajo  -> bajo acuerdo entre modelos
valor alto  -> alto acuerdo entre modelos
```

---

# 5. SynthStrip

## Ruta esperada

```text
modelos/synthstrip/synthstrip.1.pt
modelos/synthstrip/synthstrip.nocsf.1.pt
```

## Fuente

Página oficial:

```text
https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/
```

Implementación NiPreps:

```text
https://github.com/nipreps/synthstrip
```

Artículo:

```text
Hoopes, A. et al. SynthStrip: skull-stripping for any brain image.
NeuroImage, 2022.
DOI: 10.1016/j.neuroimage.2022.119474
```

---


# 6. Entornos esperados en la versión portable

La versión portable de BTVolux espera los siguientes entornos:

```text
runtime/env_app/
runtime/env_nnunet/
runtime/env_agunet/
runtime/env_plsnet/
```

---

# 7. Resumen de modelos en BTVolux

| Modelo en BTVolux | Fuente | Tipo |
|---|---|---|
| Radionics | Raidionics | Segmentación ONNX |
| nnU-Net (Task501) | nnU-Net v1 / Task501_t1c_enhancement | Segmentación T1 postcontraste |
| AGUNet | mri_brain_tumor_segmentation | Attention U-Net |
| DAGUNet | mri_brain_tumor_segmentation | Dual Attention U-Net |
| PLS-Net | mri_brain_tumor_segmentation | Red ligera 3D |
| UNet-FV | mri_brain_tumor_segmentation | U-Net Full Volume |
| UNet-Slabs | mri_brain_tumor_segmentation | U-Net por bloques/slabs |
| Media ponderada | Implementación propia en BTVolux | Consenso ponderado |