from __future__ import annotations
"""
app.py

Aplicación principal del TFG para segmentación y validación de tumores cerebrales
a partir de estudios de resonancia magnética estructurados por paciente y por estudio.

Responsabilidades principales
-----------------------------
1. Segmentación
   - Permite explorar estudios ya preparados en `Pacientes_nifti`.
   - Carga la imagen NIfTI base del estudio y la segmentación correspondiente
     al modelo seleccionado.
   - Muestra tres vistas ortogonales del volumen:
       * axial
       * coronal
       * sagital
   - Permite ejecutar modelos de segmentación desde la propia interfaz.
   - Permite calcular volumen tumoral y exportar resultados a CSV.

2. Validación
   - Trabaja únicamente con estudios que ya tienen al menos una segmentación
     disponible.
   - Permite cargar una segmentación manual de referencia (ground truth, GT).
   - Alinea la GT al espacio de la imagen del estudio.
   - Calcula el coeficiente DICE entre la GT y las predicciones de los modelos.
   - Permite visualizar la superposición entre predicción y segmentación manual.
   - Permite exportar resultados DICE a CSV.

3. Gestión de estudios
   - Permite importar estudios DICOM desde carpetas seleccionadas por el usuario.
   - Detecta la fecha del estudio y valida si el caso parece T1c/post-contraste.
   - Convierte DICOM a NIfTI para dejar el estudio listo para visualización
     y procesamiento.
   - Permite eliminar estudios o pacientes incompletos desde la app.

4. Interfaz y estado
   - Conserva la última selección de paciente, estudio y modelo.
   - Mantiene en memoria los volúmenes cargados para evitar recargas innecesarias.
   - Sincroniza la pestaña de validación con la selección activa en segmentación
     cuando el estudio ya está procesado.
   - Muestra mensajes de estado, avisos y modales de error de forma centralizada.

Dependencias dentro del proyecto
--------------------------------
- `config.py`: rutas, carpetas base y configuración general.
- `preprocessing.py`: importación DICOM, validación T1c y conversión a NIfTI.
- `pipeline.py`: ejecución de los modelos de segmentación.
- `calcular_volumen.py`: cálculo de volúmenes tumorales.
- Este archivo reutiliza además helpers de NIfTI y alineación para validación.

Estructura del archivo
----------------------
1. Configuración y constantes de la app
2. Helpers de filesystem e importación
3. Carga NIfTI y localización de segmentaciones
4. Visualización de cortes y crosshair
5. Persistencia de la última sesión
6. Validación y cálculo de DICE
7. Exportación a CSV
8. Helpers de interfaz
9. Definición de UI
10. Lógica reactiva del servidor
"""

import time
import json
from datetime import datetime
from pathlib import Path
import shutil
from typing import List, Optional, Tuple, Dict
import os
import asyncio
import queue
import threading
import config

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from nibabel.processing import resample_from_to
import base64
import mimetypes
import csv
from shiny import App, ui, render, reactive

from config import (
    PACIENTES_DICOM_DIR,
    PACIENTES_NIFTI_DIR,
    LAST_SESSION,
    APP_LOGS_DIR,
    APP_DATA_DIR,
    APP_RESULTS_DIR,
    T1C_PROTOCOLS_TXT,
)

from preprocessing import get_dicom_study_date, prepare_study_from_dicom, convertir_dicom_a_nifti   
from pipeline import pipeline_study, pipeline_all_existing_studies

from calcular_volumen import (
    compute_study_volumes,
    format_volume_text,
    OUT_CSV,
    iter_studies,
    build_weighted_ensemble_seg_for_study,
    MODEL_DICE_WEIGHTS,
)

#Try compatibilidad
try:
    from config import list_studies as _list_studies  # type: ignore
except Exception:
    _list_studies = None

try:
    from config import get_patient_paths as _get_patient_paths  # type: ignore
except Exception:
    _get_patient_paths = None

# ============================================================
# 1) CONFIGURACIÓN / CONSTANTES 
# ============================================================

MODEL_OPTIONS: Dict[str, str] = {
    "Radionics": "radionics",
    "nnU-Net (Task501)": "nnunet_task501",
    "AGUNet": "agunet",
    "DAGUNet": "dagunet",
    "PLS-Net": "pls-net",
    "UNet-FV": "unet-fv",
    "UNet-Slabs": "unet-slabs",
    "Media ponderada": "media_ponderada",
}

MODEL_COLORS: Dict[str, str] = {
    "radionics": "#ef4444",      # rojo
    "nnunet_task501": "#e5ff00", # amarillo
    "agunet": "#ff00e1",         # verde
    "dagunet": "#a855f7",        # morado
    "pls-net": "#f59e0b",        # ámbar
    "unet-fv": "#06b6d4",        # cian
    "unet-slabs": "#e297ee",     # rosa
    "media_ponderada": "#5f7f00",# rosa
}

VAL_MODEL_KEYS = list(MODEL_COLORS.keys())
VAL_MODEL_LABELS = [label for label, key in MODEL_OPTIONS.items() if key in MODEL_COLORS]

VALIDATION_GT_COLOR = "#22cb60"
VALIDATION_OVERLAP_COLOR = "#3b82f6"

APP_DIR = Path(__file__).resolve().parent

IMAGES_DIR = APP_DATA_DIR / "images"

DICE_ALL_CSV = APP_RESULTS_DIR / "dice_todos_los_pacientes.csv"

ATLAS_BASE_DIR = APP_DATA_DIR / "atlas"
ATLAS_CACHE_DIR = ATLAS_BASE_DIR / "cache"
ATLAS_WORK_DIR = ATLAS_BASE_DIR / "work"
ATLAS_RESULTS_DIR = APP_RESULTS_DIR / "atlas"

HIDDEN_PATIENT_DIRS = {"nnUNet_cropped_data", "nnUNet_raw_data",}

# ============================================================
# 2) FILESYSTEM + CONVERSIÓN DICOM->NIfTI
# =========================================================

def scan_dicom_patients() -> List[str]:
    """Lista IDs (subcarpetas) dentro de PACIENTES_DICOM_DIR."""
    if not PACIENTES_DICOM_DIR.exists():
        return []
    return sorted([p.name for p in PACIENTES_DICOM_DIR.iterdir() if p.is_dir()])


def scan_nifti_patients() -> List[str]:
    """Lista IDs dentro de PACIENTES_NIFTI_DIR sin romper si la ruta no existe."""
    try:
        if not PACIENTES_NIFTI_DIR.exists() or not PACIENTES_NIFTI_DIR.is_dir():
            return []

        out = []
        for p in PACIENTES_NIFTI_DIR.iterdir():
            if not p.is_dir() or p.name in HIDDEN_PATIENT_DIRS:
                continue

            has_any_study = any(
                child.is_dir() and len(child.name) >= 8 and child.name[:8].isdigit()
                for child in p.iterdir()
            )
            if has_any_study:
                out.append(p.name)

        return sorted(out)

    except FileNotFoundError:
        return []
    except OSError:
        return []

def list_studies(pid: str) -> List[str]:
    """
    Devuelve fechas disponibles para un paciente.
    Si la carpeta raíz no existe, devuelve lista vacía.
    """
    if not pid:
        return []

    if _list_studies is not None:
        try:
            return list(_list_studies(pid))
        except FileNotFoundError:
            return []
        except OSError:
            return []
        except Exception:
            pass

    try:
        root = PACIENTES_NIFTI_DIR / pid
        if not root.exists() or not root.is_dir():
            return []

        dates = []
        for p in root.iterdir():
            if not p.is_dir():
                continue
            name = p.name.strip()
            if len(name) >= 8 and name[:8].isdigit():
                dates.append(name)

        return sorted(set(dates))

    except FileNotFoundError:
        return []
    except OSError:
        return []
    
def get_paths(pid: str, study_date: Optional[str] = None) -> Dict[str, Path]:
    """
    Wrapper de compatibilidad: si config.py define get_patient_paths(...) lo usamos.
    Si no, construimos rutas estándar en PACIENTES_NIFTI_DIR.

    Retorna dict con claves:
      - patient_dir
      - study_dir (si study_date)
      - nifti_dir (alias de study_dir)
    """
    if _get_patient_paths is None:
        raise RuntimeError("No encuentro get_patient_paths en config.py")

    if study_date is None:
        return dict(_get_patient_paths(pid))  # type: ignore

    try:
        return dict(_get_patient_paths(pid, study_date))  # type: ignore
    except TypeError:
        return dict(_get_patient_paths(pid))  # type: ignore
    
def get_existing_paths(pid: str, study_date: str) -> Dict[str, Path]:
    """
    Devuelve rutas del estudio SIN crear carpetas.
    Úsalo para lectura/consulta, nunca para escritura.
    """
    study_root = PACIENTES_NIFTI_DIR / pid / str(study_date).strip()
    nifti_dir = study_root / "nifti"
    ants_dir = study_root / "ants"
    radionics_dir = study_root / "radionics"
    nnunet_dir = study_root / "nnunet"
    manual_dir = study_root / "manual"

    return {
        "root": study_root,
        "pid": Path(pid),
        "study_date": Path(str(study_date).strip()),

        "dicom_dir": study_root / "dicom",
        "nifti_dir": nifti_dir,
        "ants_dir": ants_dir,
        "radionics_dir": radionics_dir,
        "nnunet_dir": nnunet_dir,

        "nifti_img": nifti_dir / f"{pid}_0000.nii.gz",
        "ants_brain": ants_dir / f"{pid}_brain.nii.gz",
        "ants_mask": ants_dir / f"{pid}_brainmask.nii.gz",
        "synth_brain": ants_dir / f"{pid}_synthstrip_brain.nii.gz",
        "synth_mask": ants_dir / f"{pid}_synthstrip_mask.nii.gz",

        "rad_seg": radionics_dir / f"{pid}_radionics_seg.nii.gz",
        "rad_prob": radionics_dir / f"{pid}_radionics_prob.nii.gz",

        "manual_dir": manual_dir,
        "manual_seg": manual_dir / f"{pid}_manual_seg.nii.gz",
        "manual_aligned": manual_dir / f"{pid}_manual_aligned.nii.gz",
    }

def study_integrity(pid: str, study_date: str) -> Dict[str, object]:
    """
    Comprueba si el estudio está completo SIN crear carpetas nuevas.
    """
    study_root = PACIENTES_NIFTI_DIR / pid / str(study_date).strip()
    dicom_dir = study_root / "dicom"
    nifti_img = study_root / "nifti" / f"{pid}_0000.nii.gz"

    has_study_root = study_root.exists() and study_root.is_dir()
    has_dicom = dicom_dir.exists() and dicom_dir.is_dir() and any(dicom_dir.rglob("*"))
    has_nifti = nifti_img.exists()

    if not has_study_root:
        status = "error_missing_study_folder"
        message = (
            "Falta la carpeta del estudio. "
            "El caso está incompleto y no se puede trabajar con él. "
            "Vuelve a cargarlo."
        )
    elif not has_nifti:
        status = "error_missing_nifti"
        message = (
            "Falta el NIfTI principal del estudio. "
            "El caso está incompleto y no se puede trabajar con él. "
            "Vuelve a cargarlo."
        )
    elif not has_dicom:
        status = "warning_missing_dicom"
        message = (
            "Falta la carpeta DICOM de este estudio. "
            "Se puede seguir trabajando porque el NIfTI existe, "
            "pero no se podrá reconstruir desde DICOM."
        )
    else:
        status = "ok"
        message = ""

    return {
        "status": status,
        "message": message,
        "has_study_root": has_study_root,
        "has_dicom": has_dicom,
        "has_nifti": has_nifti,
    }

def sync_from_selected_folder(src_root: Path, default_pid: str = "") -> List[Tuple[str, str, str]]:
    """
    Importa DICOM desde una carpeta elegida por el usuario.

    Casos soportados:
      A) src_root contiene subcarpetas -> cada subcarpeta se interpreta como paciente_id.
      B) src_root contiene DICOM directamente -> se interpreta como UN estudio, paciente_id = default_pid o nombre carpeta.

    Devuelve lista de acciones: (pid, study_date, acción).
    """
    actions: List[Tuple[str, str, str]] = []
    src_root = Path(src_root)

    if not src_root.exists():
        return [("—", "—", f"error: no existe {src_root}")]

    # Detectar si hay subcarpetas (modo multi-paciente)
    subdirs = [p for p in src_root.iterdir() if p.is_dir()]

    # Heurística: si hay subcarpetas, asumimos que cada una es un paciente
    if subdirs:
        for pdir in sorted(subdirs):
            pid = pdir.name
            try:
                d = get_dicom_study_date(pdir)
            except Exception as e:
                actions.append((pid, "??????", f"error: {e}"))
                continue
            try:
                study_date, _P = prepare_study_from_dicom(pid, dicom_source_dir=str(pdir), force_copy=False)
                # convierte a nifti siempre (para que el visor ya muestre imagen)
                convertir_dicom_a_nifti(pid, study_date, force=False)
                actions.append((pid, study_date, "created"))
            except Exception as e:
                actions.append((pid, d, f"error: {e}"))

        return actions

    # Si no hay subcarpetas: asumimos que src_root es un único estudio DICOM
    pid = (default_pid or src_root.name).strip() or src_root.name
    try:
        d = get_dicom_study_date(src_root)
    except Exception as e:
        return [(pid, "??????", f"error: {e}")]

    try:
        study_date, _P = prepare_study_from_dicom(pid, dicom_source_dir=str(src_root), force_copy=False)
        convertir_dicom_a_nifti(pid, study_date, force=False)
        return [(pid, study_date, "created")]
    except Exception as e:
        return [(pid, d, f"error: {e}")]
    

def find_manual_seg_for_study(pid: str, study_date: str) -> Optional[Path]:
    P = get_existing_paths(pid, study_date)
    manual_dir = P["manual_dir"]

    cands = [
        P["manual_seg"],
        *sorted(manual_dir.glob("*.nii.gz")),
        *sorted(manual_dir.glob("*.nii")),
    ]
    return _pick_first_existing(cands)


def save_manual_seg_for_study(pid: str, study_date: str, src_path: Path) -> Path:
    P = get_paths(pid, study_date)
    dst = P["manual_seg"]

    dst.parent.mkdir(parents=True, exist_ok=True)

    img = nib.load(str(src_path))
    nib.save(img, str(dst))

    # invalida alineada anterior si existía
    aligned = P["manual_aligned"]
    if aligned.exists():
        aligned.unlink()

    return dst

def csv_value_present(row: Optional[Dict[str, object]], field: str) -> bool:
    if not isinstance(row, dict):
        return False

    value = row.get(field)

    if value is None:
        return False

    if isinstance(value, str) and value.strip() == "":
        return False

    return True

def load_manual_gt_for_validation(pid: str, study_date: str, ref_img_path: Path) -> Optional[np.ndarray]:
    P = get_paths(pid, study_date)
    gt_path = find_manual_seg_for_study(pid, study_date)
    if gt_path is None or not gt_path.exists():
        return None

    aligned_path = P["manual_aligned"]

    if aligned_path.exists() and aligned_path.stat().st_mtime >= gt_path.stat().st_mtime:
        return load_nifti_data(aligned_path)

    arr = align_mask_to_ref(gt_path, ref_img_path)
    if arr is None:
        return None

    ref_img = load_nifti_img(ref_img_path)
    if ref_img is not None:
        out = nib.Nifti1Image(arr.astype(np.uint8), ref_img.affine, ref_img.header)
        nib.save(out, str(aligned_path))

    return arr

# ============================================================
# 3) CARGA NIfTI + DESCUBRIR OUTPUTS DE MODELOS
# ============================================================

def load_nifti_data(path: Optional[Path]) -> Optional[np.ndarray]:
    """
    Carga un NIfTI como float32 para evitar duplicar memoria con float64.
    Si no existe o falla, devuelve None.
    """
    if path is None or not Path(path).exists():
        return None

    try:
        img = nib.load(str(path))
        data = img.get_fdata(dtype=np.float32)

        # Por si algún archivo viene como 4D con un único canal
        if data.ndim == 4 and data.shape[-1] == 1:
            data = data[..., 0]

        return data

    except Exception as e:
        print(f"[load_nifti_data] ERROR cargando {path}: {e}")
        return None
    
def load_nifti_img(path: Optional[Path]):
    """
    Carga un archivo NIfTI y devuelve el objeto de imagen de nibabel.

    Si la ruta es nula, el archivo no existe o la carga falla, devuelve None.
    Esta función se usa cuando interesa conservar metadatos espaciales de la
    imagen, no solo el array de intensidades.
    """
    if path is None or not Path(path).exists():
        return None
    try:
        return nib.load(str(path))
    except Exception:
        return None


def load_nifti_canonical(path: Optional[Path]):
    """
    Carga un NIfTI y lo convierte a orientación canónica cuando es posible.

    Esto permite trabajar con una orientación espacial más consistente entre
    imágenes y máscaras antes de remuestrear o comparar volúmenes.
    Si no se puede convertir, devuelve la imagen tal cual.
    """
    img = load_nifti_img(path)
    if img is None:
        return None
    try:
        return nib.as_closest_canonical(img)
    except Exception:
        return img


def align_mask_to_ref(mask_path: Optional[Path], ref_path: Optional[Path]):
    """
    Alinea una máscara binaria al espacio de una imagen de referencia.

    Pasos:
    1. Cargar máscara e imagen de referencia en orientación canónica.
    2. Remuestrear la máscara al grid de la referencia usando vecino más próximo.
    3. Binarizar el resultado.
    4. Aplicar la corrección fija de orientación usada también en la app.

    Esta función es clave para que el visor y el cálculo de DICE trabajen
    con la misma geometría.
    """
    if mask_path is None or ref_path is None:
        return None
    if not Path(mask_path).exists() or not Path(ref_path).exists():
        return None

    try:
        ref_img = load_nifti_canonical(Path(ref_path))
        mask_img = load_nifti_canonical(Path(mask_path))

        if ref_img is None or mask_img is None:
            return None

        mask_res = resample_from_to(mask_img, ref_img, order=0)
        data = mask_res.get_fdata()
        data = (data > 0).astype(np.uint8)

        # giro 180º real del volumen en el plano de visualización
        data = np.flip(data, axis=0)
        data = np.flip(data, axis=1)
        data = np.flip(data, axis=1)

        return data

    except Exception:
        return None
    
def _pick_first_existing(paths: List[Path]) -> Optional[Path]:
    """Devuelve el primer Path existente en disco, o None."""
    for p in paths:
        if p.exists():
            return p
    return None



def find_image_and_seg(pid: str, study_date: str, model_key: str) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Localiza la imagen base del estudio y la segmentación correspondiente
    al modelo solicitado.

    Estrategia:
    - busca primero la imagen NIfTI canónica del estudio,
    - después busca la salida del modelo siguiendo los nombres estándar
      definidos por el proyecto,
    - si no encuentra el nombre exacto, intenta localizar archivos
      compatibles por patrón.

    Se usa tanto en la pestaña de segmentación como en validación y
    en el cálculo de métricas.
    """
    P = get_existing_paths(pid, study_date)

    # Imagen, busca sufijos conocidos
    img_candidates: List[Path] = []
    if "nifti_img" in P:
        img_candidates.append(P["nifti_img"])
    if "nifti_dir" in P:
        img_candidates += [
            P["nifti_dir"] / f"{pid}_0000.nii.gz",
            P["nifti_dir"] / f"{pid}_0000.nii",
        ]
    img_path = _pick_first_existing(img_candidates)

    #Segmentacion segun modelo
    seg_path: Optional[Path] = None

    if model_key == "radionics":
        if "rad_seg" in P and P["rad_seg"].exists():
            seg_path = P["rad_seg"]
        else:
            rad_dir = P.get("radionics_dir", P.get("root", Path(".")) / "radionics")
            cands = sorted(list(rad_dir.glob(f"{pid}_radionics*_seg*.nii*")))
            seg_path = cands[0] if cands else None

    elif model_key.startswith("nnunet_"):
        task = model_key.replace("nnunet_", "")  # task501...
        base = (P.get("nnunet_dir", P.get("root", Path(".")) / "nnunet") / task)
        seg_path = _pick_first_existing([
            base / f"{pid}_{task}_seg_clean.nii.gz",
            base / f"{pid}_{task}_seg_clean.nii",
            base / f"{pid}_{task}_seg.nii.gz",
            base / f"{pid}_{task}_seg.nii",
        ])
        if seg_path is None:
            cands = sorted(list(base.glob(f"{pid}_{task}_seg*.nii*")))
            seg_path = cands[0] if cands else None

    elif model_key in {"agunet", "dagunet", "pls-net", "unet-fv", "unet-slabs"}:
        out_dir = (P.get("root", Path(".")) / model_key)
        seg_path = _pick_first_existing([
            out_dir / f"{pid}_{model_key}_seg.nii.gz",
            out_dir / f"{pid}_{model_key}_seg.nii",
        ])
        if seg_path is None:
            cands = sorted([p for p in out_dir.glob("*.nii*")
                            if "seg" in p.name.lower() and "prob" not in p.name.lower()])
            seg_path = cands[0] if cands else None

    elif model_key == "media_ponderada":
        out_dir = (P.get("root", Path(".")) / "media_ponderada")
        seg_path = _pick_first_existing([
            out_dir / f"{pid}_media_ponderada_seg.nii.gz",
            out_dir / f"{pid}_media_ponderada_seg.nii",
        ])
        if seg_path is None:
            cands = sorted([
                p for p in out_dir.glob("*.nii*")
                if "seg" in p.name.lower()
            ])
            seg_path = cands[0] if cands else None

    elif model_key == "all":
        seg_path = find_image_and_seg(pid, study_date, "radionics")[1]
        if seg_path is None:
            seg_path = find_image_and_seg(pid, study_date, "nnunet_task501")[1]

    return img_path, (seg_path if (seg_path and seg_path.exists()) else None)

def load_mask_to_ref_no_flip(mask_path: Path, ref_path: Path) -> Optional[np.ndarray]:
    """
    Carga una máscara, la remuestrea al espacio de referencia y la binariza.
    No aplica flips de visualización.
    """
    try:
        ref_img = load_nifti_img(ref_path)
        mask_img = load_nifti_img(mask_path)

        if ref_img is None or mask_img is None:
            return None

        if mask_img.shape != ref_img.shape or not np.allclose(mask_img.affine, ref_img.affine):
            mask_img = resample_from_to(mask_img, ref_img, order=0)

        data = mask_img.get_fdata()
        return (data > 0).astype(np.float32)

    except Exception:
        return None


def build_weighted_heatmap_for_study(
    pid: str,
    study_date: str,
    ref_img_path: Optional[Path],
) -> Optional[np.ndarray]:
    """
    Construye un mapa de calor ponderado en [0,1] a partir de las
    segmentaciones disponibles de los modelos individuales.
    """
    if ref_img_path is None or not Path(ref_img_path).exists():
        return None

    acc = None
    total_weight = 0.0
    used_models = 0

    for model_key, weight in MODEL_DICE_WEIGHTS.items():
        _imgp, seg_path = find_image_and_seg(pid, study_date, model_key)

        if seg_path is None or not Path(seg_path).exists():
            continue

        mask = load_mask_to_ref_no_flip(Path(seg_path), Path(ref_img_path))
        if mask is None:
            continue

        if acc is None:
            acc = np.zeros(mask.shape, dtype=np.float32)

        acc += float(weight) * mask
        total_weight += float(weight)
        used_models += 1

    if acc is None or total_weight <= 0 or used_models < 2:
        return None

    heatmap = acc / total_weight
    return np.clip(heatmap, 0.0, 1.0).astype(np.float32)

# ============================================================
# 4) VISOR (PLOTEO + OVERLAY + CROSSHAIR)
# ============================================================
#Plot_slice visor pestaña segmentación
def plot_slice(
    img_data: Optional[np.ndarray],
    seg_data: Optional[np.ndarray],
    idx: int,
    axis: int,
    alpha: float,
    show_seg: bool,
    seg_color: str = "#ef4444",
    cross: dict | None = None,
    draw_cross: bool = True,
    heatmap: bool = False,
) -> None:
    # Fondo oscuro para no deslumbrar
    fig = plt.figure(figsize=(5, 5), facecolor="#0f1115")
    ax = plt.gca()
    ax.set_facecolor("#0f1115")

    if img_data is None and seg_data is None:
        plt.text(
            0.5, 0.5, "No hay imagen/segmentación cargada",
            ha="center", va="center", color="#cfd3dc"
        )
        plt.axis("off")
        return

    # ===== Imagen =====
    
    if img_data is not None:
        idx = max(0, min(int(idx), img_data.shape[axis] - 1))

        # axis: 2=axial (z fijo), 1=coronal (y fijo), 0=sagital (x fijo)
        # Nota: luego aplicamos np.rot90(sl) para que la orientación en pantalla
        # sea más consistente con el visor (imshow usa origen arriba-izquierda).

        if axis == 2:        # axial: (X,Y)
            sl = img_data[:, :, idx]
        elif axis == 1:      # coronal: (X,Z)
            sl = img_data[:, idx, :]
        else:                # sagital: (Y,Z)
            sl = img_data[idx, :, :]

        sl = np.rot90(sl, k=1)

        # contraste robusto
        vmin, vmax = np.percentile(sl, (1, 99))
        if vmax <= vmin:
            vmin, vmax = float(np.min(sl)), float(np.max(sl))
            if vmax <= vmin:
                vmax = vmin + 1.0

        plt.imshow(sl, cmap="gray", interpolation="nearest", vmin=vmin, vmax=vmax)

    # ===== Segmentación =====
    if seg_data is not None and show_seg:
        idx2 = max(0, min(int(idx), seg_data.shape[axis] - 1))

        if axis == 2:
            ms = seg_data[:, :, idx2]
        elif axis == 1:
            ms = seg_data[:, idx2, :]
        else:
            ms = seg_data[idx2, :, :]

        ms = np.rot90(ms, k=1)

        # Si viene como probabilidad float -> binariza
        if heatmap:
            ms = ms.astype(np.float32)
            ms_masked = np.ma.masked_where(ms <= 0, ms)

            if np.any(ms > 0):
                plt.imshow(
                    ms_masked,
                    cmap="turbo",
                    alpha=float(alpha),
                    interpolation="nearest",
                    vmin=0,
                    vmax=1,
                )
        else:
            # Si viene como probabilidad float -> binariza
            if ms.dtype.kind == "f":
                ms = (ms >= 0.5).astype(np.uint8)

            if np.any(ms):
                ms_masked = np.ma.masked_where(ms == 0, ms)

                cmap1 = ListedColormap([seg_color])
                plt.imshow(
                    ms_masked,
                    cmap=cmap1,
                    alpha=float(alpha),
                    interpolation="nearest",
                    vmin=0,
                    vmax=1,
                )

    # ===== Crosshair (independiente de la segmentación) =====
    # Nota: el eje vertical se invierte (Y-1-*) por el origen de imshow y el rot90 aplicado.
    if draw_cross and (cross is not None) and (img_data is not None):
        cx, cy, cz = int(cross["x"]), int(cross["y"]), int(cross["z"])

        if axis == 2:
            # axial: slice = z
            if int(idx) == cz:
                x_disp = cx
                y_disp = (img_data.shape[1] - 1 - cy)
                plt.axvline(x_disp, linewidth=1)
                plt.axhline(y_disp, linewidth=1)

        elif axis == 1:
            # coronal: slice = y
            if int(idx) == cy:
                x_disp = cx
                y_disp = (img_data.shape[2] - 1 - cz)
                plt.axvline(x_disp, linewidth=1)
                plt.axhline(y_disp, linewidth=1)

        else:
            # sagital: slice = x
            if int(idx) == cx:
                x_disp = cy
                y_disp = (img_data.shape[2] - 1 - cz)
                plt.axvline(x_disp, linewidth=1)
                plt.axhline(y_disp, linewidth=1)

    plt.axis("off")
    # Evita warning de tight_layout
    plt.tight_layout(pad=0)

#Plot_slice visor pestaña validación
def plot_validation_slice(
    img_data: Optional[np.ndarray],
    pred_data: Optional[np.ndarray],
    gt_data: Optional[np.ndarray],
    idx: int,
    plane: str,
    alpha_pred: float = 0.35,
    alpha_gt: float = 0.35,
    pred_color: str = "#ef4444",
    show_overlap: bool = True,
) -> None:
    fig = plt.figure(figsize=(9, 9), facecolor="#0f1115")
    ax = plt.gca()
    ax.set_facecolor("#0f1115")

    if img_data is None:
        plt.text(0.5, 0.5, "No hay imagen cargada", ha="center", va="center", color="#cfd3dc")
        plt.axis("off")
        return

    axis_map = {"sagital": 0, "coronal": 1, "axial": 2}
    axis = axis_map.get(plane, 2)

    idx = max(0, min(int(idx), img_data.shape[axis] - 1))

    if axis == 2:
        sl = img_data[:, :, idx]
    elif axis == 1:
        sl = img_data[:, idx, :]
    else:
        sl = img_data[idx, :, :]

    sl = np.rot90(sl, k=1)

    vmin, vmax = np.percentile(sl, (1, 99))
    if vmax <= vmin:
        vmin, vmax = float(np.min(sl)), float(np.max(sl))
        if vmax <= vmin:
            vmax = vmin + 1.0

    plt.imshow(sl, cmap="gray", interpolation="nearest", vmin=vmin, vmax=vmax)

    pred = None
    gt = None

    if pred_data is not None:
        if axis == 2:
            pred = pred_data[:, :, idx]
        elif axis == 1:
            pred = pred_data[:, idx, :]
        else:
            pred = pred_data[idx, :, :]
        pred = np.rot90(pred, k=1)
        pred = (pred > 0).astype(np.uint8)

    if gt_data is not None:
        if axis == 2:
            gt = gt_data[:, :, idx]
        elif axis == 1:
            gt = gt_data[:, idx, :]
        else:
            gt = gt_data[idx, :, :]
        gt = np.rot90(gt, k=1)
        gt = (gt > 0).astype(np.uint8)

    if pred is None and gt is None:
        plt.axis("off")
        plt.tight_layout()
        return

    if pred is None:
        gt_only = gt
        if np.any(gt_only):
            gt_masked = np.ma.masked_where(gt_only == 0, gt_only)
            plt.imshow(
                gt_masked,
                cmap=ListedColormap([VALIDATION_GT_COLOR]),
                alpha=alpha_gt,
                interpolation="nearest",
                vmin=0,
                vmax=1,
            )
        plt.axis("off")
        plt.tight_layout()
        return

    if gt is None:
        pred_only = pred
        if np.any(pred_only):
            pred_masked = np.ma.masked_where(pred_only == 0, pred_only)
            plt.imshow(
                pred_masked,
                cmap=ListedColormap([pred_color]),
                alpha=alpha_pred,
                interpolation="nearest",
                vmin=0,
                vmax=1,
            )
        plt.axis("off")
        plt.tight_layout()
        return

    if show_overlap:
        overlap = ((pred > 0) & (gt > 0)).astype(np.uint8)
        pred_only = ((pred > 0) & (gt == 0)).astype(np.uint8)
        gt_only = ((gt > 0) & (pred == 0)).astype(np.uint8)

        if np.any(pred_only):
            pred_masked = np.ma.masked_where(pred_only == 0, pred_only)
            plt.imshow(
                pred_masked,
                cmap=ListedColormap([pred_color]),
                alpha=alpha_pred,
                interpolation="nearest",
                vmin=0,
                vmax=1,
            )

        if np.any(gt_only):
            gt_masked = np.ma.masked_where(gt_only == 0, gt_only)
            plt.imshow(
                gt_masked,
                cmap=ListedColormap([VALIDATION_GT_COLOR]),
                alpha=alpha_gt,
                interpolation="nearest",
                vmin=0,
                vmax=1,
            )

        if np.any(overlap):
            overlap_masked = np.ma.masked_where(overlap == 0, overlap)
            plt.imshow(
                overlap_masked,
                cmap=ListedColormap([VALIDATION_OVERLAP_COLOR]),
                alpha=0.75,
                interpolation="nearest",
                vmin=0,
                vmax=1,
            )
    else:
        if np.any(pred):
            pred_masked = np.ma.masked_where(pred == 0, pred)
            plt.imshow(
                pred_masked,
                cmap=ListedColormap([pred_color]),
                alpha=alpha_pred,
                interpolation="nearest",
                vmin=0,
                vmax=1,
            )

        if np.any(gt):
            gt_masked = np.ma.masked_where(gt == 0, gt)
            plt.imshow(
                gt_masked,
                cmap=ListedColormap([VALIDATION_GT_COLOR]),
                alpha=alpha_gt,
                interpolation="nearest",
                vmin=0,
                vmax=1,
            )

    plt.axis("off")
    plt.tight_layout()
# ============================================================
# 5) ÚLTIMA SESIÓN 
# ============================================================

def load_last_session() -> dict:
    """Carga selección anterior (patient_id, study_date, model) si existe."""
    if not LAST_SESSION.exists():
        return {}
    try:
        return json.loads(LAST_SESSION.read_text(encoding="utf-8")) #Cargamos json con info última sesión   
    except Exception:
        return {}

def save_last_session(pid: str, sdate: str, model_label: str) -> None:
    """Guarda selección actual para restaurarla al iniciar.
    - Paciente ID
    - Fecha del estudio
    - Modelo empleado
    - Fecha de última sesión"""
    try:
        LAST_SESSION.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "patient_id": pid,
            "study_date": sdate,
            "model": model_label,
            "ts": datetime.now().isoformat(timespec="seconds"),
        }
        LAST_SESSION.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass

# ============================================================
# 6) VALIDACIÓN
# ============================================================

def dice_score(gt: np.ndarray, pred: np.ndarray, eps: float = 1e-8) -> float:
    """
    Calcula el coeficiente DICE entre una máscara de referencia (GT) y una máscara predicha.

    Ambas entradas se binarizan previamente, considerando como tumor todos los vóxeles con
    valor mayor que 0. Después se calcula el solapamiento entre ambas máscaras mediante la
    fórmula del coeficiente DICE:

        DICE = 2 * |A ∩ B| / (|A| + |B|)

    donde:
    - A es la máscara manual de referencia (ground truth),
    - B es la máscara predicha por el modelo,
    - |A ∩ B| es el número de vóxeles compartidos por ambas.
    """
    gt = (gt > 0).astype(np.uint8)
    pred = (pred > 0).astype(np.uint8)
    inter = np.sum(gt * pred)
    return float((2.0 * inter + eps) / (np.sum(gt) + np.sum(pred) + eps))

def has_prediction(pid: str, sdate: str, model_key: str) -> bool:
    """
    Comprueba si existe una segmentación predicha para un paciente, una fecha de estudio
    y un modelo concretos.
    Esta función se emplea para saber si un estudio ya ha sido procesado por un modelo
    determinado y para filtrar los casos disponibles en la pestaña de validación.
    """
    _, segp = find_image_and_seg(pid, sdate, model_key)
    return bool(segp and Path(segp).exists())

def get_available_model_seg_paths(pid: str, sdate: str) -> Dict[str, Path]:
    """
    Devuelve {model_key: seg_path} para todas las segmentaciones disponibles
    del estudio.
    """
    out: Dict[str, Path] = {}
    for mk in VAL_MODEL_KEYS:
        _, segp = find_image_and_seg(pid, sdate, mk)
        if segp is not None and Path(segp).exists():
            out[mk] = Path(segp)
    return out

def get_model_label_from_key(model_key: str) -> str:
    """
    Devuelve la etiqueta visible en la interfaz asociada a una clave interna
    de modelo.

    Ejemplo:
    - 'nnunet_task501' -> 'nnU-Net (Task501)'

    Si no encuentra correspondencia, devuelve la propia clave.
    """
    for label, key in MODEL_OPTIONS.items():
        if key == model_key:
            return label
    return model_key

def study_is_processed(pid: str, sdate: str) -> bool:
    """
    Determina si un estudio esta procesado.

    Un estudio se considera procesado cuando existe al menos una segmentación predicha por
    cualquiera de los modelos incluidos en `VAL_MODEL_KEYS`. Para ello recorre todas las
    claves de modelos válidas y comprueba si alguna tiene predicción disponible mediante
    `has_prediction(...)`.
    """
    for mk in VAL_MODEL_KEYS:
        if has_prediction(pid, sdate, mk):
            return True
    return False

def scan_processed_patients() -> List[str]:
    """
    Devuelve la lista de pacientes que tienen al menos un estudio procesado.
    La función recorre todos los pacientes disponibles en `Pacientes_nifti` y comprueba si
    alguno de sus estudios cumple la condición de estudio procesado mediante
    `study_is_processed(...)`. Solo se incluyen en la lista final aquellos pacientes para
    los que existe al menos una fecha con segmentación disponible.
    """
    pids = scan_nifti_patients()
    return [pid for pid in pids if any(study_is_processed(pid, d) for d in list_studies(pid))]

def list_processed_studies(pid: str) -> List[str]:
    """
    Obtiene las fechas de estudio procesadas para un paciente concreto.

    A partir del identificador del paciente, la función recupera todas las fechas de estudio
    disponibles y conserva únicamente aquellas que cumplen la condición de estudio procesado,
    es decir, que tienen al menos una segmentación generada por algún modelo.
    """
    return [d for d in list_studies(pid) if study_is_processed(pid, d)]

def compute_study_dices(
    pid: str,
    sdate: str,
    gt_path: Path,
    progress_cb=None,
) -> Dict[str, Optional[float]]:
    """
    Calcula los coeficientes DICE de un estudio entre la segmentación manual
    y todas las predicciones disponibles de los modelos.

    Flujo:
    1. Elegir una imagen de referencia del estudio.
    2. Alinear la GT manual a esa referencia.
    3. Recorrer todos los modelos válidos.
    4. Alinear cada predicción a la misma referencia.
    5. Calcular DICE cuando haya datos compatibles.

    El resultado se devuelve como una fila lista para mostrarse en la app
    o guardarse en CSV.
    """
    if not gt_path.exists():
        raise FileNotFoundError(f"No existe la segmentación manual: {gt_path}")

    ref_img_path = None
    for mk in VAL_MODEL_KEYS:
        imgp, _ = find_image_and_seg(pid, sdate, mk)
        if imgp is not None and Path(imgp).exists():
            ref_img_path = Path(imgp)
            break

    if ref_img_path is None:
        raise FileNotFoundError(f"No encuentro la imagen de referencia para {pid}/{sdate}")
    
    try:
        build_weighted_ensemble_seg_for_study(
            pid,
            sdate,
            threshold=0.5,
            force=False,
        )
    except Exception:
        pass

    gt = load_manual_gt_for_validation(pid, sdate, ref_img_path)
    if gt is None:
        raise RuntimeError("No se pudo alinear la segmentación manual a la imagen de referencia.")

    row: Dict[str, Optional[float]] = {"paciente_id": pid, "study_date": sdate}

    for mk in VAL_MODEL_KEYS:
        model_label = get_model_label_from_key(mk)

        if progress_cb is not None:
            progress_cb(f"Calculando DICE con {model_label}...")

        _imgp, pred_path = find_image_and_seg(pid, sdate, mk)
        if pred_path is None or not Path(pred_path).exists():
            row[mk] = None
            if progress_cb is not None:
                progress_cb(f"Sin predicción para {model_label}.")
            continue

        pred = align_mask_to_ref(Path(pred_path), ref_img_path)
        if pred is None or pred.shape != gt.shape:
            row[mk] = None
            if progress_cb is not None:
                progress_cb(f"No se pudo alinear la predicción de {model_label}.")
            continue

        row[mk] = round(dice_score(gt, pred), 4)

        if progress_cb is not None:
            progress_cb(f"DICE calculado con {model_label}: {row[mk]:.4f}")

    return row

def format_dice_text(model_key: str, study_dices: Dict[str, Optional[float]]) -> str:
    """
    Formatea el valor DICE de un modelo concreto para mostrarlo como texto
    legible en la interfaz.

    Si el modelo no tiene valor disponible en el diccionario de resultados,
    devuelve un mensaje indicándolo.
    """
    d = study_dices.get(model_key)
    if d is None:
        return f"No hay DICE disponible para {model_key}."
    return f"DICE ({model_key}): {float(d):.4f}"

# ============================================================
# 7) CSV
# ============================================================

def upsert_row_csv(
    csv_path: Path,
    row: Dict[str, object],
    key_fields: tuple[str, ...] = ("paciente_id", "study_date"),
) -> Path:
    """
    Inserta o actualiza una fila en un CSV usando un conjunto de claves
    identificadoras.

    Comportamiento:
    - si el CSV no existe, lo crea,
    - si ya existe una fila con la misma clave, la actualiza,
    - si no existe, añade una nueva fila.

    Se usa para mantener CSV acumulados de volúmenes y DICE sin duplicar
    estudios ya procesados.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    row_str = {k: ("" if v is None else str(v)) for k, v in row.items()}
    fieldnames = list(row_str.keys())
    rows: list[dict[str, str]] = []

    if csv_path.exists():
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            prev_fields = list(reader.fieldnames or [])

            for fn in prev_fields:
                if fn not in fieldnames:
                    fieldnames.append(fn)

            for prev_row in reader:
                merged = {fn: prev_row.get(fn, "") for fn in fieldnames}
                rows.append(merged)

    found = False
    for existing in rows:
        same_key = all(str(existing.get(k, "")) == str(row_str.get(k, "")) for k in key_fields)
        if same_key:
            for fn in fieldnames:
                existing[fn] = row_str.get(fn, existing.get(fn, ""))
            found = True
            break

    if not found:
        rows.append({fn: row_str.get(fn, "") for fn in fieldnames})

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return csv_path

def load_study_row_from_csv(csv_path: Path, pid: str, sdate: str, required_value_fields: tuple[str, ...] = ()) -> Optional[Dict[str, object]]:
    if not csv_path.exists():
        return None

    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if str(row.get("paciente_id", "")).strip() != str(pid).strip():
                    continue
                if str(row.get("study_date", "")).strip() != str(sdate).strip():
                    continue

                if required_value_fields and not any(str(row.get(c, "")).strip() for c in required_value_fields):
                    return None

                out = {}
                for k, v in row.items():
                    v = "" if v is None else str(v).strip()
                    if k in {"paciente_id", "study_date", "manual_path"}:
                        out[k] = v
                    elif v == "":
                        out[k] = None
                    else:
                        try:
                            out[k] = float(v)
                        except ValueError:
                            out[k] = v
                return out
    except Exception:
        return None

    return None


def delete_rows_from_csv(csv_path: Path, patient_id: str, study_date: Optional[str] = None) -> int:
    if not csv_path.exists():
        return 0

    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)

        kept = []
        removed = 0

        for row in rows:
            same_patient = str(row.get("paciente_id", "")).strip() == str(patient_id).strip()
            same_study = True if study_date is None else str(row.get("study_date", "")).strip() == str(study_date).strip()

            if same_patient and same_study:
                removed += 1
            else:
                kept.append(row)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(kept)

        return removed
    except Exception:
        return 0

def save_study_volumes_to_global_csv(study_vols: Dict[str, Optional[float]]) -> Path:
    """
    Guarda o actualiza en el CSV global acumulado los volúmenes calculados
    para un estudio concreto.
    """
    return upsert_row_csv(OUT_CSV, study_vols)

def build_single_study_volumes_csv(study_vols: Dict[str, Optional[float]], out_csv: Path) -> Path:
    """
    Genera un CSV con una sola fila correspondiente a un único estudio.

    Se usa para permitir la descarga de resultados de volumen del paciente
    y estudio actualmente seleccionados.
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    row = {k: ("" if v is None else str(v)) for k, v in study_vols.items()}
    fieldnames = list(row.keys())

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)

    return out_csv

def save_study_dices_to_global_csv(study_dices: Dict[str, Optional[float]]) -> Path:
    """
    Guarda o actualiza en el CSV global acumulado los valores DICE de un
    estudio concreto.
    """
    return upsert_row_csv(DICE_ALL_CSV, study_dices)

def build_single_study_dice_csv(study_dices: Dict[str, Optional[float]], out_csv: Path) -> Path:
    """
    Genera un CSV con una única fila de resultados DICE para el estudio
    actualmente seleccionado.
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    row = {k: ("" if v is None else str(v)) for k, v in study_dices.items()}
    fieldnames = list(row.keys())

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)

    return out_csv

def build_dice_csv(
    gt_path: Path,
    patient_id: Optional[str] = None,
    study_date: Optional[str] = None,
    out_csv: Optional[Path] = None,
) -> Path:
    """
    Calcula y exporta un CSV de DICE para uno o varios estudios.

    Permite filtrar por paciente y/o por fecha de estudio. Para cada estudio
    válido calcula el DICE frente a la segmentación manual indicada por
    `gt_path` y guarda los resultados en un CSV.
    """
    if not gt_path.exists():
        raise FileNotFoundError(f"No existe la segmentación manual: {gt_path}")

    APP_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    studies = iter_studies(patient_id=patient_id, study_date=study_date)
    if not studies:
        raise FileNotFoundError("No hay estudios para calcular DICE con los filtros indicados.")

    cols = ["paciente_id", "study_date"] + VAL_MODEL_KEYS
    rows = []

    for pid, sdate, _ in studies:
        raw = compute_study_dices(pid, sdate, gt_path)
        row = {"paciente_id": pid, "study_date": str(raw["study_date"])}
        for col in VAL_MODEL_KEYS:
            value = raw.get(col)
            row[col] = "" if value is None else str(value)
        rows.append(row)

    dest = out_csv or DICE_ALL_CSV
    with open(dest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    return dest

# ============================================================
# 8) HELPERS UI
# ============================================================
def _img_to_data_uri(path: Path) -> str:
    """
    Convierte un archivo de imagen en una cadena data URI para incrustarlo
    directamente en la interfaz HTML.

    Si la imagen no existe, devuelve cadena vacía.
    """
    if not path.exists():
        return ""
    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"

def app_logo_top_right():
    """
    Construye el bloque UI que muestra el logo de la aplicación en la esquina
    superior derecha.

    Si la imagen no existe, devuelve un contenedor vacío.
    """
    logo_path = IMAGES_DIR / "icono_verde.png"
    src = _img_to_data_uri(logo_path)
    if not src:
        return ui.div()

    return ui.div(
        ui.tags.img(src=src, alt="Logo app", class_="app-corner-logo"),
        class_="app-corner-logo-wrap",
    )

def app_config_bottom_left():
    icon_path = IMAGES_DIR / "config_app.png"
    src = _img_to_data_uri(icon_path)
    if not src:
        return ui.div()

    return ui.tags.button(
        {
            "id": "open_storage_config_btn",
            "type": "button",
            "class": "btn action-button app-config-fab",
            "title": "Configuración de rutas",
            "aria-label": "Configuración de rutas",
            "value": "0",
        },
        ui.tags.img(src=src, alt="Configuración", class_="app-config-fab-icon"),
    )

def icon_button(
    button_id: str,
    image_name: str,
    alt_text: str,
    tone: str = "neutral",
    extra_class: str = "",
):
    """
    Crea un botón de acción con icono e imagen personalizada para la interfaz.

    """
    img_path = IMAGES_DIR / image_name
    src = _img_to_data_uri(img_path)

    btn_class = f"btn action-button action-image-btn tone-{tone} {extra_class}".strip()

    # Fallback si no existe la imagen
    if not src:
        return ui.tags.button(
            {
                "id": button_id,
                "type": "button",
                "class": btn_class,
                "title": alt_text,
                "aria-label": alt_text,
                "value": "0",
            },
            alt_text,
        )

    return ui.tags.button(
        {
            "id": button_id,
            "type": "button",
            "class": btn_class,
            "title": alt_text,      # tooltip nativo al pasar el ratón
            "aria-label": alt_text,
            "value": "0",
        },
        ui.tags.span(
            ui.tags.img(src=src, alt=alt_text, class_="action-btn-icon"),
            class_="action-btn-inner",
        ),
        ui.tags.span(alt_text, class_="visually-hidden"),
    )

def mini_icon_button(button_id: str, image_name: str, alt_text: str, active: bool = False):
    """
    Crea un botón pequeño con icono, pensado para acciones compactas o
    controles secundarios de la interfaz.
    """
    img_path = IMAGES_DIR / image_name
    src = _img_to_data_uri(img_path)

    cls = "btn mini-action-btn active" if active else "btn mini-action-btn"
    if not src:
        return ui.input_action_button(button_id, alt_text)

    return ui.tags.button(
        {"id": button_id, "type": "button", "class": cls},
        ui.tags.img(src=src, alt=alt_text, class_="mini-action-icon"),
        ui.tags.span(alt_text, class_="visually-hidden"),
    )

def weighted_heatmap_legend_ui():
    """
    Leyenda visual para el mapa de calor de media ponderada.

    El visor usa cmap='turbo' con valores entre 0 y 1:
    - 0: bajo acuerdo ponderado
    - 1: alto acuerdo ponderado
    """
    return ui.div(
        ui.div(
            "Leyenda media ponderada",
            style="""
                font-size:13px;
                font-weight:700;
                color:#dbe4f3;
                margin-bottom:6px;
            """,
        ),
        ui.div(
            style="""
                height:14px;
                border-radius:999px;
                border:1px solid #334155;
                background: linear-gradient(
                    90deg,
                    #30123b 0%,
                    #4145ab 12%,
                    #4675ed 25%,
                    #39a2fc 37%,
                    #1bcfd4 50%,
                    #24eca6 62%,
                    #61fc6c 75%,
                    #b4ec32 87%,
                    #eba81e 94%,
                    #e33333 100%
                );
                margin-bottom:6px;
            """,
        ),
        ui.div(
            ui.span("0.0 bajo acuerdo"),
            ui.span("1.0 alto acuerdo"),
            style="""
                display:flex;
                justify-content:space-between;
                gap:8px;
                color:#9aa3b6;
                font-size:11px;
            """,
        ),
        ui.div(
            "Azul/morado: pocos modelos coinciden. Rojo: mayor acuerdo ponderado entre modelos.",
            style="""
                margin-top:6px;
                color:#9aa3b6;
                font-size:11px;
                line-height:1.35;
            """,
        ),
        style="""
            margin-top:10px;
            padding:10px 12px;
            border-radius:12px;
            background:#111522;
            border:1px solid #222838;
        """,
    )
# ============================================================
# 9) UI
# ============================================================

seg_ui = ui.page_sidebar(
    ui.sidebar(
        ui.div(
            ui.h4("Acciones"),
            ui.div(
                icon_button("add_patient_btn", "anadir_paciente.png", "Añadir paciente", tone="add"),
                icon_button("add_patients_btn", "anadir_varios.png", "Añadir varios pacientes", tone="add"),
                icon_button("delete_patient_btn", "eliminar_paciente.png", "Eliminar paciente", tone="danger", extra_class="span-2"),
                icon_button("run_btn", "ejecutar_modelo.png", "Ejecutar modelo", tone="run"),
                icon_button("run_all_btn", "ejecutar_todos.png", "Ejecutar modelos", tone="run"),
                icon_button("run_all_patients_models_btn","todos_pacientes_modelos.png","Ejecutar todos los modelos en todos los pacientes",tone="run",extra_class="span-2"),
                icon_button("calc_vol_btn", "calcular_volumen.png", "Calcular volumen", tone="analysis"),
                icon_button("toggle_coords_btn", "coordenadas.png", "Mostrar coordenadas", tone="viewer"),
                icon_button("download_csv_study_btn", "descargar_volumen.png", "Descargar CSV paciente", tone="download"),
                icon_button("download_csv_all_btn", "descargar_varios.png", "Descargar CSV todos", tone="download"),
                icon_button("download_seg_btn", "descargar_seg.png", "Descargar segmentación", tone="download", extra_class="span-2"),
                class_="action-button-stack",
            )
        ),
        ui.div(
            ui.input_slider("alpha", "Transparencia", 0, 1, 0.4, step=0.1),
            ui.input_checkbox("show_seg", "Mostrar segmentación", value=True),
            style="margin-top:10px;",
        ),
        width=340,
    ),

    ui.tags.style(
        """
        :root{
          --bg:#0b0d10;
          --panel:#0f1115;
          --panel2:#141823;
          --panel3:#101521;
          --text:#e6e8ee;
          --muted:#aeb4c2;
          --border:#222838;
          --accent:#d64b4b;
          --accent2:#3b82f6;
          --ok:#22c55e;
          --warn:#f59e0b;
        }

        body{ background: var(--bg) !important; color: var(--text) !important; }
        .container-fluid{ background: var(--bg) !important; }

        header.navbar, .navbar, .navbar-default{
          background: var(--panel) !important;
          border-bottom: 1px solid var(--border) !important;
          box-shadow: none !important;
        }

        .navbar-brand, .navbar .navbar-brand{
          color: var(--text) !important;
          font-weight: 600;
        }

        .bslib-sidebar-layout .sidebar{
          background: var(--panel) !important;
          color: var(--text) !important;
          border-right: 1px solid var(--border);
          min-width: 340px;
        }

        .form-control, .form-select, .selectize-input, .selectize-dropdown{
          background: var(--panel2) !important;
          color: var(--text) !important;
          border: 1px solid var(--border) !important;
          border-radius: 10px !important;
        }

        .selectize-dropdown, .selectize-dropdown-content{
          background: var(--panel2) !important;
          color: var(--text) !important;
        }

        .card{
          background: var(--panel) !important;
          color: var(--text) !important;
          border: 1px solid var(--border) !important;
          box-shadow: 0 10px 28px rgba(0,0,0,.35);
          border-radius: 14px !important;
        }

        .card-header{
          background: linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,0)) !important;
          border-bottom: 1px solid var(--border) !important;
          color: var(--text) !important;
          font-weight: 600 !important;
        }

        .btn{
          background: var(--panel2) !important;
          color: var(--text) !important;
          border: 1px solid var(--border) !important;
        }

        .btn:hover{ filter: brightness(1.08); }

        .action-button-stack{
          display:grid;
          grid-template-columns: 1fr 1fr;
          gap:10px;
        }

        .action-image-btn{
          width:100%;
          min-height:72px;
          padding:10px !important;
          border-radius:14px !important;
          text-align:center !important;
          border:1px solid #263047 !important;
          box-shadow: 0 6px 14px rgba(0,0,0,.24);
          transition: all .18s ease;
          display:flex !important;
          align-items:center;
          justify-content:center;
        }


        .action-btn-inner{
          display:flex;
          align-items:center;
          justify-content:center;
          width:100%;
        }

        .action-image-btn:hover{
          transform: translateY(-1px);
          box-shadow: 0 0 0 1px rgba(255,255,255,.06), 0 10px 22px rgba(0,0,0,.32);
        }

        .action-btn-icon{
          width:44px;
          height:44px;
          object-fit:contain;
          flex:0 0 auto;
        }

        .action-btn-label{
          display:none;
        }

        .action-image-btn.span-2{
          grid-column: 1 / -1;
        }

        /* TONOS */
        .action-image-btn.tone-add{
          background: linear-gradient(180deg, #10251d 0%, #0c1d18 100%) !important;
          border-color:#1f5b49 !important;
        }

        .action-image-btn.tone-add:hover{
          border-color:#2faa7f !important;
        }

        .action-image-btn.tone-run{
          background: linear-gradient(180deg, #142235 0%, #101b2c 100%) !important;
          border-color:#2b4d7c !important;
        }

        .action-image-btn.tone-run:hover{
          border-color:#4d8df0 !important;
        }

        .action-image-btn.tone-analysis{
          background: linear-gradient(180deg, #2a1d12 0%, #20160e 100%) !important;
          border-color:#7a552e !important;
        }

        .action-image-btn.tone-analysis:hover{
          border-color:#d18a39 !important;
        }

        .action-image-btn.tone-viewer{
          background: linear-gradient(180deg, #1f1730 0%, #181225 100%) !important;
          border-color:#594288 !important;
        }

        .action-image-btn.tone-viewer:hover{
          border-color:#8e69d6 !important;
        }

        .action-image-btn.tone-download{
          background: linear-gradient(180deg, #14242a 0%, #101d22 100%) !important;
          border-color:#35616d !important;
        }

        .action-image-btn.tone-download:hover{
          border-color:#4eb1c8 !important;
        }

        .action-image-btn.tone-danger{
          background: linear-gradient(180deg, #2b1518 0%, #211012 100%) !important;
          border-color:#7b3138 !important;
        }

        .action-image-btn.tone-danger:hover{
          border-color:#d35b66 !important;
        }

        .mini-action-btn{
          display:flex !important;
          align-items:center;
          justify-content:center;
          width:52px;
          height:52px;
          border-radius:12px !important;
          padding:6px !important;
          background:#111827 !important;
          border:1px solid #263047 !important;
        }

        .mini-action-btn.active{
          border-color:#3b82f6 !important;
          box-shadow: 0 0 0 1px rgba(59,130,246,.20);
        }

        .mini-action-icon{
          width:32px;
          height:32px;
          object-fit:contain;
        }

        .study-card{
          margin-bottom:16px;
        }

        .viewer-plot-card .shiny-plot-output{
          min-height: 460px;
        }

        .status-card{
          margin-top:16px;
        }

        .status-bar-wrap{
          display:flex;
          align-items:center;
          justify-content:space-between;
          gap:16px;
          padding:14px 16px;
          background:#0f1522;
          border:1px solid #222838;
          border-radius:12px;
          min-height:88px;
        }

        .status-lines{
          flex:1 1 auto;
          min-width:0;
        }

        .status-line{
          white-space:nowrap;
          overflow:hidden;
          text-overflow:ellipsis;
          color:#d7dcea;
          line-height:1.55;
          font-size:14px;
        }

        .status-line.dim{
          color:#9aa3b6;
        }

        .status-spinner{
          flex:0 0 auto;
          display:flex;
          align-items:center;
          justify-content:center;
          width:34px;
          height:34px;
        }

        .status-empty{
          color:#9aa3b6;
        }

        .irs--shiny .irs-bar, .irs--shiny .irs-single, .irs--shiny .irs-from, .irs--shiny .irs-to{
          background: var(--accent) !important;
          border-color: var(--accent) !important;
        }

        .irs--shiny .irs-handle{ border-color: var(--accent) !important; }
        .irs--shiny .irs-line{ background: #1b2130 !important; border-color: var(--border) !important; }

        .floating-coords{
          position: fixed;
          right: 14px;
          bottom: 14px;
          z-index: 9999;
          background: rgba(15,17,21,.92);
          border: 1px solid #222838;
          padding: 10px 12px;
          border-radius: 10px;
          font-size: 12px;
          color: #e6e8ee;
          min-width: 180px;
        }

        .floating-coords.hidden{
          display:none !important;
        }

        .validation-big .shiny-plot-output{
          min-height: 720px;
        }

        label{ color: var(--muted) !important; }
        hr{ border-color: var(--border); opacity: 1; }
        pre{ background: #111522 !important; color: var(--text) !important; border: 1px solid var(--border) !important; }

        .app-corner-logo-wrap{
          position:fixed;
          top:14px;
          right:18px;
          z-index:9998;
          pointer-events:none;
        }

        .app-corner-logo{
          width:100px;
          height:130px;
          object-fit:contain;
          filter: drop-shadow(0 6px 14px rgba(0,0,0,.35));
          opacity:.96;
        }

        .validation-main-row{
          display:flex;
          align-items:stretch;
          gap:16px;
        }

        .validation-side-info{
          width:240px;
          min-width:240px;
          background:#101521;
          border:1px solid #222838;
          border-radius:14px;
          padding:16px;
        }

        .validation-plot-wrap{
          flex:1 1 auto;
          min-width:0;
        }

        .validation-plot-wrap .shiny-plot-output{
          min-height:720px;
        }

        .dice-summary-box{
          display:flex;
          flex-direction:column;
          gap:10px;
          margin-top:4px;
        }

        .dice-summary-main{
          font-size:14px;
          line-height:1.45;
          color:#dbe4f3;
          background:#111522;
          border:1px solid #222838;
          border-radius:10px;
          padding:10px 12px;
        }

        .dice-summary-subtitle{
          font-size:13px;
          font-weight:700;
          color:#9fb0cc;
          margin-top:2px;
        }

        .dice-model-list{
          display:flex;
          flex-direction:column;
          gap:6px;
        }

        .dice-model-row{
          display:flex;
          justify-content:space-between;
          align-items:center;
          gap:10px;
          padding:6px 8px;
          border-radius:8px;
          background:#101b31;
          border:1px solid #1d2940;
        }

        .dice-model-name{
          font-size:13px;
          color:#dbe4f3;
          line-height:1.2;
        }

        .dice-model-value{
          font-size:13px;
          font-weight:700;
          color:#86efac;
          white-space:nowrap;
        }

        .validation-legend-box{
          margin-top:2px;
          padding:10px 12px;
          border-radius:10px;
          background:#111522;
          border:1px solid #222838;
          height:80%;
          display:flex;
          flex-direction:column;
        }

        .validation-legend-title{
          font-size:13px;
          font-weight:700;
          color:#9fb0cc;
          margin-bottom:8px;
        }

        .validation-legend-list{
          display:flex;
          flex-direction:column;
          justify-content:space-evenly;
          flex:1;
        }

        .validation-legend-row{
          display:flex;
          align-items:center;
          gap:10px;
          min-width:0;
        }

        .validation-legend-swatch{
          width:14px;
          height:14px;
          border-radius:4px;
          border:1px solid rgba(255,255,255,.18);
          flex:0 0 14px;
        }

        .validation-legend-label{
          font-size:13px;
          color:#dbe4f3;
          line-height:1.2;
        }

        .validation-legend-tag{
          margin-left:auto;
          font-size:11px;
          color:#9fb0cc;
          background:#0f1522;
          border:1px solid #263047;
          border-radius:999px;
          padding:2px 7px;
          white-space:nowrap;
        }

        .validation-bottom-controls{
          display:flex;
          align-items:flex-end;
          gap:18px;
          margin-top:14px;
          padding-top:12px;
          border-top:1px solid #222838;
        }

        .validation-bottom-plane{
          flex:0 0 250px;
        }

        .validation-bottom-slice{
          flex:0 1 520px;
          min-width:320px;
          margin-left:300px;
          margin-right:250px;
        }

        .validation-bottom-checks{
          flex:0 0 320px;
          display:flex;
          flex-direction:column;
          justify-content:center;
          gap:4px;
          padding-bottom:6px;
        }

        .atlas-summary-box{
          padding:12px 14px;
          border-radius:12px;
          background:#111522;
          border:1px solid #222838;
          color:#dbe4f3;
          line-height:1.55;
          white-space:pre-wrap;
        }

        .atlas-table-wrap{
          overflow-x:auto;
        }

        .atlas-table{
          width:100%;
          border-collapse:collapse;
          font-size:13px;
          color:#dbe4f3;
        }

        .atlas-table th, .atlas-table td{
          padding:10px 8px;
          border-bottom:1px solid #1d2940;
          text-align:left;
          vertical-align:top;
        }

        .atlas-table th{
          color:#9fb0cc;
          font-weight:700;
          background:#101521;
        }

        .atlas-muted{
          color:#9aa3b6;
        }

        .atlas-small-box{
          padding:10px 12px;
          border-radius:10px;
          background:#111522;
          border:1px solid #222838;
          color:#dbe4f3;
          line-height:1.5;
          white-space:pre-wrap;
        }

        .nav-tabs .nav-link{
          color:#e6e8ee !important;
          background:transparent !important;
          border-color:transparent !important;
        }

        .nav-tabs .nav-link:hover,
        .nav-tabs .nav-link:focus{
          color:#ffffff !important;
          background:#111522 !important;
          border-color:#222838 !important;
        }

        .nav-tabs .nav-link.active,
        .nav-tabs .nav-item.show .nav-link{
          color:#ffffff !important;
          background:#0f1115 !important;
          border-color:#222838 #222838 #0f1115 !important;
        }

        .app-config-fab{
          position: fixed;
          left: 18px;
          bottom: 18px;
          z-index: 9998;
          width: 62px;
          height: 62px;
          border-radius: 999px !important;
          padding: 0 !important;
          display: flex !important;
          align-items: center;
          justify-content: center;
          background: #0f1522 !important;
          border: 1px solid #263047 !important;
          box-shadow: 0 8px 18px rgba(0,0,0,.35);
        }

        .app-config-fab:hover{
          transform: translateY(-1px);
          box-shadow: 0 0 0 1px rgba(255,255,255,.06), 0 10px 22px rgba(0,0,0,.32);
        }

        .app-config-fab-icon{
          width: 38px;
          height: 38px;
          object-fit: contain;
        }

        .delete-patient-modal-footer{
          width:100%;
          min-height:90px;
          display:flex;
          flex-direction:column;
          justify-content:flex-start;
          gap: 14px;
          padding:4px 0 0px 0;
        }

        .delete-csv-check{
          width:100%;
          display:flex;
          justify-content:flex-start;
          align-items:center;
          margin:0;
          padding-left:22px;
        }

        .delete-csv-check .form-check{
          display:flex !important;
          align-items:flex-start !important;
          justify-content:flex-start !important;
          gap:8px !important;
          margin:0 !important;
          padding:0 !important;
        }

        .delete-csv-check .form-check-input{
          width:18px !important;
          height:18px !important;
          min-width:18px !important;
          border:2px solid #111827 !important;
          background-color:#ffffff !important;
          margin:0 !important;
          cursor:pointer;
        }

        .delete-csv-check .form-check-input:checked{
          background-color:#2563eb !important;
          border-color:#2563eb !important;
        }

        .delete-csv-check .form-check-label,
        .delete-csv-check label{
          color:#111827 !important;
          font-weight:600 !important;
          font-size:15px !important;
          margin:0 !important;
          line-height:1.35 !important;
          user-select:none;
        }

        .delete-patient-buttons-row{
          width:100%;
          display:flex;
          justify-content:space-between;
          align-items:center;
          gap:20px;
        }

        .delete-patient-buttons-row .btn{
          min-width:120px;
        }
        """
    ),

    ui.tags.script("""
    (function() {
      function clamp(v, lo, hi){ return Math.max(lo, Math.min(hi, v)); }

      function getDims(which){
          const d = window.__tfg_dims__ || {};
          return d[which] || null;
      }

      function enabled(){
        return !!window.__tfg_crosshair_enabled__;
      }

      function attachClick(which, outputId){
          const root = document.getElementById(outputId);
          if (!root) return;

          root.addEventListener("click", function(ev){
            if (!enabled()) return;

            const img = root.querySelector("img");
            const dims = getDims(which);
            if (!img || !dims) return;

            const rect = img.getBoundingClientRect();
            const rx = (ev.clientX - rect.left) / rect.width;
            const ry = (ev.clientY - rect.top) / rect.height;

            const x = clamp(Math.floor(rx * dims.w), 0, dims.w - 1);
            const y = clamp(Math.floor(ry * dims.h), 0, dims.h - 1);

            if (window.Shiny && Shiny.setInputValue){
                Shiny.setInputValue("click_coords", { which: which, x: x, y: y }, {priority: "event"});
            }
          });
      }

      document.addEventListener("DOMContentLoaded", function(){
          attachClick("axial", "axial_plot");
          attachClick("coronal", "coronal_plot");
          attachClick("sagital", "sagital_plot");
      });
    })();
    """),

    ui.div(
        ui.card(
            ui.card_header("Explorar estudio"),
            ui.layout_columns(
                ui.input_select("patient_id", "Paciente (ID)", choices=[]),
                ui.input_select("study_date", "Fecha (YYYYMMDD)", choices=[]),
                ui.div(
                    ui.input_select("model", "Modelo", choices=list(MODEL_OPTIONS.keys()), selected="Radionics"),
                    ui.output_ui("weighted_heatmap_legend"),
                ),
                col_widths=[4, 4, 4],
                fill=False,
            ),
            ui.div(
                ui.strong("Volumen:"),
                ui.output_text("volume_result"),
                style="""
                    margin-top:6px;
                    padding:8px 12px;
                    border-radius:10px;
                    background:#111522;
                    border:1px solid #222838;
                    color:#e6e8ee;
                """,
            ),
            class_="study-card",
        ),

        ui.layout_columns(
            ui.card(
                ui.card_header(
                    ui.div(
                        {"style": "display:flex; justify-content:space-between; align-items:center; width:100%;"},
                        ui.span("Axial"),
                        ui.output_ui("seg_badge_axial"),
                    ),
                ),
                ui.output_plot("axial_plot"),
                ui.input_slider("axial_slice", "Corte (Z)", 0, 200, 100),
                class_="viewer-plot-card",
            ),
            ui.card(
                ui.card_header(
                    ui.div(
                        {"style": "display:flex; justify-content:space-between; align-items:center; width:100%;"},
                        ui.span("Coronal"),
                        ui.output_ui("seg_badge_coronal"),
                    ),
                ),
                ui.output_plot("coronal_plot"),
                ui.input_slider("coronal_slice", "Corte (Y)", 0, 200, 100),
                class_="viewer-plot-card",
            ),
            ui.card(
                ui.card_header(
                    ui.div(
                        {"style": "display:flex; justify-content:space-between; align-items:center; width:100%;"},
                        ui.span("Sagital"),
                        ui.output_ui("seg_badge_sagital"),
                    ),
                ),
                ui.output_plot("sagital_plot"),
                ui.input_slider("sagital_slice", "Corte (X)", 0, 200, 100),
                class_="viewer-plot-card",
            ),
            col_widths=[4, 4, 4],
            
        ),

        ui.card(
            ui.card_header("Estado de ejecución"),
            ui.output_ui("run_status_bar"),
            class_="status-card",
        ),
        ui.output_ui("storage_path_warning_ui"),
        ui.output_ui("study_integrity_alert"),
        ui.output_ui("import_alert_ui"),
        ui.output_ui("dims_js"),
        ui.output_ui("crosshair_js"),

        ui.div(
            ui.output_text("coords_readout"),
            class_="floating-coords",
            id="coords_panel",
        ),
    ),
    title="BTVolux",
)

val_ui = ui.page_sidebar(
    ui.sidebar(
        ui.div(
            ui.h4("Acciones"),
            ui.div(
                icon_button("browse_gt_btn", "seg_manual.png", "Cargar GT manual", tone="add"),
                icon_button("calc_dice", "calcular_DICE.png", "Calcular DICE", tone="analysis"),
                icon_button("download_dice_csv_study_btn", "descargar_volumen.png", "Descargar CSV DICE paciente", tone="download"),
                icon_button("download_dice_csv_all_btn", "descargar_varios.png", "Descargar CSV DICE varios", tone="download"),
                class_="action-button-stack",
            ),
        ),
        ui.hr(),
        ui.div(
            ui.h4("Explorar estudio"),
            ui.input_select("val_patient_id", "Paciente (ID)", choices=[]),
            ui.input_select("val_study_date", "Fecha", choices=[]),
            ui.input_select("val_model", "Modelo", choices=VAL_MODEL_LABELS, selected="Radionics"),
        ),
        width=340,
    ),

    ui.div(
        ui.card(
            ui.card_header("Visor de validación"),
            ui.div(
                ui.div(
                    ui.h5("Resultado DICE", style="margin-top:0; margin-bottom:12px;"),
                    ui.output_ui("pred_badge_validation"),
                    ui.div(style="height:8px;"),
                    ui.output_ui("manual_badge_validation"),
                    ui.div(style="height:14px;"),
                    ui.output_ui("dice_badge_validation"),
                    class_="validation-side-info",
                ),
                ui.div(
                    ui.output_plot("validation_plot"),
                    class_="validation-plot-wrap",
                ),
                ui.div(
                    ui.h5("Leyenda", style="margin-top:0; margin-bottom:12px;"),
                    ui.output_ui("validation_legend_ui"),
                    class_="validation-side-info",
                ),
                class_="validation-main-row",
            ),
            ui.div(
                ui.div(
                    ui.input_select(
                        "val_plane",
                        "Plano",
                        choices={"axial": "Axial", "coronal": "Coronal", "sagital": "Sagital"},
                        selected="axial",
                    ),
                    class_="validation-bottom-plane",
                ),
                ui.div(
                    ui.input_slider("val_slice", "Corte", 0, 200, 100),
                    class_="validation-bottom-slice",
                ),
                ui.div(
                    ui.input_checkbox("val_show_pred", "Mostrar segmentación del modelo", value=True),
                    ui.input_checkbox("val_show_gt", "Mostrar segmentación manual", value=True),
                    ui.input_checkbox("val_show_overlap", "Mostrar solape", value=True),
                    class_="validation-bottom-checks",
                ),
                class_="validation-bottom-controls",
            ),
            class_="validation-big",
        ),
        ui.card(
            ui.card_header("Estado de validación"),
            ui.output_ui("validation_status_bar"),
            class_="status-card",
        ),
    ),

    title="BTVolux",
)

atlas_ui = ui.page_sidebar(
    ui.sidebar(
        ui.div(
            ui.h4("Acciones"),
            ui.div(
                icon_button("analyze_atlas_btn", "calcular_atlas.png", "Analizar atlas", tone="analysis", extra_class="span-2"),
                class_="action-button-stack",
            ),
        ),
        ui.hr(),
        ui.div(
            ui.h4("Explorar estudio"),
            ui.input_select("atlas_patient_id", "Paciente (ID)", choices=[]),
            ui.input_select("atlas_study_date", "Fecha", choices=[]),
            ui.input_select("atlas_model", "Modelo", choices=VAL_MODEL_LABELS, selected="Radionics"),
        ),
        width=340,
    ),

    ui.div(
        ui.card(
            ui.card_header("Resumen anatómico"),
            ui.output_ui("atlas_summary_ui"),
        ),
        ui.card(
            ui.card_header("Regiones con solape"),
            ui.output_ui("atlas_regions_ui"),
        ),
        ui.card(
            ui.card_header("Registro / atlas"),
            ui.output_ui("atlas_registration_ui"),
        ),
        ui.card(
            ui.card_header("Estado del atlas"),
            ui.output_ui("atlas_status_bar"),
            class_="status-card",
        ),
    ),

    title="BTVolux",
)

app_ui = ui.page_fluid(
    app_config_bottom_left(),
    app_logo_top_right(),
    ui.navset_tab(
        ui.nav_panel("Segmentación", seg_ui),
        ui.nav_panel("Validación (DICE)", val_ui),
        ui.nav_panel("Atlas anatómico", atlas_ui),
        id="main_tabs",
    )
)


# ============================================================
# 10) SERVER 
# ============================================================

def server(input, output, session):
    """
    Implementa toda la lógica reactiva de la aplicación Shiny.

    Esta función coordina el estado de la interfaz, la carga de estudios,
    la ejecución de modelos, la validación mediante DICE y la actualización
    dinámica de los visores de imagen.

    Bloques principales
    -------------------
    1. Estado reactivo
       - Guarda en memoria la selección actual del usuario.
       - Mantiene cacheados los volúmenes NIfTI y las segmentaciones cargadas.
       - Almacena estados auxiliares de importación, ejecución, validación
         y mensajes de interfaz.

    2. Helpers internos
       - Gestionan logs y mensajes de estado.
       - Refrescan selectores de pacientes y estudios.
       - Recargan el visor principal.
       - Abren modales, borran estudios o pacientes y actualizan archivos
         auxiliares como `dicom_t1c_keywords.txt`.

    3. Inicialización
       - Restaura la última sesión guardada.
       - Inicializa las selecciones de segmentación y validación.
       - Intenta arrancar validación con el mismo paciente y estudio que
         segmentación cuando ese caso ya está procesado.

    4. Acciones de segmentación
       - Importación de estudios DICOM.
       - Ejecución de uno o varios modelos.
       - Descarga de segmentaciones.
       - Cálculo y exportación de volúmenes.

    5. Acciones de validación
       - Carga de segmentación manual.
       - Cálculo de DICE.
       - Exportación de resultados DICE.

    6. Reactivos automáticos
       - Sincronizan los selectores entre pestañas.
       - Recargan automáticamente el visor cuando cambia la selección.
       - Actualizan sliders, caches y estado visual según el caso cargado.

    7. Crosshair y navegación espacial
       - Convierte clicks sobre los visores 2D en coordenadas de vóxel 3D.
       - Sincroniza las tres vistas ortogonales con el punto seleccionado.

    8. Outputs
       - Renderizan plots, badges, barras de estado, avisos y textos
         mostrados en la interfaz.
    """

    # ===================== 1) ESTADO REACTIVO =====================
    # Guardamos en caché arrays NIfTI en memoria para evitar recargar desde disco
    # cada vez que el usuario mueve un slider.

    #Estado de importación
    import_dir = reactive.Value("")  # carpeta elegida para importar DICOM
    import_preview = reactive.Value("Ninguna carpeta seleccionada.")
    import_alert = reactive.Value("")
    import_mode = reactive.Value("single")   # "single" | "multi"
    import_status_text = reactive.Value("")
    sync_log = reactive.Value("")
    sync_status_msg = reactive.Value("")
    last_imported_pid = reactive.Value("")
    last_imported_date = reactive.Value("")

    # Estado carpetas raiz
    storage_patients_dir_text = reactive.Value(str(config.PACIENTES_NIFTI_DIR))
    storage_results_dir_text = reactive.Value(str(config.APP_RESULTS_DIR))
    storage_path_warning = reactive.Value("")

    #Estado de modales y borrados
    last_modal_message = reactive.Value("")
    modal_delete_pid = reactive.Value("")
    modal_delete_sdate = reactive.Value("")
    modal_delete_patient_pid = reactive.Value("")
    missing_nifti_warned_key = reactive.Value("")

    #Estado principal del visor de segmentación
    current_img_path = reactive.Value(None)  # type: ignore
    current_seg_path = reactive.Value(None)  # type: ignore
    img_data = reactive.Value(None)          # type: ignore
    seg_data = reactive.Value(None)          # type: ignore
    status_text = reactive.Value("")
    volume_text = reactive.Value("")
    study_volumes_cache = reactive.Value(None)
    viewer_case = reactive.Value(None)
    main_status_lines = reactive.Value([])
    download_status_msg = reactive.Value("")

    #Estado de ejecución de modelos
    run_is_busy = reactive.Value(False)
    run_thread = reactive.Value(None)
    run_job_info = reactive.Value(None)
    run_log_path = reactive.Value("")

    #Estado de coordenadas / crosshair
    crosshair_enabled = reactive.Value(True)
    #Punto 3D seleccionado para las coordenadas
    cross_voxel = reactive.Value(None)  # dict: {"x":int,"y":int,"z":int}

    #Estado de validación
    manual_gt_path = reactive.Value("")
    val_gt_data = reactive.Value(None)
    val_img_data = reactive.Value(None)
    val_pred_data = reactive.Value(None)
    dice_text = reactive.Value("")
    study_dices_cache = reactive.Value(None)
    validation_status_msg = reactive.Value("")
   
    last_seg_pid = reactive.Value("")
    last_val_pid = reactive.Value("")

    # Estado de atlas anatómico
    atlas_rows = reactive.Value([])
    atlas_summary_text = reactive.Value("")
    atlas_registration_info = reactive.Value(None)
    atlas_status_msg = reactive.Value("Selecciona paciente, estudio y modelo para analizar atlas.")
    atlas_last_case = reactive.Value(None)
    last_atlas_pid = reactive.Value("")

    # ===================== 2) HELPERS ESTADO Y LOGS =====================

    def _new_status_log(prefix: str) -> Path:
        """
        Genera la ruta de un nuevo archivo de log con marca temporal dentro de
        la carpeta de logs de la aplicación.
        """
        APP_LOGS_DIR.mkdir(parents=True, exist_ok=True)
        return APP_LOGS_DIR / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    def _push_main_status(msg: str) -> None:
        """
        Añade un mensaje al historial corto de estado mostrado en la interfaz.

        Solo conserva las tres líneas más recientes para evitar saturar el panel
        de estado.
        """
        msg = str(msg).strip()
        if not msg:
            return

        prev = list(main_status_lines.get() or [])
        prev.append(msg)
        main_status_lines.set(prev[-3:])
        status_text.set(msg)

    def _set_main_status(msg: str, append: bool = True) -> None:
        """
        Actualiza el estado principal de la app y, si hay un log activo asociado
        a una ejecución larga, escribe también el mensaje en dicho archivo.
        """
        msg = str(msg).strip()
        if not msg:
            return

        path_str = run_log_path.get()
        path = Path(path_str) if path_str else None

        # Solo escribir a archivo si ya hay un log activo (ejecución de modelos)
        if path is not None and str(path).strip():
            try:
                mode = "a" if append and path.exists() else "w"
                with open(path, mode, encoding="utf-8") as f:
                    if append and path.exists() and path.stat().st_size > 0:
                        f.write("\n")
                    f.write(msg)
            except Exception:
                pass

        _push_main_status(msg)

    def _reset_main_status(first_msg: str = "") -> None:
        """
        Limpia el estado principal en memoria y desactiva el log de archivo actual.

        Puede dejar opcionalmente un primer mensaje visible en el panel de estado.
        """
        run_log_path.set("")
        if first_msg:
            msg = first_msg.strip()
            main_status_lines.set([msg])
            status_text.set(msg)
        else:
            main_status_lines.set([])
            status_text.set("")

    def _start_run_file_log(prefix: str, first_msg: str = "") -> Path:
        """
        Inicia un log de archivo para una ejecución larga de modelos.

        Además de crear el archivo, lo registra como log activo y actualiza
        el estado visible en la interfaz.
        """
        path = _new_status_log(prefix)
        run_log_path.set(str(path))

        if first_msg:
            msg = first_msg.strip()
            with open(path, "w", encoding="utf-8") as f:
                f.write(msg)
            main_status_lines.set([msg])
            status_text.set(msg)
        else:
            with open(path, "w", encoding="utf-8") as f:
                f.write("")
            main_status_lines.set([])
            status_text.set("")

        return path

    def _set_import_status(msg: str) -> None:
        """
        Actualiza el mensaje de estado de importación y lo replica también en
        el panel principal de estado.
        """
        msg = str(msg).strip()
        import_status_text.set(msg)
        _push_main_status(msg)

    def _set_validation_status(msg: str, also_main: bool = False, reset_main: bool = False) -> None:
        """
        Actualiza el mensaje de estado mostrado en la pestaña de validación.

        Los parámetros `also_main` y `reset_main` están reservados para una posible
        ampliación futura, pero actualmente el estado se guarda solo en el canal
        de validación.
        """
        msg = str(msg).strip()
        validation_status_msg.set(msg)

    # ===================== 3) HELPERS SELECTS/SINCRONIZACION =====================
    def refresh_patient_choices(keep_selected: str = "") -> None:
        """
        Refresca el selector de pacientes (`input.patient_id`) leyendo los IDs
        disponibles en disco (PACIENTES_NIFTI_DIR).

        - Si `keep_selected` existe en la lista de pacientes, lo mantiene seleccionado.
        - Si no existe (p.ej. porque se borró o cambió), selecciona el primer paciente.
        - Si no hay pacientes, deja selección vacía.
        """
        pids = scan_nifti_patients()
        selected = keep_selected if keep_selected in pids else (pids[0] if pids else "")
        ui.update_select("patient_id", choices=pids, selected=selected)

    def refresh_study_choices(pid: str, keep_selected: str = "") -> None:
        """
        Refresca el selector de fechas de estudio (`input.study_date`) para un paciente.

        Comportamiento:
        - Si `pid` está vacío, limpia el selector (sin choices).
        - Obtiene la lista de estudios/fechas con `list_studies(pid)` (YYYYMMDD).
        - Si `keep_selected` sigue existiendo, lo mantiene; si no, selecciona la última
        fecha disponible (normalmente el estudio más reciente).
        """
        if not pid:
            ui.update_select("study_date", choices=[], selected="")
            return
        dates = list_studies(pid)
        # Mantener fecha anterior si existe; si no, seleccionar la más reciente (última).
        selected = keep_selected if keep_selected in dates else (dates[-1] if dates else "")
        ui.update_select("study_date", choices=dates, selected=selected)

    def refresh_validation_patient_choices(keep_selected: str = "") -> None:
        """
        La función obtiene la lista de pacientes que tienen al menos un estudio procesado
        mediante `scan_processed_patients()` y actualiza el input `val_patient_id` con
        esos valores. Si el paciente previamente seleccionado sigue estando disponible,
        lo mantiene seleccionado; en caso contrario, selecciona el primero de la lista.
        """
        pids = scan_processed_patients()
        selected = keep_selected if keep_selected in pids else (pids[0] if pids else "")
        ui.update_select("val_patient_id", choices=pids, selected=selected)

    def refresh_validation_study_choices(pid: str, keep_selected: str = "") -> None:
        """
        Actualiza el selector de fechas de estudio de la pestaña de validación para
        un paciente determinado.
        """
        if not pid:
            ui.update_select("val_study_date", choices=[], selected="")
            return
        dates = list_processed_studies(pid)
        selected = keep_selected if keep_selected in dates else (dates[-1] if dates else "")
        ui.update_select("val_study_date", choices=dates, selected=selected)

    def refresh_atlas_patient_choices(keep_selected: str = "") -> None:
        """
        Actualiza el selector de pacientes de la pestaña Atlas usando solo casos
        que tienen al menos una segmentación disponible.
        """
        pids = scan_processed_patients()
        selected = keep_selected if keep_selected in pids else (pids[0] if pids else "")
        ui.update_select("atlas_patient_id", choices=pids, selected=selected)

    def refresh_atlas_study_choices(pid: str, keep_selected: str = "") -> None:
        """
        Actualiza el selector de estudios procesados de la pestaña Atlas para el
        paciente seleccionado.
        """
        if not pid:
            ui.update_select("atlas_study_date", choices=[], selected="")
            return
        dates = list_processed_studies(pid)
        selected = keep_selected if keep_selected in dates else (dates[-1] if dates else "")
        ui.update_select("atlas_study_date", choices=dates, selected=selected)

    # ===================== 4) HELPERS LECTURA/CALCULO =====================
    # ---------- Load (paths + NIfTI + sliders) ----------
    def reload_viewer(pid: str, sdate: str, model_key: str) -> None:
        """
        Recarga el visor principal de segmentación.

        Si cambia solo el modelo dentro del mismo paciente/estudio, mantiene los
        slices actuales. Si cambia paciente o estudio, centra el visor.
        """
        valid_dates = list_studies(pid)

        if not pid or not sdate or sdate not in valid_dates:
            current_img_path.set(None)
            current_seg_path.set(None)
            img_data.set(None)
            seg_data.set(None)
            volume_text.set("")
            status_text.set("Listo para segmentar")
            viewer_case.set(None)
            return

        integ = study_integrity(pid, sdate)

        if integ["status"] == "error_missing_study_folder":
            current_img_path.set(None)
            current_seg_path.set(None)
            img_data.set(None)
            seg_data.set(None)
            volume_text.set("")
            status_text.set(str(integ["message"]))
            viewer_case.set(None)
            show_blocking_modal("Estudio incompleto", str(integ["message"]), kind="error")
            return

        if integ["status"] == "error_missing_nifti":
            current_img_path.set(None)
            current_seg_path.set(None)
            img_data.set(None)
            seg_data.set(None)
            volume_text.set("")
            status_text.set(str(integ["message"]))
            viewer_case.set(None)

            modal_delete_pid.set(pid)
            modal_delete_sdate.set(sdate)

            warn_key = f"{pid}|{sdate}|{model_key}|missing_nifti"
            if missing_nifti_warned_key.get() != warn_key:
                missing_nifti_warned_key.set(warn_key)
                show_blocking_modal(
                    "Estudio incompleto",
                    str(integ["message"]),
                    kind="error",
                    allow_delete_study=True,
                )
            return

        with reactive.isolate():
            previous_case = viewer_case.get()

        same_study = (
            isinstance(previous_case, dict)
            and previous_case.get("pid") == pid
            and previous_case.get("sdate") == sdate
        )

        imgp, segp = find_image_and_seg(pid, sdate, model_key)
        current_img_path.set(imgp)

        idata = load_nifti_data(imgp)

        if model_key == "media_ponderada":
            sdata = build_weighted_heatmap_for_study(pid, sdate, imgp)
            current_seg_path.set(segp)
        else:
            sdata = load_nifti_data(segp)
            current_seg_path.set(segp)

        img_data.set(idata)
        seg_data.set(sdata)

        if idata is not None:
            X, Y, Z = idata.shape

            if same_study:
                try:
                    with reactive.isolate():
                        ax_in = input.axial_slice()
                        co_in = input.coronal_slice()
                        sa_in = input.sagital_slice()

                    ax = max(0, min(int(ax_in), Z - 1))
                    co = max(0, min(int(co_in), Y - 1))
                    sa = max(0, min(int(sa_in), X - 1))

                except Exception:
                    ax = int(Z // 2)
                    co = int(Y // 2)
                    sa = int(X // 2)

                ui.update_slider("axial_slice", min=0, max=int(Z - 1), value=ax)
                ui.update_slider("coronal_slice", min=0, max=int(Y - 1), value=co)
                ui.update_slider("sagital_slice", min=0, max=int(X - 1), value=sa)

                with reactive.isolate():
                    old_cross = cross_voxel.get()

                if isinstance(old_cross, dict):
                    cross_voxel.set({
                        "x": max(0, min(int(old_cross.get("x", sa)), X - 1)),
                        "y": max(0, min(int(old_cross.get("y", co)), Y - 1)),
                        "z": max(0, min(int(old_cross.get("z", ax)), Z - 1)),
                    })
                else:
                    cross_voxel.set({"x": sa, "y": co, "z": ax})

            else:
                cx = int(X // 2)
                cy = int(Y // 2)
                cz = int(Z // 2)

                cross_voxel.set({"x": cx, "y": cy, "z": cz})

                ui.update_slider("axial_slice", min=0, max=int(Z - 1), value=cz)
                ui.update_slider("coronal_slice", min=0, max=int(Y - 1), value=cy)
                ui.update_slider("sagital_slice", min=0, max=int(X - 1), value=cx)

            viewer_case.set({"pid": pid, "sdate": sdate})

        if integ["status"] == "warning_missing_dicom":
            status_text.set(str(integ["message"]))
            show_blocking_modal("Falta DICOM del estudio", str(integ["message"]), kind="warning")
        else:
            status_text.set("Listo para segmentar")
            
    def _save_csv_with_dialog(src_csv: Path, default_name: str) -> str:
        """
        Abre un diálogo del sistema para que el usuario elija dónde guardar una
        copia de un CSV ya generado por la aplicación.
        """
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        dst_file = filedialog.asksaveasfilename(
            title="Guardar CSV como...",
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=[("CSV", "*.csv")],
        )
        root.destroy()

        if not dst_file:
            return "Descarga cancelada."

        dst_path = Path(dst_file)
        shutil.copy2(src_csv, dst_path)
        return f"CSV guardado en: {dst_path}"
    
    # ===================== 5) HELPERS MANTENIMIENTO =====================
    #---------- Boton para borrar el estudio si no hay NIFTI --------
    def delete_study(pid: str, sdate: str) -> bool:
        """
        Borra la carpeta completa del estudio:
        Pacientes_nifti/<pid>/<study_date>
        Si el paciente se queda sin estudios, borra también su carpeta.
        """
        try:
            study_root = PACIENTES_NIFTI_DIR / pid / str(sdate).strip()

            if study_root.exists() and study_root.is_dir():
                shutil.rmtree(study_root, ignore_errors=False)

            patient_root = PACIENTES_NIFTI_DIR / pid
            if patient_root.exists() and patient_root.is_dir():
                remaining_studies = [p for p in patient_root.iterdir() if p.is_dir()]
                if not remaining_studies:
                    patient_root.rmdir()

            delete_rows_from_csv(OUT_CSV, pid, sdate)
            delete_rows_from_csv(DICE_ALL_CSV, pid, sdate)

            return True
        except Exception as e:
            status_text.set(f"No se pudo borrar el estudio {pid}/{sdate}: {e}")
            return False
        
    def delete_patient(pid: str) -> bool:
        """
        Elimina por completo un paciente de la carpeta de estudios procesados,
        incluyendo todos sus estudios asociados.
        """
        try:
            patient_root = PACIENTES_NIFTI_DIR / pid
            if patient_root.exists() and patient_root.is_dir():
                shutil.rmtree(patient_root, ignore_errors=False)
            return True
        except Exception as e:
            status_text.set(f"No se pudo borrar el paciente {pid}: {e}")
            return False
        
    def append_t1c_keywords_to_txt(raw_text: str) -> tuple[bool, str]:
        """
        Añade una o varias keywords al archivo dicom_t1c_keywords.txt.
        Admite separadores por coma, punto y coma o salto de línea.
        Evita duplicados ignorando mayúsculas/minúsculas.
        """
        try:
            txt_path = Path(T1C_PROTOCOLS_TXT)
            txt_path.parent.mkdir(parents=True, exist_ok=True)

            raw_text = str(raw_text).strip()
            if not raw_text:
                return False, "Escribe al menos una palabra clave."

            parts = []
            normalized = raw_text.replace(";", "\n").replace(",", "\n")
            for piece in normalized.splitlines():
                token = piece.strip()
                if token:
                    parts.append(token)

            if not parts:
                return False, "No se detectó ninguna palabra clave válida."

            existing = []
            if txt_path.exists():
                existing = [
                    ln.strip() for ln in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                    if ln.strip()
                ]

            existing_lower = {x.lower() for x in existing}
            added = []

            for kw in parts:
                if kw.lower() not in existing_lower:
                    existing.append(kw)
                    existing_lower.add(kw.lower())
                    added.append(kw)

            txt_path.write_text("\n".join(existing) + "\n", encoding="utf-8")

            if added:
                return True, f"Keywords añadidas al dicom_t1c_keywords.txt: {', '.join(added)}"
            return True, "Las keywords ya estaban en dicom_t1c_keywords.txt."
        except Exception as e:
            return False, f"No se pudieron guardar las keywords T1c: {e}"

    # ---------- Crosshair mapping (click 2D -> voxel 3D) ----------
    def click_to_voxel(h: dict, vol_shape: tuple[int,int,int]) -> dict | None:
        """Convierte coordenadas de click en la imagen mostrada a voxel 3D.

        Importante:
          - En plot_slice aplicamos np.rot90 al slice para una orientación más natural en pantalla.
          - imshow tiene origen arriba-izquierda, por eso invertimos el eje vertical (Y-1 - y_disp).
          - Usamos dims_js para que JS conozca (w,h) de cada vista tras rot90.
        """
        X, Y, Z = vol_shape
        which = h.get("which")
        x_disp = int(h.get("x", 0))
        y_disp = int(h.get("y", 0))

        if which == "axial":
            z = int(input.axial_slice())
            x = max(0, min(x_disp, X-1))
            y = max(0, min((Y - 1 - y_disp), Y-1))
            return {"x": x, "y": y, "z": z}

        if which == "coronal":
            y = int(input.coronal_slice())
            x = max(0, min(x_disp, X-1))
            z = max(0, min((Z - 1 - y_disp), Z-1))
            return {"x": x, "y": y, "z": z}

        if which == "sagital":
            x = int(input.sagital_slice())
            y = max(0, min(x_disp, Y-1))
            z = max(0, min((Z - 1 - y_disp), Z-1))
            return {"x": x, "y": y, "z": z}

        return None


    # ===================== INIT (última sesión) =====================
    # Restaurar selección anterior al abrir la app.

    last = load_last_session()

    # ---------- Fix carpeta no disponible -------
    if not Path(config.PACIENTES_NIFTI_DIR).exists():
        storage_path_warning.set(
            f"La carpeta de estudios guardada no existe o no está disponible: {config.PACIENTES_NIFTI_DIR}. "
            "Abre configuración y selecciona otra ruta."
        )
    else:
        storage_path_warning.set("")

    # ---------- Pestaña Segmentación ----------
    pids0 = scan_nifti_patients()
    last_pid = last.get("patient_id", "")
    if last_pid not in pids0:
        last_pid = pids0[0] if pids0 else ""
    ui.update_select("patient_id", choices=pids0, selected=last_pid)

    last_date = ""
    if last_pid:
        dates = list_studies(last_pid)
        last_date = last.get("study_date", "")
        if last_date not in dates:
            last_date = dates[-1] if dates else ""
        ui.update_select("study_date", choices=dates, selected=last_date)
    else:
        ui.update_select("study_date", choices=[], selected="")

    model_labels = list(MODEL_OPTIONS.keys())
    last_model = last.get("model", "Radionics")
    if last_model not in model_labels:
        last_model = "Radionics"
    ui.update_select("model", choices=model_labels, selected=last_model)

    # ---------- Pestaña Validación ----------
    # Intentamos arrancar con el mismo paciente/fecha que en Segmentación,
    # pero solo si ese caso está procesado.
    val_pids0 = scan_processed_patients()

    val_pid0 = last_pid if last_pid in val_pids0 else (val_pids0[0] if val_pids0 else "")
    ui.update_select("val_patient_id", choices=val_pids0, selected=val_pid0)

    if val_pid0:
        val_dates0 = list_processed_studies(val_pid0)

        # Preferimos la misma fecha que en Segmentación si existe en validación
        val_date0 = last_date if last_date in val_dates0 else (val_dates0[-1] if val_dates0 else "")
        ui.update_select("val_study_date", choices=val_dates0, selected=val_date0)
    else:
        ui.update_select("val_study_date", choices=[], selected="")

    ui.update_select(
        "val_model",
        choices=VAL_MODEL_LABELS,
        selected=last_model if last_model in VAL_MODEL_LABELS else "Radionics",
    )

    # Estado inicial del visor de validación
    ui.update_select(
        "val_plane",
        choices={"axial": "Axial", "coronal": "Coronal", "sagital": "Sagital"},
        selected="axial",
    )
    ui.update_slider("val_slice", min=0, max=0, value=0)

    # ---------- Pestaña Atlas ----------
    atlas_pids0 = scan_processed_patients()
    atlas_pid0 = last_pid if last_pid in atlas_pids0 else (atlas_pids0[0] if atlas_pids0 else "")
    ui.update_select("atlas_patient_id", choices=atlas_pids0, selected=atlas_pid0)

    if atlas_pid0:
        atlas_dates0 = list_processed_studies(atlas_pid0)
        atlas_date0 = last_date if last_date in atlas_dates0 else (atlas_dates0[-1] if atlas_dates0 else "")
        ui.update_select("atlas_study_date", choices=atlas_dates0, selected=atlas_date0)
    else:
        ui.update_select("atlas_study_date", choices=[], selected="")

    ui.update_select(
        "atlas_model",
        choices=VAL_MODEL_LABELS,
        selected=last_model if last_model in VAL_MODEL_LABELS else "Radionics",
    )

    # ===================== 4) EVENTOS MODALES =====================
    #Helper
    def show_blocking_modal(
        title: str,
        message: str,
        kind: str = "error",
        allow_delete_study: bool = False,
        allow_add_t1c_keyword: bool = False,
    ):
        """
        Muestra un modal bloqueante para errores o avisos importantes.

        Puede incluir acciones adicionales según el contexto, como borrar un
        estudio incompleto o añadir nuevas keywords T1c.
        """
        key = f"{title}::{message}::delete={allow_delete_study}::t1c={allow_add_t1c_keyword}"
        if last_modal_message.get() == key:
            return

        last_modal_message.set(key)

        if kind == "warning":
            title_color = "#f8fafc"
            body_color = "#f8fafc"
            panel_bg = "#2117a1"
            border_color = "#2117a1"
        else:
            title_color = "#f8fafc"
            body_color = "#f8fafc"
            panel_bg = "#3a1616"
            border_color = "#b91c1c"

        footer_children = []

        if allow_add_t1c_keyword:
            footer_children.append(
                ui.div(
                    ui.input_text(
                        "modal_t1c_keyword_input",
                        "",
                        placeholder="Ej: t1c, post-contrast, gd, contrast enhanced",
                        width="320px",
                    ),
                    ui.input_action_button("modal_add_t1c_keyword_btn", "Añadir palabra T1c"),
                    style="display:flex; gap:10px; align-items:end; flex-wrap:wrap; margin-right:auto;",
                )
            )

        if allow_delete_study:
            footer_children.append(
                ui.input_action_button("modal_delete_study_btn", "Borrar estudio")
            )

        footer_children.append(
            ui.input_action_button("modal_close_btn", "Cerrar")
        )

        ui.modal_show(
            ui.modal(
                ui.div(
                    ui.h4(
                        title,
                        style=f"margin:0 0 12px 0; color:{title_color}; font-weight:700;"
                    ),
                    ui.div(
                        message,
                        style=f"color:{body_color}; white-space:pre-wrap; line-height:1.5;"
                    ),
                    style=(
                        f"background:{panel_bg}; "
                        f"border:1px solid {border_color}; "
                        f"border-radius:12px; "
                        f"padding:16px;"
                    ),
                ),
                title="",
                easy_close=False,
                footer=ui.div(
                    *footer_children,
                    style="display:flex; justify-content:flex-end; gap:10px; flex-wrap:wrap; width:100%;",
                ),
            )
        )
    
    def show_storage_config_modal():
        ui.modal_show(
            ui.modal(
                ui.div(
                    ui.h4(
                        "Configuración de rutas",
                        style="margin:0 0 12px 0; color:#f8fafc; font-weight:700;",
                    ),
                    ui.div(
                        "Aquí puedes cambiar la carpeta raíz de estudios y la carpeta de resultados. "
                        "Los cambios se guardarán para futuros arranques de la app.",
                        style="color:#d7dcea; line-height:1.5; margin-bottom:14px;",
                    ),
                    ui.div(
                        "Si una ruta guardada no existe porque el disco externo no está conectado, puedes cambiarla aquí.",
                        style=(
                            "color:#fde68a; "
                            "background:#3b2f12; "
                            "border:1px solid #a16207; "
                            "border-radius:10px; "
                            "padding:10px 12px; "
                            "margin-bottom:14px;"
                        ),
                    ),
                    ui.input_text(
                        "storage_patients_dir_input",
                        "Carpeta Pacientes_nifti",
                        value=storage_patients_dir_text.get(),
                        width="100%",
                    ),
                    ui.div(style="height:8px;"),
                    ui.input_action_button("browse_storage_patients_dir_btn", "Elegir carpeta estudios"),
                    ui.div(style="height:14px;"),
                    ui.input_text(
                        "storage_results_dir_input",
                        "Carpeta de resultados",
                        value=storage_results_dir_text.get(),
                        width="100%",
                    ),
                    ui.div(style="height:8px;"),
                    ui.input_action_button("browse_storage_results_dir_btn", "Elegir carpeta resultados"),
                    style=(
                        "background:#101521; "
                        "border:1px solid #222838; "
                        "border-radius:12px; "
                        "padding:16px;"
                    ),
                ),
                title="",
                easy_close=False,
                footer=ui.div(
                    ui.input_action_button("save_storage_config_btn", "Guardar"),
                    ui.input_action_button("modal_close_btn", "Cancelar"),
                    style="display:flex; justify-content:flex-end; gap:10px;",
                ),
            )
        )
    #Eventos modales
    @reactive.Effect
    @reactive.event(input.modal_close_btn)
    def _close_modal():
        ui.modal_remove()

    @reactive.Effect
    @reactive.event(input.open_storage_config_btn)
    def _open_storage_config():
        show_storage_config_modal()

    @reactive.Effect
    @reactive.event(input.modal_add_t1c_keyword_btn)
    def _modal_add_t1c_keyword():
        raw_text = input.modal_t1c_keyword_input()

        ok, msg = append_t1c_keywords_to_txt(raw_text)

        if ok:
            import_alert.set(
                "Palabra(s) T1c añadida(s) correctamente. Cierra este aviso y vuelve a importar el estudio."
            )
            _push_main_status(msg)
            ui.modal_remove()
            last_modal_message.set("")
        else:
            _push_main_status(msg)
            show_blocking_modal(
                "No se pudo añadir la keyword T1c",
                msg,
                kind="error",
                allow_add_t1c_keyword=True,
            )

    @reactive.Effect
    @reactive.event(input.confirm_delete_patient_btn)
    def _confirm_delete_patient():
        pid = modal_delete_patient_pid.get()
        ui.modal_remove()

        if not pid:
            return

        delete_csv_rows = bool(input.delete_patient_csv_rows())
        ok = delete_patient(pid)

        if ok and delete_csv_rows:
            n_vol = delete_rows_from_csv(OUT_CSV, pid)
            n_dice = delete_rows_from_csv(DICE_ALL_CSV, pid)
            _push_main_status(f"Filas eliminadas de CSV: volumenes={n_vol}, dice={n_dice}")

        modal_delete_patient_pid.set("")

        if not ok:
            return

        _push_main_status(f"Paciente eliminado: {pid}")

        refresh_patient_choices(keep_selected="")
        new_pid = input.patient_id() or ""
        refresh_study_choices(new_pid, keep_selected="")

        refresh_validation_patient_choices(keep_selected="")
        new_val_pid = input.val_patient_id() or ""
        refresh_validation_study_choices(new_val_pid, keep_selected="")

        refresh_atlas_patient_choices(keep_selected="")
        new_atlas_pid = input.atlas_patient_id() or ""
        refresh_atlas_study_choices(new_atlas_pid, keep_selected="")

        current_img_path.set(None)
        current_seg_path.set(None)
        img_data.set(None)
        seg_data.set(None)
        val_img_data.set(None)
        val_pred_data.set(None)
        volume_text.set("")

    @reactive.Effect
    @reactive.event(input.browse_storage_patients_dir_btn)
    def _browse_storage_patients_dir():
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        folder = filedialog.askdirectory(title="Selecciona la carpeta raíz de estudios")
        root.destroy()

        if folder:
            storage_patients_dir_text.set(folder)
            ui.update_text("storage_patients_dir_input", value=folder)


    @reactive.Effect
    @reactive.event(input.browse_storage_results_dir_btn)
    def _browse_storage_results_dir():
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        folder = filedialog.askdirectory(title="Selecciona la carpeta de resultados")
        root.destroy()

        if folder:
            storage_results_dir_text.set(folder)
            ui.update_text("storage_results_dir_input", value=folder)

    @reactive.Effect
    @reactive.event(input.save_storage_config_btn)
    def _save_storage_config():
        try:
            patients_dir = Path(str(input.storage_patients_dir_input()).strip()).expanduser()
            results_dir = Path(str(input.storage_results_dir_input()).strip()).expanduser()

            if not str(patients_dir).strip():
                raise ValueError("La carpeta de estudios no puede estar vacía.")
            if not str(results_dir).strip():
                raise ValueError("La carpeta de resultados no puede estar vacía.")

            patients_dir.mkdir(parents=True, exist_ok=True)
            results_dir.mkdir(parents=True, exist_ok=True)

            saved = config.save_storage_settings(
                pacientes_nifti_dir=patients_dir,
                results_dir=results_dir,
            )

            storage_patients_dir_text.set(str(saved["pacientes_nifti_dir"]))
            storage_results_dir_text.set(str(saved["results_dir"]))

            ui.modal_remove()
            show_blocking_modal(
                "Configuración guardada",
                "Las nuevas rutas se han guardado correctamente.\n\nReinicia la app para aplicar los cambios en todos los módulos.",
                kind="warning",
            )
        except Exception as e:
            show_blocking_modal(
                "Error guardando configuración",
                str(e),
                kind="error",
            )

    @reactive.Effect
    @reactive.event(input.patient_id, input.study_date, input.model)
    def _reset_modal_guard_on_seg_selection_change():
        last_modal_message.set("")

    @reactive.Effect
    @reactive.event(input.val_patient_id, input.val_study_date, input.val_model)
    def _reset_modal_guard_on_val_selection_change():
        last_modal_message.set("")

    @reactive.Effect
    @reactive.event(input.val_patient_id, input.val_study_date, input.val_model)
    def _clear_validation_messages_on_selection_change():
        dice_text.set("")
        study_dices_cache.set(None)

    @reactive.Effect
    @reactive.event(input.atlas_patient_id, input.atlas_study_date, input.atlas_model)
    def _clear_atlas_on_selection_change():
        atlas_rows.set([])
        atlas_summary_text.set("")
        atlas_registration_info.set(None)
        atlas_last_case.set(None)
        atlas_status_msg.set("Selecciona paciente, estudio y modelo para analizar atlas.")
    # ===================== 5) ACCIONES =====================

    # 1. Acciones segmentacion
    # ---- Sync DICOM -> NIfTI  ----
    def _import_from_dialog(mode: str):
        """
        Gestiona la importación de estudios DICOM desde una carpeta elegida por
        el usuario.

        Modos:
        - 'single': importa un único estudio o paciente.
        - 'multi': interpreta cada subcarpeta como un paciente distinto.

        Durante la importación:
        - detecta fecha de estudio,
        - valida si el estudio parece T1c,
        - copia DICOM,
        - convierte a NIfTI,
        - actualiza los selectores y el visor si la importación termina bien.
        """
        try:
            import tkinter as tk
            from tkinter import filedialog

            _set_import_status("Seleccionando carpeta para importar...")

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)

            folder = filedialog.askdirectory(
                title="Selecciona carpeta del paciente" if mode == "single" else "Selecciona carpeta raíz con varios pacientes"
            )
            root.destroy()

            if not folder:
                _set_import_status("Importación cancelada.")
                return

            import_dir.set(folder)
            import_preview.set(folder)
            import_mode.set(mode)
            import_alert.set("")
            last_modal_message.set("")

            root_path = Path(folder)

            if mode == "multi":
                subdirs = [p for p in root_path.iterdir() if p.is_dir()]
                total = len(subdirs)

                if total == 0:
                    _set_import_status("No se encontraron carpetas de pacientes para importar.")
                    return

                _reset_main_status(f"Detectados {total} pacientes para importar.")
                _set_main_status(f"Iniciando importación múltiple de {total} pacientes...")

                actions = []
                t1c_errors = []
                other_errors = []

                for idx, pdir in enumerate(sorted(subdirs), start=1):
                    pid = pdir.name
                    _set_main_status(f"[{idx}/{total}] Analizando paciente {pid}...")

                    try:
                        detected_date = get_dicom_study_date(pdir)
                        _set_main_status(f"[{idx}/{total}] Fecha detectada para {pid}: {detected_date}")
                    except Exception as e:
                        detected_date = "??????"
                        act = f"error: {e}"
                        actions.append((pid, detected_date, act))
                        other_errors.append(f"{pid} / {detected_date}: {act}")
                        _set_main_status(f"[{idx}/{total}] Error detectando fecha de {pid}: {e}")
                        continue

                    try:
                        _set_main_status(f"[{idx}/{total}] Preparando estudio {pid}/{detected_date}...")
                        study_date, _P = prepare_study_from_dicom(pid, dicom_source_dir=str(pdir), force_copy=False)

                        _set_main_status(f"[{idx}/{total}] Convirtiendo DICOM a NIfTI para {pid}/{study_date}...")
                        convertir_dicom_a_nifti(pid, study_date, force=False)

                        act = "created"
                        actions.append((pid, study_date, act))
                        _set_main_status(f"[{idx}/{total}] Importado correctamente: {pid}/{study_date}")
                    except Exception as e:
                        act = f"error: {e}"
                        actions.append((pid, detected_date, act))
                        if "t1c" in act.lower() or "post-contraste" in act.lower():
                            t1c_errors.append(f"{pid} / {detected_date}: {act}")
                            _set_main_status(f"[{idx}/{total}] Rechazado por T1c: {pid}/{detected_date}")
                        else:
                            other_errors.append(f"{pid} / {detected_date}: {act}")
                            _set_main_status(f"[{idx}/{total}] Error importando {pid}/{detected_date}: {e}")

                finished_ok = sum(1 for _pid, _d, act in actions if str(act).lower() == "created")
                _set_main_status(f"Importación múltiple terminada: {finished_ok}/{total} pacientes importados.")
                _set_import_status(f"Importación terminada: {finished_ok}/{total} pacientes importados.")

            else:
                pid_hint = root_path.name
                _reset_main_status(f"Importación individual iniciada para: {pid_hint}")
                _set_main_status(f"Analizando carpeta seleccionada: {pid_hint}...")

                actions = sync_from_selected_folder(root_path, default_pid="")
                for pid, d, act in actions:
                    if str(act).lower() == "created":
                        _set_main_status(f"Importado correctamente: {pid}/{d}")
                    else:
                        _set_main_status(f"Error en importación de {pid}/{d}: {act}")
                t1c_errors = []
                other_errors = []

                for pid, d, act in actions:
                    act_low = str(act).lower()
                    if act_low.startswith("error:"):
                        if "t1c" in act_low or "post-contraste" in act_low:
                            t1c_errors.append(f"{pid} / {d}: {act}")
                        else:
                            other_errors.append(f"{pid} / {d}: {act}")

                if actions:
                    pid0, d0, act0 = actions[0]
                    if str(act0).lower() == "created":
                        _set_import_status(f"Importación terminada: {pid0} / {d0}")
                    else:
                        _set_import_status(f"Importación terminada con errores: {pid0} / {d0}")

            if t1c_errors:
                msg = (
                    "Importación rechazada porque el estudio no parece T1c/post-contraste.\n\n"
                    "Si sabes que este estudio sí es válido, añade una o varias palabras clave T1c "
                    "al dicom_t1c_keywords.txt desde este aviso y vuelve a importarlo.\n\n"
                    + "\n".join(t1c_errors)
                )
                import_alert.set(msg)
                show_blocking_modal(
                    "Estudio no válido para segmentación",
                    msg,
                    kind="error",
                    allow_add_t1c_keyword=True,
                )

            lines = [f"- {pid} / {d}: {act}" for pid, d, act in actions]
            sync_log.set("\n".join(lines) if lines else "No se importó nada.")

            created_actions = [(pid, d, act) for pid, d, act in actions if str(act).lower() == "created"]
            if created_actions:
                last_pid_imported, last_date_imported, _ = created_actions[-1]
                last_imported_pid.set(last_pid_imported)
                last_imported_date.set(last_date_imported)

                refresh_patient_choices(keep_selected=last_pid_imported)
                refresh_study_choices(last_pid_imported, keep_selected=last_date_imported)
                ui.update_select("patient_id", selected=last_pid_imported)
                ui.update_select("study_date", selected=last_date_imported)

                refresh_validation_patient_choices(keep_selected=last_pid_imported)
                refresh_validation_study_choices(last_pid_imported, keep_selected=last_date_imported)
                ui.update_select("val_patient_id", selected=last_pid_imported)
                ui.update_select("val_study_date", selected=last_date_imported)

                refresh_atlas_patient_choices(keep_selected=last_pid_imported)
                refresh_atlas_study_choices(last_pid_imported, keep_selected=last_date_imported)
                ui.update_select("atlas_patient_id", selected=last_pid_imported)
                ui.update_select("atlas_study_date", selected=last_date_imported)

                model_key = MODEL_OPTIONS[input.model()]
                reload_viewer(last_pid_imported, last_date_imported, model_key)

        except Exception as e:
            _set_import_status(f"Error al importar: {e}")


    @reactive.Effect
    @reactive.event(input.add_patient_btn)
    def _add_patient_clicked():
        _import_from_dialog("single")


    @reactive.Effect
    @reactive.event(input.add_patients_btn)
    def _add_patients_clicked():
        _import_from_dialog("multi")

    @reactive.Effect
    @reactive.event(input.delete_patient_btn)
    def _delete_patient_clicked():
        pid = input.patient_id()
        if not pid:
            status_text.set("Selecciona un paciente para eliminar.")
            return

        modal_delete_patient_pid.set(pid)

        ui.modal_show(
            ui.modal(
                ui.div(
                    ui.h4(
                        "Confirmar borrado de paciente",
                        style="margin:0 0 12px 0; color:#f8fafc; font-weight:700;",
                    ),
                    ui.div(
                        f"Vas a eliminar el paciente {pid} con todos sus estudios. Esta acción no se puede deshacer.\n\n¿Seguro que quieres continuar?",
                        style="color:#f8fafc; white-space:pre-wrap; line-height:1.5;",
                    ),
                    style=(
                        "background:#3a1616; "
                        "border:1px solid #b91c1c; "
                        "border-radius:12px; "
                        "padding:16px;"
                    ),
                ),
                title="",
                easy_close=False,
                footer=ui.div(
                    ui.div(
                        ui.input_checkbox(
                            "delete_patient_csv_rows",
                            "Eliminar también sus filas de volumenes.csv y dice.csv",
                            value=True,
                        ),
                        class_="delete-csv-check",
                    ),

                    ui.div(
                        ui.input_action_button("confirm_delete_patient_btn", "Sí, eliminar"),
                        ui.input_action_button("modal_close_btn", "Cancelar"),
                        class_="delete-patient-buttons-row",
                    ),

                    class_="delete-patient-modal-footer",
                ),
                size="m",
            )
        )

    # ---- Ejecutar modelo/pipeline ----
    @reactive.Effect
    @reactive.event(input.run_btn)
    def _run_clicked():
        pid = input.patient_id()
        sdate = input.study_date()
        model_key = MODEL_OPTIONS[input.model()]

        if not pid or not sdate:
            status_text.set("Selecciona paciente y estudio.")
            return

        if model_key == "media_ponderada":
            try:
                _reset_main_status(f"Generando segmentación NIfTI de media ponderada para {pid}/{sdate}...")

                seg_path = build_weighted_ensemble_seg_for_study(
                    pid,
                    sdate,
                    threshold=0.5,
                    force=True,
                )

                if seg_path is None:
                    _set_main_status("No se pudo generar la media ponderada: hacen falta al menos dos segmentaciones disponibles.")
                    return

                _set_main_status(f"Segmentación media ponderada guardada: {seg_path.name}")
                reload_viewer(pid, sdate, model_key)

            except Exception as e:
                _set_main_status(f"Error generando media ponderada: {e}")

            return

        log_file = _start_run_file_log(
            "run",
            f"Ejecutando {model_key} en {pid}/{sdate}...",
        )

        log_file = Path(log_file)

        run_is_busy.set(True)
        status_text.set(f"⏳ Ejecutando {model_key} en {pid}/{sdate}...")

        def worker():
            try:
                if model_key == "radionics":
                    errors = pipeline_study(
                        pid, sdate,
                        mode="radionics",
                        progress_log_path=str(log_file),
                    )

                elif model_key == "nnunet_task501":
                    errors = pipeline_study(
                        pid, sdate,
                        mode="nnunet",
                        run_501=True,
                        progress_log_path=str(log_file),
                    )

                elif model_key in {"agunet", "dagunet", "pls-net", "unet-fv", "unet-slabs"}:
                    errors = pipeline_study(
                        pid, sdate,
                        mode=model_key,
                        progress_log_path=str(log_file),
                    )

                elif model_key == "all":
                    errors = pipeline_study(
                        pid, sdate,
                        mode="all",
                        run_501=True, run_902=False, run_903=False, run_904=False,
                        progress_log_path=str(log_file),
                    )
                else:
                    errors = [f"Modelo no soportado: {model_key}"]

                if errors:
                    with open(log_file, "w", encoding="utf-8") as f:
                        f.write("Terminado con fallos")
                else:
                    with open(log_file, "w", encoding="utf-8") as f:
                        f.write("Segmentación lista")

            except Exception as e:
                try:
                    with open(log_file, "w", encoding="utf-8") as f:
                        f.write(f"[APP ERROR] {e}")
                except Exception:
                    pass

        t = threading.Thread(target=worker, daemon=True)
        run_thread.set(t)
        run_job_info.set((pid, sdate, model_key))
        t.start()

    @reactive.Effect
    @reactive.event(input.run_all_btn)
    def _run_all_clicked():
        pid = input.patient_id()
        sdate = input.study_date()
        model_key = "all"

        if not pid or not sdate:
            status_text.set("Selecciona paciente y estudio.")
            return

        log_file = _start_run_file_log(
            "run_all",
            f"Ejecutando todos los modelos en {pid}/{sdate}...",
        )
        log_file = Path(log_file)

        run_is_busy.set(True)
        status_text.set(f" Ejecutando todos los modelos en {pid}/{sdate}...")

        def worker():
            try:
                errors = pipeline_study(
                    pid,
                    sdate,
                    mode="all",
                    run_501=True,
                    progress_log_path=str(log_file),
                )

                if errors:
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write("\nTerminado con fallos")
                else:
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write("\nTodos los modelos terminados")
            except Exception as e:
                try:
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write(f"\n[APP ERROR] {e}")
                except Exception:
                    pass

        t = threading.Thread(target=worker, daemon=True)
        run_thread.set(t)
        run_job_info.set((pid, sdate, model_key))
        t.start()

    @reactive.Effect
    @reactive.event(input.confirm_run_all_patients_btn)
    def _confirm_run_all_patients():
        if run_is_busy.get():
            ui.modal_remove()
            status_text.set("Ya hay una ejecución en curso.")
            return

        ui.modal_remove()

        current_pid = input.patient_id() or ""
        current_sdate = input.study_date() or ""

        log_file = _start_run_file_log(
            "run_all_patients",
            "Iniciando ejecución de todos los modelos en todos los estudios disponibles...",
        )
        log_file = Path(log_file)

        run_is_busy.set(True)
        status_text.set(" Ejecutando todos los modelos en todos los estudios disponibles...")

        def worker():
            try:
                errors = pipeline_all_existing_studies(
                    mode="all",
                    skip_completed=True,
                    run_501=True,
                    progress_log_path=str(log_file),
                )

                if errors:
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write("\nTerminado con fallos")
                else:
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write("\nLote terminado correctamente")
            except Exception as e:
                try:
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write(f"\n[APP ERROR] {e}")
                except Exception:
                    pass

        t = threading.Thread(target=worker, daemon=True)
        run_thread.set(t)
        run_job_info.set((current_pid, current_sdate, "all_patients"))
        t.start()

    @reactive.Effect
    @reactive.event(input.run_all_patients_models_btn)
    def _run_all_patients_models_clicked():
        if run_is_busy.get():
            status_text.set("Ya hay una ejecución en curso.")
            return

        pids = scan_nifti_patients()
        total_patients = len(pids)
        total_studies = sum(len(list_studies(pid)) for pid in pids)

        if total_patients == 0 or total_studies == 0:
            status_text.set("No hay pacientes/estudios disponibles en Pacientes_nifti.")
            return

        ui.modal_show(
            ui.modal(
                ui.div(
                    ui.h4(
                        "Confirmar ejecución masiva",
                        style="margin:0 0 12px 0; color:#f8fafc; font-weight:700;",
                    ),
                    ui.div(
                        (
                            f"Se van a ejecutar todos los modelos en {total_studies} estudios "
                            f"de {total_patients} pacientes disponibles en la carpeta.\n\n"
                            "Los modelos que ya estén hechos se saltarán automáticamente.\n\n"
                            "Esta operación puede tardar bastante. ¿Seguro que quieres continuar?"
                        ),
                        style="color:#f8fafc; white-space:pre-wrap; line-height:1.5;",
                    ),
                    style=(
                        "background:#15263f; "
                        "border:1px solid #2b4d7c; "
                        "border-radius:12px; "
                        "padding:16px;"
                    ),
                ),
                title="",
                easy_close=False,
                footer=ui.div(
                    ui.input_action_button("confirm_run_all_patients_btn", "Sí, ejecutar"),
                    ui.input_action_button("modal_close_btn", "Cancelar"),
                    style="display:flex; justify-content:flex-end; gap:10px;",
                ),
            )
        )

    @reactive.Effect
    @reactive.event(input.download_seg_btn)
    def _download_seg_clicked():
        src = current_seg_path.get()

        if not src or not Path(src).exists():
            download_status_msg.set("No hay segmentación cargada para descargar.")
            _push_main_status("No hay segmentación cargada para descargar.")
            return

        try:
            import tkinter as tk
            from tkinter import filedialog

            pid = input.patient_id() or "paciente"
            sdate = input.study_date() or "fecha"
            model_label = input.model().replace(" ", "_").replace("(", "").replace(")", "")

            src_path = Path(src)
            if src_path.suffixes[-2:] == [".nii", ".gz"]:
                default_ext = ".nii.gz"
            else:
                default_ext = src_path.suffix or ".nii.gz"

            default_name = f"{pid}_{sdate}_{model_label}_seg{default_ext}"

            _push_main_status(f"Preparando descarga de segmentación: {pid}/{sdate}")

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            dst_file = filedialog.asksaveasfilename(
                title="Guardar segmentación como...",
                defaultextension=default_ext,
                initialfile=default_name,
                filetypes=[("NIfTI comprimido", "*.nii.gz"), ("NIfTI", "*.nii")],
            )
            root.destroy()

            if not dst_file:
                download_status_msg.set("Descarga cancelada.")
                _push_main_status("Descarga de segmentación cancelada.")
                return

            dst_path = Path(dst_file)
            shutil.copy2(src_path, dst_path)
            download_status_msg.set(f"Segmentación guardada en: {dst_path}")
            _push_main_status(f"Segmentación guardada: {dst_path.name}")

        except Exception as e:
            download_status_msg.set(f"Error al guardar la segmentación: {e}")
            _push_main_status(f"Error al guardar la segmentación: {e}")
    
    @reactive.Effect
    @reactive.event(input.calc_vol_btn)
    def _calc_volume_clicked():
        pid = input.patient_id()
        sdate = input.study_date()
        model_key = MODEL_OPTIONS[input.model()]

        if not pid or not sdate:
            volume_text.set("Selecciona paciente y estudio.")
            return

        try:
            existing = load_study_row_from_csv(
                OUT_CSV,
                pid,
                sdate,
                required_value_fields=(
                    "radionics",
                    "agunet",
                    "dagunet",
                    "pls-net",
                    "unet-fv",
                    "unet-slabs",
                    "nnunet_task501",
                ),
            )

            if existing is not None and csv_value_present(existing, "media_ponderada"):
                study_volumes_cache.set(existing)
                volume_text.set(format_volume_text(model_key, existing))
                _reset_main_status(f"Volumen ya calculado para {pid}/{sdate}. Mostrando valor guardado.")
                return

            if existing is not None and not csv_value_present(existing, "media_ponderada"):
                _reset_main_status(
                    f"Volumen existente para {pid}/{sdate}, pero falta media ponderada. Recalculando..."
                )
            else:
                _reset_main_status(f"Calculando volumen para {pid}/{sdate}...")

            study_vols = compute_study_volumes(pid, sdate)
            study_volumes_cache.set(study_vols)

            _set_main_status("Guardando volumen del estudio en el CSV global acumulado...")
            save_study_volumes_to_global_csv(study_vols)

            volume_text.set(format_volume_text(model_key, study_vols))
            _set_main_status(f"Volumen calculado para {pid}/{sdate}")

            if model_key == "media_ponderada":
                reload_viewer(pid, sdate, model_key)

        except Exception as e:
            volume_text.set(f"Error al calcular volumen ({model_key}): {e}")
            _set_main_status(f"Error al calcular volumen: {e}")
    
    @reactive.Effect
    @reactive.event(input.download_csv_study_btn)
    def _download_csv_study_clicked():
        pid = input.patient_id()
        sdate = input.study_date()

        if not pid or not sdate:
            download_status_msg.set("Selecciona paciente y estudio antes de descargar el CSV.")
            return

        try:
            cached = study_volumes_cache.get()
            if isinstance(cached, dict) and cached.get("paciente_id") == pid and cached.get("study_date") == sdate:
                study_vols = cached
            else:
                _reset_main_status(f"Calculando volumen para exportar {pid}/{sdate}...")
                study_vols = compute_study_volumes(pid, sdate)
                study_volumes_cache.set(study_vols)
                save_study_volumes_to_global_csv(study_vols)

            tmp_csv = APP_RESULTS_DIR / f"volumenes_{pid}_{sdate}.csv"
            csv_path = build_single_study_volumes_csv(study_vols, tmp_csv)
            download_status_msg.set(_save_csv_with_dialog(csv_path, f"volumenes_{pid}_{sdate}.csv"))
            _set_main_status(f"CSV del estudio preparado: {pid}/{sdate}")
        except Exception as e:
            download_status_msg.set(f"Error al generar el CSV del estudio: {e}")
            _set_main_status(f"Error al exportar CSV del estudio: {e}")


    @reactive.Effect
    @reactive.event(input.download_csv_all_btn)
    def _download_csv_all_clicked():
        try:
            if not OUT_CSV.exists():
                download_status_msg.set("Aún no hay volúmenes calculados para exportar.")
                return

            download_status_msg.set(_save_csv_with_dialog(OUT_CSV, "volumenes_todos_los_pacientes.csv"))
            _set_main_status("CSV global de volúmenes exportado.")
        except Exception as e:
            download_status_msg.set(f"Error al generar el CSV global: {e}")
            _set_main_status(f"Error al exportar CSV global de volúmenes: {e}")

    @reactive.Effect
    @reactive.event(input.toggle_coords_btn)
    def _toggle_coords_clicked():
        crosshair_enabled.set(not crosshair_enabled.get())

    # 2. Acciones validacion
    # ---- Botón buscar carpeta ----
    @reactive.Effect
    @reactive.event(input.browse_gt_btn)
    def _browse_gt():
        import tkinter as tk
        from tkinter import filedialog

        pid = input.val_patient_id()
        sdate = input.val_study_date()

        if not pid or not sdate:
            _set_validation_status("Selecciona antes paciente y estudio.")
            return

        _set_validation_status("Seleccionando segmentación manual...")

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        file = filedialog.askopenfilename(
            title="Seleccionar segmentación manual",
            filetypes=[("NIfTI", "*.nii *.nii.gz")],
        )

        root.destroy()

        if not file:
            _set_validation_status("Selección de GT manual cancelada.")
            return

        try:
            dst = save_manual_seg_for_study(pid, sdate, Path(file))
            manual_gt_path.set(str(dst))
            _set_validation_status(f"GT manual guardada en el estudio: {dst.name}")
            study_dices_cache.set(None)
            dice_text.set("")
        except Exception as e:
            _set_validation_status(f"Error guardando GT manual: {e}")

    @reactive.Effect
    @reactive.event(input.download_dice_csv_study_btn)
    def _download_dice_csv_study_clicked():
        pid = input.val_patient_id()
        sdate = input.val_study_date()
        gt_path = manual_gt_path.get()

        if not pid or not sdate:
            dice_text.set("")
            _set_validation_status("Falta seleccionar paciente y estudio.")
            return

        if not gt_path or not Path(gt_path).exists():
            dice_text.set("")
            _set_validation_status("Falta seleccionar segmentación manual.")
            return

        try:
            cached = study_dices_cache.get()
            if isinstance(cached, dict) and cached.get("paciente_id") == pid and cached.get("study_date") == sdate:
                study_dices = cached
            else:
                _set_validation_status(f"Iniciando cálculo DICE para exportar {pid}/{sdate}...")
                study_dices = compute_study_dices(
                    pid,
                    sdate,
                    Path(gt_path),
                    progress_cb=_set_validation_status,
                )
                study_dices_cache.set(study_dices)
                save_study_dices_to_global_csv(study_dices)

            tmp_csv = APP_RESULTS_DIR / f"dice_{pid}_{sdate}.csv"
            csv_path = build_single_study_dice_csv(study_dices, tmp_csv)
            _set_validation_status(f"CSV DICE del estudio preparado: {pid}/{sdate}")
            _save_csv_with_dialog(csv_path, f"dice_{pid}_{sdate}.csv")

        except Exception as e:
            _set_validation_status(f"Error al exportar CSV DICE del estudio: {e}")
            
    @reactive.Effect
    @reactive.event(input.download_dice_csv_all_btn)
    def _download_dice_csv_all_clicked():
        gt_path = manual_gt_path.get()

        if not gt_path or not Path(gt_path).exists():
            _set_validation_status("Falta seleccionar segmentación manual.")
            return

        try:
            if not DICE_ALL_CSV.exists():
                _set_validation_status("No hay DICE acumulados todavía.")
                return

            _set_validation_status("Exportando CSV global de DICE...")
            _save_csv_with_dialog(DICE_ALL_CSV, "dice_todos_los_pacientes.csv")
            _set_validation_status("CSV global de DICE exportado.")

        except Exception as e:
            _set_validation_status(f"Error al exportar CSV DICE global: {e}")


    @reactive.Effect
    @reactive.event(input.analyze_atlas_btn)
    def _analyze_atlas_clicked():
        pid = input.atlas_patient_id()
        sdate = input.atlas_study_date()

        if not pid or not sdate:
            atlas_rows.set([])
            atlas_summary_text.set("")
            atlas_registration_info.set(None)
            atlas_last_case.set(None)
            atlas_status_msg.set("Falta seleccionar paciente y estudio.")
            return

        model_key = MODEL_OPTIONS[input.atlas_model()]
        img_path, pred_path = find_image_and_seg(pid, sdate, model_key)

        if img_path is None or not Path(img_path).exists():
            atlas_rows.set([])
            atlas_summary_text.set("")
            atlas_registration_info.set(None)
            atlas_last_case.set(None)
            atlas_status_msg.set("No se encontró la imagen T1/NIfTI del estudio.")
            return

        if pred_path is None or not Path(pred_path).exists():
            atlas_rows.set([])
            atlas_summary_text.set("")
            atlas_registration_info.set(None)
            atlas_last_case.set(None)
            atlas_status_msg.set(f"No hay segmentación disponible para {get_model_label_from_key(model_key)}.")
            return

        try:
            from atlas_utils import analyze_tumor_with_harvard_oxford, summarize_top_regions

            work_dir = ATLAS_WORK_DIR / pid / str(sdate) / model_key
            work_dir.mkdir(parents=True, exist_ok=True)

            atlas_status_msg.set(
                f"Analizando atlas para {pid}/{sdate} con {get_model_label_from_key(model_key)}..."
            )

            result = analyze_tumor_with_harvard_oxford(
                tumor_mask_path=Path(pred_path),
                t1_img_path=Path(img_path),
                work_dir=work_dir,
                data_dir=ATLAS_CACHE_DIR,
                min_overlap_voxels=5,
            )

            rows = list(result.get("top_regions", []))
            atlas_rows.set(rows)
            atlas_summary_text.set(summarize_top_regions(rows))
            atlas_registration_info.set(result.get("registration"))
            atlas_last_case.set({
                "paciente_id": pid,
                "study_date": sdate,
                "model_key": model_key,
            })

            atlas_status_msg.set(
                f"Atlas analizado para {pid}/{sdate} con {get_model_label_from_key(model_key)}."
            )

        except Exception as e:
            atlas_rows.set([])
            atlas_summary_text.set("")
            atlas_registration_info.set(None)
            atlas_last_case.set(None)
            atlas_status_msg.set(f"Error al analizar atlas: {e}")

    @reactive.Effect
    @reactive.event(input.calc_dice)
    def _calc_dice_clicked():
        pid = input.val_patient_id()
        sdate = input.val_study_date()
        model_key = MODEL_OPTIONS[input.val_model()]

        if not pid or not sdate:
            dice_text.set("")
            _set_validation_status("Falta seleccionar paciente y fecha.")
            return

        existing = load_study_row_from_csv(
            DICE_ALL_CSV,
            pid,
            sdate,
            required_value_fields=tuple(VAL_MODEL_KEYS),
        )

        if existing is not None and csv_value_present(existing, "media_ponderada"):
            study_dices_cache.set(existing)
            dice_text.set(format_dice_text(model_key, existing))
            _set_validation_status(f"DICE ya calculado para {pid}/{sdate}. Mostrando valores guardados.")
            return

        if existing is not None and not csv_value_present(existing, "media_ponderada"):
            _set_validation_status(
                f"DICE existente para {pid}/{sdate}, pero falta media ponderada. Recalculando..."
            )

        gt_path = manual_gt_path.get()
        if not gt_path or not Path(gt_path).exists():
            dice_text.set("")
            _set_validation_status("Falta seleccionar segmentación manual.")
            return

        try:
            # Si el modelo seleccionado es Media ponderada, primero hay que generar su NIfTI.
            # Si no se hace aquí, find_image_and_seg() no la encuentra y corta antes de calcular DICE.
            if model_key == "media_ponderada":
                _set_validation_status(f"Generando segmentación media ponderada para {pid}/{sdate}...")

                seg_path = build_weighted_ensemble_seg_for_study(
                    pid,
                    sdate,
                    threshold=0.5,
                    force=True,
                )

                if seg_path is None:
                    dice_text.set("")
                    _set_validation_status(
                        "No se pudo generar la media ponderada: hacen falta al menos dos segmentaciones disponibles."
                    )
                    return

            img_path, pred_path = find_image_and_seg(pid, sdate, model_key)

            if img_path is None or not Path(img_path).exists():
                dice_text.set("")
                _set_validation_status(f"No hay imagen de referencia disponible para {pid}/{sdate}.")
                return

            if pred_path is None or not Path(pred_path).exists():
                dice_text.set("")
                _set_validation_status(f"No hay predicción disponible para {model_key}.")
                return

            _set_validation_status(f"Iniciando cálculo DICE para {pid}/{sdate}...")

            all_dices = compute_study_dices(
                pid,
                sdate,
                Path(gt_path),
                progress_cb=_set_validation_status,
            )

            # Releer por si compute_study_dices acaba de generar la media ponderada
            img_path, pred_path = find_image_and_seg(pid, sdate, model_key)

            study_dices_cache.set(all_dices)

            _set_validation_status("Guardando DICE en app_data/results/dice.csv...")
            save_study_dices_to_global_csv(all_dices)

            dice_text.set(format_dice_text(model_key, all_dices))

            if pred_path is None or not Path(pred_path).exists():
                _set_validation_status(f"DICE calculado, pero no se pudo cargar la predicción visible de {model_key}.")
                return

            _set_validation_status("Comprobando alineación GT vs predicción visible...")
            gt = align_mask_to_ref(Path(gt_path), img_path)
            pred = align_mask_to_ref(Path(pred_path), img_path)

            if gt is None or pred is None:
                _set_validation_status("DICE guardado, pero no se pudieron alinear GT y predicción para la vista.")
                return

            if gt.shape != pred.shape:
                _set_validation_status(f"DICE guardado, pero hay distinto tamaño en vista: GT={gt.shape} vs Pred={pred.shape}")
                return

            _set_validation_status(f"DICE calculado para {pid}/{sdate}")

        except Exception as e:
            dice_text.set("")
            _set_validation_status(f"Error al calcular DICE: {e}")

    # 3. Eventos de mantenimiento
    @reactive.Effect
    @reactive.event(input.patient_id, input.study_date, input.model)
    def _reset_missing_nifti_guard():
        missing_nifti_warned_key.set("")
    
    @reactive.Effect
    @reactive.event(input.patient_id, input.study_date, input.model)
    def _refresh_volume_text_on_selection_change():
        pid = input.patient_id()
        sdate = input.study_date()

        if not pid or not sdate:
            volume_text.set("")
            return

        cached = study_volumes_cache.get()

        if isinstance(cached, dict) and cached.get("paciente_id") == pid and cached.get("study_date") == sdate:
            volume_text.set(format_volume_text(MODEL_OPTIONS[input.model()], cached))
        else:
            volume_text.set("")

    @reactive.Effect
    @reactive.event(input.val_patient_id, input.val_study_date, input.val_model)
    def _refresh_dice_text_on_selection_change():
        pid = input.val_patient_id()
        sdate = input.val_study_date()
        if not pid or not sdate:
            dice_text.set("")
            return

        cached = study_dices_cache.get()
        if isinstance(cached, dict) and cached.get("paciente_id") == pid and cached.get("study_date") == sdate:
            dice_text.set(format_dice_text(MODEL_OPTIONS[input.val_model()], cached))

    @reactive.Effect
    @reactive.event(input.modal_delete_study_btn)
    def _delete_study_from_modal():
        pid = modal_delete_pid.get()
        sdate = modal_delete_sdate.get()

        if not pid or not sdate:
            ui.modal_remove()
            return

        ok = delete_study(pid, sdate)

        ui.modal_remove()
        last_modal_message.set("")
        modal_delete_pid.set("")
        modal_delete_sdate.set("")

        if not ok:
            return

        status_text.set(f"Estudio eliminado: {pid}/{sdate}")

        # Refrescar selects de segmentación
        current_pid = input.patient_id()
        current_sdate = input.study_date()

        refresh_patient_choices(keep_selected=current_pid)
        new_pid = input.patient_id() or ""
        if current_pid == pid and current_sdate == sdate:
            pids = scan_nifti_patients()
            new_pid = current_pid if current_pid in pids else (pids[0] if pids else "")

        refresh_study_choices(new_pid, keep_selected="")
        ui.update_select("patient_id", selected=new_pid)

        # Refrescar selects de validación
        current_val_pid = input.val_patient_id()
        refresh_validation_patient_choices(keep_selected=current_val_pid)
        val_pids = scan_processed_patients()
        new_val_pid = current_val_pid if current_val_pid in val_pids else (val_pids[0] if val_pids else "")
        refresh_validation_study_choices(new_val_pid, keep_selected="")
        ui.update_select("val_patient_id", selected=new_val_pid)

        # Limpiar visor actual
        current_img_path.set(None)
        current_seg_path.set(None)
        img_data.set(None)
        seg_data.set(None)
        val_img_data.set(None)
        val_pred_data.set(None)
        volume_text.set("")


    # ===================== 6) REACTIVES AUTOMÁTICOS =====================

    @reactive.Effect
    @reactive.event(input.patient_id)
    def _on_patient_change():
        pid = input.patient_id()
        volume_text.set("")
        study_volumes_cache.set(None)
        viewer_case.set(None)

        if not pid:
            ui.update_select("study_date", choices=[], selected="")
            last_seg_pid.set("")
            return

        dates = list_studies(pid)
        selected = dates[-1] if dates else ""

        ui.update_select("study_date", choices=dates, selected=selected)
        last_seg_pid.set(pid)

        if selected:
            _push_main_status(f"Paciente cambiado a {pid}. Estudio seleccionado: {selected}")
        else:
            _push_main_status(f"Paciente cambiado a {pid}. Sin estudios disponibles.")

    @reactive.Effect
    @reactive.event(input.patient_id, input.study_date)
    def _sync_validation_with_seg_selection():
        pid = input.patient_id()
        sdate = input.study_date()

        if not pid:
            return

        processed_pids = scan_processed_patients()
        if pid not in processed_pids:
            return

        ui.update_select("val_patient_id", selected=pid)

        processed_dates = list_processed_studies(pid)
        if sdate and sdate in processed_dates:
            ui.update_select("val_study_date", choices=processed_dates, selected=sdate)
        else:
            selected = processed_dates[-1] if processed_dates else ""
            ui.update_select("val_study_date", choices=processed_dates, selected=selected)

    @reactive.Effect
    @reactive.event(input.patient_id, input.study_date)
    def _sync_atlas_with_seg_selection():
        pid = input.patient_id()
        sdate = input.study_date()

        if not pid:
            return

        processed_pids = scan_processed_patients()
        if pid not in processed_pids:
            return

        ui.update_select("atlas_patient_id", selected=pid)

        processed_dates = list_processed_studies(pid)
        if sdate and sdate in processed_dates:
            ui.update_select("atlas_study_date", choices=processed_dates, selected=sdate)
        else:
            selected = processed_dates[-1] if processed_dates else ""
            ui.update_select("atlas_study_date", choices=processed_dates, selected=selected)

    @reactive.Effect
    @reactive.event(input.val_patient_id)
    def _on_val_patient_change():
        pid = input.val_patient_id()

        if not pid:
            ui.update_select("val_study_date", choices=[], selected="")
            last_val_pid.set("")
            return

        dates = list_processed_studies(pid)
        selected = dates[-1] if dates else ""

        ui.update_select("val_study_date", choices=dates, selected=selected)
        last_val_pid.set(pid)

    @reactive.Effect
    @reactive.event(input.atlas_patient_id)
    def _on_atlas_patient_change():
        pid = input.atlas_patient_id()

        if not pid:
            ui.update_select("atlas_study_date", choices=[], selected="")
            last_atlas_pid.set("")
            return

        dates = list_processed_studies(pid)
        selected = dates[-1] if dates else ""

        ui.update_select("atlas_study_date", choices=dates, selected=selected)
        last_atlas_pid.set(pid)

    @reactive.Effect
    @reactive.event(input.patient_id, input.study_date, input.model)
    def _auto_load():
        pid = input.patient_id()
        sdate = input.study_date()
        model_label = input.model()

        if not pid or not sdate or not model_label:
            current_img_path.set(None)
            current_seg_path.set(None)
            img_data.set(None)
            seg_data.set(None)
            volume_text.set("")
            status_text.set("Listo para segmentar")
            return

        dates = list_studies(pid)

        if not dates:
            current_img_path.set(None)
            current_seg_path.set(None)
            img_data.set(None)
            seg_data.set(None)
            volume_text.set("")
            status_text.set("Listo para segmentar")
            return

        if sdate not in dates:
            ui.update_select("study_date", choices=dates, selected=dates[-1])
            status_text.set("Actualizando estudio seleccionado...")
            return

        model_key = MODEL_OPTIONS[model_label]

        _push_main_status(f"Cargando imagen {pid}/{sdate} con {model_key}...")

        try:
            reload_viewer(pid, sdate, model_key)
            _push_main_status(f"Imagen cargada: {pid}/{sdate}")
        except Exception as e:
            current_img_path.set(None)
            current_seg_path.set(None)
            img_data.set(None)
            seg_data.set(None)
            status_text.set(f"Error cargando imagen: {e}")
            _push_main_status(f"Error cargando imagen: {e}")
        
    @reactive.Effect
    def _persist_selection():
        pid = input.patient_id()
        sdate = input.study_date()
        model_label = input.model()
        if pid and sdate and model_label:
            save_last_session(pid, sdate, model_label)

    @reactive.Effect
    @reactive.event(input.val_patient_id, input.val_study_date, input.val_model)
    def _load_validation_base():
        pid = input.val_patient_id()
        sdate = input.val_study_date()
        model_key = MODEL_OPTIONS[input.val_model()]

        if not pid or not sdate:
            val_img_data.set(None)
            val_pred_data.set(None)
            val_gt_data.set(None)
            manual_gt_path.set("")
            _set_validation_status("Selecciona paciente y estudio en validación.")
            return

        valid_dates = list_processed_studies(pid)
        if sdate not in valid_dates:
            val_img_data.set(None)
            val_pred_data.set(None)
            val_gt_data.set(None)
            manual_gt_path.set("")
            _set_validation_status("Cambiando estudio...")
            return

        try:
            img_path, pred_path = find_image_and_seg(pid, sdate, model_key)
            img = load_nifti_data(img_path)
            pred = load_nifti_data(pred_path)

            val_img_data.set(img)
            val_pred_data.set(pred)

            gt_path = find_manual_seg_for_study(pid, sdate)
            manual_gt_path.set(str(gt_path) if gt_path else "")

            _set_validation_status(f"Validación base cargada para {pid}/{sdate}.")
        except Exception as e:
            val_img_data.set(None)
            val_pred_data.set(None)
            val_gt_data.set(None)
            manual_gt_path.set("")
            _set_validation_status(f"Error cargando validación: {e}")

    @reactive.Effect
    @reactive.event(input.val_patient_id, input.val_study_date, input.val_model, manual_gt_path)
    def _load_validation_manual():
        pid = input.val_patient_id()
        sdate = input.val_study_date()

        if not pid or not sdate:
            val_gt_data.set(None)
            return

        gt_path = manual_gt_path.get()
        if not gt_path:
            val_gt_data.set(None)
            return

        model_key = MODEL_OPTIONS[input.val_model()]
        img_path, _ = find_image_and_seg(pid, sdate, model_key)

        if img_path is None or not Path(img_path).exists():
            val_gt_data.set(None)
            return

        try:
            _set_validation_status("Cargando GT manual del estudio...")
            gt_arr = load_manual_gt_for_validation(pid, sdate, Path(img_path))
            val_gt_data.set(gt_arr)

            if gt_arr is not None:
                _set_validation_status(f"GT manual lista para {pid}/{sdate}.")
            else:
                _set_validation_status(f"No hay GT manual guardada para {pid}/{sdate}.")
        except Exception as e:
            val_gt_data.set(None)
            _set_validation_status(f"Error cargando GT manual: {e}")

    @reactive.Effect
    def _update_val_slice_slider():
        pid = input.val_patient_id()
        sdate = input.val_study_date()

        if not pid or not sdate:
            ui.update_slider("val_slice", min=0, max=0, value=0)
            return

        model_key = MODEL_OPTIONS[input.val_model()]
        img_path, _ = find_image_and_seg(pid, sdate, model_key)

        if not img_path or not Path(img_path).exists():
            ui.update_slider("val_slice", min=0, max=0, value=0)
            return

        shape = nib.load(str(img_path)).shape
        plane = input.val_plane()

        if plane == "sagital":
            max_idx = shape[0] - 1
        elif plane == "coronal":
            max_idx = shape[1] - 1
        else:
            max_idx = shape[2] - 1

        ui.update_slider("val_slice", min=0, max=max_idx, value=max_idx // 2)

    @reactive.Effect
    def _watch_run_thread():
        reactive.invalidate_later(0.5)

        t = run_thread.get()
        job = run_job_info.get()

        if t is None or job is None:
            return

        if t.is_alive():
            return

        pid, sdate, model_key = job

        run_is_busy.set(False)
        run_log_path.set("")

        if model_key == "all_patients":
            try:
                current_pid = input.patient_id() or pid
                current_sdate = input.study_date() or sdate

                refresh_patient_choices(keep_selected=current_pid)
                if current_pid:
                    refresh_study_choices(current_pid, keep_selected=current_sdate)

                refresh_validation_patient_choices(keep_selected=input.val_patient_id() or "")
                if input.val_patient_id():
                    refresh_validation_study_choices(input.val_patient_id(), keep_selected=input.val_study_date() or "")

                refresh_atlas_patient_choices(keep_selected=input.atlas_patient_id() or "")
                if input.atlas_patient_id():
                    refresh_atlas_study_choices(input.atlas_patient_id(), keep_selected=input.atlas_study_date() or "")

                current_model_key = MODEL_OPTIONS[input.model()]
                if current_pid and current_sdate:
                    reload_viewer(current_pid, current_sdate, current_model_key)
            except Exception:
                pass

            status_text.set("Ejecución masiva terminada para todos los estudios disponibles.")
            _push_main_status("Ejecución masiva terminada para todos los estudios disponibles.")
        else:
            try:
                reload_viewer(pid, sdate, model_key)
            except Exception:
                pass

            status_text.set(f"Segmentación terminada para {pid}/{sdate} con {model_key}")
            _push_main_status(f"Segmentación terminada para {pid}/{sdate} con {model_key}")

        run_thread.set(None)
        run_job_info.set(None)

    # ===================== 7) CROSSHAIR =====================

    @reactive.Effect
    def _on_click_set_crosshair():
        if not crosshair_enabled.get():
            return

        h = input.click_coords()
        idata = img_data.get()
        if h is None or idata is None:
            return

        vox = click_to_voxel(h, idata.shape)
        if vox is None:
            return

        cross_voxel.set(vox)
        ui.update_slider("axial_slice", value=int(vox["z"]))
        ui.update_slider("coronal_slice", value=int(vox["y"]))
        ui.update_slider("sagital_slice", value=int(vox["x"]))

    @reactive.Effect
    def _clear_crosshair_when_off():
        if not crosshair_enabled.get():
            cross_voxel.set(None)

    # ===================== 8) OUTPUTS (UI + plots) =====================

    @output
    @render.ui
    def run_status_bar():
        reactive.invalidate_later(0.5)

        p = run_log_path.get()
        busy = bool(run_is_busy.get())

        lines = []
        if p and busy:
            try:
                path = Path(p)
                if path.exists():
                    txt = path.read_text(encoding="utf-8", errors="ignore").strip()
                    if txt:
                        raw_lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
                        lines = raw_lines[-3:]
            except Exception as e:
                lines = [f"[LOG ERROR] {e}"]

        if not lines:
            hist = list(main_status_lines.get() or [])
            if hist:
                lines = hist[-3:]

        if not lines:
            fallback = status_text.get().strip() or import_status_text.get().strip()
            lines = [fallback] if fallback else ["Listo para segmentar"]

        line_nodes = []
        for i, line in enumerate(lines):
            cls = "status-line" if i == len(lines) - 1 else "status-line dim"
            line_nodes.append(ui.div(line, class_=cls))

        spinner = ui.div(
            ui.tags.div(class_="spinner-border text-primary", role="status")
            if busy else ui.div(""),
            class_="status-spinner",
        )

        return ui.div(
            ui.div(*line_nodes, class_="status-lines"),
            spinner,
            class_="status-bar-wrap",
        )

    @output
    @render.ui
    def weighted_heatmap_legend():
        model_key = MODEL_OPTIONS.get(input.model(), "")

        if model_key != "media_ponderada":
            return ui.div()

        return weighted_heatmap_legend_ui()
        
    @output
    @render.ui
    def validation_status_bar():
        msg = validation_status_msg.get().strip() or "Selecciona paciente y estudio en validación."

        return ui.div(
            ui.div(
                ui.div(msg, class_="status-line"),
                class_="status-lines",
            ),
            ui.div("", class_="status-spinner"),
            class_="status-bar-wrap",
        )

    @output
    @render.text
    def sync_status():
        return sync_status_msg.get()
    
    @output
    @render.ui
    def import_alert_ui():
        msg = import_alert.get()
        if not msg:
            return ui.div()

        return ui.div(
            msg,
            style="""
                padding:10px 12px;
                border-radius:10px;
                background:#3a1616;
                border:1px solid #b91c1c;
                color:#fecaca;
                margin-bottom:10px;
                font-weight:600;
            """,
        )
    
    @output
    @render.ui
    def study_integrity_alert():
        pid = input.patient_id()
        sdate = input.study_date()

        if not pid or not sdate:
            return ui.div()

        integ = study_integrity(pid, sdate)
        status = integ["status"]
        msg = integ["message"]

        if status == "ok":
            return ui.div()

        if status == "warning_missing_dicom":
            return ui.div(
                msg,
                style="""
                    padding:10px 12px;
                    border-radius:10px;
                    background:#3b2f12;
                    border:1px solid #a16207;
                    color:#fde68a;
                    margin-bottom:10px;
                """,
            )

        return ui.div(
            msg,
            style="""
                padding:10px 12px;
                border-radius:10px;
                background:#3a1616;
                border:1px solid #b91c1c;
                color:#fecaca;
                margin-bottom:10px;
                font-weight:600;
            """,
        )
    
    @output
    @render.ui
    def storage_path_warning_ui():
        msg = storage_path_warning.get()
        if not msg:
            return ui.div()

        return ui.div(
            msg,
            style="""
                padding:10px 12px;
                border-radius:10px;
                background:#3b2f12;
                border:1px solid #a16207;
                color:#fde68a;
                margin-bottom:10px;
                font-weight:600;
            """,
        )

    def _make_seg_badge():
        imgp = current_img_path.get()
        sdata = seg_data.get()
        model_key = MODEL_OPTIONS.get(input.model(), "")

        if not imgp:
            txt, bg = "Sin imagen", "#6b7280"
        elif sdata is None:
            txt, bg = "Seg: no disponible", "#b45309"
        elif model_key == "media_ponderada":
            txt, bg = "Mapa: consenso", "#2563eb"
        else:
            txt, bg = "Seg: cargada", "#15803d"

        return ui.span(
            txt,
            style=f"""
                background:{bg};
                color:white;
                padding:2px 8px;
                border-radius:999px;
                font-size:12px;
                line-height:18px;
                white-space:nowrap;
            """,
        )
    
    def badge_pill(text: str, bg: str, border: str = "", fg: str = "#ffffff"):
        if not border:
            border = bg
        return ui.span(
            text,
            style=(
                f"display:inline-block; "
                f"padding:4px 10px; "
                f"border-radius:999px; "
                f"font-size:12px; "
                f"font-weight:600; "
                f"color:{fg}; "
                f"background:{bg}; "
                f"border:1px solid {border};"
            ),
        )

    @output
    @render.ui
    def seg_badge_axial():
        return _make_seg_badge()

    @output
    @render.ui
    def seg_badge_coronal():
        return _make_seg_badge()

    @output
    @render.ui
    def seg_badge_sagital():
        return _make_seg_badge()
    
    @output
    @render.ui
    def pred_badge_validation():
        pred = val_pred_data.get()
        model_key = MODEL_OPTIONS.get(input.val_model(), "radionics")
        model_color = MODEL_COLORS.get(model_key, "#ef4444")

        if pred is not None:
            return badge_pill("Pred: cargada", bg=model_color, border=model_color)

        return badge_pill("Pred: no", bg="#374151", border="#4b5563")
    @output
    @render.ui
    def manual_badge_validation():
        gt = val_gt_data.get()
        if gt is not None:
            return badge_pill("Manual: cargada", bg="#2563eb", border="#1d4ed8")
        return badge_pill("Manual: no", bg="#1e293b", border="#334155")
    
    @output
    @render.ui
    def dice_badge_validation():
        pid = input.val_patient_id()
        sdate = input.val_study_date()
        selected_model_key = MODEL_OPTIONS[input.val_model()]
        cached = study_dices_cache.get()

        if not isinstance(cached, dict):
            return ui.div(
                badge_pill("DICE: —", bg="#1f2937", border="#374151"),
                ui.div(
                    "Calcula primero el DICE para ver los resultados.",
                    class_="dice-summary-main",
                ),
                class_="dice-summary-box",
            )

        if cached.get("paciente_id") != pid or cached.get("study_date") != sdate:
            return ui.div(
                badge_pill("DICE: —", bg="#1f2937", border="#374151"),
                ui.div(
                    "Calcula primero el DICE para este estudio.",
                    class_="dice-summary-main",
                ),
                class_="dice-summary-box",
            )

        selected_value = cached.get(selected_model_key)
        selected_label = get_model_label_from_key(selected_model_key)

        if selected_value is not None:
            top_badge = badge_pill(
                f"DICE: {float(selected_value):.4f}",
                bg="#14532d",
                border="#166534",
            )
            main_text = f"DICE del modelo seleccionado ({selected_label}): {float(selected_value):.4f}"
        else:
            top_badge = badge_pill("DICE: —", bg="#1f2937", border="#374151")
            main_text = f"DICE del modelo seleccionado ({selected_label}): no disponible"

        model_rows = []

        for mk in VAL_MODEL_KEYS:
            label = get_model_label_from_key(mk)
            value = cached.get(mk)

            if value is None:
                sort_value = -1.0
                value_txt = "—"
            else:
                sort_value = float(value)
                value_txt = f"{sort_value:.4f}"

            model_rows.append((sort_value, label, value_txt))

        model_rows = sorted(model_rows, key=lambda x: x[0], reverse=True)

        rows = []
        for sort_value, label, value_txt in model_rows:
            rows.append(
                ui.div(
                    ui.span(label, class_="dice-model-name"),
                    ui.span(value_txt, class_="dice-model-value"),
                    class_="dice-model-row",
                )
            )

        return ui.div(
            top_badge,
            ui.div(main_text, class_="dice-summary-main"),
            ui.div("Todos los modelos", class_="dice-summary-subtitle"),
            ui.div(*rows, class_="dice-model-list"),
            class_="dice-summary-box",
        )
    
    @output
    @render.ui
    def validation_legend_ui():
        selected_model_key = MODEL_OPTIONS[input.val_model()]
        show_pred = bool(input.val_show_pred())
        show_gt = bool(input.val_show_gt())
        show_overlap = bool(input.val_show_overlap())

        rows = []

        for mk in VAL_MODEL_KEYS:
            label = get_model_label_from_key(mk)
            color = MODEL_COLORS.get(mk, "#ef4444")

            children = [
                ui.span(class_="validation-legend-swatch", style=f"background:{color};"),
                ui.span(label, class_="validation-legend-label"),
            ]

            if mk == selected_model_key and show_pred:
                children.append(ui.span("visible", class_="validation-legend-tag"))

            rows.append(ui.div(*children, class_="validation-legend-row"))

        gt_children = [
            ui.span(class_="validation-legend-swatch", style=f"background:{VALIDATION_GT_COLOR};"),
            ui.span("GT manual", class_="validation-legend-label"),
        ]
        if show_gt:
            gt_children.append(ui.span("visible", class_="validation-legend-tag"))

        rows.append(ui.div(*gt_children, class_="validation-legend-row"))

        overlap_children = [
            ui.span(class_="validation-legend-swatch", style=f"background:{VALIDATION_OVERLAP_COLOR};"),
            ui.span("Solape / unión", class_="validation-legend-label"),
        ]
        if show_pred and show_gt and show_overlap:
            overlap_children.append(ui.span("visible", class_="validation-legend-tag"))

        rows.append(ui.div(*overlap_children, class_="validation-legend-row"))

        return ui.div(
            ui.div("Leyenda de colores", class_="validation-legend-title"),
            ui.div(*rows, class_="validation-legend-list"),
            class_="validation-legend-box",
        )

    @output
    @render.text
    def volume_result():
        return volume_text.get() or ""
    
    @output
    @render.ui
    def atlas_status_bar():
        msg = atlas_status_msg.get().strip() or "Selecciona paciente, estudio y modelo para analizar atlas."

        return ui.div(
            ui.div(
                ui.div(msg, class_="status-line"),
                class_="status-lines",
            ),
            ui.div("", class_="status-spinner"),
            class_="status-bar-wrap",
        )

    @output
    @render.ui
    def atlas_summary_ui():
        txt = atlas_summary_text.get().strip()
        if not txt:
            txt = (
                "Pulsa 'Analizar atlas' para registrar la segmentación del modelo al espacio MNI "
                "y calcular el solape con las regiones del atlas Harvard-Oxford."
            )
        return ui.div(txt, class_="atlas-summary-box")

    @output
    @render.ui
    def atlas_regions_ui():
        rows = list(atlas_rows.get() or [])
        if not rows:
            return ui.div("Todavía no hay regiones analizadas.", class_="atlas-small-box atlas-muted")

        header = ui.tags.tr(
            ui.tags.th("Región"),
            ui.tags.th("Grupo"),
            ui.tags.th("% tumor"),
            ui.tags.th("% informe"),
            ui.tags.th("Vóxeles"),
        )

        body_rows = []
        for r in rows[:15]:
            body_rows.append(
                ui.tags.tr(
                    ui.tags.td(str(r.get("region_name", ""))),
                    ui.tags.td(str(r.get("atlas_group", ""))),
                    ui.tags.td(f"{float(r.get('tumor_pct', 0.0)):.1f}"),
                    ui.tags.td(f"{float(r.get('reported_pct', 0.0)):.1f}"),
                    ui.tags.td(str(r.get("overlap_voxels", ""))),
                )
            )

        return ui.div(
            ui.tags.table(
                ui.tags.thead(header),
                ui.tags.tbody(*body_rows),
                class_="atlas-table",
            ),
            class_="atlas-table-wrap",
        )

    @output
    @render.ui
    def atlas_registration_ui():
        info = atlas_registration_info.get()
        case = atlas_last_case.get()

        lines = []
        if isinstance(case, dict):
            label = get_model_label_from_key(case.get("model_key", ""))
            lines.append(f"Caso analizado: {case.get('paciente_id', '')} / {case.get('study_date', '')} / {label}")

        if isinstance(info, dict):
            template_path = info.get("template_path")
            reg_t1 = info.get("registered_t1_path")
            reg_mask = info.get("registered_mask_path")

            if template_path:
                lines.append(f"Plantilla MNI: {template_path}")
            if reg_t1:
                lines.append(f"T1 registrado: {reg_t1}")
            if reg_mask:
                lines.append(f"Máscara registrada: {reg_mask}")

        if not lines:
            lines = ["Todavía no hay información de registro disponible."]

        return ui.div("\n".join(lines), class_="atlas-small-box")
    

    @output
    @render.plot
    def validation_plot():
        img_data_local = val_img_data.get()
        pred_data_local = val_pred_data.get()
        gt_data_local = val_gt_data.get()
        
        model_key = MODEL_OPTIONS[input.val_model()]
        pred_color = MODEL_COLORS.get(model_key, "#ef4444")

        if img_data_local is None:
            fig = plt.figure(figsize=(6, 6))
            plt.text(0.5, 0.5, "Selecciona paciente y estudio", ha="center", va="center")
            plt.axis("off")
            return

        plane = input.val_plane()
        idx = input.val_slice()

        plot_validation_slice(
            img_data=img_data_local,
            pred_data=pred_data_local if input.val_show_pred() else None,
            gt_data=gt_data_local if input.val_show_gt() else None,
            idx=idx,
            plane=plane,
            pred_color=pred_color,
            show_overlap=bool(input.val_show_overlap()),
        )

                    
    @output
    @render.ui
    def dims_js():
        idata = img_data.get()
        if idata is None:
            return ui.tags.script("window.__tfg_dims__ = {};")

        # Dimensiones después del rot90 y cómo imshow interpreta (w = cols, h = rows)
        # axial: sl = img[:,:,z] -> (X,Y) -> rot90 -> (Y,X) => h=Y, w=X
        axial_w = int(idata.shape[0])
        axial_h = int(idata.shape[1])

        # coronal: sl = img[:,y,:] -> (X,Z) -> rot90 -> (Z,X) => h=Z, w=X
        cor_w = int(idata.shape[0])
        cor_h = int(idata.shape[2])

        # sagital: sl = img[x,:,:] -> (Y,Z) -> rot90 -> (Z,Y) => h=Z, w=Y
        sag_w = int(idata.shape[1])
        sag_h = int(idata.shape[2])

        return ui.tags.script(f"""
         window.__tfg_dims__ = {{
            axial:   {{ w: {axial_w}, h: {axial_h} }},
            coronal: {{ w: {cor_w},   h: {cor_h} }},
            sagital: {{ w: {sag_w},   h: {sag_h} }}
        }};
        """)
    
    @output
    @render.ui
    def crosshair_js():
        enabled = "true" if crosshair_enabled.get() else "false"
        panel_class = "floating-coords" if crosshair_enabled.get() else "floating-coords hidden"
        return ui.tags.script(f"""
            window.__tfg_crosshair_enabled__ = {enabled};
            (function(){{
                const p = document.getElementById("coords_panel");
                if (p) p.className = "{panel_class}";
            }})();
        """)
    
    @output
    @render.text
    def coords_readout():
        if not crosshair_enabled.get():
            return "Coordenadas ocultas"
        v = cross_voxel.get()
        if v is None:
            return "Click para fijar punto"
        return f"Voxel: x={v['x']} y={v['y']} z={v['z']}"
        
    
    # ---- Plots (tres vistas) ----
    @output
    @render.plot
    def axial_plot():
        model_key = MODEL_OPTIONS[input.model()]
        color = MODEL_COLORS.get(model_key, "#ef4444")
        plot_slice(
            img_data.get(),
            seg_data.get(),
            input.axial_slice(),
            axis=2,
            alpha=input.alpha(),
            show_seg=input.show_seg(),
            seg_color=color,
            cross=cross_voxel.get(),
            draw_cross=crosshair_enabled.get(),
            heatmap=(model_key == "media_ponderada"),
        )


    @output
    @render.plot
    def coronal_plot():
        model_key = MODEL_OPTIONS[input.model()]
        color = MODEL_COLORS.get(model_key, "#ef4444")
        plot_slice(
            img_data.get(),
            seg_data.get(),
            input.coronal_slice(),
            axis=1,
            alpha=input.alpha(),
            show_seg=input.show_seg(),
            seg_color=color,
            cross=cross_voxel.get(),
            draw_cross=crosshair_enabled.get(),
            heatmap=(model_key == "media_ponderada"),
        )


    @output
    @render.plot
    def sagital_plot():
        model_key = MODEL_OPTIONS[input.model()]
        color = MODEL_COLORS.get(model_key, "#ef4444")
        plot_slice(
            img_data.get(),
            seg_data.get(),
            input.sagital_slice(),
            axis=0,
            alpha=input.alpha(),
            show_seg=input.show_seg(),
            seg_color=color,
            cross=cross_voxel.get(),
            draw_cross=crosshair_enabled.get(),
            heatmap=(model_key == "media_ponderada"),
        )

app = App(app_ui, server)