from __future__ import annotations
"""
atlas_utils.py

Utilidades para relacionar una segmentación tumoral con un atlas anatómico.

Objetivo
--------
Este módulo está pensado como apoyo a una futura tercera pestaña de la app,
donde se quiera estimar qué regiones anatómicas podrían verse afectadas por el
tumor.

Flujo recomendado
-----------------
1. Partir de:
   - el T1 del estudio (NIfTI)
   - la máscara tumoral del estudio (NIfTI binario)
2. Registrar el T1 del paciente al espacio MNI152.
3. Llevar la máscara tumoral al mismo espacio MNI.
4. Cargar un atlas anatómico en espacio MNI.
5. Calcular el solape del tumor con las regiones del atlas.

Atlas usado por defecto
-----------------------
Se usa Harvard-Oxford (cortical + subcortical) en versión determinista
(maxprob-thr25-2mm), descargado mediante nilearn.

Notas importantes
-----------------
- `nilearn.datasets.fetch_atlas_harvard_oxford(...)` descarga el atlas y lo
  cachea localmente. En un empaquetado final conviene precargar esos ficheros.
- Para obtener resultados anatómicos razonables, la máscara debe estar en
  espacio MNI. Re-muestrear sin registrar NO equivale a registrar.
- El registro espacial se implementa con ANTsPy si está disponible.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import nibabel as nib
from nilearn.datasets import fetch_atlas_harvard_oxford, load_mni152_template
from nilearn.image import resample_to_img

try:
    import ants  # type: ignore
except Exception:  # pragma: no cover - entorno sin ANTs
    ants = None


# ============================================================================
# Configuración por defecto
# ============================================================================

DEFAULT_TEMPLATE_RESOLUTION = 2
DEFAULT_ATLAS_NAMES = (
    "cort-maxprob-thr25-2mm",
    "sub-maxprob-thr25-2mm",
)


@dataclass
class AtlasSpec:
    """Especificación mínima de un atlas anatómico cargado."""

    name: str
    maps_path: Path
    labels: List[str]
    group: str


# ============================================================================
# Helpers internos
# ============================================================================


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _to_path(p: str | Path) -> Path:
    return p if isinstance(p, Path) else Path(p)


def _save_img_if_needed(img: Any, out_path: Path) -> Path:
    """
    Guarda una imagen nibabel-like en disco si todavía no existe.
    """
    out_path = _to_path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(out_path))
    return out_path


def _load_mask_binary(mask_path: str | Path) -> nib.Nifti1Image:
    """
    Carga una máscara y la binariza (>0).
    """
    mask_path = _to_path(mask_path)
    if not mask_path.exists():
        raise FileNotFoundError(f"No existe la máscara: {mask_path}")

    img = nib.load(str(mask_path))
    data = (img.get_fdata() > 0).astype(np.uint8)
    return nib.Nifti1Image(data, img.affine, img.header)


# ============================================================================
# Carga de plantilla y atlas
# ============================================================================


def save_mni_template(
    out_path: str | Path,
    resolution: int = DEFAULT_TEMPLATE_RESOLUTION,
) -> Path:
    """
    Guarda en disco la plantilla MNI152 de nilearn.

    Parámetros
    ----------
    out_path : str | Path
        Ruta de salida donde guardar la plantilla.
    resolution : int, default=2
        Resolución solicitada a nilearn para la plantilla MNI152.

    Devuelve
    --------
    Path
        Ruta al fichero guardado.
    """
    template_img = load_mni152_template(resolution=resolution)
    return _save_img_if_needed(template_img, _to_path(out_path))

def fetch_harvard_oxford_atlases(
    data_dir: Optional[str | Path] = None,
    symmetric_split: bool = False,
) -> List[AtlasSpec]:
    """
    Descarga/carga los atlas Harvard-Oxford cortical y subcortical.

    Usa las versiones deterministas `maxprob-thr25-2mm`, adecuadas para obtener
    regiones no solapadas y calcular porcentajes de afectación por región.

    Parámetros
    ----------
    data_dir : str | Path | None
        Carpeta de caché donde nilearn puede guardar los atlas.
    symmetric_split : bool, default=False
        Si es True, pide a nilearn dividir regiones simétricas izquierda/derecha
        cuando el atlas lo permite.

    Devuelve
    --------
    list[AtlasSpec]
        Lista con dos atlas: cortical y subcortical.
    """
    atlas_specs: List[AtlasSpec] = []

    for atlas_name in DEFAULT_ATLAS_NAMES:
        bunch = fetch_atlas_harvard_oxford(
            atlas_name,
            data_dir=data_dir,
            symmetric_split=symmetric_split,
            resume=True,
            verbose=1,
        )

        maps_obj = getattr(bunch, "maps", None)
        filename_obj = getattr(bunch, "filename", None)

        if isinstance(maps_obj, (str, bytes, Path)):
            maps_path = Path(maps_obj)

        elif isinstance(filename_obj, (str, bytes, Path)):
            maps_path = Path(filename_obj)

        elif maps_obj is not None and hasattr(maps_obj, "to_filename"):
            atlas_dir = _ensure_dir(_to_path(data_dir) if data_dir is not None else Path("atlas_cache"))
            maps_path = atlas_dir / f"{atlas_name}.nii.gz"
            if not maps_path.exists():
                nib.save(maps_obj, str(maps_path))

        else:
            raise TypeError(
                f"No se pudo obtener una ruta válida para el atlas {atlas_name}. "
                f"Tipo de bunch.maps: {type(maps_obj)}"
            )

        labels = list(getattr(bunch, "labels", []))
        group = "cortical" if atlas_name.startswith("cort") else "subcortical"

        atlas_specs.append(
            AtlasSpec(
                name=atlas_name,
                maps_path=maps_path,
                labels=labels,
                group=group,
            )
        )

    return atlas_specs

# ============================================================================
# Registro a espacio MNI con ANTs
# ============================================================================


def register_t1_and_mask_to_mni(
    t1_img_path: str | Path,
    tumor_mask_path: str | Path,
    work_dir: str | Path,
    resolution: int = DEFAULT_TEMPLATE_RESOLUTION,
    force: bool = False,
) -> Dict[str, Path]:
    """
    Registra un T1 al espacio MNI y aplica la transformación a la máscara tumoral.

    Requiere ANTsPy (`import ants`). El registro se hace con `type_of_transform`
    = `SyN`, razonable para una primera versión anatómica de la app.

    Parámetros
    ----------
    t1_img_path : str | Path
        NIfTI T1 del paciente.
    tumor_mask_path : str | Path
        Máscara tumoral binaria en el espacio del T1.
    work_dir : str | Path
        Carpeta de trabajo donde guardar plantilla, T1 registrado, máscara
        registrada y transformaciones.
    resolution : int, default=2
        Resolución de la plantilla MNI de destino.
    force : bool, default=False
        Si es False y los ficheros ya existen, reutiliza resultados previos.

    Devuelve
    --------
    dict[str, Path]
        Rutas a los principales artefactos generados.
    """
    if ants is None:
        raise RuntimeError(
            "ANTsPy no está disponible. Instálalo antes de usar el registro a atlas."
        )

    t1_img_path = _to_path(t1_img_path)
    tumor_mask_path = _to_path(tumor_mask_path)
    work_dir = _ensure_dir(_to_path(work_dir))

    if not t1_img_path.exists():
        raise FileNotFoundError(f"No existe la imagen T1: {t1_img_path}")
    if not tumor_mask_path.exists():
        raise FileNotFoundError(f"No existe la máscara tumoral: {tumor_mask_path}")

    template_path = work_dir / f"mni152_template_{resolution}mm.nii.gz"
    reg_t1_path = work_dir / f"{t1_img_path.stem.replace('.nii', '')}_to_mni.nii.gz"
    reg_mask_path = work_dir / f"{tumor_mask_path.stem.replace('.nii', '')}_to_mni.nii.gz"

    if (
        not force
        and template_path.exists()
        and reg_t1_path.exists()
        and reg_mask_path.exists()
    ):
        return {
            "template_path": template_path,
            "registered_t1_path": reg_t1_path,
            "registered_mask_path": reg_mask_path,
        }

    save_mni_template(template_path, resolution=resolution)

    fixed = ants.image_read(str(template_path))
    moving = ants.image_read(str(t1_img_path))

    reg = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform="SyN",
    )

    warped_t1 = reg["warpedmovout"]
    ants.image_write(warped_t1, str(reg_t1_path))

    moving_mask = ants.image_read(str(tumor_mask_path))
    warped_mask = ants.apply_transforms(
        fixed=fixed,
        moving=moving_mask,
        transformlist=reg["fwdtransforms"],
        interpolator="nearestNeighbor",
    )
    ants.image_write(warped_mask, str(reg_mask_path))

    return {
        "template_path": template_path,
        "registered_t1_path": reg_t1_path,
        "registered_mask_path": reg_mask_path,
    }


# ============================================================================
# Solape tumor - atlas
# ============================================================================


def compute_region_overlaps(
    tumor_mask_mni_path: str | Path,
    atlas_spec: AtlasSpec,
    min_overlap_voxels: int = 1,
) -> List[Dict[str, Any]]:
    """
    Calcula el solape de una máscara tumoral en espacio MNI con un atlas.

    La máscara tumoral debe estar ya registrada a MNI. Este paso NO realiza
    registro, solo remuestreo del atlas al grid de la máscara si hace falta.

    Parámetros
    ----------
    tumor_mask_mni_path : str | Path
        Máscara tumoral binaria en espacio MNI.
    atlas_spec : AtlasSpec
        Atlas anatómico cargado.
    min_overlap_voxels : int, default=1
        Número mínimo de vóxeles en solape para conservar una región.

    Devuelve
    --------
    list[dict]
        Una lista de regiones con métricas de solape.
    """
    tumor_img = _load_mask_binary(tumor_mask_mni_path)
    atlas_img = nib.load(str(atlas_spec.maps_path))

    atlas_res = resample_to_img(
        atlas_img,
        tumor_img,
        interpolation="nearest",
        force_resample=True,
        copy_header=True,
    )

    tumor = (tumor_img.get_fdata() > 0).astype(np.uint8)
    atlas = atlas_res.get_fdata().astype(np.int32)

    tumor_voxels = int(np.count_nonzero(tumor))
    if tumor_voxels == 0:
        return []

    rows: List[Dict[str, Any]] = []
    atlas_labels = atlas_spec.labels

    region_ids = sorted(int(v) for v in np.unique(atlas) if int(v) > 0)
    for region_id in region_ids:
        region_mask = atlas == region_id
        region_voxels = int(np.count_nonzero(region_mask))
        if region_voxels == 0:
            continue

        overlap_voxels = int(np.count_nonzero(region_mask & (tumor > 0)))
        if overlap_voxels < int(min_overlap_voxels):
            continue

        tumor_pct = 100.0 * overlap_voxels / tumor_voxels
        region_pct = 100.0 * overlap_voxels / region_voxels

        label = (
            atlas_labels[region_id]
            if region_id < len(atlas_labels)
            else f"Region_{region_id}"
        )

        rows.append(
            {
                "atlas_group": atlas_spec.group,
                "atlas_name": atlas_spec.name,
                "region_id": region_id,
                "region_name": label,
                "overlap_voxels": overlap_voxels,
                "tumor_pct": round(tumor_pct, 3),
                "region_pct": round(region_pct, 3),
            }
        )

    rows.sort(key=lambda r: (r["tumor_pct"], r["overlap_voxels"]), reverse=True)
    return rows


# ============================================================================
# Flujo alto nivel
# ============================================================================


def analyze_tumor_with_harvard_oxford(
    tumor_mask_path: str | Path,
    t1_img_path: Optional[str | Path] = None,
    work_dir: Optional[str | Path] = None,
    data_dir: Optional[str | Path] = None,
    min_overlap_voxels: int = 5,
    symmetric_split: bool = False,
) -> Dict[str, Any]:
    """
    Ejecuta un análisis completo tumor-vs-atlas con Harvard-Oxford.

    Comportamiento:
    - Si se proporciona `t1_img_path`, primero registra el estudio a MNI y usa
      la máscara tumoral registrada.
    - Si no se proporciona, asume que `tumor_mask_path` ya está en espacio MNI.

    Parámetros
    ----------
    tumor_mask_path : str | Path
        Máscara tumoral del caso.
    t1_img_path : str | Path | None
        T1 del estudio, necesario si la máscara todavía no está en MNI.
    work_dir : str | Path | None
        Carpeta de trabajo para resultados intermedios. Obligatoria si se va a
        registrar el caso con ANTs.
    data_dir : str | Path | None
        Carpeta de caché para atlas de nilearn.
    min_overlap_voxels : int, default=5
        Filtro mínimo para considerar una región afectada.
    symmetric_split : bool, default=False
        Pide a nilearn dividir regiones simétricas cuando sea posible.

    Devuelve
    --------
    dict[str, Any]
        Diccionario con:
        - `mask_mni_path`
        - `registration` (si hubo registro)
        - `rows` (todas las regiones con solape)
        - `top_regions` (las regiones ya ordenadas)
    """
    tumor_mask_path = _to_path(tumor_mask_path)

    if t1_img_path is not None:
        if work_dir is None:
            raise ValueError("Si pasas `t1_img_path`, también debes pasar `work_dir`.")
        reg_info = register_t1_and_mask_to_mni(
            t1_img_path=t1_img_path,
            tumor_mask_path=tumor_mask_path,
            work_dir=work_dir,
        )
        mask_mni_path = reg_info["registered_mask_path"]
    else:
        reg_info = None
        mask_mni_path = tumor_mask_path

    atlases = fetch_harvard_oxford_atlases(
        data_dir=data_dir,
        symmetric_split=symmetric_split,
    )

    all_rows: List[Dict[str, Any]] = []
    for atlas_spec in atlases:
        all_rows.extend(
            compute_region_overlaps(
                tumor_mask_mni_path=mask_mni_path,
                atlas_spec=atlas_spec,
                min_overlap_voxels=min_overlap_voxels,
            )
        )

        all_rows.sort(key=lambda r: (r["tumor_pct"], r["overlap_voxels"]), reverse=True)

        total_reported_overlap = sum(int(r.get("overlap_voxels", 0)) for r in all_rows)

        for r in all_rows:
            if total_reported_overlap > 0:
                r["reported_pct"] = round(
                    100.0 * int(r.get("overlap_voxels", 0)) / total_reported_overlap,
                    3,
                )
            else:
                r["reported_pct"] = 0.0

        return {
            "mask_mni_path": Path(mask_mni_path),
            "registration": reg_info,
            "rows": all_rows,
            "top_regions": all_rows,
        }

def summarize_top_regions(
    rows: List[Dict[str, Any]],
    top_n: int = 5,
    min_tumor_pct: float = 1.0,
) -> str:
    """
    Genera un resumen textual corto de las regiones con mayor solape.

    Este texto está pensado para mostrarlo en la futura pestaña de atlas de la
    app, no como informe clínico definitivo.
    """
    if not rows:
        return "No se detectó solape significativo entre el tumor y las regiones del atlas."

    picked = [r for r in rows if float(r.get("tumor_pct", 0.0)) >= float(min_tumor_pct)]
    if not picked:
        return "El tumor no alcanza el umbral mínimo de solape definido para informar regiones."

    picked = picked[: max(1, int(top_n))]
    parts = []
    for r in picked:
        parts.append(
            f"{r['region_name']} ({r['atlas_group']}, {r['tumor_pct']:.1f}% del tumor; {r.get('reported_pct', 0.0):.1f}% del informe)"
        )

    return "Posibles regiones anatómicas afectadas: " + "; ".join(parts) + "."
