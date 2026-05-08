from __future__ import annotations
"""
calcular_volumen.py
Cálculo de volumen tumoral (mL) a partir de segmentaciones NIfTI.

Estructura esperada (NUEVA, por estudio):
Pacientes_nifti/<PACIENTE_ID>/<FECHA_DICOM>/
  radionics/<ID>_radionics_seg.nii(.gz)
  agunet/<ID>_agunet_seg.nii(.gz)
  dagunet/<ID>_dagunet_seg.nii(.gz)
  pls-net/<ID>_pls-net_seg.nii(.gz)
  unet-fv/<ID>_unet-fv_seg.nii(.gz)
  unet-slabs/<ID>_unet-slabs_seg.nii(.gz)
  nnunet/task501/<ID>_task501_seg_clean.nii(.gz)  (si existe; si no, _seg)
Salida:
- Resultados/volumenes.csv

Notas importantes:
- El cálculo usa la máscara binaria (voxeles > 0). Si el archivo es un mapa de probabilidad (float 0..1),
  se umbraliza con prob_thr (por defecto 0.5).
- El script NO debe fallar si falta alguna segmentación: deja la celda vacía.
- Recorre TODOS los estudios (ID+FECHA). Esto permite varios MR por paciente.
"""

from pathlib import Path
import csv
from typing import Dict, Iterable, List, Optional, Tuple

from config import list_studies as config_list_studies, normalize_study_folder_name, APP_RESULTS_DIR

import SimpleITK as sitk
import numpy as np

from config import (
    list_studies as config_list_studies,
    normalize_study_folder_name,
    APP_RESULTS_DIR,
    PACIENTES_NIFTI_DIR,
)
APP = Path(__file__).resolve().parent

PACIENTES = PACIENTES_NIFTI_DIR
OUT_CSV = APP_RESULTS_DIR / "volumenes.csv"

AGU_KEYS = ["agunet", "dagunet", "pls-net", "unet-fv", "unet-slabs"]
NNUNET_TASKS = ["task501"]

WEIGHTED_MODEL_KEY = "media_ponderada"

MODEL_DICE_WEIGHTS = {
    "radionics": 0.8654131578947368,
    "nnunet_task501": 0.8029644736842105,
    "agunet": 0.8283644736842105,
    "dagunet": 0.8231407894736842,
    "pls-net": 0.8211302631578947,
    "unet-fv": 0.7921368421052631,
    "unet-slabs": 0.8276710526315789,
}

MODEL_LABELS = {
    "radionics": "Radionics",
    "nnunet_task501": "nnU-Net (Task501)",
    "agunet": "AGUNet",
    "dagunet": "DAGUNet",
    "pls-net": "PLS-Net",
    "unet-fv": "UNet-FV",
    "unet-slabs": "UNet-Slabs",
    "media_ponderada": "Media ponderada",
}

def volume_ml(seg_path: Path, prob_thr: float = 0.5) -> float:
    img = sitk.ReadImage(str(seg_path))
    arr = sitk.GetArrayFromImage(img)

    is_float = arr.dtype.kind == "f"
    sample = arr[::10, ::10, ::10]
    uniq = np.unique(sample)
    looks_prob = is_float and (len(uniq) > 3) and (float(np.nanmax(arr)) <= 1.0)

    if looks_prob:
        mask = arr >= prob_thr
    else:
        mask = arr > 0

    vox = int(np.count_nonzero(mask))
    sx, sy, sz = img.GetSpacing()  # mm
    return (vox * sx * sy * sz) / 1000.0


def _exists_any(p: Path) -> Optional[Path]:
    if p.exists():
        return p
    # probar .nii si venía .nii.gz o viceversa
    if p.suffixes[-2:] == [".nii", ".gz"]:
        p2 = p.with_suffix("")  # quita .gz
        if p2.exists():
            return p2
    if p.suffix == ".nii":
        p2 = p.with_suffix(".nii.gz")
        if p2.exists():
            return p2
    return None


def find_radionics(pid: str, study_dir: Path) -> Optional[Path]:
    p = study_dir / "radionics" / f"{pid}_radionics_seg.nii.gz"
    p2 = _exists_any(p)
    if p2 is not None:
        return p2
    # fallback por si alguien guardó con otro sufijo
    c = list((study_dir / "radionics").glob(f"{pid}_radionics_seg.nii*"))
    return c[0] if c else None


def find_agu(pid: str, study_dir: Path, key: str) -> Optional[Path]:
    p = study_dir / key / f"{pid}_{key}_seg.nii.gz"
    p2 = _exists_any(p)
    if p2 is not None:
        return p2
    c = list((study_dir / key).glob(f"{pid}_{key}_seg.nii*"))
    return c[0] if c else None


def find_nnunet(pid: str, study_dir: Path, task: str) -> Optional[Path]:
    base = study_dir / "nnunet" / task
    # preferimos clean
    p_clean = base / f"{pid}_{task}_seg_clean.nii.gz"
    p2 = _exists_any(p_clean)
    if p2 is not None:
        return p2
    # luego seg normal
    p = base / f"{pid}_{task}_seg.nii.gz"
    p3 = _exists_any(p)
    if p3 is not None:
        return p3
    # fallback: cualquier .nii* en esa carpeta
    if base.exists():
        c = sorted(list(base.glob(f"{pid}_{task}_seg*.nii*")))
        return c[0] if c else None
    return None

def find_model_seg(pid: str, study_dir: Path, model_key: str) -> Optional[Path]:
    if model_key == "radionics":
        return find_radionics(pid, study_dir)

    if model_key == "nnunet_task501":
        return find_nnunet(pid, study_dir, "task501")

    if model_key in AGU_KEYS:
        return find_agu(pid, study_dir, model_key)

    if model_key == WEIGHTED_MODEL_KEY:
        p = study_dir / WEIGHTED_MODEL_KEY / f"{pid}_{WEIGHTED_MODEL_KEY}_seg.nii.gz"
        return p if p.exists() else None

    return None


def _same_sitk_geometry(a: sitk.Image, b: sitk.Image) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing())
        and np.allclose(a.GetOrigin(), b.GetOrigin())
        and np.allclose(a.GetDirection(), b.GetDirection())
    )


def _resample_mask_to_ref(mask_img: sitk.Image, ref_img: sitk.Image) -> sitk.Image:
    if _same_sitk_geometry(mask_img, ref_img):
        return mask_img

    return sitk.Resample(
        mask_img,
        ref_img,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )


def build_weighted_ensemble_seg(
    pid: str,
    study_dir: Path,
    threshold: float = 0.5,
    force: bool = False,
) -> Optional[Path]:
    """
    Genera una segmentación NIfTI por voto ponderado.

    Cada modelo aporta su máscara binaria con peso igual a su DICE medio.
    El resultado se guarda en:
      <study_dir>/media_ponderada/<pid>_media_ponderada_seg.nii.gz
    """
    out_dir = study_dir / WEIGHTED_MODEL_KEY
    out_path = out_dir / f"{pid}_{WEIGHTED_MODEL_KEY}_seg.nii.gz"

    if out_path.exists() and not force:
        return out_path

    sources: list[tuple[str, Path, float]] = []

    for model_key, weight in MODEL_DICE_WEIGHTS.items():
        seg_path = find_model_seg(pid, study_dir, model_key)
        if seg_path is not None and seg_path.exists():
            sources.append((model_key, seg_path, float(weight)))

    if len(sources) < 2:
        return None

    ref_img = sitk.ReadImage(str(sources[0][1]))
    ref_arr = sitk.GetArrayFromImage(ref_img)
    acc = np.zeros(ref_arr.shape, dtype=np.float32)
    total_weight = 0.0

    for _model_key, seg_path, weight in sources:
        seg_img = sitk.ReadImage(str(seg_path))
        seg_img = sitk.Cast(seg_img > 0, sitk.sitkUInt8)
        seg_img = _resample_mask_to_ref(seg_img, ref_img)

        arr = sitk.GetArrayFromImage(seg_img).astype(np.uint8)
        acc += weight * (arr > 0).astype(np.float32)
        total_weight += weight

    if total_weight <= 0:
        return None

    prob = acc / total_weight
    mask = (prob >= float(threshold)).astype(np.uint8)

    out_img = sitk.GetImageFromArray(mask)
    out_img.CopyInformation(ref_img)

    out_dir.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(out_img, str(out_path))

    return out_path


def build_weighted_ensemble_seg_for_study(
    pid: str,
    sdate: str,
    threshold: float = 0.5,
    force: bool = False,
) -> Optional[Path]:
    studies = iter_studies(patient_id=pid, study_date=sdate)
    if not studies:
        raise FileNotFoundError(f"No encuentro el estudio {pid}/{sdate} en {PACIENTES}")

    _pid, _study_name, study_dir = studies[0]
    return build_weighted_ensemble_seg(
        pid=pid,
        study_dir=study_dir,
        threshold=threshold,
        force=force,
    )

def iter_studies(patient_id: Optional[str] = None, study_date: Optional[str] = None) -> List[Tuple[str, str, Path]]:
    """
    Devuelve lista (pid, study_date, study_dir).
    Soporta carpetas de estudio tipo YYYYMMDD y YYYYMMDD_01.
    Permite filtrar por paciente y/o estudio concreto.
    """
    out: List[Tuple[str, str, Path]] = []
    try:
        if not PACIENTES.exists() or not PACIENTES.is_dir():
            return out
    except (FileNotFoundError, OSError):
        return out

    pid_dirs: Iterable[Path]
    if patient_id:
        pid_dir = PACIENTES / patient_id
        pid_dirs = [pid_dir] if pid_dir.exists() else []
    else:
        try:
            pid_dirs = sorted([p for p in PACIENTES.iterdir() if p.is_dir()])
        except (FileNotFoundError, OSError):
            return out

    wanted_study = normalize_study_folder_name(study_date) if study_date else None

    for pid_dir in pid_dirs:
        pid = pid_dir.name
        valid_studies = set(config_list_studies(pid))
        for study_name in sorted(valid_studies):
            if wanted_study and study_name != wanted_study:
                continue
            study_dir = pid_dir / study_name
            if study_dir.exists() and study_dir.is_dir():
                out.append((pid, study_name, study_dir))
    return out


def compute_study_volumes(pid: str, sdate: str) -> Dict[str, Optional[float]]:
    studies = iter_studies(patient_id=pid, study_date=sdate)
    if not studies:
        raise FileNotFoundError(f"No encuentro el estudio {pid}/{sdate} en {PACIENTES}")

    _, study_name, sdir = studies[0]
    row: Dict[str, Optional[float]] = {"paciente_id": pid, "study_date": study_name}

    rp = find_radionics(pid, sdir)
    row["radionics"] = round(volume_ml(rp), 3) if rp else None

    for k in AGU_KEYS:
        sp = find_agu(pid, sdir, k)
        row[k] = round(volume_ml(sp), 3) if sp else None

    for t in NNUNET_TASKS:
        col = f"nnunet_{t}"
        sp = find_nnunet(pid, sdir, t)
        row[col] = round(volume_ml(sp), 3) if sp else None

    # Generar también la segmentación NIfTI de media ponderada
    weighted_seg_path = build_weighted_ensemble_seg(
        pid=pid,
        study_dir=sdir,
        threshold=0.5,
        force=False,
    )

    # El volumen de media_ponderada se calcula desde SU máscara NIfTI,
    # no como media ponderada de los volúmenes de los modelos.
    row[WEIGHTED_MODEL_KEY] = (
        round(volume_ml(weighted_seg_path), 3)
        if weighted_seg_path is not None and Path(weighted_seg_path).exists()
        else None
    )

    return row


def format_volume_text(model_key: str, study_vols: Dict[str, Optional[float]]) -> str:
    label = MODEL_LABELS.get(model_key, model_key)

    vol = study_vols.get(model_key)
    weighted_vol = study_vols.get(WEIGHTED_MODEL_KEY)

    lines = []

    if vol is None:
        lines.append(f"No hay segmentación disponible para {label}.")
    else:
        lines.append(f"Volumen ({label}): {float(vol):.3f} mL")

    if model_key != WEIGHTED_MODEL_KEY and weighted_vol is not None:
        lines.append(f"Volumen (media ponderada): {float(weighted_vol):.3f} mL")

    return "\n".join(lines)

def build_volumes_csv(patient_id: Optional[str] = None, study_date: Optional[str] = None, out_csv: Optional[Path] = None) -> Path:
    APP_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    studies = iter_studies(patient_id=patient_id, study_date=study_date)
    if not studies:
        raise FileNotFoundError(f"No hay estudios en {PACIENTES} con los filtros indicados")

    present_cols = {"radionics"}

    for pid, sdate, sdir in studies:
        if find_radionics(pid, sdir):
            present_cols.add("radionics")
        for k in AGU_KEYS:
            if find_agu(pid, sdir, k):
                present_cols.add(k)
        for t in NNUNET_TASKS:
            col = f"nnunet_{t}"
            if find_nnunet(pid, sdir, t):
                present_cols.add(col)

    cols: List[str] = ["paciente_id", "study_date", "radionics"]
    for k in AGU_KEYS:
        if k in present_cols:
            cols.append(k)
    for t in NNUNET_TASKS:
        col = f"nnunet_{t}"
        if col in present_cols:
            cols.append(col)
            
    cols.append(WEIGHTED_MODEL_KEY)

    rows: List[Dict[str, str]] = []
    for pid, sdate, _sdir in studies:
        raw_row = compute_study_volumes(pid, sdate)
        row: Dict[str, str] = {"paciente_id": pid, "study_date": str(raw_row["study_date"])}
        for col in cols:
            if col in {"paciente_id", "study_date"}:
                continue
            value = raw_row.get(col)
            row[col] = "" if value is None else str(value)
        rows.append(row)

    dest = out_csv or OUT_CSV
    with open(dest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    return dest


def main(patient_id: Optional[str] = None, study_date: Optional[str] = None, out_csv: Optional[Path] = None):
    dest = build_volumes_csv(patient_id=patient_id, study_date=study_date, out_csv=out_csv)
    print(f"[OK] Guardado -> {dest}")
    return dest


if __name__ == "__main__":
    main()