from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, List
import csv

import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to

from config import APP_RESULTS_DIR
from calcular_volumen import iter_studies
from app import find_image_and_seg


APP = Path(__file__).resolve().parent
MANUAL_DIR = Path(r"H:\Imagenes_pintadas\todas")
OUT_CSV = APP / "Resultados" / "dice.csv"

HEADER = [
    "paciente_id",
    "study_date",
    "manual_path",
    "manual_vs_radionics",
    "manual_vs_nnunet_task501",
    "manual_vs_agunet",
    "manual_vs_dagunet",
    "manual_vs_pls-net",
    "manual_vs_unet-fv",
    "manual_vs_unet-slabs",
    "radionics_vs_unet-fv",
    "radionics_vs_unet-slabs",
    "nnunet_task501_vs_radionics",
    "nnunet_task501_vs_pls-net",
    "nnunet_task501_vs_unet-fv",
    "nnunet_task501_vs_unet-slabs",
    "agunet_vs_radionics",
    "agunet_vs_nnunet_task501",
    "agunet_vs_dagunet",
    "agunet_vs_pls-net",
    "agunet_vs_unet-fv",
    "agunet_vs_unet-slabs",
    "dagunet_vs_radionics",
    "dagunet_vs_nnunet_task501",
    "dagunet_vs_pls-net",
    "dagunet_vs_unet-fv",
    "dagunet_vs_unet-slabs",
    "pls-net_vs_radionics",
    "pls-net_vs_unet-fv",
    "pls-net_vs_unet-slabs",
    "unet-fv_vs_unet-slabs",
]

MODEL_KEYS = [
    "radionics",
    "nnunet_task501",
    "agunet",
    "dagunet",
    "pls-net",
    "unet-fv",
    "unet-slabs",
]


def load_nifti_img(path: Optional[Path]):
    if path is None or not Path(path).exists():
        return None
    try:
        return nib.load(str(path))
    except Exception:
        return None


def load_nifti_canonical(path: Optional[Path]):
    img = load_nifti_img(path)
    if img is None:
        return None
    try:
        return nib.as_closest_canonical(img)
    except Exception:
        return img


def align_mask_to_ref(mask_path: Optional[Path], ref_path: Optional[Path]):
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

        # Igual que en app.py
        data = np.flip(data, axis=0)
        data = np.flip(data, axis=1)
        data = np.flip(data, axis=1)

        return data

    except Exception:
        return None


def dice_score(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    a = (a > 0).astype(np.uint8)
    b = (b > 0).astype(np.uint8)
    inter = np.sum(a * b)
    return float((2.0 * inter + eps) / (np.sum(a) + np.sum(b) + eps))


def find_manual_seg(pid: str, manual_root: Path) -> Optional[Path]:
    """
    Busca por ID sin grado, admitiendo también barra baja final.
    Ejemplo:
      123131_G2 -> 123131.nii / 123131.nii.gz / 123131_.nii / 123131_.nii.gz
    """
    base_id = str(pid).split("_")[0].strip()

    direct = [
        manual_root / f"{base_id}.nii.gz",
        manual_root / f"{base_id}.nii",
        manual_root / f"{base_id}_.nii.gz",
        manual_root / f"{base_id}_.nii",
    ]
    for p in direct:
        if p.exists():
            return p

    patterns = [
        f"{base_id}.nii.gz",
        f"{base_id}.nii",
        f"{base_id}_.nii.gz",
        f"{base_id}_.nii",
    ]
    for pat in patterns:
        rec = list(manual_root.rglob(pat))
        if rec:
            return rec[0]

    return None


def get_ref_image_path(pid: str, sdate: str) -> Optional[Path]:
    for mk in MODEL_KEYS:
        imgp, _ = find_image_and_seg(pid, sdate, mk)
        if imgp is not None and Path(imgp).exists():
            return Path(imgp)
    return None


def get_model_seg_paths(pid: str, sdate: str) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for mk in MODEL_KEYS:
        _, segp = find_image_and_seg(pid, sdate, mk)
        if segp is not None and Path(segp).exists():
            out[mk] = Path(segp)
    return out


def init_empty_row(pid: str, sdate: str) -> Dict[str, object]:
    row: Dict[str, object] = {col: "" for col in HEADER}
    row["paciente_id"] = pid
    row["study_date"] = sdate
    return row


def set_pair_dice(row: Dict[str, object], a_key: str, b_key: str, value: object) -> None:
    direct = f"{a_key}_vs_{b_key}"
    reverse = f"{b_key}_vs_{a_key}"

    if direct in row:
        row[direct] = value
    elif reverse in row:
        row[reverse] = value


def compute_study_dices(pid: str, sdate: str, manual_root: Path) -> Dict[str, object]:
    row = init_empty_row(pid, sdate)

    manual_path = find_manual_seg(pid, manual_root)
    row["manual_path"] = manual_path.name if manual_path else ""

    ref_img_path = get_ref_image_path(pid, sdate)
    if ref_img_path is None:
        return row

    model_seg_paths = get_model_seg_paths(pid, sdate)
    if not model_seg_paths:
        return row

    aligned_models: Dict[str, np.ndarray] = {}

    for mk, seg_path in model_seg_paths.items():
        arr = align_mask_to_ref(seg_path, ref_img_path)
        if arr is not None:
            aligned_models[mk] = arr

    manual_arr = None
    if manual_path is not None:
        manual_arr = align_mask_to_ref(manual_path, ref_img_path)

    for mk in MODEL_KEYS:
        col = f"manual_vs_{mk}"
        if manual_arr is None or mk not in aligned_models:
            row[col] = ""
            continue

        pred = aligned_models[mk]
        if pred.shape != manual_arr.shape:
            row[col] = ""
            continue

        row[col] = round(dice_score(manual_arr, pred), 4)

    for i, mk1 in enumerate(MODEL_KEYS):
        for mk2 in MODEL_KEYS[i + 1:]:
            if mk1 not in aligned_models or mk2 not in aligned_models:
                set_pair_dice(row, mk1, mk2, "")
                continue

            a = aligned_models[mk1]
            b = aligned_models[mk2]

            if a.shape != b.shape:
                set_pair_dice(row, mk1, mk2, "")
                continue

            val = round(dice_score(a, b), 4)
            set_pair_dice(row, mk1, mk2, val)

    return row


def load_existing_rows(csv_path: Path) -> Dict[tuple[str, str], Dict[str, str]]:
    if not csv_path.exists():
        return {}

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    out: Dict[tuple[str, str], Dict[str, str]] = {}
    for row in rows:
        key = (
            str(row.get("paciente_id", "")).strip(),
            str(row.get("study_date", "")).strip(),
        )
        out[key] = row
    return out


def row_should_be_recomputed(row: Dict[str, str]) -> bool:
    return str(row.get("manual_path", "")).strip() == ""


def normalize_existing_row(row: Dict[str, str]) -> Dict[str, object]:
    out: Dict[str, object] = {col: "" for col in HEADER}
    for col in HEADER:
        if col == "manual_path":
            out[col] = Path(str(row.get(col, "")).strip()).name if str(row.get(col, "")).strip() else ""
        else:
            out[col] = row.get(col, "")
    return out


def build_dice_csv(manual_root: Path, out_csv: Path = OUT_CSV) -> Path:
    APP_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    studies = iter_studies()
    if not studies:
        raise FileNotFoundError("No hay estudios en Pacientes_nifti.")

    existing = load_existing_rows(out_csv)

    rows: List[Dict[str, object]] = []

    for pid, sdate, _study_dir in studies:
        key = (str(pid).strip(), str(sdate).strip())
        prev = existing.get(key)

        if prev is not None and not row_should_be_recomputed(prev):
            rows.append(normalize_existing_row(prev))
            continue

        rows.append(compute_study_dices(pid, sdate, manual_root))

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)

    return out_csv


def main():
    out = build_dice_csv(MANUAL_DIR, OUT_CSV)
    print(f"[OK] CSV guardado en: {out}")


if __name__ == "__main__":
    main()