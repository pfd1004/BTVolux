"""
segmentation.py

Implementación de los modelos de segmentación distintos de Radionics.

Incluye:
- nnU-Net v1
- familia AGUNet (AGUNet, DAGUNet, PLS-Net, UNet-FV, UNet-Slabs)

Responsabilidades:
- Preparar los inputs en el formato esperado por cada modelo.
- Lanzar la inferencia externa.
- Renombrar y guardar salidas en rutas canónicas del proyecto.
- Generar segmentaciones binarias a partir de mapas de probabilidad cuando sea necesario.

Idea clave:
- Cada modelo puede producir salidas con nombres distintos en su propio repositorio,
  pero este archivo las normaliza a un formato común para el resto del proyecto.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import SimpleITK as sitk

from config import (
    get_patient_paths,
    NNUNET_RAW_DATA_BASE,
    NNUNET_PREPROCESSED,
    NNUNET_RESULTS,
    NNUNET_TASK501_NAME,
    NNUNET_PYTHON_EXE,
    NNUNET_PREDICT_EXE,
    AGUNET_MAIN,
    AGUNET_PYTHON_EXE,
    PLSNET_PYTHON_EXE,
    AGUNET_PROB_THRESHOLD,
)
# =========================
# nnU-Net
# =========================

def ensure_nnunet_env() -> None:
    if "nnUNet_raw_data_base" not in os.environ:
        os.environ["nnUNet_raw_data_base"] = str(NNUNET_RAW_DATA_BASE)
    if "nnUNet_preprocessed" not in os.environ:
        os.environ["nnUNet_preprocessed"] = str(NNUNET_PREPROCESSED)
    if "RESULTS_FOLDER" not in os.environ:
        os.environ["RESULTS_FOLDER"] = str(NNUNET_RESULTS)


def _clean_dir(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def _pick_first_nii(folder: Path) -> Path:
    cands = sorted(list(folder.glob("*.nii.gz")) + list(folder.glob("*.nii")))
    if not cands:
        raise FileNotFoundError(f"No hay NIfTI en {folder}")
    return cands[0]

def run_nnunet_predict(
    input_dir: Path,
    output_dir: Path,
    task_name: str,
    model: str = "3d_fullres",
    folds: str = "all",
    step_size: str = "0.25",
    n_threads_preproc: int = 1,
    n_threads_save: int = 1,
) -> None:
    ensure_nnunet_env()
    output_dir.mkdir(parents=True, exist_ok=True)

    python_exe = Path(NNUNET_PYTHON_EXE)

    if python_exe.exists():
        cmd = [
            str(python_exe),
            "-m", "nnunet.inference.predict_simple",
            "-i", str(input_dir),
            "-o", str(output_dir),
            "-t", task_name,
            "-m", model,
            "-f", str(folds),
            "--step_size", str(step_size),
            "--num_threads_preprocessing", str(int(n_threads_preproc)),
            "--num_threads_nifti_save", str(int(n_threads_save)),
        ]
    else:
        cmd = [
            str(NNUNET_PREDICT_EXE),
            "-i", str(input_dir),
            "-o", str(output_dir),
            "-t", task_name,
            "-m", model,
            "-f", str(folds),
            "--step_size", str(step_size),
            "--num_threads_preprocessing", str(int(n_threads_preproc)),
            "--num_threads_nifti_save", str(int(n_threads_save)),
        ]

    print("[nnU-Net]", " ".join(cmd))

    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"

    subprocess.run(cmd, check=True, cwd=str(output_dir), env=env)

# ============================================================
# Postproceso / limpieza con brainmask
# ============================================================
def cleanup_seg_with_brainmask(
    seg_path: Path,
    brain_mask_path: Path,
    out_path: Path,
    dilate: int = 2,
    keep_largest_cc: bool = True,
) -> None:
    seg = sitk.ReadImage(str(seg_path))
    mask = sitk.ReadImage(str(brain_mask_path))

    seg_bin = sitk.Cast(seg > 0, sitk.sitkUInt8)
    mask_bin = sitk.Cast(mask > 0, sitk.sitkUInt8)

    if dilate and dilate > 0:
        mask_bin = sitk.BinaryDilate(mask_bin, [int(dilate)] * 3)

    cleaned = sitk.And(seg_bin, mask_bin)

    if keep_largest_cc:
        cc = sitk.ConnectedComponent(cleaned)
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(cc)
        if stats.GetNumberOfLabels() > 0:
            largest = max(stats.GetLabels(), key=lambda l: stats.GetPhysicalSize(l))
            cleaned = sitk.Cast(cc == largest, sitk.sitkUInt8)

    cleaned.CopyInformation(seg)
    sitk.WriteImage(cleaned, str(out_path))


def segmentar_paciente_nnunet(
    paciente_id: str,
    study_date: str,
    use_ants_brain: bool = True,
    run_501: bool = True,
    cleanup_with_brainmask: bool = True,
    cleanup_dilate: int = 2,
    cleanup_keep_largest_cc: bool = True,
    cleanup_overwrite: bool = False,
) -> None:
    """
    Ejecuta nnU-Net para un estudio concreto.

    Lógica de entrada:
    - prioriza el volumen skull-stripped de SynthStrip
    - si no existe, usa el volumen skull-stripped de ANTs
    - si tampoco existe, usa el NIfTI original

    Lógica de salida:
    - guarda la predicción en una ruta canónica del estudio
    - opcionalmente genera una versión limpia usando la brainmask

    Objetivo:
    desacoplar la salida real de nnU-Net del resto del proyecto.
    """
    P = get_patient_paths(paciente_id, study_date)

    if use_ants_brain and P["synth_brain"].exists():
        in_nii = P["synth_brain"]
    elif use_ants_brain and P["ants_brain"].exists():
        in_nii = P["ants_brain"]
    else:
        in_nii = P["nifti_img"]

    if not in_nii.exists():
        raise FileNotFoundError(f"No existe input nnU-Net: {in_nii}")

    tmp_in = P["nnunet_in_dir"]
    _clean_dir(tmp_in)
    tmp_nii = tmp_in / f"{paciente_id}_0000.nii.gz"
    shutil.copy2(in_nii, tmp_nii)

    tasks = [
        ("task501", NNUNET_TASK501_NAME, run_501),
    ]

    for task_key, task_name, enabled in tasks:
        if not enabled:
            continue

        out_dir = P["nnunet_dir"] / task_key
        out_dir.mkdir(parents=True, exist_ok=True)

        tmp_out = P["tmp_dir"] / f"nnunet_out_{task_key}"
        _clean_dir(tmp_out)

        print(f"[nnU-Net:{task_key}] {paciente_id} ({study_date})")
        run_nnunet_predict(tmp_in, tmp_out, task_name, model="3d_fullres", folds="all", step_size="0.25")

        produced = _pick_first_nii(tmp_out)

        final_seg = out_dir / f"{paciente_id}_{task_key}_seg.nii.gz"
        if final_seg.exists():
            final_seg.unlink()
        shutil.copy2(produced, final_seg)
        print(f"[nnU-Net:{task_key}] Guardado -> {final_seg}")

        if cleanup_with_brainmask:
            brain_mask = None
            if P["synth_mask"].exists():
                brain_mask = P["synth_mask"]
            elif P["ants_mask"].exists():
                brain_mask = P["ants_mask"]

            if brain_mask is not None:
                clean_seg = out_dir / f"{paciente_id}_{task_key}_seg_clean.nii.gz"
                cleanup_seg_with_brainmask(
                    seg_path=final_seg,
                    brain_mask_path=brain_mask,
                    out_path=clean_seg,
                    dilate=int(cleanup_dilate),
                    keep_largest_cc=bool(cleanup_keep_largest_cc),
                )
                print(f"[nnU-Net:{task_key}] Clean -> {clean_seg}")

                if cleanup_overwrite:
                    shutil.copy2(clean_seg, final_seg)
                    print(f"[nnU-Net:{task_key}] Overwrite seg -> {final_seg}")


# =========================
# AGUNet
# =========================

AGU_MODE_MAP = {
    "agunet": "AGUNet",
    "dagunet": "DAGUNet",
    "pls-net": "PLS-Net",
    "unet-fv": "UNet-FV",
    "unet-slabs": "UNet-Slabs",
}


def _suffixes_str(p: Path) -> str:
    return "".join(p.suffixes) if p.suffixes else p.suffix


def threshold_prob_to_seg(prob_path: Path, seg_path: Path, thr: float = 0.5) -> None:
    p = sitk.ReadImage(str(prob_path))
    seg = sitk.BinaryThreshold(p, lowerThreshold=float(thr), upperThreshold=1e9, insideValue=1, outsideValue=0)
    seg = sitk.Cast(seg, sitk.sitkUInt8)
    sitk.WriteImage(seg, str(seg_path))

def _get_agunet_python_for_mode(mode_key: str) -> str:
    mode_key = mode_key.lower().strip()
    if mode_key == "pls-net":
        return str(PLSNET_PYTHON_EXE)
    return str(AGUNET_PYTHON_EXE)

def run_agunet_patient(
    paciente_id: str,
    study_date: str,
    gpu_id: int = 0,
    force: bool = False,
    mode_key: str = "agunet",
) -> Path:
    """
    Ejecuta un modelo de la familia AGUNet sobre un estudio.

    Pasos:
    1. Elegir el ejecutable Python adecuado para el modelo.
    2. Lanzar el script externo del repositorio original.
    3. Localizar el output real generado por ese repositorio.
    4. Renombrarlo al formato canónico del proyecto:
       <id>_<mode>_prob.nii(.gz)
    5. Umbralizar la probabilidad para generar:
       <id>_<mode>_seg.nii.gz

    Esta función encapsula toda la lógica de compatibilidad con repos externos.
    """
    mode_key = mode_key.lower().strip()
    if mode_key not in AGU_MODE_MAP:
        raise ValueError(f"mode_key inválido: {mode_key}. Usa: {sorted(AGU_MODE_MAP.keys())}")

    model_name = AGU_MODE_MAP[mode_key]
    P = get_patient_paths(paciente_id, study_date)

    out_dir_key = f"{mode_key}_dir"
    out_dir = P[out_dir_key] if out_dir_key in P else (P["root"] / mode_key)
    out_dir.mkdir(parents=True, exist_ok=True)

    in_nii = P["nifti_img"]
    if not in_nii.exists():
        raise FileNotFoundError(f"No existe input para {model_name}: {in_nii}")

    python_exe = _get_agunet_python_for_mode(mode_key)

    if not Path(python_exe).exists():
        raise FileNotFoundError(f"No encuentro Python para {model_name}: {python_exe}")
    if not Path(AGUNET_MAIN).exists():
        raise FileNotFoundError(f"No encuentro AGUNET_MAIN: {AGUNET_MAIN}")

    existing_prob = sorted(list(out_dir.glob(f"{paciente_id}_{mode_key}_prob.nii*")))
    seg_out = out_dir / f"{paciente_id}_{mode_key}_seg.nii.gz"

    if existing_prob and not force:
        prob_out = Path(existing_prob[0])
        if not seg_out.exists():
            threshold_prob_to_seg(prob_out, seg_out, thr=AGUNET_PROB_THRESHOLD)
            print(f"[{model_name}] Umbralizado (post-SKIP) -> {seg_out}")
        print(f"[{model_name}] SKIP: ya existe {prob_out.name}")
        return prob_out

    cmd = [
        python_exe,
        str(AGUNET_MAIN),
        "-i", str(in_nii),
        "-o", str(out_dir),
        "-m", model_name,
        "-g", str(gpu_id),
    ]
    print(f"[{model_name}] {paciente_id} ({study_date}): {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    root = P["root"]

    # Algunos repos externos no generan siempre el mismo nombre de salida.
    # Por eso se buscan varias posibilidades y, en último término, se toma
    # el NIfTI más reciente compatible.
    def _pick_repo_output(root_dir: Path, out_dir2: Path, mk: str) -> Optional[Path]:
        exact_names = [
            f"{mk}-pred_Tumor.nii.gz",
            f"{mk}-pred_tumor.nii.gz",
            f"{mk}-pred_Tumor.nii",
            f"{mk}-pred_tumor.nii",
            "pred_Tumor.nii.gz",
            "pred_tumor.nii.gz",
            "pred_Tumor.nii",
            "pred_tumor.nii",
        ]
        for name in exact_names:
            p = out_dir2 / name
            if p.exists():
                return p
            p = root_dir / name
            if p.exists():
                return p

        ignore_substr = ["_prob", "_seg"]
        patterns = [
            f"{mk}-pred_*tumor*.nii.gz",
            f"{mk}-pred_*tumor*.nii",
            "*-pred_*tumor*.nii.gz",
            "*-pred_*tumor*.nii",
            "*pred*tumor*.nii.gz",
            "*pred*tumor*.nii",
            "*.nii.gz",
            "*.nii",
        ]

        def hits(folder: Path, pat: str):
            lst = list(folder.glob(pat))
            return [p for p in lst if not any(s in p.name for s in ignore_substr)]

        for pat in patterns:
            cand = hits(out_dir2, pat) + hits(root_dir, pat)
            if cand:
                cand.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                return cand[0]
        return None

    produced = _pick_repo_output(root, out_dir, mode_key)
    if produced is None:
        raise FileNotFoundError(f"[{model_name}] No encuentro output .nii/.nii.gz tras ejecutar en {out_dir}")

    ext = _suffixes_str(produced)
    prob_out = out_dir / f"{paciente_id}_{mode_key}_prob{ext}"
    if prob_out.exists():
        prob_out.unlink()

    shutil.move(str(produced), str(prob_out))
    print(f"[{model_name}] Guardado prob -> {prob_out}")

    if (not seg_out.exists()) or force:
        threshold_prob_to_seg(prob_out, seg_out, thr=AGUNET_PROB_THRESHOLD)
        print(f"[{model_name}] Umbralizado -> {seg_out}")

    return prob_out
