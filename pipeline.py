from __future__ import annotations
"""
pipeline.py

Orquestador principal del procesamiento por estudio.

Responsabilidades:
- Asegurar que el estudio tenga NIfTI base.
- Preparar skull-stripping/brainmask cuando un modelo lo necesite.
- Lanzar los distintos modelos de segmentación.
- Permitir ejecución individual o conjunta.
- Continuar con el resto de modelos aunque uno falle.

Filosofía:
- preprocessing.py prepara el estudio.
- radionics.py y segmentation.py ejecutan modelos concretos.
- pipeline.py coordina el flujo completo.
"""

from typing import Optional

from config import (
    RADIONICS_MODEL_DIR,
    get_all_patient_ids,
    get_patient_paths,
    BRAINMASK_METHOD,
    list_studies as config_list_studies,
)
from preprocessing import (
    prepare_study_from_dicom,
    convertir_dicom_a_nifti,
    brain_extraction_ants,
    brain_extraction_synthstrip,
)

from radionics import run_radionics
from segmentation import segmentar_paciente_nnunet, run_agunet_patient

# ============================================================
# Helpers de preparación previa
# ============================================================

def _ensure_nifti(pid: str, study_date: str) -> None:
    P = get_patient_paths(pid, study_date)
    if not P["nifti_img"].exists():
        convertir_dicom_a_nifti(pid, study_date)


def _ensure_ants_brainmask(pid: str, study_date: str, modality_ants: str = "t1") -> None:
    P = get_patient_paths(pid, study_date)
    if not (P["ants_brain"].exists() and P["ants_mask"].exists()):
        brain_extraction_ants(
            paciente_id=pid,
            study_date=study_date,
            input_nii=P["nifti_img"],
            modality=modality_ants,
            force=False,
        )


def _ensure_synthstrip_brainmask(pid: str, study_date: str) -> None:
    P = get_patient_paths(pid, study_date)
    if not (P["synth_brain"].exists() and P["synth_mask"].exists()):
        brain_extraction_synthstrip(
            paciente_id=pid,
            study_date=study_date,
            input_nii=str(P["nifti_img"]),
            force=False,
        )

def _exists_any_nifti(p) -> bool:
    p = str(p)
    from pathlib import Path

    path = Path(p)
    if path.exists():
        return True

    if "".join(path.suffixes[-2:]) == ".nii.gz":
        if path.with_suffix("").exists():
            return True

    if path.suffix == ".nii":
        if path.with_suffix(".nii.gz").exists():
            return True

    return False


def _model_output_exists(paciente_id: str, study_date: str, mode: str) -> bool:
    P = get_patient_paths(paciente_id, study_date)

    if mode == "radionics":
        return _exists_any_nifti(P["rad_seg"])

    if mode == "nnunet":
        base = P["nnunet_dir"] / "task501"
        candidates = [
            base / f"{paciente_id}_task501_seg_clean.nii.gz",
            base / f"{paciente_id}_task501_seg_clean.nii",
            base / f"{paciente_id}_task501_seg.nii.gz",
            base / f"{paciente_id}_task501_seg.nii",
        ]
        return any(p.exists() for p in candidates)

    if mode in {"agunet", "dagunet", "pls-net", "unet-fv", "unet-slabs"}:
        out_dir = P[f"{mode}_dir"]
        candidates = [
            out_dir / f"{paciente_id}_{mode}_seg.nii.gz",
            out_dir / f"{paciente_id}_{mode}_seg.nii",
        ]
        return any(p.exists() for p in candidates)

    if mode == "all":
        required = [
            _model_output_exists(paciente_id, study_date, "radionics"),
            _model_output_exists(paciente_id, study_date, "nnunet"),
            _model_output_exists(paciente_id, study_date, "agunet"),
            _model_output_exists(paciente_id, study_date, "dagunet"),
            _model_output_exists(paciente_id, study_date, "pls-net"),
            _model_output_exists(paciente_id, study_date, "unet-fv"),
            _model_output_exists(paciente_id, study_date, "unet-slabs"),
        ]
        return all(required)

    return False

# ============================================================
# Pipeline principal por estudio
# ============================================================

def pipeline_study(
    paciente_id: str,
    study_date: str,
    mode: str = "radionics",
    modality_ants: str = "t1",
    gpu_id_agunet: int = 0,
    use_ants_for_nnunet: bool = True,
    use_ants_for_radionics: bool = False,
    make_mask_for_radionics_cleanup: bool = True,
    rad_mask_dilate: int = 1,
    rad_keep_largest_cc: bool = True,
    run_501: bool = True,
    progress_cb=None,
    progress_log_path: Optional[str] = None,
    skip_completed: bool = False,
) -> list[str]:
    """
    Ejecuta el pipeline de segmentación para un estudio concreto.

    Orden general:
    1. Asegurar NIfTI base.
    2. Preparar brainmask si algún modelo la necesita.
    3. Ejecutar Radionics si corresponde.
    4. Ejecutar familia AGUNet si corresponde.
    5. Ejecutar nnU-Net si corresponde.
    6. Devolver una lista de errores sin interrumpir el resto del pipeline.

    Notas:
    - Radionics trabaja directamente sobre el NIfTI base.
    - nnU-Net puede usar skull-stripping previo.
    - mode="all" intenta ejecutar todos los modelos disponibles.
    - Si un modelo falla, el pipeline continúa con los demás.
    """
    # Importante:
    # el log se escribe en modo "w" para conservar solo el último estado visible
    # en la app. Si se quisiera un histórico completo por ejecución, habría que
    # cambiarlo a modo "a".
    def emit(msg: str) -> None:
        print(msg)

        if progress_log_path:
            try:
                with open(progress_log_path, "w", encoding="utf-8") as f:
                    f.write(msg + "\n")
            except Exception:
                pass

        if progress_cb is not None:
            try:
                progress_cb(msg)
            except Exception:
                pass
    mode = mode.lower().strip()
    allowed = {
        "radionics",
        "nnunet",
        "all",
        "agunet",
        "dagunet",
        "pls-net",
        "unet-fv",
        "unet-slabs",
    }
    if mode not in allowed:
        raise ValueError("mode debe ser: " + " | ".join(sorted(allowed)))

    errors: list[str] = []
    P = get_patient_paths(paciente_id, study_date)

    # 1) Asegurar NIfTI base
    _ensure_nifti(paciente_id, study_date)

    # 2) Preparación SOLO para nnU-Net
    # nnU-Net lo necesita siempre si se ejecuta solo o dentro de "all"
    if mode in {"nnunet", "all"}:
        try:
            emit("[nnU-Net] Preparando brainmask...")
            if BRAINMASK_METHOD.lower() == "synthstrip":
                _ensure_synthstrip_brainmask(paciente_id, study_date)
                emit("[nnU-Net] SynthStrip terminado")
            else:
                _ensure_ants_brainmask(
                    paciente_id,
                    study_date,
                    modality_ants=modality_ants,
                )
                emit("[nnU-Net] ANTs terminado")
        except Exception as e:
            msg = f"Preparación nnU-Net: {e}"
            errors.append(msg)
            print(f"[nnU-Net prep] ERROR: {e}")

            # si el usuario pidió solo nnU-Net, paramos aquí porque no puede continuar
            if mode == "nnunet":
                return errors

    # 3) Radionics
    # IMPORTANTE: nunca usa SynthStrip, nunca usa ANTs, nunca usa máscara externa
    if mode in {"radionics", "all"}:
        if skip_completed and _model_output_exists(paciente_id, study_date, "radionics"):
            emit("[Radionics] SKIP: ya existe segmentación")
        else:
            try:
                emit("[Radionics] Ejecutando...")
                run_radionics(
                    input_nii=str(P["nifti_img"]),
                    model_dir=str(RADIONICS_MODEL_DIR),
                    output_nii=str(P["rad_seg"]),
                    thr=0.5,
                    step=80,
                    use_gpu=True,
                    output_prob_nii=None,
                    ants_mask_nii=None,
                    mask_dilate=rad_mask_dilate,
                    keep_largest_cc=rad_keep_largest_cc,
                )
                emit("[Radionics] Terminado")
                print(f"[Radionics] Guardado -> {P['rad_seg']}")
            except Exception as e:
                msg = f"Radionics: {e}"
                errors.append(msg)
                print(f"[Radionics] ERROR: {e}")

    # 4) Familia AGUNet
    if mode == "all":
        for agu_mode in ["agunet", "dagunet", "pls-net", "unet-fv", "unet-slabs"]:
            if skip_completed and _model_output_exists(paciente_id, study_date, agu_mode):
                emit(f"[{agu_mode}] SKIP: ya existe segmentación")
                continue

            try:
                emit(f"[{agu_mode}] Ejecutando...")
                run_agunet_patient(
                    paciente_id=paciente_id,
                    study_date=study_date,
                    gpu_id=gpu_id_agunet,
                    force=False,
                    mode_key=agu_mode,
                )
                emit(f"[{agu_mode}] Terminado")
            except Exception as e:
                msg = f"{agu_mode}: {e}"
                errors.append(msg)
                print(f"[{agu_mode}] ERROR: {e}")

    elif mode in {"agunet", "dagunet", "pls-net", "unet-fv", "unet-slabs"}:
        if skip_completed and _model_output_exists(paciente_id, study_date, mode):
            emit(f"[{mode}] SKIP: ya existe segmentación")
        else:
            try:
                run_agunet_patient(
                    paciente_id=paciente_id,
                    study_date=study_date,
                    gpu_id=gpu_id_agunet,
                    force=False,
                    mode_key=mode,
                )
            except Exception as e:
                msg = f"{mode}: {e}"
                errors.append(msg)
                print(f"[{mode}] ERROR: {e}")

    # 5) nnU-Net
    if mode in {"nnunet", "all"}:
        if skip_completed and _model_output_exists(paciente_id, study_date, "nnunet"):
            emit("[nnU-Net] SKIP: ya existe segmentación")
        else:
            try:
                emit("[nnU-Net] Ejecutando Task501...")
                segmentar_paciente_nnunet(
                    paciente_id=paciente_id,
                    study_date=study_date,
                    use_ants_brain=True,
                    run_501=run_501,
                )
                emit("[nnU-Net] Terminado")
            except Exception as e:
                msg = f"nnU-Net: {e}"
                errors.append(msg)
                print(f"[nnU-Net] ERROR: {e}")

    # 6) Resumen final
    if errors:
        print("\n===== RESUMEN DE FALLOS =====")
        for err in errors:
            print(f"- {err}")
    else:
        print("\n✅ Todos los modelos se ejecutaron correctamente")

    return errors


def pipeline_patient(
    paciente_id: str,
    dicom_source_dir: Optional[str] = None,
    mode: str = "radionics",
    modality_ants: str = "t1",
    gpu_id_agunet: int = 0,
    force_copy_dicom: bool = False,
    **kwargs
) -> None:
    """
    Usa LA FECHA del DICOM que estás añadiendo:
    - Copia a Pacientes_nifti/<id>/<fecha>/dicom
    - Y procesa ese estudio.
    """
    study_date, _P = prepare_study_from_dicom(
        paciente_id=paciente_id,
        dicom_source_dir=dicom_source_dir,
        force_copy=force_copy_dicom,
    )

    print(f"[Pipeline] {paciente_id} -> estudio {study_date}")
    pipeline_study(
        paciente_id=paciente_id,
        study_date=study_date,
        mode=mode,
        modality_ants=modality_ants,
        gpu_id_agunet=gpu_id_agunet,
        **kwargs
    )


def pipeline_all(mode: str = "radionics", **kwargs) -> None:
    ids = get_all_patient_ids()
    print(f"[Pipeline] Pacientes detectados: {len(ids)}")

    for pid in ids:
        try:
            print(f"\n========== {pid} ({mode}) ==========")
            pipeline_patient(pid, dicom_source_dir=None, mode=mode, **kwargs)
        except Exception as e:
            print(f"[ERROR] {pid}: {e}")

def pipeline_all_existing_studies(
    mode: str = "all",
    skip_completed: bool = True,
    progress_cb=None,
    progress_log_path: Optional[str] = None,
    **kwargs
) -> list[str]:
    def emit(msg: str) -> None:
        print(msg)

        if progress_log_path:
            try:
                with open(progress_log_path, "w", encoding="utf-8") as f:
                    f.write(msg + "\n")
            except Exception:
                pass

        if progress_cb is not None:
            try:
                progress_cb(msg)
            except Exception:
                pass

    cases = []
    for pid in get_all_patient_ids():
        for study_date in config_list_studies(pid):
            cases.append((pid, study_date))

    if not cases:
        emit("[Lote] No hay estudios disponibles en Pacientes_nifti.")
        return []

    emit(f"[Lote] Estudios detectados: {len(cases)}")

    all_errors: list[str] = []

    for idx, (pid, study_date) in enumerate(cases, start=1):
        emit(f"[{idx}/{len(cases)}] Procesando {pid}/{study_date}...")

        try:
            errors = pipeline_study(
                paciente_id=pid,
                study_date=study_date,
                mode=mode,
                progress_cb=progress_cb,
                progress_log_path=progress_log_path,
                skip_completed=skip_completed,
                **kwargs,
            )

            for err in errors:
                all_errors.append(f"{pid}/{study_date}: {err}")

        except Exception as e:
            all_errors.append(f"{pid}/{study_date}: {e}")

    if all_errors:
        emit(f"[Lote] Terminado con {len(all_errors)} errores.")
    else:
        emit("[Lote] Terminado correctamente.")

    return all_errors
