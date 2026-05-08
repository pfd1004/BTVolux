"""
preprocessing.py

Preprocesado de estudios DICOM antes de la segmentación.

Responsabilidades:
- Leer la fecha del estudio desde los DICOM.
- Validar si el estudio parece T1c/post-contraste.
- Copiar el estudio DICOM a la estructura interna del proyecto.
- Convertir DICOM a NIfTI.
- Generar brainmask y brain-extracted volume con ANTs o SynthStrip.

Reglas de diseño:
- La unidad de trabajo del proyecto es el estudio, no solo el paciente.
- La carpeta de destino se decide a partir de la fecha del DICOM.
- Si dos estudios comparten fecha base, se usa un sufijo _01, _02, etc.
- La validación T1c es flexible y se basa en palabras clave editables
  en app_data/config/dicom_t1c_keywords.txt.
"""
from __future__ import annotations
from pathlib import Path
import logging
import shutil
import subprocess
from datetime import datetime
from typing import Optional, Tuple

import dicom2nifti
import ants

from config import (
    get_patient_paths,
    normalize_study_date,
    SYNTHSTRIP_NO_CSF,
    SYNTHSTRIP_BORDER,
    SYNTHSTRIP_USE_GPU,
    PACIENTES_DICOM_DIR,
    SYNTHSTRIP_MODEL_STD,
    SYNTHSTRIP_MODEL_NOCSF,
    PACIENTES_NIFTI_DIR,
    T1C_PROTOCOLS_TXT,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("preprocessing")

try:
    import pydicom
except Exception:
    pydicom = None
    logger.warning("pydicom no está disponible: la fecha del estudio usará fallback (mtime).")

try:
    import antspynet
except Exception:
    antspynet = None
    logger.warning("antspynet no está disponible. Se usará ants.get_mask como fallback.")


# -------------------------
# DICOM helpers
# -------------------------

def _iter_candidate_files(dicom_dir: Path, max_files: int = 300):
    n = 0
    for p in dicom_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.name.lower() in {"dicomdir", "thumbs.db"}:
            continue
        yield p
        n += 1
        if n >= max_files:
            break


def get_dicom_study_date(dicom_dir: Path) -> str:
    """
    Extrae la fecha del estudio a partir de los metadatos DICOM.

    Prioridad de tags:
    1. StudyDate
    2. SeriesDate
    3. AcquisitionDate
    4. ContentDate

    Si no se encuentra ninguna fecha válida, usa como fallback la fecha
    de modificación del primer fichero legible.

    Devuelve siempre una fecha normalizada en formato YYYYMMDD.
    """
    dicom_dir = Path(dicom_dir)
    if not dicom_dir.exists():
        raise FileNotFoundError(f"No existe dicom_dir: {dicom_dir}")

    if pydicom is not None:
        for f in _iter_candidate_files(dicom_dir):
            try:
                ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
            except Exception:
                continue

            for tag in ("StudyDate", "SeriesDate", "AcquisitionDate", "ContentDate"):
                v = getattr(ds, tag, None)
                if v:
                    s = str(v).strip().replace(".", "-").replace("/", "-")
                    try:
                        return normalize_study_date(s)
                    except Exception:
                        pass

    # fallback: mtime del primer fichero
    for f in _iter_candidate_files(dicom_dir, max_files=1):
        dt = datetime.fromtimestamp(f.stat().st_mtime)
        return dt.strftime("%Y%m%d")

    return datetime.now().strftime("%Y%m%d")

def get_dicom_study_uid(dicom_dir: Path) -> str:
    """
    Devuelve StudyInstanceUID del primer DICOM legible.
    Si no se encuentra, devuelve cadena vacía.
    """
    dicom_dir = Path(dicom_dir)
    if not dicom_dir.exists() or pydicom is None:
        return ""

    for f in _iter_candidate_files(dicom_dir):
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
        except Exception:
            continue

        uid = getattr(ds, "StudyInstanceUID", None)
        if uid:
            return str(uid).strip()

    return ""


def find_existing_study_with_same_uid(paciente_id: str, src_dicom_dir: Path, base_study_date: str) -> str:
    """
    Si ya existe un estudio del mismo paciente y misma fecha base con el mismo
    StudyInstanceUID, devuelve el nombre de esa carpeta de estudio.
    Si no, devuelve "".
    """
    src_uid = get_dicom_study_uid(src_dicom_dir)
    if not src_uid:
        return ""

    patient_root = PACIENTES_NIFTI_DIR / paciente_id
    if not patient_root.exists():
        return ""

    base_study_date = normalize_study_date(base_study_date)

    for study_dir in sorted(patient_root.iterdir()):
        if not study_dir.is_dir():
            continue

        study_name = study_dir.name.strip()
        if not study_name.startswith(base_study_date):
            continue

        existing_dicom_dir = study_dir / "dicom"
        if not existing_dicom_dir.exists():
            continue

        existing_uid = get_dicom_study_uid(existing_dicom_dir)
        if existing_uid and existing_uid == src_uid:
            return study_name

    return ""

# ============================================================
# Validación T1c / post-contraste
# ============================================================

def ensure_t1c_protocols_file() -> None:
    """
    Crea el fichero editable de palabras clave T1c si no existe.
    """
    if T1C_PROTOCOLS_TXT.exists():
        return

    default_lines = [
        "# Palabras clave para detectar estudios T1c / post-contraste",
        "# Una línea = una palabra o patrón esperado en tags DICOM",
        "# Se compara en minúsculas contra:",
        "#   SeriesDescription",
        "#   ProtocolName",
        "#   SequenceName",
        "#   ImageType",
        "#   ContrastBolusAgent",
        "",
        "t1c",
        "t1 ce",
        "t1+c",
        "t1_c",
        "t1gd",
        "post",
        "postcontrast",
        "post-contrast",
        "post contrast",
        "mprage c+",
        "mprage post",
        "spgr c+",
        "tfe post",
        "bravo c+",
        "gd",
        "gadolinium",
        "contrast",
        "ce",
    ]
    T1C_PROTOCOLS_TXT.write_text("\n".join(default_lines) + "\n", encoding="utf-8")


def load_t1c_protocol_keywords() -> list[str]:
    ensure_t1c_protocols_file()

    lines = T1C_PROTOCOLS_TXT.read_text(
        encoding="utf-8",
        errors="ignore",
    ).splitlines()

    keywords = []
    for line in lines:
        s = line.strip().lower()
        if not s or s.startswith("#"):
            continue
        keywords.append(s)
    return keywords


def extract_dicom_text_signature(dicom_dir: Path) -> str:
    """
    Construye una firma textual del primer DICOM legible usando tags útiles
    para identificar el tipo de secuencia.
    """
    if pydicom is None:
        return ""

    for f in _iter_candidate_files(dicom_dir):
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
        except Exception:
            continue

        fields = []
        for tag in (
            "SeriesDescription",
            "ProtocolName",
            "SequenceName",
            "ImageType",
            "ContrastBolusAgent",
        ):
            v = getattr(ds, tag, None)
            if v is None:
                continue

            if isinstance(v, (list, tuple)):
                txt = " ".join(str(x) for x in v)
            else:
                txt = str(v)

            txt = txt.strip()
            if txt:
                fields.append(f"{tag}={txt}")

        if fields:
            return " | ".join(fields)

    return ""


def is_t1c_dicom_dir(dicom_dir: Path) -> tuple[bool, list[str], str]:
    """
    Determina si un estudio DICOM parece corresponder a una secuencia
    T1 con contraste/post-contraste.

    La comprobación se basa en buscar palabras clave configurables en
    varios campos DICOM relevantes, como SeriesDescription o ProtocolName.

    Devuelve:
    - bool: si el estudio parece T1c
    - list[str]: keywords que han hecho match
    - str: firma textual generada a partir de los tags leídos
    """
    dicom_dir = Path(dicom_dir)
    keywords = load_t1c_protocol_keywords()
    signature = extract_dicom_text_signature(dicom_dir).lower()

    matches = [kw for kw in keywords if kw in signature]
    return bool(matches), matches, signature


# ============================================================
# Gestión de carpetas y copia del estudio
# ============================================================

def ensure_patient_folders(paciente_id: str, study_date: str) -> dict[str, Path]:
    """Crea (si no existen) las subcarpetas del estudio y devuelve paths."""
    return get_patient_paths(paciente_id, study_date)

def allocate_study_folder_name(
    paciente_id: str,
    base_study_date: str,
    dicom_src_dir: Path | str | None = None,
) -> str:
    """
    Decide el nombre final de carpeta para un estudio.

    Comportamiento:
    - Si ya existe un estudio del mismo paciente, misma fecha base y mismo
      StudyInstanceUID, reutiliza esa carpeta.
    - Si no existe, crea:
        YYYYMMDD
        YYYYMMDD_01
        YYYYMMDD_02
        ...

    Esto evita duplicados reales y a la vez permite varios estudios
    distintos en la misma fecha.
    """
    base_study_date = normalize_study_date(base_study_date)
    patient_root = PACIENTES_NIFTI_DIR / paciente_id
    patient_root.mkdir(parents=True, exist_ok=True)

    if dicom_src_dir is not None:
        same_uid_study = find_existing_study_with_same_uid(
            paciente_id=paciente_id,
            src_dicom_dir=Path(dicom_src_dir),
            base_study_date=base_study_date,
        )
        if same_uid_study:
            return same_uid_study

    first = patient_root / base_study_date
    if not first.exists():
        return base_study_date

    idx = 1
    while True:
        candidate = f"{base_study_date}_{idx:02d}"
        if not (patient_root / candidate).exists():
            return candidate
        idx += 1

def copy_dicom_to_patient_folder(
    paciente_id: str,
    dicom_src_dir: Path | str | None = None,
    force: bool = False,
) -> tuple[str, Path]:
    """
    Copia un estudio DICOM a su carpeta destino por fecha:

      <dicom_src_dir> (default: Pacientes/<id>/)
        -> Pacientes_nifti/<id>/<study_date>/dicom/

    Devuelve (study_date, dst_dir).
    """
    if dicom_src_dir is None:
        dicom_src_dir = PACIENTES_DICOM_DIR / paciente_id
    dicom_src_dir = Path(dicom_src_dir)

    if not dicom_src_dir.exists():
        raise FileNotFoundError(f"No existe DICOM origen: {dicom_src_dir}")

    base_study_date = get_dicom_study_date(dicom_src_dir)

    if force:
        study_date = base_study_date
    else:
        study_date = allocate_study_folder_name(
            paciente_id,
            base_study_date,
            dicom_src_dir=dicom_src_dir,
        )

    P = get_patient_paths(paciente_id, study_date)
    dst = P["dicom_dir"]

    dst.mkdir(parents=True, exist_ok=True)

    if force and dst.exists():
        shutil.rmtree(dst)
        dst.mkdir(parents=True, exist_ok=True)

    logger.info(f"[COPY DICOM] {dicom_src_dir} -> {dst} (study_date={study_date})")
    shutil.copytree(dicom_src_dir, dst, dirs_exist_ok=True)
    return study_date, dst


def prepare_study_from_dicom(
    paciente_id: str,
    dicom_source_dir: Optional[str] = None,
    force_copy: bool = False,
) -> Tuple[str, dict[str, Path]]:
    """
    - Lee la fecha del DICOM que acabas de meter.
    - Crea Pacientes_nifti/<id>/<fecha>/ (subcarpetas).
    - Copia el DICOM a .../<fecha>/dicom/ (si no existía o si force_copy=True).
    - Devuelve (study_date, paths_dict).
    """
    src = Path(dicom_source_dir) if dicom_source_dir else (PACIENTES_DICOM_DIR / paciente_id)

    ok_t1c, matched, signature = is_t1c_dicom_dir(src)
    if not ok_t1c:
        raise ValueError(
            "El estudio DICOM no parece ser T1c/post-contraste. "
            f"Firma leída: {signature or '[sin tags útiles]'} | "
            f"Edita el fichero {T1C_PROTOCOLS_TXT.name} en app_data/config si el hospital usa otra nomenclatura."
        )

    study_date, _ = copy_dicom_to_patient_folder(
        paciente_id,
        dicom_src_dir=src,
        force=force_copy,
    )
    P = ensure_patient_folders(paciente_id, study_date)
    return study_date, P


def _find_dicom_source_dir(paciente_id: str, study_date: str) -> Path:
    """
    Preferencia: Pacientes_nifti/<id>/<study_date>/dicom/ si tiene contenido.
    Fallback:    Pacientes/<id>/.
    """
    P = get_patient_paths(paciente_id, study_date)
    if P["dicom_dir"].exists() and any(P["dicom_dir"].iterdir()):
        return P["dicom_dir"]

    src = PACIENTES_DICOM_DIR / paciente_id
    if src.exists() and any(src.rglob("*")):
        return src

    raise FileNotFoundError(f"No encuentro DICOM en {P['dicom_dir']} ni en {src}")

# ============================================================
# Conversión DICOM -> NIfTI
# ============================================================

def convertir_dicom_a_nifti(paciente_id: str, study_date: str, force: bool = False) -> Path:
    """Convierte DICOM a NIfTI y devuelve el NIfTI canónico (<id>_0000.nii.gz)."""
    P = get_patient_paths(paciente_id, study_date)
    out_nifti = P["nifti_img"]
    if out_nifti.exists() and not force:
        return out_nifti

    dicom_dir = _find_dicom_source_dir(paciente_id, study_date)
    nifti_dir = P["nifti_dir"]
    nifti_dir.mkdir(parents=True, exist_ok=True)

    if force:
        for f in nifti_dir.glob("*.nii*"):
            try:
                f.unlink()
            except Exception:
                pass

    logger.info(f"[DICOM→NIfTI] {dicom_dir} -> {nifti_dir}")
    dicom2nifti.convert_directory(
        dicom_directory=str(dicom_dir),
        output_folder=str(nifti_dir),
        compression=True,
        reorient=True,
    )

    return renombrar_nifti(paciente_id, study_date)


def renombrar_nifti(paciente_id: str, study_date: str) -> Path:
    """Renombra el primer NIfTI encontrado dentro de nifti/ a <id>_0000.nii.gz."""
    P = get_patient_paths(paciente_id, study_date)
    nifti_dir = P["nifti_dir"]

    candidates = sorted(list(nifti_dir.glob("*.nii.gz")) + list(nifti_dir.glob("*.nii")))
    if not candidates:
        raise FileNotFoundError(f"No se encontró ningún NIfTI en {nifti_dir}")

    src = candidates[0]
    dst = P["nifti_img"]

    if src.resolve() != dst.resolve():
        if dst.exists():
            dst.unlink()
        logger.info(f"[RENOMBRE] {src.name} -> {dst.name}")
        src.rename(dst)

    return dst

# ============================================================
# Brain extraction
# ============================================================

def brain_extraction_ants(
    paciente_id: str,
    study_date: str,
    input_nii: Path,
    modality: str = "t1",
    force: bool = False,
) -> tuple[Path, Path]:
    """
    Genera brain y brainmask con ANTs (antspynet si existe) de forma robusta.
    """
    P = get_patient_paths(paciente_id, study_date)
    out_brain = P["ants_brain"]
    out_mask = P["ants_mask"]

    if out_brain.exists() and out_mask.exists() and not force:
        return out_brain, out_mask

    input_nii = Path(input_nii)
    if not input_nii.exists():
        raise FileNotFoundError(f"Input NIfTI no existe: {input_nii}")

    logger.info(f"[ANTs] Leyendo: {input_nii}")
    img = ants.image_read(str(input_nii))

    def _mask_fraction(m) -> float:
        arr = m.numpy()
        return float((arr > 0).mean())

    if antspynet is None:
        logger.warning("[ANTs] antspynet no disponible: usando ants.get_mask (fallback).")
        mask = ants.get_mask(img)
    else:
        logger.info(f"[ANTs] antspynet.brain_extraction (modality={modality})")
        prob_mask = antspynet.brain_extraction(img, modality=modality)
        mask = ants.threshold_image(prob_mask, 0.20, 1.0)

        for op in [("FillHoles", None), ("MC", 2), ("MD", 1), ("GetLargestComponent", None)]:
            try:
                if op[1] is None:
                    mask = ants.iMath(mask, op[0])
                else:
                    mask = ants.iMath(mask, op[0], op[1])
            except Exception:
                pass

        frac = _mask_fraction(mask)
        logger.info(f"[ANTs] brainmask fraction={frac:.4f}")
        if frac < 0.08:
            logger.warning("[ANTs] Máscara pequeña. Fallback a ants.get_mask().")
            mask = ants.get_mask(img)
            try:
                mask = ants.iMath(mask, "FillHoles")
            except Exception:
                pass

    brain = img * mask

    out_mask.parent.mkdir(parents=True, exist_ok=True)
    ants.image_write(brain, str(out_brain))
    ants.image_write(mask, str(out_mask))

    logger.info(f"[ANTs] Brain -> {out_brain}")
    logger.info(f"[ANTs] Mask  -> {out_mask}")
    return out_brain, out_mask


# -------------------------
# Brain extraction (SynthStrip)
# -------------------------

def brain_extraction_synthstrip(
    paciente_id: str,
    study_date: str,
    input_nii: str | Path,
    force: bool = False,
    no_csf: bool | None = None,
    border: int | None = None,
) -> tuple[Path, Path]:
    """
    SynthStrip (Windows-friendly) usando el CLI `nipreps-synthstrip`.

    Requiere:
      pip install -U nipreps-synthstrip
    """
    import shutil as _shutil

    P = get_patient_paths(paciente_id, study_date)
    out_brain = P["synth_brain"]
    out_mask = P["synth_mask"]

    if (not force) and out_brain.exists() and out_mask.exists():
        return out_brain, out_mask

    out_brain.parent.mkdir(parents=True, exist_ok=True)

    if no_csf is None:
        no_csf = SYNTHSTRIP_NO_CSF
    if border is None:
        border = SYNTHSTRIP_BORDER

    model_path = SYNTHSTRIP_MODEL_NOCSF if bool(no_csf) else SYNTHSTRIP_MODEL_STD
    if not model_path.exists():
        raise RuntimeError(
            "No encuentro el modelo SynthStrip:\n"
            f"  {model_path}\n\n"
            "Colócalo en modelos/synthstrip/ con nombres:\n"
            "  synthstrip.1.pt\n"
            "  synthstrip.nocsf.1.pt"
        )

    cmd = _shutil.which("nipreps-synthstrip")
    if not cmd:
        raise RuntimeError(
            "No encuentro el comando `nipreps-synthstrip`.\n"
            "Instala: pip install -U nipreps-synthstrip\n"
            "Comprueba: where nipreps-synthstrip"
        )

    in_path = Path(input_nii)
    if not in_path.exists():
        raise FileNotFoundError(f"Input NIfTI no existe: {in_path}")

    def _torch_cuda_available() -> bool:
        try:
            import torch
            return bool(torch.cuda.is_available())
        except Exception:
            return False


    use_gpu = False

    if SYNTHSTRIP_USE_GPU is True:
        use_gpu = True
    elif SYNTHSTRIP_USE_GPU == "auto":
        use_gpu = _torch_cuda_available()
    else:
        use_gpu = False

    args = [
        cmd,
        "-i", str(in_path),
        "-o", str(out_brain),
        "-m", str(out_mask),
        "-b", str(int(border)),
        "--model", str(model_path),
    ]

    if use_gpu:
        args.append("-g")
        logger.info("[SynthStrip] Ejecutando en GPU.")
    else:
        logger.info("[SynthStrip] Ejecutando en CPU.")

    logger.info(f"[SynthStrip] {' '.join(args)}")
    subprocess.run(args, check=True)

    logger.info(f"[SynthStrip] Brain -> {out_brain}")
    logger.info(f"[SynthStrip] Mask  -> {out_mask}")
    return out_brain, out_mask
