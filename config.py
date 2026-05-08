from __future__ import annotations
"""
config.py

Archivo central de configuración del proyecto.

Responsabilidades:
- Definir todas las rutas base del proyecto.
- Definir la estructura canónica de carpetas por paciente y por estudio.
- Centralizar nombres de modelos, ejecutables y archivos auxiliares.
- Proporcionar helpers para normalizar fechas y nombres de carpetas de estudio.

Estructura objetivo del estudio:
Pacientes_nifti/<PACIENTE_ID>/<YYYYMMDD o YYYYMMDD_XX>/
    dicom/
    nifti/
    ants/
    radionics/
    nnunet/
    agunet/
    dagunet/
    pls-net/
    unet-fv/
    unet-slabs/
    tmp/

Notas:
- Un mismo paciente puede tener varios estudios.
- Si existen varios estudios con la misma fecha base, se usa el sufijo _01, _02, etc.
- Este archivo debe actuar como fuente única de verdad para rutas y nombres canónicos.
"""

from pathlib import Path
from datetime import datetime
import re
import json

# ============================================================
# Rutas base del proyecto
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

# Entradas base
PACIENTES_DICOM_DIR = BASE_DIR / "Pacientes"
DEFAULT_PACIENTES_NIFTI_DIR = BASE_DIR / "Pacientes_nifti"
DEFAULT_RESULTS_DIR = BASE_DIR / "app_data" / "results"

APP_DATA_DIR = BASE_DIR / "app_data"
APP_STATE_DIR = APP_DATA_DIR / "state"
APP_CONFIG_DIR = APP_DATA_DIR / "config"
APP_LOGS_DIR = APP_DATA_DIR / "logs"

STORAGE_SETTINGS_JSON = APP_CONFIG_DIR / "storage_settings.json"

# ============================================================
# Configuración de nnU-Net
# ============================================================

# nnU-Net v1
NNUNET_BASE_DIR = BASE_DIR / "modelos" / "nnunet"
NNUNET_RAW_DATA_BASE = NNUNET_BASE_DIR / "nnUNet_raw_data_base"
NNUNET_PREPROCESSED = NNUNET_BASE_DIR / "nnUNet_preprocessed"
NNUNET_RESULTS = NNUNET_BASE_DIR / "nnUNet_results"
NNUNET_TASK501_NAME = "Task501_t1c_enhancement"

# ============================================================
# Configuración de skull-stripping / brainmask
# ============================================================

# Skull-stripping / Brainmask
BRAINMASK_METHOD = "synthstrip"  # "ants" | "synthstrip"
SYNTHSTRIP_BORDER = 3
SYNTHSTRIP_NO_CSF = False
SYNTHSTRIP_USE_GPU = "auto"
SYNTHSTRIP_MODEL_DIR = BASE_DIR / "modelos" / "synthstrip"
SYNTHSTRIP_MODEL_STD = SYNTHSTRIP_MODEL_DIR / "synthstrip.1.pt"
SYNTHSTRIP_MODEL_NOCSF = SYNTHSTRIP_MODEL_DIR / "synthstrip.nocsf.1.pt"

# ============================================================
# Configuración de modelos externos y ejecutables
# ============================================================

# Radionics
RADIONICS_MODEL_DIR = BASE_DIR / "modelos" / "radionics" / "MRI_Meningioma"

# AGUNet (repo externo)
AGUNET_MAIN = BASE_DIR / "modelos" / "mri_brain_tumor_segmentation" / "main.py"
AGUNET_PROB_THRESHOLD = 0.5

# ============================================================
# Entornos / ejecutables
# ============================================================

RUNTIME_DIR = BASE_DIR / "runtime"

# Si existe runtime/env_app, asumimos versión portable empaquetada.
IS_PORTABLE = (RUNTIME_DIR / "env_app").exists()

AGUNET_CONDA_ENV = "agusnet"
NNUNET_CONDA_ENV = "nnunet_v1"
PLSNET_CONDA_ENV = "agunet_plsnet"

if IS_PORTABLE:
    NNUNET_PYTHON_EXE = str(
        RUNTIME_DIR / "env_nnunet" / "python.exe"
    )

    NNUNET_PREDICT_EXE = str(
        RUNTIME_DIR / "env_nnunet" / "Scripts" / "nnUNet_predict.exe"
    )

    AGUNET_PYTHON_EXE = str(
        RUNTIME_DIR / "env_agunet" / "python.exe"
    )

    PLSNET_PYTHON_EXE = str(
        RUNTIME_DIR / "env_plsnet" / "python.exe"
    )

    CONDA_EXE = ""

else:
    NNUNET_PYTHON_EXE = r"C:\Users\Pablo\anaconda3\envs\nnunet_v1\python.exe"
    NNUNET_PREDICT_EXE = r"C:\Users\Pablo\anaconda3\envs\nnunet_v1\Scripts\nnUNet_predict.exe"
    AGUNET_PYTHON_EXE = r"C:\Users\Pablo\anaconda3\envs\agusnet\python.exe"
    PLSNET_PYTHON_EXE = r"C:\Users\Pablo\anaconda3\envs\agunet_plsnet\python.exe"
    CONDA_EXE = r"C:\Users\Pablo\anaconda3\Scripts\conda.exe"

def ensure_dir(p: Path, create: bool = True) -> Path:
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p

def _load_storage_settings() -> dict:
    if not STORAGE_SETTINGS_JSON.exists():
        return {}
    try:
        return json.loads(STORAGE_SETTINGS_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {}
    
def _resolve_app_path(value: str | Path | None, default: Path) -> Path:
    """
    Resuelve rutas de configuración.

    - Si la ruta es absoluta, se usa tal cual.
    - Si la ruta es relativa, se interpreta relativa a BASE_DIR.
    - Si viene vacía, se usa el valor por defecto.
    """
    if value is None or str(value).strip() == "":
        p = default
    else:
        p = Path(value).expanduser()

    if not p.is_absolute():
        p = BASE_DIR / p

    return p.resolve()


def get_storage_settings() -> dict[str, Path]:
    raw = _load_storage_settings()

    pacientes_nifti_dir = _resolve_app_path(
        raw.get("pacientes_nifti_dir", "Pacientes_nifti"),
        DEFAULT_PACIENTES_NIFTI_DIR,
    )

    results_dir = _resolve_app_path(
        raw.get("results_dir", "app_data/results"),
        DEFAULT_RESULTS_DIR,
    )

    # Fallback portable:
    # si el JSON apunta a una ruta antigua/no disponible, pero existe la
    # carpeta local Pacientes_nifti junto a la app, usar la local.
    if not pacientes_nifti_dir.exists() and DEFAULT_PACIENTES_NIFTI_DIR.exists():
        pacientes_nifti_dir = DEFAULT_PACIENTES_NIFTI_DIR.resolve()

    results_dir.mkdir(parents=True, exist_ok=True)

    return {
        "pacientes_nifti_dir": pacientes_nifti_dir,
        "results_dir": results_dir,
    }


def save_storage_settings(
    pacientes_nifti_dir: str | Path | None = None,
    results_dir: str | Path | None = None,
) -> dict[str, Path]:
    current = _load_storage_settings()

    if pacientes_nifti_dir is not None:
        current["pacientes_nifti_dir"] = str(Path(pacientes_nifti_dir).expanduser().resolve())

    if results_dir is not None:
        current["results_dir"] = str(Path(results_dir).expanduser().resolve())

    STORAGE_SETTINGS_JSON.parent.mkdir(parents=True, exist_ok=True)
    STORAGE_SETTINGS_JSON.write_text(json.dumps(current, indent=2), encoding="utf-8")
    return get_storage_settings()



#Settings para poder mover carpeta raiz
_storage_settings = get_storage_settings()

APP_RESULTS_DIR = ensure_dir(_storage_settings["results_dir"])
PACIENTES_NIFTI_DIR = ensure_dir(_storage_settings["pacientes_nifti_dir"], create=False)

LAST_SESSION = APP_STATE_DIR / "last_session.json"
T1C_PROTOCOLS_TXT = APP_CONFIG_DIR / "dicom_t1c_keywords.txt"

_STUDY_FOLDER_RE = re.compile(r"^(?P<base>\d{8})(?:_(?P<idx>\d+))?$")



def normalize_study_folder_name(s: str) -> str:
    """
    Normaliza el nombre de una carpeta de estudio a formato canónico.

    Entradas admitidas:
    - YYYYMMDD
    - YYYYMMDD_1
    - YYYYMMDD_01

    Salida:
    - YYYYMMDD
    - YYYYMMDD_01
    - YYYYMMDD_02
    - ...

    Se usa para asegurar que toda la app trate igual los estudios con
    sufijo de desambiguación.
    """
    s = str(s).strip()
    m = _STUDY_FOLDER_RE.match(s)
    if not m:
        # si viene solo fecha normalizable, devolver YYYYMMDD
        base = normalize_study_date(s)
        return base

    base = normalize_study_date(m.group("base"))
    idx = m.group("idx")
    if idx is None:
        return base
    return f"{base}_{int(idx):02d}"


def get_study_base_date(study_folder_name: str) -> str:
    """
    Devuelve todas las rutas canónicas asociadas a un estudio.

    Además de devolverlas, crea en disco las carpetas necesarias si todavía
    no existen. Debe usarse cuando se va a escribir o preparar un estudio.

    Para lectura sin crear carpetas conviene usar una función separada.
    """
    s = normalize_study_folder_name(study_folder_name)
    return s.split("_")[0]

def normalize_study_date(s: str) -> str:
    """
    Normaliza fechas del DICOM a YYYYMMDD.
    Acepta:
      - YYYYMMDD
      - YYYY-MM-DD
      - YYYY/MM/DD
      - YYYY.MM.DD
    """
    s = str(s).strip()
    if s.isdigit() and len(s) == 8:
        return s

    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y%m%d")
        except Exception:
            pass
    cleaned = s.replace("-", "").replace("/", "").replace(".", "")
    if cleaned.isdigit() and len(cleaned) == 8:
        return cleaned

    raise ValueError(f"No puedo normalizar fecha: {s}")


def get_study_root(paciente_id: str, study_date: str) -> Path:
    """
    Devuelve la carpeta raíz canónica de un estudio.

    La ruta final siempre sigue la estructura:
        Pacientes_nifti/<PACIENTE_ID>/<STUDY_DATE>/

    Antes de construirla, `study_date` se normaliza con
    `normalize_study_folder_name(...)`, de forma que se acepten tanto
    nombres simples (`YYYYMMDD`) como nombres con sufijo de desambiguación
    (`YYYYMMDD_01`, `YYYYMMDD_02`, ...).

    Además, si la carpeta todavía no existe, se crea en disco.
    """
    study_folder = normalize_study_folder_name(study_date)
    return ensure_dir(PACIENTES_NIFTI_DIR / paciente_id / study_folder)


def get_patient_paths(paciente_id: str, study_date: str) -> dict[str, Path]:
    """
    Devuelve todas las rutas canónicas asociadas a un estudio.

    Esta función centraliza la estructura interna del proyecto y debe usarse
    siempre que se necesiten rutas de trabajo para un paciente y un estudio
    concretos. Además de devolver las rutas, crea automáticamente las carpetas
    necesarias si todavía no existen.

    La estructura objetivo del estudio es:

        Pacientes_nifti/<PACIENTE_ID>/<STUDY_DATE>/
            dicom/
            nifti/
            ants/
            radionics/
            nnunet/
            agunet/
            dagunet/
            pls-net/
            unet-fv/
            unet-slabs/
            manual/
            tmp/
                nnunet_input/
    """
    root = get_study_root(paciente_id, study_date)

    dicom_dir = ensure_dir(root / "dicom")
    nifti_dir = ensure_dir(root / "nifti")
    ants_dir = ensure_dir(root / "ants")
    radionics_dir = ensure_dir(root / "radionics")
    nnunet_dir = ensure_dir(root / "nnunet")
    tmp_dir = ensure_dir(root / "tmp")
    nnunet_in_dir = ensure_dir(tmp_dir / "nnunet_input")

    # modelos tipo AGUNet 
    agunet_dir = ensure_dir(root / "agunet")
    dagunet_dir = ensure_dir(root / "dagunet")
    pls_net_dir = ensure_dir(root / "pls-net")
    unet_fv_dir = ensure_dir(root / "unet-fv")
    unet_slabs_dir = ensure_dir(root / "unet-slabs")

    #segmentacion manual
    manual_dir = ensure_dir(root / "manual")

    return {
        "root": root,
        "pid": paciente_id,
        "study_date": normalize_study_folder_name(study_date),
        "study_date_base": get_study_base_date(study_date),

        "dicom_dir": dicom_dir,
        "nifti_dir": nifti_dir,
        "ants_dir": ants_dir,
        "radionics_dir": radionics_dir,
        "nnunet_dir": nnunet_dir,
        "tmp_dir": tmp_dir,
        "nnunet_in_dir": nnunet_in_dir,

        "agunet_dir": agunet_dir,
        "dagunet_dir": dagunet_dir,
        "pls-net_dir": pls_net_dir,
        "unet-fv_dir": unet_fv_dir,
        "unet-slabs_dir": unet_slabs_dir,

        "manual_dir": manual_dir,
        "manual_seg": manual_dir / f"{paciente_id}_manual_seg.nii.gz",
        "manual_aligned": manual_dir / f"{paciente_id}_manual_aligned.nii.gz",

        # Archivos canónicos
        "nifti_img": nifti_dir / f"{paciente_id}_0000.nii.gz",
        "ants_brain": ants_dir / f"{paciente_id}_brain.nii.gz",
        "ants_mask": ants_dir / f"{paciente_id}_brainmask.nii.gz",
        "synth_brain": ants_dir / f"{paciente_id}_synthstrip_brain.nii.gz",
        "synth_mask": ants_dir / f"{paciente_id}_synthstrip_mask.nii.gz",

        "rad_seg": radionics_dir / f"{paciente_id}_radionics_seg.nii.gz",
        "rad_prob": radionics_dir / f"{paciente_id}_radionics_prob.nii.gz", 
    }


def get_all_patient_ids() -> list[str]:
    """Lista IDs dentro de Pacientes_nifti/; si está vacío, usa Pacientes/."""
    ids: list[str] = []
    if PACIENTES_NIFTI_DIR.exists():
        ids = sorted([p.name for p in PACIENTES_NIFTI_DIR.iterdir() if p.is_dir()])
    if ids:
        return ids
    if PACIENTES_DICOM_DIR.exists():
        return sorted([p.name for p in PACIENTES_DICOM_DIR.iterdir() if p.is_dir()])
    return []


def list_studies(paciente_id: str) -> list[str]:
    """
    Lista estudios dentro de Pacientes_nifti/<id>/.
    Soporta:
      - YYYYMMDD
      - YYYYMMDD_01
      - YYYYMMDD_02
    """
    root = PACIENTES_NIFTI_DIR / paciente_id
    if not root.exists():
        return []

    studies = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        try:
            studies.append(normalize_study_folder_name(p.name))
        except Exception:
            continue

    studies = sorted(set(studies))
    return studies