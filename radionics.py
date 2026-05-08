"""
radionics.py

Inferencia del modelo Radionics en formato ONNX mediante sliding window.

Responsabilidades:
- Reescalar la imagen a resolución de trabajo.
- Aplicar recorte espacial para reducir coste computacional.
- Normalizar intensidades de forma robusta.
- Ejecutar inferencia por ventanas.
- Reconstruir el volumen de probabilidad completo.
- Reproyectar la salida al espacio original.
- Generar segmentación binaria final.

Notas de implementación:
- El volumen interno se maneja en orden Z, Y, X.
- La probabilidad se reconstruye agregando ventanas solapadas.
- Puede aplicarse una brainmask para reducir falsos positivos.
- Opcionalmente puede conservarse solo la componente conexa más grande.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import SimpleITK as sitk

import torch
import onnxruntime as ort


PATCH = (160, 160, 160)   # (Z,Y,X)
DEFAULT_STEP = 80         # stride

# Reescalado geométrico del volumen.
# Se usa interpolación distinta para imágenes continuas y para etiquetas.
def resample(img: sitk.Image, out_spacing=(1.0, 1.0, 1.0), is_label: bool = False) -> sitk.Image:
    out_spacing = tuple(float(x) for x in out_spacing)
    in_spacing = img.GetSpacing()
    in_size = img.GetSize()

    out_size = [
        int(np.round(in_size[i] * (in_spacing[i] / out_spacing[i])))
        for i in range(3)
    ]

    r = sitk.ResampleImageFilter()
    r.SetOutputSpacing(out_spacing)
    r.SetSize(out_size)
    r.SetOutputDirection(img.GetDirection())
    r.SetOutputOrigin(img.GetOrigin())
    r.SetTransform(sitk.Transform())
    r.SetDefaultPixelValue(0)
    r.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkBSpline)
    return r.Execute(img)


# Recorte al bounding box mínimo con señal distinta de cero.
# Reduce memoria y acelera la inferencia.
def crop_minimum_background(vol_zyx: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int, int, int]]:
    """Recorta al bbox mínimo donde vol > 0. Devuelve (vol_crop, bbox)."""
    mask = vol_zyx > 0
    if not np.any(mask):
        bbox = (0, vol_zyx.shape[0], 0, vol_zyx.shape[1], 0, vol_zyx.shape[2])
        return vol_zyx, bbox

    zz, yy, xx = np.where(mask)
    z0, z1 = int(zz.min()), int(zz.max()) + 1
    y0, y1 = int(yy.min()), int(yy.max()) + 1
    x0, x1 = int(xx.min()), int(xx.max()) + 1
    return vol_zyx[z0:z1, y0:y1, x0:x1], (z0, z1, y0, y1, x0, x1)


def crop_by_mask(vol_zyx: np.ndarray, mask_zyx: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int, int, int]]:
    """Recorta al bbox mínimo donde mask > 0. Devuelve (vol_crop, bbox)."""
    m = mask_zyx > 0
    if not np.any(m):
        bbox = (0, vol_zyx.shape[0], 0, vol_zyx.shape[1], 0, vol_zyx.shape[2])
        return vol_zyx, bbox

    zz, yy, xx = np.where(m)
    z0, z1 = int(zz.min()), int(zz.max()) + 1
    y0, y1 = int(yy.min()), int(yy.max()) + 1
    x0, x1 = int(xx.min()), int(xx.max()) + 1
    return vol_zyx[z0:z1, y0:y1, x0:x1], (z0, z1, y0, y1, x0, x1)

# Normalización robusta por percentiles.
# Evita que outliers de intensidad dominen la escala.
def clip_scale_0_1(vol: np.ndarray) -> np.ndarray:
    """
    Normaliza robusta a [0,1] por percentiles.
    Alineado con pre_processing.ini: 0 - 99.995
    """
    v = vol.astype(np.float32, copy=False)
    lo = np.percentile(v, 0.0)
    hi = np.percentile(v, 99.995)
    if hi <= lo:
        hi = lo + 1.0
    v = np.clip(v, lo, hi)
    v = (v - lo) / (hi - lo)
    return v.astype(np.float32, copy=False)


def clip_scale_0_1_masked(vol: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Igual que clip_scale_0_1 pero calculando percentiles SOLO dentro de mask.
    Además pone a 0 lo de fuera.
    """
    v = vol.astype(np.float32, copy=False)
    m = mask > 0
    if not np.any(m):
        return clip_scale_0_1(v)

    vals = v[m]
    lo = np.percentile(vals, 0.0)
    hi = np.percentile(vals, 99.995)
    if hi <= lo:
        hi = lo + 1.0

    v = np.clip(v, lo, hi)
    v = (v - lo) / (hi - lo)
    v[~m] = 0.0
    return v.astype(np.float32, copy=False)


def _sliding_window_coords(length: int, patch: int, step: int) -> list[int]:
    if length <= patch:
        return [0]
    coords = list(range(0, length - patch + 1, step))
    last = length - patch
    if coords[-1] != last:
        coords.append(last)
    return coords


def softmax_last(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)

# Inferencia por sliding window.
# El modelo procesa patches fijos y luego se promedian las predicciones
# en zonas solapadas para reconstruir el volumen completo.
def predict_volume_probability(
    sess: ort.InferenceSession,
    vol_zyx: np.ndarray,
    step: int,
) -> np.ndarray:
    """
    Ejecuta inferencia por ventanas sobre un volumen 3D ya recortado y normalizado.

    Entrada:
    - vol_zyx: volumen en orden [Z, Y, X]

    Salida:
    - mapa de probabilidad 3D en la misma geometría que el volumen de entrada

    Estrategia:
    - si el volumen es más pequeño que el patch, se rellena con padding
    - se recorren ventanas solapadas
    - las probabilidades se acumulan y se normalizan por el número de veces
      que cada vóxel ha sido predicho
    """
    Z, Y, X = vol_zyx.shape
    pz, py, px = PATCH

    pad_z = max(0, pz - Z)
    pad_y = max(0, py - Y)
    pad_x = max(0, px - X)
    if pad_z or pad_y or pad_x:
        vol_pad = np.pad(vol_zyx, ((0, pad_z), (0, pad_y), (0, pad_x)), mode="constant")
    else:
        vol_pad = vol_zyx

    Zp, Yp, Xp = vol_pad.shape
    probs = np.zeros((Zp, Yp, Xp), dtype=np.float32)
    counts = np.zeros((Zp, Yp, Xp), dtype=np.float32)

    input0 = sess.get_inputs()[0]
    input_name = input0.name
    in_shape = input0.shape
    output_name = sess.get_outputs()[0].name

    # Detecta NDHWC (1,160,160,160,1) vs NCDHW (1,1,160,160,160)
    is_ndhwc = True
    if isinstance(in_shape, (list, tuple)) and len(in_shape) == 5:
        if (in_shape[1] in (1, None)) and (in_shape[2] in (160, None)):
            is_ndhwc = False
        else:
            is_ndhwc = True

    z_coords = _sliding_window_coords(Zp, pz, step)
    y_coords = _sliding_window_coords(Yp, py, step)
    x_coords = _sliding_window_coords(Xp, px, step)

    for z in z_coords:
        for y in y_coords:
            for x in x_coords:
                patch = vol_pad[z:z+pz, y:y+py, x:x+px].astype(np.float32, copy=False)

                if is_ndhwc:
                    inp = patch[None, :, :, :, None]  # (1,160,160,160,1)
                else:
                    inp = patch[None, None, :, :, :]  # (1,1,160,160,160)

                out = sess.run([output_name], {input_name: inp})[0]
                out = np.asarray(out)
                out = np.squeeze(out)

                # Normaliza salida a prob (160,160,160)
                if out.ndim == 4 and out.shape[-1] in (2, 3, 4):
                    out = softmax_last(out)[..., -1]
                elif out.ndim == 4 and out.shape[-1] == 1:
                    out = out[..., 0]
                elif out.ndim == 4 and out.shape[0] in (2, 3, 4):
                    out = softmax_last(np.moveaxis(out, 0, -1))[..., -1]
                elif out.ndim == 4 and out.shape[0] == 1:
                    out = out[0]
                elif out.ndim != 3:
                    raise ValueError(f"Salida inesperada del modelo: {out.shape}")

                if out.shape != (pz, py, px):
                    raise ValueError(f"Salida no coincide con patch: {out.shape} vs {(pz,py,px)}")

                probs[z:z+pz, y:y+py, x:x+px] += out
                counts[z:z+pz, y:y+py, x:x+px] += 1.0

    counts[counts == 0] = 1.0
    probs = probs / counts
    probs = probs[:Z, :Y, :X]
    return probs.astype(np.float32, copy=False)


# ====== Limpieza con mask + largest component ======

from typing import Optional

def _auto_ants_mask_from_input(input_nii: str) -> Optional[Path]:
    """
    Auto-detecta máscara brainmask en estructura antigua o nueva.

    Antigua:
      Pacientes_nifti/<ID>/nifti/<ID>_0000.nii.gz
      Pacientes_nifti/<ID>/ants/<ID>_brainmask.nii.gz
      Pacientes_nifti/<ID>/ants/<ID>_synthstrip_mask.nii.gz

    Nueva (por estudio):
      Pacientes_nifti/<ID>/<YYYYMMDD>/nifti/<ID>_0000.nii.gz
      Pacientes_nifti/<ID>/<YYYYMMDD>/ants/<ID>_brainmask.nii.gz
      Pacientes_nifti/<ID>/<YYYYMMDD>/ants/<ID>_synthstrip_mask.nii.gz
    """
    try:
        in_path = Path(input_nii)

        # Caso NUEVO: .../<ID>/<FECHA>/nifti/xxx.nii.gz
        # in_path.parent = nifti
        # in_path.parent.parent = <FECHA>
        # in_path.parent.parent.parent = <ID>
        study_root = in_path.parent.parent
        patient_root = study_root.parent

        # Si study_root parece una fecha YYYYMMDD, usamos estructura nueva
        if study_root.name.isdigit() and len(study_root.name) == 8:
            pid = patient_root.name
            cand = [
                study_root / "ants" / f"{pid}_synthstrip_mask.nii.gz",
                study_root / "ants" / f"{pid}_brainmask.nii.gz",
            ]
            for m in cand:
                if m.exists():
                    return m
            return None

        # Caso ANTIGUO: .../<ID>/nifti/xxx.nii.gz  -> patient_root = in_path.parent.parent
        patient_root_old = in_path.parent.parent
        pid = patient_root_old.name
        cand = [
            patient_root_old / "ants" / f"{pid}_synthstrip_mask.nii.gz",
            patient_root_old / "ants" / f"{pid}_brainmask.nii.gz",
        ]
        for m in cand:
            if m.exists():
                return m
        return None

    except Exception:
        return None



def _make_mask_permissive(mask_img: sitk.Image, fillholes=True, close_radius=3, dilate_radius=2) -> sitk.Image:
    """
    Hace la máscara más permisiva para no comerse meningiomas periféricos:
    - fill holes
    - closing
    - dilate
    """
    m = sitk.Cast(mask_img, sitk.sitkUInt8)
    if fillholes:
        m = sitk.BinaryFillhole(m)
    if close_radius and close_radius > 0:
        m = sitk.BinaryMorphologicalClosing(m, [close_radius] * 3)
    if dilate_radius and dilate_radius > 0:
        m = sitk.BinaryDilate(m, [dilate_radius] * 3)
    return m


def _apply_mask_to_prob(prob_img: sitk.Image, ref_img: sitk.Image, mask_path: Path,
                        dilate_radius: int = 2, close_radius: int = 3, fillholes: bool = True) -> sitk.Image:
    """
    Aplica una máscara (en el espacio de ref_img) sobre prob_img.
    """
    m0 = sitk.ReadImage(str(mask_path))
    m0 = sitk.Cast(m0, sitk.sitkUInt8)

    # Resample nearest a espacio de ref_img
    m0r = sitk.Resample(m0, ref_img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    m0r = _make_mask_permissive(m0r, fillholes=fillholes, close_radius=close_radius, dilate_radius=dilate_radius)

    return sitk.Mask(prob_img, m0r, outsideValue=0.0)


def _keep_largest_component(binary_mask: sitk.Image) -> sitk.Image:
    cc = sitk.ConnectedComponent(binary_mask)
    rel = sitk.RelabelComponent(cc, sortByObjectSize=True)
    out = sitk.BinaryThreshold(rel, 1, 1, 1, 0)
    return sitk.Cast(out, sitk.sitkUInt8)


def run_radionics(
    input_nii: str,
    model_dir: str,
    output_nii: str,
    thr: float = 0.5,
    step: int = DEFAULT_STEP,
    use_gpu: bool = True,
    output_prob_nii: str | None = None,
    ants_mask_nii: str | None = None,   # si None -> auto
    mask_dilate: int = 1,               # 1-3 recomendado
    keep_largest_cc: bool = True,
) -> None:
    """
    Ejecuta la inferencia completa del modelo Radionics sobre un NIfTI.

    Flujo:
    1. Leer imagen original.
    2. Reescalar a 1 mm isotrópico.
    3. Aplicar máscara cerebral si existe.
    4. Recortar volumen de trabajo.
    5. Normalizar intensidades.
    6. Ejecutar ONNX con sliding window.
    7. Reconstruir el volumen completo.
    8. Reproyectar la probabilidad al espacio original.
    9. Umbralizar para obtener segmentación binaria.
    10. Limpiar componentes si se ha pedido.

    Salidas:
    - output_nii: segmentación binaria final
    - output_prob_nii: mapa de probabilidad, solo si se solicita
    """
    model_path = Path(model_dir) / "model.onnx"
    if not model_path.exists():
        raise FileNotFoundError(f"No encuentro model.onnx en: {model_path}")

    out_dir = Path(output_nii).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) leer + resample a 1mm
    img0 = sitk.ReadImage(str(input_nii))
    img1 = resample(img0, out_spacing=(1.0, 1.0, 1.0), is_label=False)
    vol = sitk.GetArrayFromImage(img1).astype(np.float32)  # [Z,Y,X]

    # 1.5) cargar máscara si existe (ANTs o SynthStrip) y resamplearla al espacio 1mm
    if ants_mask_nii is None:
        mask_path = _auto_ants_mask_from_input(input_nii)
    else:
        p = Path(ants_mask_nii)
        mask_path = p if p.exists() else None

    mask_1mm_np = None
    if mask_path is not None:
        m0 = sitk.ReadImage(str(mask_path))
        m0 = sitk.Cast(m0, sitk.sitkUInt8)

        # resample nearest al espacio de img1 (1mm)
        m1 = sitk.Resample(m0, img1, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)

        # Permisiva para meningiomas periféricos
        m1 = _make_mask_permissive(
            m1,
            fillholes=True,
            close_radius=3,
            dilate_radius=max(2, int(mask_dilate)),
        )

        mask_1mm_np = sitk.GetArrayFromImage(m1).astype(np.uint8)

    # 2) crop (mejor con máscara que con vol>0)
    if mask_1mm_np is not None:
        vol_crop, bbox = crop_by_mask(vol, mask_1mm_np)
        z0, z1, y0, y1, x0, x1 = bbox
        mask_crop = mask_1mm_np[z0:z1, y0:y1, x0:x1]
        # 3) normalizar usando percentiles SOLO dentro del cerebro
        vol_crop = clip_scale_0_1_masked(vol_crop, mask_crop)
    else:
        vol_crop, bbox = crop_minimum_background(vol)
        vol_crop = clip_scale_0_1(vol_crop)

    # 4) sesión ONNX (GPU primero, fallback CPU)
    # 4) sesión ONNX (GPU primero, fallback CPU)
    if use_gpu:
        try:
            import torch
            print(f"[Radionics] torch.cuda.is_available() = {torch.cuda.is_available()}")
            try:
                ort.preload_dlls()
                print("[Radionics] ORT preload_dlls() OK")
            except Exception as e:
                print(f"[Radionics] preload_dlls no disponible o falló: {e}")
        except Exception as e:
            print(f"[Radionics] No se pudo importar torch antes de ORT: {e}")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]

    try:
        sess = ort.InferenceSession(str(model_path), providers=providers)
        print(f"[Radionics] Providers activos: {sess.get_providers()}")
    except Exception as e:
        print(f"[Radionics] Fallo al crear sesión CUDA, fallback a CPU: {e}")
        sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        print(f"[Radionics] Providers activos: {sess.get_providers()}")

    # 5) predicción prob en crop
    prob_crop = predict_volume_probability(sess, vol_crop, step=step)

    # 6) un-crop al tamaño del volumen resampleado
    prob_full = np.zeros_like(vol, dtype=np.float32)
    z0, z1, y0, y1, x0, x1 = bbox
    prob_full[z0:z1, y0:y1, x0:x1] = prob_crop

    # 7) prob en 1mm -> resample a espacio original
    prob_img_1mm = sitk.GetImageFromArray(prob_full)
    prob_img_1mm.CopyInformation(img1)
    res_prob = sitk.Resample(prob_img_1mm, img0, sitk.Transform(), sitk.sitkBSpline, 0.0, sitk.sitkFloat32)

    # 7.5) limpieza con máscara en espacio original (opcional, pero ayuda)
    if mask_path is not None:
        res_prob = _apply_mask_to_prob(
            res_prob,
            img0,
            mask_path,
            dilate_radius=max(2, int(mask_dilate)),
            close_radius=3,
            fillholes=True,
        )

    # guardar prob (solo si se pide)
    if output_prob_nii is not None:
        sitk.WriteImage(res_prob, str(output_prob_nii))

    # 8) binaria SIEMPRE
    seg = sitk.BinaryThreshold(res_prob, lowerThreshold=float(thr), upperThreshold=1e9, insideValue=1, outsideValue=0)
    seg = sitk.Cast(seg, sitk.sitkUInt8)

    if keep_largest_cc:
        seg = _keep_largest_component(seg)

    sitk.WriteImage(seg, str(output_nii))


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Radionics ONNX inference (sliding window)")
    p.add_argument("-i", "--input", required=True, help="Input NIfTI (.nii/.nii.gz)")
    p.add_argument("-m", "--model", required=True, help="Carpeta que contiene model.onnx")
    p.add_argument("-o", "--output", required=True, help="Output segmentation binaria NIfTI (.nii.gz)")
    p.add_argument("--prob", default=None, help="Output probability NIfTI (.nii.gz) (opcional)")
    p.add_argument("--thr", type=float, default=0.5, help="Umbral binario")
    p.add_argument("--step", type=int, default=DEFAULT_STEP, help="Stride sliding window")
    p.add_argument("--cpu", action="store_true", help="Forzar CPU")

    # limpieza opcional
    p.add_argument("--ants-mask", default=None, help="Ruta brainmask (ANTs o SynthStrip) (si no, auto-detect)")
    p.add_argument("--mask-dilate", type=int, default=1, help="Dilatación de máscara (1-3 recomendado)")
    p.add_argument("--no-largest-cc", action="store_true", help="No aplicar largest connected component")
    return p


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    run_radionics(
        input_nii=args.input,
        model_dir=args.model,
        output_nii=args.output,
        thr=args.thr,
        step=args.step,
        use_gpu=not args.cpu,
        output_prob_nii=args.prob,
        ants_mask_nii=args.ants_mask,
        mask_dilate=args.mask_dilate,
        keep_largest_cc=not args.no_largest_cc,
    )
