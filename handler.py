"""Worker Runpod · DentalSegmentator (nnU-Net v2)

    CBCT  ->  labels.nii.gz + dientes_sup.stl + dientes_inf.stl + cbct.json      (y nada más)

Hace lo mismo que el cuaderno de Colab, con los mismos parámetros, pero entrega SOLO lo que
consume el kit corona-raíz (pasos 4-6):

    labels.nii.gz     mapa de etiquetas de nnU-Net: 0 fondo · 1 maxilar · 2 mandíbula ·
                      3 dientes sup. · 4 dientes inf. · 5 canal      -> paso 5 (--cbct-nifti --label 3|4)
    dientes_sup.stl   etiqueta 3, en mm y LPS (DICOM)                -> paso 4 (--cbct) de la arcada mx
    dientes_inf.stl   etiqueta 4                                     -> paso 4 (--cbct) de la arcada md
    cbct.json         metadatos y control de calidad. Se sube EL ÚLTIMO: si existe, está todo.

y lo que necesita el visor de cortes (transversales · axial · tangencial · panorámica):

    volume.nii.gz     el CBCT en grises, int16, sobre la misma rejilla que labels.nii.gz y recortado a
                      la zona de los dientes + margen (peso acotado aunque el CBCT sea de cara entera)
    cbct.json.viewer  curva panorámica de cada arcada (mm, LPS) sacada de la segmentación, ventana de
                      grises sugerida y la geometría de las panorámicas
    pano_mx.png       panorámica ya calculada de cada arcada: vista previa inmediata y comprobación
    pano_md.png       visual de que la curva está bien puesta

Maxilar, mandíbula y canal NO se mallan ni se suben (en el cuaderno eran ~100 MB de STL). No se
pierden: siguen dentro de labels.nii.gz y se pueden mallar más adelante sin volver a la GPU.

Entrada (job["input"])
    cbct_url       URL de descarga del CBCT: .zip con la carpeta DICOM, .dcm multiframe, .nii o .nii.gz
    result_url     URL base de subida; se hace PUT a  <result_url>/<nombre>  (o usa "{name}" como hueco)
    result_urls    alternativa: {"labels.nii.gz": url, ...} (p. ej. URLs prefirmadas de S3)
    input_format   opcional: zip | dcm | nii | nii.gz. Si falta, se deduce del contenido y de la URL
    use_mirroring  opcional (true). false = ~8 veces más rápido y algo menos preciso
    name           opcional, solo para los logs

Sin result_url/result_urls no se sube nada: el output trae las métricas y, si cabe, el mapa de
etiquetas en base64 (útil para probar desde la consola de Runpod).
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import time
import traceback
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import requests
import runpod
import SimpleITK as sitk
import trimesh
from scipy import ndimage
from scipy.interpolate import splev, splprep
from scipy.spatial import cKDTree
from skimage import measure

# torch y nnunetv2 se importan dentro de get_predictor(): así este módulo se puede importar
# (y probar) en una máquina sin GPU ni nnU-Net.

MODEL_ROOT = Path(os.environ.get("DENTALSEG_MODEL_ROOT", "/models/nnunet/results"))
WORK_ROOT = Path(os.environ.get("DENTALSEG_WORK_ROOT", "/tmp/dentalseg"))
MAX_DOWNLOAD = int(os.environ.get("DENTALSEG_MAX_DOWNLOAD_MB", "3000")) * 1024 * 1024
MAX_UNZIPPED = int(os.environ.get("DENTALSEG_MAX_UNZIPPED_MB", "8000")) * 1024 * 1024
MAX_ZIP_FILES = 20_000
MAX_INLINE_B64 = 8 * 1024 * 1024        # /run admite 10 MB de payload; se deja margen
TAUBIN_ITERS = 10                       # igual que el cuaderno

LABELS = {1: "maxilar", 2: "mandibula", 3: "dientes_sup", 4: "dientes_inf", 5: "canal"}
STL_LABELS = {"dientes_sup": 3, "dientes_inf": 4}      # lo único que se malla
LABELS_FILE, META_FILE = "labels.nii.gz", "cbct.json"
VOLUME_FILE = "volume.nii.gz"                          # grises para el visor, misma rejilla que las etiquetas
ARCH_LABELS = {"mx": 3, "md": 4}
PANO_FILES = {"mx": "pano_mx.png", "md": "pano_md.png"}
VIEWER_MARGIN_XY, VIEWER_MARGIN_Z = 25.0, 20.0         # mm alrededor de los dientes que conserva volume.nii.gz
VOLUME_EXTS = (".nii.gz", ".nii", ".nrrd", ".mha", ".mhd")


class InputError(Exception):
    """El problema está en la entrada: mensaje limpio y el worker sigue sano."""


class UploadError(Exception):
    """La API no aceptó los resultados."""


# --------------------------------------------------------------------------- #
#  Modelo (una vez por worker)
# --------------------------------------------------------------------------- #

_PREDICTOR = None
MODEL_INFO: dict = {}


def get_predictor():
    global _PREDICTOR
    if _PREDICTOR is not None:
        return _PREDICTOR

    import importlib.metadata as md

    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    started = time.time()
    cands = sorted(p for p in MODEL_ROOT.rglob("nnUNetTrainer__nnUNetPlans__3d_fullres")
                   if "__MACOSX" not in p.parts and (p / "plans.json").is_file())
    if not cands:
        raise RuntimeError(f"no se encontró el modelo bajo {MODEL_ROOT}")
    model_dir = cands[0]
    names = sorted(p.name.replace("fold_", "") for p in model_dir.glob("fold_*") if p.is_dir())
    folds = tuple(int(f) if f.isdigit() else f for f in names)
    if not folds:
        raise RuntimeError(f"{model_dir} no tiene carpetas fold_*")
    first = model_dir / f"fold_{folds[0]}"
    ckpt = "checkpoint_final.pth" if (first / "checkpoint_final.pth").is_file() else "checkpoint_best.pth"
    plans = json.loads((model_dir / "plans.json").read_text())
    spacing = float(min(plans["configurations"]["3d_fullres"]["spacing"]))

    cuda = torch.cuda.is_available()
    device = torch.device("cuda", 0) if cuda else torch.device("cpu")
    predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                perform_everything_on_device=cuda, device=device,
                                verbose=False, allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(str(model_dir), use_folds=folds, checkpoint_name=ckpt)

    MODEL_INFO.update({
        "dataset": model_dir.parent.name, "folds": [str(f) for f in folds], "checkpoint": ckpt,
        "spacing_mm": spacing, "nnunetv2": md.version("nnunetv2"), "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0) if cuda else "cpu",
    })
    print(f"[modelo] {MODEL_INFO} · cargado en {time.time() - started:.1f}s", flush=True)
    _PREDICTOR = predictor
    return predictor


def run_nnunet(nifti_in: Path, out_trunc: Path, use_mirroring: bool) -> Path:
    """Predicción en el propio proceso. predict_from_files (el del cuaderno) lanza procesos con
    'spawn' y pasa tensores por /dev/shm, que en un contenedor serverless suele ser de 64 MB; la
    variante secuencial da el mismo resultado sin multiproceso."""
    predictor = get_predictor()
    predictor.use_mirroring = bool(use_mirroring)
    out_trunc.parent.mkdir(parents=True, exist_ok=True)     # la variante secuencial no crea la carpeta
    predictor.predict_from_files_sequential([[str(nifti_in)]], [str(out_trunc)],
                                            save_probabilities=False, overwrite=True)
    out = Path(str(out_trunc) + predictor.dataset_json["file_ending"])
    if not out.is_file():
        raise RuntimeError("nnU-Net no escribió la segmentación")
    return out


# --------------------------------------------------------------------------- #
#  Entrada: descarga y lectura del volumen
# --------------------------------------------------------------------------- #

def download(url: str, dest: Path) -> int:
    err = ""
    for attempt in range(1, 4):
        try:
            with requests.get(url, stream=True, timeout=(15, 180)) as r:
                if r.status_code in (401, 403, 404, 410):
                    raise InputError(f"la URL del CBCT respondió HTTP {r.status_code} (¿enlace caducado?)")
                r.raise_for_status()
                total = 0
                with dest.open("wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        total += len(chunk)
                        if total > MAX_DOWNLOAD:
                            raise InputError(f"el CBCT supera {MAX_DOWNLOAD // 2**20} MB")
                        fh.write(chunk)
            if total == 0:
                raise InputError("la URL del CBCT devolvió un archivo vacío")
            return total
        except requests.RequestException as exc:
            err = str(exc)
            if attempt < 3:
                time.sleep(5 * attempt)
    raise InputError(f"no se pudo descargar el CBCT tras 3 intentos: {err}")


def sniff(path: Path) -> str:
    """Tipo de archivo por su contenido ('' si no se reconoce)."""
    with path.open("rb") as fh:
        head = fh.read(352)
    if head[:2] == b"PK":
        return "zip"
    if head[:2] == b"\x1f\x8b":
        return "nii.gz"
    if head[128:132] == b"DICM":
        return "dcm"
    if len(head) >= 348 and head[344:347] in (b"n+1", b"ni1"):
        return "nii"
    return ""


def kind_from_name(name: str) -> str:
    name = name.lower()
    for ext in ("nii.gz", "nii", "zip", "dcm"):
        if name.endswith("." + ext) or name == ext:
            return ext
    return ""


def _junk(p: Path) -> bool:
    return ("__MACOSX" in p.parts or p.name.startswith("._")
            or p.name.lower() in ("dicomdir", ".ds_store", "thumbs.db"))


def _unzip(path: Path, dest: Path) -> None:
    try:
        with zipfile.ZipFile(path) as z:
            infos = [i for i in z.infolist() if not i.is_dir()]
            if len(infos) > MAX_ZIP_FILES:
                raise InputError(f"el zip trae {len(infos)} archivos (máximo {MAX_ZIP_FILES})")
            if sum(i.file_size for i in infos) > MAX_UNZIPPED:
                raise InputError(f"el zip descomprimido supera {MAX_UNZIPPED // 2**20} MB")
            z.extractall(dest)      # zipfile ya neutraliza rutas absolutas y '..'
    except zipfile.BadZipFile:
        raise InputError("el archivo no es un zip válido")


def _read_tree(root: Path, log) -> sitk.Image:
    """La serie DICOM con más cortes que haya bajo root; si no hay DICOM, el mayor volumen suelto."""
    per_dir: dict[Path, int] = {}
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.name.lower() == "dicomdir":
            p.unlink()              # índice sin imagen: confunde a GDCM (igual que en el cuaderno)
        elif not _junk(p):
            per_dir[p.parent] = per_dir.get(p.parent, 0) + 1

    reader = sitk.ImageSeriesReader()
    best: tuple = ()
    for d in sorted(per_dir, key=lambda k: -per_dir[k])[:50]:
        if best and per_dir[d] <= len(best):
            break                   # ninguna carpeta restante puede tener una serie más larga
        for sid in reader.GetGDCMSeriesIDs(str(d)):
            files = reader.GetGDCMSeriesFileNames(str(d), sid)
            if len(files) > len(best):
                best = files
    if best:
        log(f"serie DICOM: {len(best)} ficheros en {Path(best[0]).parent.name}/")
        if len(best) == 1:
            return sitk.ReadImage(best[0])          # multiframe
        reader.SetFileNames(best)
        return reader.Execute()

    sueltos = [p for p in root.rglob("*") if p.is_file() and not _junk(p)
               and p.name.lower().endswith(VOLUME_EXTS)]
    if sueltos:
        mayor = max(sueltos, key=lambda p: p.stat().st_size)
        log(f"sin series DICOM; se usa el volumen {mayor.name}")
        return sitk.ReadImage(str(mayor))
    raise InputError("el zip no contiene ninguna serie DICOM ni un volumen NIfTI")


def read_volume(path: Path, kind: str, tmp: Path, log) -> sitk.Image:
    try:
        if kind == "zip":
            _unzip(path, tmp / "unzipped")
            img = _read_tree(tmp / "unzipped", log)
        else:
            img = sitk.ReadImage(str(path))
    except RuntimeError as exc:     # SimpleITK lanza RuntimeError con archivos corruptos
        raise InputError(f"no se pudo leer el CBCT: {str(exc).strip().splitlines()[-1][:300]}")
    if img.GetDimension() == 4:     # algunos multiframe se leen como 4D
        img = img[:, :, :, 0]
    if img.GetDimension() != 3:
        raise InputError(f"se esperaba un volumen 3D y llegó uno de {img.GetDimension()}D")
    if img.GetNumberOfComponentsPerPixel() != 1:
        raise InputError("el volumen es multicanal (¿captura en color?): se necesita el CBCT en escala de grises")
    if min(img.GetSize()) < 16:
        raise InputError(f"volumen demasiado pequeño {img.GetSize()}: ¿se subió una sola imagen en vez de la serie?")
    return img


def resample_to_model(img: sitk.Image, model_spacing: float) -> tuple[sitk.Image, bool]:
    """nnU-Net remuestrea a la resolución del modelo de todos modos; hacerlo antes no pierde
    precisión y ahorra RAM y tiempo con CBCT finos (0,15-0,2 mm). Idéntico al cuaderno."""
    sp = np.array(img.GetSpacing())
    if sp.min() >= model_spacing:
        return img, False
    new_sp = np.maximum(sp, model_spacing)
    new_size = [int(round(s * o / n)) for s, o, n in zip(img.GetSize(), sp, new_sp)]
    out = sitk.Resample(img, new_size, sitk.Transform(), sitk.sitkLinear, img.GetOrigin(),
                        tuple(float(x) for x in new_sp), img.GetDirection(), 0, img.GetPixelID())
    return out, True


# --------------------------------------------------------------------------- #
#  Salida: mallas y subida
# --------------------------------------------------------------------------- #

def mesh_label(seg: np.ndarray, label: int, spacing_zyx, origin, direction):
    """(malla, info) de una etiqueta, en mm del paciente (LPS). Mismo método que el cuaderno
    (marching cubes + Taubin x10) con dos retoques que no cambian la geometría: se trabaja sobre
    el recorte de la etiqueta (más rápido) con un vóxel de margen (la superficie queda cerrada
    aunque el diente toque el borde del volumen), y se deja la normal hacia fuera."""
    mask = seg == label
    if not mask.any():
        return None, None
    box = ndimage.find_objects(mask.astype(np.uint8))[0]
    sub = np.pad(mask[box], 1)
    verts, faces, _, _ = measure.marching_cubes(sub.astype(np.uint8), level=0.5, spacing=tuple(spacing_zyx))
    verts += (np.array([s.start for s in box]) - 1) * np.asarray(spacing_zyx)
    verts = verts[:, ::-1] @ direction.T + origin           # z,y,x -> x,y,z -> mm en el paciente
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    trimesh.smoothing.filter_taubin(mesh, iterations=TAUBIN_ITERS)
    if mesh.volume < 0:             # invertir el orden de los ejes es un espejo: da la vuelta a las caras
        mesh.invert()

    _, n = ndimage.label(sub)
    sizes = ndimage.sum(sub, _, range(1, n + 1)) if n else [0]
    info = {
        "label": int(label), "faces": int(len(mesh.faces)), "voxels": int(mask.sum()),
        "volume_mm3": round(float(mask.sum() * np.prod(spacing_zyx)), 1),
        "bounds_mm": np.round(mesh.bounds, 2).tolist(),
        "pieces": int(n), "largest_piece_pct": round(float(100 * max(sizes) / mask.sum()), 1),
    }
    return mesh, info


def viewer_volume(nifti: Path, seg: np.ndarray, dest: Path) -> tuple[np.ndarray, dict]:
    """Escribe volume.nii.gz para el visor y devuelve (array completo z,y,x, datos del archivo).

    Siempre int16 y sobre la misma rejilla que labels.nii.gz, pero RECORTADO a la zona de los dientes
    más un margen: un CBCT de campo grande (cara entera) pasaría de 200 MB y el navegador tiene que
    descargarlo en cada caso; así el peso queda acotado. offset_ijk dice dónde cae el recorte dentro
    de la rejilla de las etiquetas:  índice_en_labels = índice_en_volume + offset_ijk."""
    img = sitk.ReadImage(str(nifti))
    if img.GetPixelID() != sitk.sitkInt16:      # uint16, float, int32...: se recorta al rango de int16
        img = sitk.Clamp(sitk.Round(img) if "float" in img.GetPixelIDTypeAsString() else img,
                         sitk.sitkInt16, -32768, 32767)
    arr = sitk.GetArrayFromImage(img)

    box = ndimage.find_objects(np.isin(seg, list(ARCH_LABELS.values())).astype(np.uint8))[0]   # z,y,x
    sp = np.array(img.GetSpacing())             # x,y,z
    margin = np.ceil(np.array([VIEWER_MARGIN_XY, VIEWER_MARGIN_XY, VIEWER_MARGIN_Z]) / sp).astype(int)
    lo = np.maximum(np.array([b.start for b in box][::-1]) - margin, 0)
    hi = np.minimum(np.array([b.stop for b in box][::-1]) + margin, np.array(img.GetSize()))
    roi = img[int(lo[0]):int(hi[0]), int(lo[1]):int(hi[1]), int(lo[2]):int(hi[2])]
    sitk.WriteImage(roi, str(dest), True)

    sub = arr[::4, ::4, ::4]
    w_lo, w_hi = (float(v) for v in np.percentile(sub, [1, 99.9]))
    info = {
        "file": VOLUME_FILE, "dtype": "int16", "size": [int(v) for v in roi.GetSize()],
        "spacing": [round(float(v), 4) for v in roi.GetSpacing()],
        "origin": [round(float(v), 3) for v in roi.GetOrigin()],
        "direction": [round(float(v), 6) for v in roi.GetDirection()],
        "offset_ijk": [int(v) for v in lo], "min": int(arr.min()), "max": int(arr.max()),
        "window": {"low": round(w_lo), "high": round(max(w_hi, w_lo + 1))},
    }
    return arr, info


def arch_curve(seg: np.ndarray, label: int, spacing_zyx, origin, direction,
               step: float = 1.0, extend: float = 12.0):
    """Curva panorámica de una arcada, en mm (LPS), sacada de la propia segmentación.

    Se proyectan los dientes sobre el plano axial. Visto desde un punto C situado en el extremo
    abierto de la U, cada rayo cruza la arcada una sola vez: se reparten los vóxeles en sectores
    angulares, de cada sector se toma el radio mediano (el centro de la banda de dientes) y por
    esos puntos pasa un spline suavizado. Los huecos de dientes ausentes los salva el spline. La
    curva se alarga `extend` mm por cada extremo y se entrega de la DERECHA del paciente a la
    izquierda (convención radiológica), con un punto cada `step` mm."""
    idx = np.argwhere(seg == label)
    if len(idx) < 2000:
        return None
    zr = idx[:, 0]
    if len(idx) > 300_000:
        idx = idx[:: len(idx) // 300_000 + 1]
    pts = (idx[:, ::-1] * np.asarray(spacing_zyx)[::-1]) @ direction.T + origin
    x, y = pts[:, 0], pts[:, 1]

    # ¿Hacia dónde abre la U? En LPS, hacia +y (posterior). El extremo abierto es el más ancho en x.
    y20, y80 = np.percentile(y, [20, 80])
    ancho = lambda m: float(np.ptp(np.percentile(x[m], [5, 95]))) if m.sum() > 50 else 0.0
    sign = 1.0 if ancho(y >= y80) >= ancho(y <= y20) else -1.0
    yy = y * sign
    cx, cy = float(np.mean(np.percentile(x, [1, 99]))), float(yy.max() + 2.0)
    vx, vy = x - cx, yy - cy                                # vy < 0 siempre
    theta, r = np.arctan2(-vy, vx), np.hypot(vx, vy)        # 0 = +x (izquierda del paciente) ... pi = -x

    n_bins = 48
    which = np.minimum((theta / np.pi * n_bins).astype(int), n_bins - 1)
    ctrl = []
    for b in range(n_bins):
        m = which == b
        if m.sum() >= 40:
            rb, tb = np.median(r[m]), np.median(theta[m])
            ctrl.append((cx + rb * np.cos(tb), (cy - rb * np.sin(tb)) * sign))
    if len(ctrl) < 8:
        return None
    P = np.array(ctrl)[::-1]                                # de -x (derecha del paciente) a +x
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    tck, _ = splprep([P[:, 0], P[:, 1]], u=d, s=len(P) * 0.6 ** 2, k=3)
    dense = np.column_stack(splev(np.linspace(0, d[-1], 1000), tck))

    def uniforme(Q):
        a = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))]
        t = np.arange(0, a[-1], step)
        return np.column_stack([np.interp(t, a, Q[:, 0]), np.interp(t, a, Q[:, 1])]), float(a[-1])

    core, core_len = uniforme(dense)
    if not 40 <= core_len <= 250:                           # una arcada mide 90-140 mm
        return None
    t0 = core[0] - core[3]; t0 /= np.linalg.norm(t0)
    t1 = core[-1] - core[-4]; t1 /= np.linalg.norm(t1)
    k = np.arange(step, extend + 1e-6, step)[:, None]
    curve, length = uniforme(np.vstack([core[0] + k[::-1] * t0, core, core[-1] + k * t1]))

    fit = float(np.median(cKDTree(dense).query(np.column_stack([x, y]))[0]))
    z_mm = (zr.min() * spacing_zyx[0], zr.max() * spacing_zyx[0])     # válido si el eje z del volumen es el del paciente
    z_mm = [round(float(origin[2] + direction[2, 2] * z), 2) for z in z_mm]
    return {
        "points_mm": np.round(curve, 2).tolist(), "step_mm": step, "length_mm": round(length, 1),
        "extended_mm": extend, "teeth_z_mm": sorted(z_mm), "fit_mm": round(fit, 2), "sectors": len(ctrl),
        "order": "derecha del paciente -> izquierda", "opens": "+y" if sign > 0 else "-y",
    }


def panoramic(vol: np.ndarray, spacing_zyx, origin, direction, points_mm, z_lo: float, z_hi: float,
              ds: float = 0.3, dz: float = 0.3, half: float = 5.0):
    """Reconstrucción panorámica: media de un bloque de ±half mm a lo largo de la normal a la
    curva. Columna c <-> longitud de arco c*ds desde el primer punto; fila f <-> z = z_hi - f*dz."""
    P = np.asarray(points_mm, float)
    a = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    s = np.arange(0, a[-1], ds)
    X, Y = np.interp(s, a, P[:, 0]), np.interp(s, a, P[:, 1])
    T = np.gradient(np.column_stack([X, Y]), axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    N = np.column_stack([T[:, 1], -T[:, 0]])
    zs = np.arange(z_hi, z_lo, -dz)
    sp_xyz = np.asarray(spacing_zyx)[::-1]
    acc = np.zeros((len(zs), len(s)), np.float32)
    offsets = np.arange(-half, half + 1e-6, 1.0)
    fondo = float(vol.min())
    for t in offsets:
        p = np.empty((len(zs), len(s), 3))
        p[..., 0], p[..., 1], p[..., 2] = X + t * N[:, 0], Y + t * N[:, 1], zs[:, None]
        ijk = ((p - origin) @ direction) / sp_xyz           # mm -> índice (la matriz de dirección es ortonormal)
        acc += ndimage.map_coordinates(vol, [ijk[..., 2], ijk[..., 1], ijk[..., 0]], order=1,
                                       mode="constant", cval=fondo, output=np.float32)
    acc /= len(offsets)
    lo, hi = np.percentile(acc, [1, 99.7])
    img = (np.clip((acc - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
    meta = {"ds_mm": ds, "dz_mm": dz, "z_top_mm": round(float(z_hi), 2), "width": int(img.shape[1]),
            "height": int(img.shape[0]), "slab_mm": 2 * half}
    return img, meta


def voxels_present(voxels: dict) -> set[int]:
    return {int(k) for k, v in voxels.items() if v > 0}


def z_extent_mm(size_xyz, spacing_zyx, origin, direction) -> tuple[float, float]:
    corners = np.array([[i, j, k] for i in (0, size_xyz[0] - 1) for j in (0, size_xyz[1] - 1)
                        for k in (0, size_xyz[2] - 1)], float)
    z = ((corners * np.asarray(spacing_zyx)[::-1]) @ direction.T + origin)[:, 2]
    return float(z.min()), float(z.max())


def result_target(inp: dict, name: str):
    urls = inp.get("result_urls") or {}
    if name in urls:
        return urls[name]
    base = inp.get("result_url")
    if not base:
        return None
    return base.replace("{name}", name) if "{name}" in base else base.rstrip("/") + "/" + name


def upload(path: Path, url: str) -> None:
    err = ""
    for attempt in range(1, 4):
        try:
            with path.open("rb") as fh:     # objeto fichero => requests manda Content-Length (no chunked)
                r = requests.put(url, data=fh, headers={"Content-Type": "application/octet-stream"},
                                 timeout=(15, 600))
            if r.status_code in (200, 201, 204):
                return
            err = f"HTTP {r.status_code}: {r.text[:200]}"
            if 400 <= r.status_code < 500 and r.status_code not in (408, 429):
                break               # token caducado, nombre no permitido...: reintentar no arregla nada
        except requests.RequestException as exc:
            err = str(exc)
        if attempt < 3:
            time.sleep(5 * attempt)
    raise UploadError(f"la API no aceptó {path.name}: {err}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
#  Handler
# --------------------------------------------------------------------------- #

def _progress(job: dict, step: str) -> None:
    """Aparece en /status como output mientras el job está IN_PROGRESS."""
    if os.environ.get("RUNPOD_WEBHOOK_POST_OUTPUT"):        # solo existe dentro de Runpod
        try:
            runpod.serverless.progress_update(job, step)
        except Exception:
            pass


def handler(job: dict) -> dict:
    inp = job.get("input") or {}
    job_id = re.sub(r"[^A-Za-z0-9_.-]", "_", str(job.get("id") or "local"))
    name = str(inp.get("name") or job_id)
    work = WORK_ROOT / job_id
    warnings: list[str] = []
    timings: dict[str, int] = {}
    stage = "input"

    def log(msg: str) -> None:
        print(f"[{name}] {msg}", flush=True)

    def lap(key: str, since: float) -> float:
        timings[key] = int((time.time() - since) * 1000)
        return time.time()

    try:
        url = inp.get("cbct_url")
        if not url or not isinstance(url, str):
            raise InputError("falta input.cbct_url")
        has_target = bool(inp.get("result_url") or inp.get("result_urls"))
        shutil.rmtree(work, ignore_errors=True)
        (work / "out").mkdir(parents=True, exist_ok=True)
        t_all = t = time.time()

        # 1) descarga ------------------------------------------------------- #
        stage = "download"; _progress(job, stage)
        raw = work / "input.bin"
        size = download(url, raw)
        kind = sniff(raw) or kind_from_name(str(inp.get("input_format") or "")) \
            or kind_from_name(unquote(urlparse(url).path))
        if not kind:
            raise InputError("formato no reconocido: se admite .zip (carpeta DICOM), .dcm, .nii o .nii.gz")
        raw = raw.rename(work / f"input.{kind}")             # SimpleITK elige el lector por la extensión
        log(f"descargado {size / 1e6:.1f} MB ({kind})")
        t = lap("download", t)

        # 2) DICOM/NIfTI -> volumen a la resolución del modelo ---------------- #
        stage = "load"; _progress(job, stage)
        img = read_volume(raw, kind, work, log)
        original = {"size": list(img.GetSize()), "spacing": [round(float(s), 4) for s in img.GetSpacing()]}
        sp = np.array(img.GetSpacing())
        if sp.max() > 1.0 or sp.max() / sp.min() > 3:
            warnings.append(f"spacing sospechoso {original['spacing']} mm: si el DICOM no declara bien el "
                            "grosor de corte, las mallas saldrán deformadas en ese eje")
        get_predictor()                                      # asegura MODEL_INFO (spacing del modelo)
        img, resampled = resample_to_model(img, MODEL_INFO["spacing_mm"])
        nifti = work / "in" / "case_0000.nii.gz"             # nnU-Net exige el sufijo _0000
        nifti.parent.mkdir(exist_ok=True)
        sitk.WriteImage(img, str(nifti))
        log(f"original {original['size']} @ {original['spacing']} -> {list(img.GetSize())} @ "
            f"{[round(float(s), 4) for s in img.GetSpacing()]} ({np.prod(img.GetSize()) / 1e6:.1f} M vóxeles)")
        del img
        raw.unlink(missing_ok=True)
        shutil.rmtree(work / "unzipped", ignore_errors=True)
        t = lap("load", t)

        # 3) nnU-Net --------------------------------------------------------- #
        stage = "segment"; _progress(job, stage)
        seg_path = run_nnunet(nifti, work / "seg" / "case", inp.get("use_mirroring", True))
        t = lap("segment", t)

        # 4) solo lo necesario: mapa de etiquetas + STL de dientes ------------ #
        stage = "mesh"; _progress(job, stage)
        out_dir = work / "out"
        seg_img = sitk.ReadImage(str(seg_path))
        seg = sitk.GetArrayFromImage(seg_img)                                 # z,y,x
        spacing_zyx = np.array(seg_img.GetSpacing())[::-1]
        origin = np.array(seg_img.GetOrigin())
        direction = np.array(seg_img.GetDirection()).reshape(3, 3)
        voxels = {str(int(l)): int(c) for l, c in zip(*np.unique(seg, return_counts=True))}
        log(f"vóxeles por etiqueta: {voxels}")
        shutil.copyfile(seg_path, out_dir / LABELS_FILE)

        meshes, empty = {}, []
        for stl_name, label in STL_LABELS.items():
            mesh, info = mesh_label(seg, label, spacing_zyx, origin, direction)
            if mesh is None:
                empty.append(stl_name)
                warnings.append(f"la etiqueta {label} ({stl_name}) salió vacía: esa arcada no entra en el CBCT")
                continue
            mesh.export(out_dir / f"{stl_name}.stl")
            meshes[stl_name] = info
            log(f"{stl_name}: {info['faces']} caras · {info['pieces']} pieza(s), la mayor {info['largest_piece_pct']}%")
            if info["largest_piece_pct"] < 90:
                warnings.append(f"{stl_name} está partido en {info['pieces']} piezas (la mayor, "
                                f"{info['largest_piece_pct']}%): usa labels.nii.gz en el paso 5, no el STL")
        if len(empty) == len(STL_LABELS):
            raise InputError("el modelo no encontró dientes en el volumen: ¿es un CBCT dental?")
        t = lap("mesh", t)

        # 4b) para el visor de cortes: grises en la rejilla de las etiquetas + curva de cada arcada -- #
        vol, vol_info = viewer_volume(nifti, seg, out_dir / VOLUME_FILE)
        if vol.shape != seg.shape:
            raise RuntimeError(f"volumen {vol.shape} y etiquetas {seg.shape} no comparten rejilla")
        curves, panos = {}, {}
        for arch, label in ARCH_LABELS.items():
            try:
                curve = arch_curve(seg, label, spacing_zyx, origin, direction)
            except Exception as exc:                         # noqa: BLE001 · la curva es un extra: no tumba el job
                curve = None
                log(f"curva {arch}: {type(exc).__name__}: {exc}")
            if curve is None:
                if label in voxels_present(voxels):
                    warnings.append(f"no se pudo trazar la curva panorámica de {arch}: habrá que dibujarla a mano")
                continue
            curves[arch] = curve
            if curve["fit_mm"] > 4 or curve["opens"] != "+y":
                warnings.append(f"curva de {arch} dudosa (ajuste {curve['fit_mm']} mm, abre hacia {curve['opens']}): "
                                "revisa la panorámica; si abre hacia -y el volumen no está en LPS")
        if curves:
            z_min, z_max = z_extent_mm(seg_img.GetSize(), spacing_zyx, origin, direction)
            zs = [z for c in curves.values() for z in c["teeth_z_mm"]]
            z_lo, z_hi = max(z_min, min(zs) - 12.0), min(z_max, max(zs) + 12.0)
            for arch, curve in curves.items():
                img, meta = panoramic(vol, spacing_zyx, origin, direction, curve["points_mm"], z_lo, z_hi)
                sitk.WriteImage(sitk.GetImageFromArray(img), str(out_dir / PANO_FILES[arch]))
                panos[arch] = {"file": PANO_FILES[arch], **meta}
                log(f"curva {arch}: {curve['length_mm']} mm, ajuste {curve['fit_mm']} mm · panorámica "
                    f"{meta['width']}x{meta['height']}")
        del vol
        t = lap("viewer", t)

        # 5) a la API -------------------------------------------------------- #
        files = {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(out_dir.iterdir())}
        result = {
            "name": name, "files": files, "uploaded": False,
            "volume": {"original": original, "size": list(seg_img.GetSize()),
                       "spacing": [round(float(s), 4) for s in seg_img.GetSpacing()],
                       "origin": [round(float(o), 3) for o in origin],
                       "direction": [round(float(d), 6) for d in direction.ravel()],
                       "resampled": bool(resampled), "frame": "LPS (DICOM), mm"},
            "labels": {str(k): v for k, v in LABELS.items()}, "voxels": voxels,
            "meshes": meshes, "empty": empty,
            # Para el visor. OJO: la geometría se toma de AQUÍ (viewer.volume.* para volume.nii.gz,
            # volume.* para labels.nii.gz; mm en LPS, como los STL y las curvas), no de la cabecera
            # NIfTI, que está en RAS (x e y cambian de signo). Vóxel (i,j,k) -> i + j*nx + k*nx*ny.
            "viewer": {"volume": vol_info, "labels": LABELS_FILE, "curves": curves, "panos": panos},
            "warnings": warnings,
            "model": dict(MODEL_INFO), "use_mirroring": bool(inp.get("use_mirroring", True)),
        }
        if has_target:
            stage = "upload"; _progress(job, stage)
            for fname in files:
                target = result_target(inp, fname)
                if target is None:
                    raise InputError(f"result_urls no trae destino para {fname}")
                upload(out_dir / fname, target)
            t = lap("upload", t)
            timings["total"] = int((time.time() - t_all) * 1000)
            result.update(uploaded=True, timings_ms=timings)
            meta = out_dir / META_FILE                        # el último: marca el resultado como completo
            meta.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            target = result_target(inp, META_FILE)
            if target is None:
                raise InputError(f"result_urls no trae destino para {META_FILE}")
            upload(meta, target)
        else:
            timings["total"] = int((time.time() - t_all) * 1000)
            result["timings_ms"] = timings
            warnings.append("sin result_url: no se subió nada (los STL no caben en la respuesta de Runpod)")
            b64 = base64.b64encode((out_dir / LABELS_FILE).read_bytes()).decode()
            if len(b64) <= MAX_INLINE_B64:
                result["labels_nii_gz_base64"] = b64
        log(f"listo en {timings['total'] / 1000:.1f}s · {timings}")
        return result

    except (InputError, UploadError) as exc:
        log(f"ERROR en '{stage}': {exc}")
        return {"error": str(exc)}
    except Exception as exc:        # noqa: BLE001
        traceback.print_exc()
        out = {"error": f"[{stage}] {type(exc).__name__}: {exc}"[:2000]}
        if stage == "segment":
            out["refresh_worker"] = True    # tras un fallo de CUDA el proceso no es de fiar: worker nuevo
        return out
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":       # imprescindible: nada de esto debe ejecutarse al importar el módulo
    if os.environ.get("DENTALSEG_PRELOAD", "1") == "1":
        get_predictor()
    runpod.serverless.start({"handler": handler})
