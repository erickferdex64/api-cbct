"""Prueba del handler SIN GPU ni nnU-Net:  python test_local.py

Sustituye la red por un umbral sobre un CBCT sintético (dos esferas = "dientes" sup. e inf.) y
comprueba todo lo demás: descarga, zip DICOM / NIfTI, remuestreo, mallas en mm del paciente
(también con la matriz de dirección girada), superficie cerrada, subida por PUT y errores.
Requisitos: numpy scipy scikit-image trimesh SimpleITK requests runpod
"""
import json, os, tempfile, threading, time, zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import trimesh

os.environ.setdefault("DENTALSEG_WORK_ROOT", tempfile.mkdtemp(prefix="dentalseg_work_"))
import handler as H

ROOT = Path(tempfile.mkdtemp(prefix="dentalseg_test_"))
SERVE, UPLOADS = ROOT / "serve", ROOT / "uploads"
SERVE.mkdir(); UPLOADS.mkdir()
BASE = {"labels.nii.gz", "dientes_sup.stl", "dientes_inf.stl", "volume.nii.gz", "cbct.json"}
ALLOWED = BASE | {"pano_mx.png", "pano_md.png"}


class Srv(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_GET(self):
        p = SERVE / self.path.strip("/").split("/")[-1]
        if not p.is_file():
            self.send_response(404); self.end_headers(); return
        data = p.read_bytes()
        self.send_response(200); self.send_header("Content-Length", str(len(data))); self.end_headers()
        self.wfile.write(data)

    def do_PUT(self):
        name = self.path.strip("/").split("/")[-1]
        n = int(self.headers.get("Content-Length") or -1)
        assert "chunked" not in (self.headers.get("Transfer-Encoding") or ""), "la subida debe llevar Content-Length"
        body = self.rfile.read(n) if n > 0 else b""
        if "/caducado/" in self.path or name not in ALLOWED:
            self.send_response(404); self.end_headers(); return
        (UPLOADS / name).write_bytes(body)
        self.send_response(201); self.end_headers()


srv = ThreadingHTTPServer(("127.0.0.1", 0), Srv)
threading.Thread(target=srv.serve_forever, daemon=True).start()
URL = f"http://127.0.0.1:{srv.server_address[1]}"

# ---- CBCT sintético: fondo -1000, esfera "dientes sup." 3000, esfera "dientes inf." 2000 ---------
SIZE, SP, ORIGIN = (150, 140, 130), 0.2, np.array([-15.0, -14.0, 5.0])      # x,y,z · más fino que el modelo
C_SUP, R_SUP = np.array([60, 70, 95]) * SP, 4.0                               # centros en mm "de índice"
C_INF, R_INF = np.array([95, 70, 6]) * SP, 3.5                                # toca el borde z=0 del volumen


def volumen(direction):
    zz, yy, xx = np.meshgrid(*[np.arange(n) * SP for n in SIZE[::-1]], indexing="ij")
    arr = np.full(SIZE[::-1], -1000, np.int16)
    arr[(xx - C_SUP[0]) ** 2 + (yy - C_SUP[1]) ** 2 + (zz - C_SUP[2]) ** 2 <= R_SUP ** 2] = 3000
    arr[(xx - C_INF[0]) ** 2 + (yy - C_INF[1]) ** 2 + (zz - C_INF[2]) ** 2 <= R_INF ** 2] = 2000
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((SP,) * 3); img.SetOrigin(tuple(ORIGIN)); img.SetDirection(tuple(np.ravel(direction)))
    return img


def escribir_dicom_zip(img, dest):
    tmp = ROOT / "dcm"; (tmp / "PACIENTE" / "SERIE1").mkdir(parents=True, exist_ok=True)
    w = sitk.ImageFileWriter(); w.KeepOriginalImageUIDOn()
    uid = "1.2.826.0.1.3680043.2.1125." + time.strftime("%Y%m%d%H%M%S")
    d = img.GetDirection()
    for i in range(img.GetDepth()):
        sl = img[:, :, i]
        for tag, val in (("0008|0060", "CT"), ("0020|000e", uid), ("0020|000d", uid + ".1"), ("0008|0016", "1.2.840.10008.5.1.4.1.1.2"),
                         ("0020|0037", "\\".join(map(str, (d[0], d[3], d[6], d[1], d[4], d[7])))),
                         ("0020|0032", "\\".join(map(str, img.TransformIndexToPhysicalPoint((0, 0, i))))),
                         ("0020|0013", str(i))):
            sl.SetMetaData(tag, val)
        w.SetFileName(str(tmp / "PACIENTE" / "SERIE1" / f"IM{i:04d}.dcm")); w.Execute(sl)
    (tmp / "DICOMDIR").write_bytes(b"\0" * 128 + b"DICM" + b"indice sin imagen")
    (tmp / "__MACOSX").mkdir(exist_ok=True); (tmp / "__MACOSX" / "._IM0000.dcm").write_bytes(b"basura")
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for p in tmp.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(tmp))


class PredictorFalso:
    """Misma interfaz que nnUNetPredictor 2.8.1 para lo que usa el handler. Como el de verdad,
    escribe un JSON en la carpeta de salida SIN crearla y nombra la salida con file_ending."""
    dataset_json = {"file_ending": ".nii.gz"}
    use_mirroring = True

    def predict_from_files_sequential(self, list_of_lists, outputs, save_probabilities=False, overwrite=True,
                                      folder_with_segs_from_prev_stage=None):
        (Path(outputs[0]).parent / "predict_from_raw_data_args.json").write_text("{}")
        img = sitk.ReadImage(list_of_lists[0][0]); a = sitk.GetArrayFromImage(img)
        mitad = a.shape[0] // 2                              # umbral a media altura del borde de cada esfera
        seg = np.zeros(a.shape, np.uint8); seg[mitad:][a[mitad:] > 1000] = 3; seg[:mitad][a[:mitad] > 500] = 4
        out = sitk.GetImageFromArray(seg); out.CopyInformation(img)
        sitk.WriteImage(out, outputs[0] + ".nii.gz", True)


FALSO = PredictorFalso()
H.get_predictor = lambda: FALSO
H.MODEL_INFO.update({"spacing_mm": 0.312, "dataset": "FALSO", "device": "cpu"})


def comprobar_volumen(out, lab):
    """volume.nii.gz: int16, misma rejilla que las etiquetas (recorte con desplazamiento entero) y el
    JSON describe exactamente lo que hay en el archivo."""
    vol = sitk.ReadImage(str(UPLOADS / "volume.nii.gz")); v = out["viewer"]["volume"]
    assert vol.GetPixelID() == sitk.sitkInt16 and list(vol.GetSize()) == v["size"]
    assert np.allclose(vol.GetSpacing(), lab.GetSpacing()) and np.allclose(vol.GetDirection(), lab.GetDirection())
    assert np.allclose(vol.GetOrigin(), v["origin"], atol=2e-3) and np.allclose(vol.GetSpacing(), v["spacing"], atol=1e-4)
    off = np.array(v["offset_ijk"]); assert np.allclose(lab.TransformIndexToPhysicalPoint([int(o) for o in off]), vol.GetOrigin(), atol=1e-4)
    g = sitk.GetArrayFromImage(vol); e = sitk.GetArrayFromImage(lab)
    e = e[off[2]:off[2] + g.shape[0], off[1]:off[1] + g.shape[1], off[0]:off[0] + g.shape[2]]
    assert e.shape == g.shape and g[e == 3].mean() > 2000 > g[e == 0].mean(), "grises y etiquetas no casan vóxel a vóxel"
    assert (sitk.GetArrayFromImage(lab) > 2).sum() == (e > 2).sum(), "el recorte se dejó dientes fuera"
    return vol


def comprobar(nombre, direction, fichero, fmt_en_url=True):
    for f in UPLOADS.iterdir(): f.unlink()
    D = np.array(direction, float).reshape(3, 3)
    out = H.handler({"id": nombre, "input": {"cbct_url": f"{URL}/tok:en:firma/{fichero if fmt_en_url else 'descarga'}",
                                             "result_url": f"{URL}/subida/tok:en:firma", "name": nombre}})
    assert "error" not in out, out
    assert out["uploaded"] and set(out["files"]) | {"cbct.json"} == BASE, out["files"]
    assert {p.name for p in UPLOADS.iterdir()} == BASE, "en la API debe quedar SOLO lo necesario"
    assert out["volume"]["resampled"] and abs(out["volume"]["spacing"][0] - 0.312) < 1e-3
    for stl, c_idx, r in (("dientes_sup.stl", C_SUP, R_SUP), ("dientes_inf.stl", C_INF, R_INF)):
        assert (UPLOADS / stl).stat().st_size == out["files"][stl]["bytes"]
        m = trimesh.load(UPLOADS / stl, process=True)
        esperado = D @ c_idx + ORIGIN                                   # centro de la esfera en mm del paciente
        assert m.is_watertight, f"{stl} no es cerrada"
        assert m.volume > 0, f"{stl} tiene las normales hacia dentro"
        if stl == "dientes_sup.stl":                                    # la inferior está cortada por el borde
            assert np.linalg.norm(m.center_mass - esperado) < 0.15, (m.center_mass, esperado)
            assert abs(m.volume / (4 / 3 * np.pi * r ** 3) - 1) < 0.06, m.volume
        else:
            assert np.linalg.norm((m.center_mass - esperado)[:2] @ np.eye(2)) < 3.0
    lab = sitk.ReadImage(str(UPLOADS / "labels.nii.gz"))
    assert np.allclose(np.array(lab.GetDirection()).reshape(3, 3), D, atol=1e-6)
    comprobar_volumen(out, lab)
    assert out["viewer"]["curves"] == {} and any("curva" in w for w in out["warnings"])   # dos esferas no son una arcada
    meta = json.loads((UPLOADS / "cbct.json").read_text(encoding="utf-8"))
    assert meta["uploaded"] and meta["meshes"]["dientes_sup"]["pieces"] == 1 and "timings_ms" in meta
    json.dumps(out)                                                     # todo serializable (nada de numpy)
    print(f"OK  {nombre:28s} {out['timings_ms']['total']:5d} ms · sup {meta['meshes']['dientes_sup']['faces']} caras")


I = np.eye(3)
GIRO = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])                   # 90° en z
FLIP = np.diag([-1.0, -1.0, 1.0])                                       # típico de NIfTI pasados de RAS a LPS
sitk.WriteImage(volumen(I), str(SERVE / "cbct.nii.gz"))
sitk.WriteImage(volumen(GIRO), str(SERVE / "giro.nii.gz"))
sitk.WriteImage(volumen(FLIP), str(SERVE / "flip.nii"))
escribir_dicom_zip(volumen(I), SERVE / "cbct.zip")

comprobar("nifti_gz", I, "cbct.nii.gz")
comprobar("nifti_direccion_girada", GIRO, "giro.nii.gz")
comprobar("nifti_sin_comprimir_flip", FLIP, "flip.nii")
comprobar("zip_dicom", I, "cbct.zip")
(SERVE / "descarga").write_bytes((SERVE / "cbct.zip").read_bytes())
comprobar("zip_sin_extension_en_url", I, "cbct.zip", fmt_en_url=False)

# ---- visor: dos arcadas de "dientes" construidas en coordenadas FÍSICAS (el paciente es el mismo
#      sea cual sea la matriz de dirección del archivo) -> curva panorámica + panorámica ------------
from scipy.spatial import cKDTree
ARCOS = {"mx": (25.0, 38.0, 32.0, 3000), "md": (23.0, 35.0, 10.0, 2000)}           # semiejes, z, gris


def elipse(a, b, grados):
    ph = np.radians(grados)
    return np.column_stack([a * np.sin(ph), -b * np.cos(ph) + 5.0])                # abre hacia +y (LPS)


def fantoma_arcadas(D, sp=0.4, dtype=np.int16):
    D = np.array(D, float); lo, hi = np.array([-40.0, -50.0, 0.0]), np.array([40.0, 30.0, 44.0])
    eje = np.argmax(np.abs(D), axis=0)                                             # eje físico de cada eje de índice
    size = [int(round((hi - lo)[eje[j]] / sp)) for j in range(3)]
    ext = (np.array(size) - 1) * sp
    esquinas = np.array([[i, j, k] for i in (0, ext[0]) for j in (0, ext[1]) for k in (0, ext[2])]) @ D.T
    origin = lo - esquinas.min(0)
    kk, jj, ii = np.meshgrid(*[np.arange(n) * sp for n in size[::-1]], indexing="ij")
    P = np.stack([ii, jj, kk], -1) @ D.T + origin
    arr = np.full(ii.shape, -1000, dtype)
    for a, b, z0, gris in ARCOS.values():
        for n, c in enumerate(elipse(a, b, np.linspace(-78, 78, 14))):
            if n != 4:                                                             # falta un diente
                arr[((P[..., 0] - c[0]) ** 2 + (P[..., 1] - c[1]) ** 2) / 3.6 ** 2 + ((P[..., 2] - z0) / 8.5) ** 2 <= 1] = gris
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((sp,) * 3); img.SetOrigin(tuple(origin)); img.SetDirection(tuple(D.ravel()))
    return img


curvas = {}
for nombre, D, dtype in (("arcadas", I, np.int16), ("arcadas_giro", GIRO, np.int16), ("arcadas_flip_float", FLIP, np.float32)):
    for f in UPLOADS.iterdir(): f.unlink()
    sitk.WriteImage(fantoma_arcadas(D, dtype=dtype), str(SERVE / f"{nombre}.nii.gz"))
    out = H.handler({"id": nombre, "input": {"cbct_url": f"{URL}/t/{nombre}.nii.gz", "result_url": f"{URL}/subida/t"}})
    assert "error" not in out, out
    assert {p.name for p in UPLOADS.iterdir()} == ALLOWED, sorted(p.name for p in UPLOADS.iterdir())
    lab = sitk.ReadImage(str(UPLOADS / "labels.nii.gz"))
    comprobar_volumen(out, lab)                                                    # también cuando el CBCT llega en float
    v = out["viewer"]; assert v["volume"]["window"]["high"] > v["volume"]["window"]["low"]
    for arch, (a, b, z0, _) in ARCOS.items():
        c = v["curves"][arch]; pts = np.array(c["points_mm"]); n_ext = int(c["extended_mm"] / c["step_mm"])
        assert pts[0, 0] < 0 < pts[-1, 0] and c["opens"] == "+y", "debe ir de la derecha del paciente (-x) a la izquierda"
        real = cKDTree(elipse(a, b, np.linspace(-100, 100, 6000)))                # el último diente sobresale de ±78°
        desvio = real.query(pts[n_ext:-n_ext])[0].max()
        assert desvio < 1.0, f"{arch}: la curva se aparta {desvio:.2f} mm del centro de la arcada"
        assert abs(c["teeth_z_mm"][0] - (z0 - 8.5)) < 1 and abs(c["teeth_z_mm"][1] - (z0 + 8.5)) < 1, c["teeth_z_mm"]
        m = v["panos"][arch]; png = sitk.GetArrayFromImage(sitk.ReadImage(str(UPLOADS / m["file"])))
        assert png.shape == (m["height"], m["width"]) and png.dtype == np.uint8
        fila = lambda z: int((m["z_top_mm"] - z) / m["dz_mm"])
        assert png[fila(z0 + 6):fila(z0 - 6)].mean() > 4 * png[fila(21.5):fila(20.5)].mean() + 5, "la panorámica no pasa por los dientes"
        curvas.setdefault(arch, []).append(pts)
    print(f"OK  {nombre:28s} curva mx {v['curves']['mx']['length_mm']} mm (ajuste {v['curves']['mx']['fit_mm']}), "
          f"md {v['curves']['md']['length_mm']} mm · panorámica {v['panos']['md']['width']}x{v['panos']['md']['height']}")
H.VIEWER_MARGIN_XY, H.VIEWER_MARGIN_Z = 6.0, 3.0                                     # forzar el recorte
for f in UPLOADS.iterdir(): f.unlink()
out = H.handler({"id": "recorte", "input": {"cbct_url": f"{URL}/t/arcadas_giro.nii.gz", "result_url": f"{URL}/subida/t"}})
lab = sitk.ReadImage(str(UPLOADS / "labels.nii.gz")); vol = comprobar_volumen(out, lab)
assert np.prod(vol.GetSize()) < 0.75 * np.prod(lab.GetSize()), vol.GetSize()
assert sum(o > 0 for o in out["viewer"]["volume"]["offset_ijk"]) == 2             # en z los dientes llenan el fantoma
assert np.allclose(out["viewer"]["curves"]["md"]["points_mm"], curvas["md"][1])       # la curva no depende del recorte
H.VIEWER_MARGIN_XY, H.VIEWER_MARGIN_Z = 25.0, 20.0
print(f"OK  recorte del volumen del visor  {lab.GetSize()} -> {vol.GetSize()}, desplazamiento {out['viewer']['volume']['offset_ijk']}")
for arch, cs in curvas.items():                                                    # misma curva física con cualquier orientación
    for otra in cs[1:]:
        assert cKDTree(cs[0]).query(otra)[0].max() < 0.8, f"{arch}: la curva depende de la matriz de dirección"
print("OK  la curva no depende de cómo venga orientado el archivo")
import shutil as _sh; _sh.copy(UPLOADS / "pano_md.png", "/tmp/pano_md_fantoma.png")

# ---- errores: mensaje limpio y sin dejar basura ---------------------------------------------------
e = H.handler({"id": "e1", "input": {}});                                         assert "cbct_url" in e["error"], e
e = H.handler({"id": "e2", "input": {"cbct_url": f"{URL}/x/no_existe.zip"}});    assert "404" in e["error"], e
(SERVE / "roto.zip").write_bytes(b"PK\x03\x04 esto no es un zip")
e = H.handler({"id": "e3", "input": {"cbct_url": f"{URL}/x/roto.zip"}});         assert "zip" in e["error"], e
(SERVE / "texto.bin").write_bytes(b"hola" * 200)
e = H.handler({"id": "e4", "input": {"cbct_url": f"{URL}/x/texto.bin"}});        assert "formato" in e["error"], e
e = H.handler({"id": "e5", "input": {"cbct_url": f"{URL}/x/cbct.nii.gz", "result_url": f"{URL}/caducado/t"}})
assert "no aceptó" in e["error"] and "refresh_worker" not in e, e
sitk.WriteImage(sitk.GetImageFromArray(np.zeros((40, 40, 40), np.int16)), str(SERVE / "vacio.nii.gz"))
e = H.handler({"id": "e6", "input": {"cbct_url": f"{URL}/x/vacio.nii.gz"}});     assert "dientes" in e["error"], e
print("OK  errores de entrada/subida con mensaje limpio")

o = H.handler({"id": "sin_destino", "input": {"cbct_url": f"{URL}/x/cbct.nii.gz"}})
assert not o["uploaded"] and "labels_nii_gz_base64" in o, o.keys()
print("OK  sin result_url: métricas + mapa de etiquetas en base64")

def boom(*a, **k): raise RuntimeError("CUDA error: fallo simulado")
FALSO.predict_from_files_sequential = boom
e = H.handler({"id": "e7", "input": {"cbct_url": f"{URL}/x/cbct.nii.gz"}})
assert e.get("refresh_worker") is True and "segment" in e["error"], e
print("OK  fallo en la inferencia -> refresh_worker")
assert not any(Path(os.environ["DENTALSEG_WORK_ROOT"]).iterdir()), "quedaron temporales"
print("OK  sin temporales tras cada job\n\nTODO CORRECTO")
