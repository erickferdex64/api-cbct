"""Se ejecuta durante el `docker build`: deja los pesos de DentalSegmentator DENTRO de la imagen.
Así el arranque en frío no depende de Zenodo (en el cuaderno la descarga tardó 4,5 min).

Pesos: Zenodo 10.5281/zenodo.10829675 · licencia CC BY 4.0.
"""
import hashlib
import os
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

URL = os.environ.get(
    "DENTALSEG_WEIGHTS_URL",
    "https://zenodo.org/records/10829675/files/Dataset112_DentalSegmentator_v100.zip?download=1")
MD5 = os.environ.get("DENTALSEG_WEIGHTS_MD5", "b71cd5230168d28a4f71b078265b76be")
DEST = Path(os.environ.get("DENTALSEG_MODEL_ROOT", "/models/nnunet/results"))
ZIP = Path("/tmp/dentalseg_weights.zip")


def descargar() -> None:
    for intento in range(1, 6):
        try:
            h, n = hashlib.md5(), 0
            req = urllib.request.Request(URL, headers={"User-Agent": "dentalseg-worker-build"})
            with urllib.request.urlopen(req, timeout=120) as r, ZIP.open("wb") as fh:
                while chunk := r.read(1 << 20):
                    fh.write(chunk); h.update(chunk); n += len(chunk)
            if h.hexdigest() != MD5:
                raise RuntimeError(f"md5 {h.hexdigest()} != {MD5} ({n / 1e6:.0f} MB)")
            print(f"pesos descargados: {n / 1e6:.0f} MB, md5 correcto", flush=True)
            return
        except Exception as exc:        # noqa: BLE001
            print(f"intento {intento}/5 fallido: {exc}", flush=True)
            time.sleep(15 * intento)
    sys.exit("no se pudieron descargar los pesos")


descargar()
DEST.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(ZIP) as z:
    z.extractall(DEST)
ZIP.unlink()
shutil.rmtree(DEST / "__MACOSX", ignore_errors=True)

modelos = [p for p in DEST.rglob("nnUNetTrainer__nnUNetPlans__3d_fullres") if (p / "plans.json").is_file()]
if not modelos or not list(modelos[0].glob("fold_*/checkpoint_*.pth")):
    sys.exit(f"el zip no trae el modelo esperado bajo {DEST}")
print("modelo:", modelos[0], "| folds:", sorted(p.name for p in modelos[0].glob("fold_*")), flush=True)
