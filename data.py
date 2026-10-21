"""Download the immutable UCI benchmark and validate its basic contract."""

import hashlib
import io
import json
from pathlib import Path
import zipfile
import pandas as pd
import requests

URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"


def load_data(source=None):
    folder = Path(".data")
    folder.mkdir(exist_ok=True)
    archive = Path(source) if source else folder / "credit.zip"
    if not archive.exists():
        response = requests.get(URL, timeout=120)
        response.raise_for_status()
        archive.write_bytes(response.content)
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    with zipfile.ZipFile(archive) as z:
        name = next(n for n in z.namelist() if n.endswith(".xls"))
        frame = pd.read_excel(io.BytesIO(z.read(name)), header=1)
    frame = frame.rename(columns={frame.columns[-1]: "target"})
    if (
        len(frame) != 30000
        or not frame.ID.is_unique
        or not frame.target.isin([0, 1]).all()
    ):
        raise ValueError("Unexpected UCI data contract")
    manifest = dict(
        source_url=URL,
        sha256=checksum,
        rows=len(frame),
        unit="account",
        history="April–September 2005",
        label="October 2005 default",
        limitations="Historical Taiwan snapshot; no out-of-time account cohort available.",
    )
    Path("manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return frame
