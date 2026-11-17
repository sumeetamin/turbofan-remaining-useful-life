"""Fetch NASA's official C-MAPSS archive and extract the FD001 training file."""
from io import BytesIO
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

URL = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
ARCHIVE = Path("data/cmapss.zip")
OUT = Path("data/train_FD001.txt")


def main():
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE.exists():
        urlretrieve(URL, ARCHIVE)
    with ZipFile(ARCHIVE) as outer:
        nested_name = next(n for n in outer.namelist() if n.endswith("CMAPSSData.zip"))
        with ZipFile(BytesIO(outer.read(nested_name))) as inner:
            member = next(n for n in inner.namelist() if n.endswith("train_FD001.txt"))
            OUT.write_bytes(inner.read(member))
    print(f"Extracted {OUT}; NASA describes C-MAPSS as engine degradation simulation data.")


if __name__ == "__main__":
    main()

