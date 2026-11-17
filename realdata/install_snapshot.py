"""Verify and install only the two bundled, public-derived NOAA arrays."""
from hashlib import sha256
from pathlib import Path
from shutil import copyfile

HERE = Path(__file__).resolve().parent
SNAPSHOTS = {
    "2021.npz": "2a1cdee94efc93b6a6732460d7ca3c85bdce5a11640f323b457a137aaad259fa",
    "2022.npz": "e94f988bbc2b22ca84b5832e50b1f28931ec334a614c1f271762a9688df530f9",
}

def main():
    for name, digest in SNAPSHOTS.items():
        if sha256((HERE/"snapshot"/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"bad bundled data digest: {name}")
    out=HERE/"dev"
    out.mkdir(exist_ok=True)
    for name in SNAPSHOTS:
        copyfile(HERE/"snapshot"/name, out/name)
    print("Verified bundled public-derived NOAA arrays (2021 and 2022 only)")

if __name__ == "__main__":
    main()
