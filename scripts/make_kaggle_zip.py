"""Package only the Kaggle source files, with paths rooted inside the ZIP."""
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT=Path(__file__).resolve().parents[1]
FILES=(
    ROOT/"src",
    ROOT/"notebooks"/"train_gpu.py",
    ROOT/"notebooks"/"run_diamond.py",
    ROOT/"requirements-train.txt",
)
OUTPUT=ROOT/"protfunc-source.zip"

def main():
    missing=[str(p) for p in FILES if not p.exists()]
    if missing: raise FileNotFoundError("Required Kaggle source file(s) missing: "+", ".join(missing))
    with ZipFile(OUTPUT,"w",compression=ZIP_DEFLATED) as zf:
        for entry in FILES:
            paths=sorted(entry.rglob("*") if entry.is_dir() else [entry])
            for path in paths:
                if path.is_dir() or "__pycache__" in path.parts or ".venv" in path.parts: continue
                zf.write(path,path.relative_to(ROOT).as_posix())
    print(f"Created {OUTPUT}")

if __name__=="__main__": main()
