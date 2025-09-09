# pointcloud-segmentation-viewer

**pointcloud-segmentation-viewer** is a modular, extensible toolkit for semantic segmentation experiments on 3D point clouds.
It provides a simple CLI, training scaffolding (DGCNN-style), and an Open3D visualizer so you can iterate locally and demo results quickly.

> Built to start local and grow toward a web viewer (Three.js + FastAPI).

## Features

* **Train/evaluate** DGCNN-style models on point clouds (ShapeNetPart config included).
* **Visualize** point clouds locally with Open3D.
* **Native C++ extension** (`knn`, pybind11) to showcase Python↔C++ integration—ready to extend with fast kernels.
* Clean CLI powered by **Typer** (`pcseg`).

---

## Quickstart

```bash
# 0) (Recommended) Use a fresh environment named pcseg
python -m venv .venv && source .venv/bin/activate   # or conda create -n pcseg python=3.10

# 1) Install in editable mode (builds the C++ extension too)
python -m pip install -U pip wheel
python -m pip install -e .

# 2) Sanity-check the native extension
python - <<'PY'
import knn
print("knn.version():", knn.version())  # -> "stub"
PY

# 3) Discover the CLI
pcseg --help
```

You should see the `train` and `view` commands listed.

---

## Datasets & Config

Edit the config to point to your local data:

```yaml
# python/pcseg/cfg/default.yaml
dataset:
  name: shapenetpart
  root: /ABS/PATH/TO/shapenetpart_hdf5_2048   # <-- set this
  num_points: 2048
  num_part_classes: 50

# Model/optimizer/scheduler settings are in the same file and can be tuned.
```

---

## Usage

### 1) Train

```bash
# via CLI (preferred)
pcseg train --config python/pcseg/cfg/default.yaml

# fallback if you prefer running the module/script directly
python -m pcseg.engine.train --config python/pcseg/cfg/default.yaml
```

> Tip: run `pcseg train --help` to see the flags exposed by your Typer command.

### 2) Visualize (Open3D)

```bash
# show help
python -m pcseg.visualize_open3d -h

# example (replace with your PLY/PCD path)
python -m pcseg.visualize_open3d /path/to/example.ply
```

---

## Project Layout

```
.
├── cpp/
│   ├── CMakeLists.txt            # pybind11 build: knn module
│   └── knn.cpp                   # stub: knn.version() -> "stub"
├── python/
│   ├── pcseg/
│   │   ├── cfg/
│   │   │   └── default.yaml      # training config (edit dataset.root)
│   │   ├── data/
│   │   │   └── shapenet_part.py
│   │   ├── engine/
│   │   │   ├── train.py
│   │   │   └── utils.py
│   │   ├── models/
│   │   │   └── dgcnn.py
│   │   ├── __init__.py           # package marker
│   │   ├── cli.py                # Typer CLI: `pcseg`
│   │   └── visualize_open3d.py   # local viewer
│   ├── tests/
│   └── __init__.py
├── 1801.07829v2.pdf              # DGCNN paper
├── CMakeLists.txt                # adds cpp/ subdir
├── pyproject.toml                # packaging, build backend, console script
├── README.md
└── repo_tree.py                  # helper to print ASCII tree
```

---

## Development Notes

### Build system

* **Build backend:** scikit-build-core + CMake + pybind11.
* **Console script:** `pcseg = "pcseg.cli:app"`.

### C++ extension (manual build, optional)

If you want to build and import without installing the wheel:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target knn -j
PYTHONPATH="$(pwd)/build/cpp:$PYTHONPATH" python -c "import knn; print(knn.version())"
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'pcseg'`**

* Make sure `python/pcseg/__init__.py` exists (it can be empty).
* Check `pyproject.toml` includes:

  ```toml
  [tool.setuptools]
  package-dir = {"" = "python"}

  [tool.setuptools.packages.find]
  where = ["python"]
  include = ["pcseg*"]
  ```
* Reinstall: `pip install -v -e .`

**OpenMP error on macOS (Torch/Open3D):**
`OMP: Error #15: Initializing libomp.dylib...`

* Quick workaround (dev only): `export KMP_DUPLICATE_LIB_OK=TRUE`
* Best practice: **lazy import** heavy libs inside CLI subcommands (avoid importing torch/open3d at module import).
* Environment hygiene: keep OpenMP stack consistent (e.g., conda-forge).

**`ModuleNotFoundError: No module named 'knn'`**

* Ensure the extension built: `pip install -e . -v` or the manual CMake build above.
* If you built manually, add `build/cpp` to `PYTHONPATH` for the session.

---

## Roadmap (ideas)

* `pcseg demo` command: tiny synthetic point cloud → mock forward → Open3D color overlay.
* Add metrics logging (TensorBoard/Rich) and checkpoints.
* Extend `knn` with a real k-NN kernel and integrate into DGCNN graph construction.

---

## References

* Yue Wang, Yongbin Sun, Ziwei Liu, Sanjay E. Sarma, Michael M. Bronstein, and Justin M. Solomon. “Dynamic Graph CNN for Learning on Point Clouds.” *ACM Transactions on Graphics* 38, no. 5 (2019). [arXiv:1801.07829](https://arxiv.org/abs/1801.07829)

---
