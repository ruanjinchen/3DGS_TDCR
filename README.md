Read https://github.com/graphdeco-inria/gaussian-splatting for information and installation instructions.

Please install submodules/diff-gaussian-rasterization from https://github.com/skhu101/GauHuman.

The dataset can be downloaded from https://pan.baidu.com/s/1hAnDpUJtX0hZL6jsVMOMYQ?pwd=p7eh pw: p7eh (For China) or https://drive.google.com/file/d/1arR7oxNg6O2Buzq38YOQxa1WLoulmJ0q/view?usp=sharing (For other countries).

Unzip the dataset and put it in the root path of the project.

## Compile Commands
### Prepare for H100 and A100
```bash
export TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0"
export CUDA_HOME="$CONDA_PREFIX"
export CPATH="$CONDA_PREFIX/include:$CONDA_PREFIX/targets/x86_64-linux/include:${CPATH}"
export LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib:${LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDAHOSTCXX=/usr/bin/g++
```

### PyTorch3D
#### Install
```bash
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
python -m pip install . --no-build-isolation -v
```
#### Check
```bash
cd ..

python - <<'PY'
import torch
import pytorch3d
from pytorch3d.ops import knn_points
x = torch.randn(2, 100, 3, device="cuda")
y = torch.randn(2, 200, 3, device="cuda")
d, idx, nn = knn_points(x, y, K=3)
print("pytorch3d ok:", getattr(pytorch3d, "__version__", "no_version"))
print("knn_points ok:", d.shape, idx.shape)
PY
```


### Diff-gaussian-rasterization
#### Compile
```bash
cd install submodules/diff-gaussian-rasterization
python -m pip install . --no-build-isolation -v
```
#### Check
```bash
cd ../..

python - <<'PY'
import os, glob
import diff_gaussian_rasterization as dgr
print("diff_gaussian_rasterization imported from:", dgr.__file__)
pkgdir = os.path.dirname(dgr.__file__)
print("package dir:", pkgdir)
print("shared libs:", glob.glob(os.path.join(pkgdir, "*.so")))
PY
```

### simple_knn
#### Compile
```bash
cd install submodules/simple-knn
python -m pip install . --no-build-isolation -v
```
#### Check
```bash
cd ../..

python - <<'PY'
import torch
from simple_knn._C import distCUDA2
x = torch.randn(1024, 3, device="cuda", dtype=torch.float32)
d2 = distCUDA2(x)
print("distCUDA2 OK:", d2.shape, d2.dtype, d2.device, "mean=", float(d2.mean()))
PY

```
### Uninstall and Clean
#### diff-gaussian-rasterization
```bash
cd submodules/diff-gaussian-rasterization
python -m pip uninstall -y diff_gaussian_rasterization diff_gaussian_rasterization || true
rm -rf build dist *.egg-info
```
#### simple-knn 
```bash
cd submodules/simple-knn
python -m pip uninstall -y simple_knn simple-knn || true
rm -rf build dist *.egg-info
```