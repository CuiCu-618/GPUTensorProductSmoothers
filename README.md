# GPUTensorProductSmoothers
A GPU implementation of vertex-patch smoothers for higher order finite element methods in two and three dimensions.

## Getting started

```bash
git clone git@github.com:CuiCu-618/GPUTensorProductSmoothers.git
cd GPUTensorProductSmoothers/
mkdir build
cd build/
cp ../scripts/gputps-setup.sh .
bash ./gputps-setup.sh
make release
make heat
./apps/heat
```
