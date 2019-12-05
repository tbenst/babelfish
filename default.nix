let
  overlays = [
    (self: super: {
      pythonOverrides = python-self: python-super: {
        opencv3 = python-super.opencv3.override {
          enableCuda = true;
          enableFfmpeg = true;
        };
        pytorch = python-super.pytorch.override {
          cudaSupport = true;
          mklSupport = true;
        };
        /* numpy = python-super.numpy.override { blas = super.mkl; }; */
      };
      python = super.python.override {packageOverrides = self.pythonOverrides;};
    }
  )];
  pkgs = import <nixpkgs> { inherit overlays;};
  mkDerivation = import ./autotools.nix pkgs;
in
with pkgs.python37Packages;
buildPythonPackage rec {
  name = "mypackage";
  src = ./babelfish;
  doCheck = false;
  checkPhase = ''
    python -m unittest discover
  '';
  propagatedBuildInputs = [
    bokeh
    cython
    dill
    future
    h5py
    joblib
    matplotlib
    moviepy
    nose
    numpy
    opencv3
    pandas
    pims
    pytest
    pytorch
    scikitlearn
    scikitimage
    scipy
    seaborn
    torchvision
    tqdm
    tifffile
 ];
}
