let
  overlays = [
    (self: super: {
      pythonOverrides = python-self: python-super: {
        opencv3 = python-super.opencv3.override {
          enableCuda = true;
          enableFfmpeg = true;
        };
#        pytorch = python-super.pytorch.override {
#          cudaSupport = true;
          # mklSupport = true;
#        };
        /* numpy = python-super.numpy.override { blas = super.mkl; }; */
      };
      python37 = super.python37.override {packageOverrides = self.pythonOverrides;};
#      cudatoolkit = super.cudatoolkit_10_1;
#      cudnn_cudatoolkit = super.cudnn_cudatoolkit_10_0;
      # TODO switch to ffmpeg on left side for GPU (moviepy uses via imageio)
      ffmpeg = super.ffmpeg-full.override {
        nonfreeLicensing = true;
        nvenc = true; # nvidia support
      };
      ffmpeg-full = super.ffmpeg-full.override {
        nonfreeLicensing = true;
        nvenc = true; # nvidia support
      };
    }
  )];
  pkgs = import <nixpkgs> { inherit overlays;};
  secrets = import ./secrets.nix;
in
with pkgs.python37Packages;
buildPythonPackage rec {
  name = "babelfish";
  src = ./.;
  doCheck = false;
  checkPhase = ''
    python -m unittest discover
  '';

  inherit (secrets) MLFLOW_TRACKING_URI MLFLOW_TRACKING_USERNAME MLFLOW_TRACKING_PASSWORD;

  propagatedBuildInputs = [
    bokeh
    cython
    dill
    future
    h5py
    joblib
    matplotlib
    mlflow
    moviepy
    mypy # dev env
    nose
    numpy
    opencv3
    pandas
    pims
    pylint # dev env
    pytest
    pytorch
    pytorch-lightning
    requests
    scikitlearn
    scikitimage
    scipy
    seaborn
    tables
    torchvision
    tqdm
    tifffile
 ];
}
