{ pkgs ? import <nixpkgs> {}
, python37Packages ? pkgs.python37Packages
}:
with python37Packages;

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
