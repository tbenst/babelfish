{ pkgs ? import <nixpkgs> {}
, python37Packages ? pkgs.python37Packages
}:
with python37Packages;

buildPythonPackage rec {
  name = "mypackage";
  src = /home/tyler/code/babelfish;
  doCheck = true;
  checkPhase = ''
    python -m unittest discover
  '';
  propagatedBuildInputs = [
    dill
    h5py
    joblib
    matplotlib
    numpy
    opencv3
    pandas
    pytest
    pytorch
    scikitlearn
    seaborn
    torchvision
    tqdm
    tifffile
 ];
}
