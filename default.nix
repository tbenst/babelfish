{ pkgs ? import <nixpkgs> {}
, pythonPackages ? pkgs.python37Packages
}:
with pythonPackages;

buildPythonPackage rec {
  name = "mypackage";
  src = /home/tyler/code/babel_fish;
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
