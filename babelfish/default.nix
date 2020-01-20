# this file defines the Nix package
{ lib, buildPythonPackage
, babelfish-models
, bokeh
, boto3
, cython
, click
, dill
# , ffmpeg
, future
, h5py
, joblib
, matplotlib
, mlflow
, moviepy
, nose
, numba
, numpy
, opencv3
, pandas
, pims
, pytest
, pytorch
, pytorch-lightning
, requests
, scipy
, seaborn
, tables
, torchvision
, tqdm
, tifffile
}:

buildPythonPackage rec {
  pname = "babelfish";
  version = "0.1.0";
  src = ./.;
  doCheck = false;
  checkPhase = ''
    python -m unittest discover
  '';

  # nativeBuildInputs = [ ffmpeg ];

  propagatedBuildInputs = [
    babelfish-models
    bokeh
    boto3
    cython
    click
    dill
    future
    h5py
    joblib
    matplotlib
    mlflow
    moviepy
    nose
    numpy
    opencv3
    pandas
    pims
    pytest
    pytorch
    pytorch-lightning
    requests
    scipy
    seaborn
    tables
    torchvision
    tqdm
    tifffile
 ];

   meta = with lib; {
    description = "Zebrafish analysis";
    homepage = "https://github.com/tbenst/babelfish";
    maintainers = [ maintainers.tbenst ];
    license = licenses.gpl3;
  };
}