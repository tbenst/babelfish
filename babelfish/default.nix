# this file defines the Nix package
{ lib, buildPythonPackage
, altair
, babelfish-models
, bokeh
, boto3
, cython
, click
, dill
, ffmpeg
, fire
, future
, hdf5plugin
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
, pyarrow
, pytorch-lightning
, pynwb
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

  nativeBuildInputs = [ ffmpeg ];

  propagatedBuildInputs = [
    altair
    babelfish-models
    bokeh
    boto3
    cython
    click
    dill
    fire
    future
    h5py
    hdf5plugin
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
    pyarrow
    pynwb
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
