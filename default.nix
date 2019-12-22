{ lib, buildPythonPackage
, bokeh
, cython
, dill
, future
, h5py
, joblib
, matplotlib
, mlflow
, moviepy
, nose
, numpy
, opencv3
, pandas
, pims
, pytest
, pytorch
, pytorch-lightning
, requests
, scikitlearn
, scikitimage
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
    nose
    numpy
    opencv3
    pandas
    pims
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

   meta = with lib; {
    description = "Zebrafish analysis";
    homepage = "https://github.com/tbenst/babelfish";
    maintainers = [ maintainers.tbenst ];
    license = licenses.gpl3;
  };
}
