{ lib
, buildPythonPackage
, fetchPypi
, h5py
, python
}:

buildPythonPackage rec {
  pname = "hdf5plugin";
  version = "2.1.2";

  src = fetchPypi {
    inherit pname version;
    sha256 = "11da0aa6114ce20d98574f4c620dc1424b43e51ea5a219e2d203144f33c05193";
  };

  propagatedBuildInputs = [
    h5py
  ];
  
  checkPhase = ''
    ${python.interpreter} test/test.py
  '';

  meta = with lib; {
    description = "HDF5 Plugins for windows,MacOS and linux";
    homepage = "https://github.com/silx-kit/hdf5plugin";
    license = licenses.https://github.com/silx-kit/hdf5plugin/blob/master/LICENSE;
    maintainers = [ maintainers.tbenst ];
  };
}