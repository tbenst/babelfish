{ stdenv, lib, python3, python3Packages
, fd
, ffmpeg
, fzf
, less
, strace
, zstd
, bash
}:
let 
  python-env = python3.buildEnv.override {
    extraLibs = with python3Packages; [
      av
      babelfish
      babelfish-models
      fire
      numba
      pynwb
    ];
    ignoreCollisions = true;
  };
in
stdenv.mkDerivation {
  name = "babelfish-scripts";
  buildInputs = [
    python-env
    fd
    ffmpeg
    fzf
    less
    strace
    zstd
    bash
  ];
  src = ./.;
  # unpackPhase = "true";
  installPhase = ''
    mkdir -p $out/bin
    mkdir -p $out/lib
    touch $out/lib/__init__.py
    find $src -executable -type f -exec cp -t $out/bin {} +
    find $src -name "*.py" -exec cp -t $out/lib {} +
  '';
  # doCheck = false;
}