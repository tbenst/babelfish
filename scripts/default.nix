{ stdenv, lib, python3, python3Packages
, fd
, ffmpeg
, fzf
, strace
, zstd
, bash
}:
let 
  python-env = python3.buildEnv.override {
    extraLibs = with python3Packages; [
      babelfish
      babelfish-models
      numba
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
    strace
    zstd
    bash
  ];
  src = ./.;
  # unpackPhase = "true";
  installPhase = ''
    mkdir -p $out/bin
    find $src -executable -type f -exec cp -t $out/bin {} +
  '';
  # doCheck = false;
}