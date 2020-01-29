let
  pkgs = import ./nixpkgs.nix;
  python-env = pkgs.python3.buildEnv.override {
    extraLibs = with pkgs.python3Packages; [
      apache-airflow
      babelfish
      babelfish-models
      jupyter
      mypy
      pyarrow
      pylint
      stytra
      seqnmf
    ];
    ignoreCollisions = true;
  };
in
pkgs.singularity-tools.buildImage {
  name = "babelfish"; 
  diskSize = 1024*25;
  contents = with pkgs; [ bash ]; 
  # runAsRoot = '' 
  #   #!${pkgs.runtimeShell}
  #   mkdir -p /data
  # '';

  # runScript = "/usr/bin/env bash";
}