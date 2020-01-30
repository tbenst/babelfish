let
  pkgs = import ./nixpkgs.nix;
  python-env = pkgs.python3.buildEnv.override {
    extraLibs = with pkgs.python3Packages; [
      babelfish
      babelfish-models
      numba
    ];
    ignoreCollisions = true;
  };
in
pkgs.singularity-tools.buildImage {
  name = "babelfish"; 
  diskSize = 1024*100;
  contents = with pkgs; [
    python-env
    babelfish-scripts
  ]; 
}