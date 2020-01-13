let
  pkgs = import ./nix/nixpkgs.nix;
  secrets = import ./nix/secrets.nix;
  python-env = pkgs.python3.withPackages(ps: with ps; [
    babelfish
    jupyter_core
    mypy
    pylint
  ]);

in
# TODO how to hotload? don't install babelfish, just propagatedBuildInputs
pkgs.mkShell {
    inherit (secrets) MLFLOW_TRACKING_URI MLFLOW_TRACKING_USERNAME MLFLOW_TRACKING_PASSWORD;
    buildInputs = [ python-env ];
}
