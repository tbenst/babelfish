# WIP dev shell
let
  overlays = import ./overlays.nix;
  pkgs = import <nixpkgs> { inherit overlays;};
  secrets = import ./secrets.nix;
in
with pkgs.python37Packages;

(pkgs.callPackage ./default.nix).overrideAttrs (old: {
  inherit (secrets) MLFLOW_TRACKING_URI MLFLOW_TRACKING_USERNAME MLFLOW_TRACKING_PASSWORD;
  propagatedBuildInputs = old.propagatedBuildInputs ++ [
    mypy
    pylint

  ]
}) {}
