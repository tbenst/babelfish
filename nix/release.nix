let
  pkgs = import ./nixpkgs;
in {
  babelfish = pkgs.python3Packages.babelfish;
}
