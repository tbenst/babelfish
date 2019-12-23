# this file is used for Continuous Integration
# { supportedSystems ? [ "x86_64-linux" ]}:
let
  nixpkgsSHA = "07be0186f298e5b16897a168eae6ab01a5540fc4";
  pkgs = import (fetchTarball
    "https://github.com/tbenst/nixpkgs/archive/${nixpkgsSHA}.tar.gz") {
      system = builtins.currentSystem;
      overlays = import ./overlays.nix;
    };
  jobs = rec {
    hello = pkgs.hello;
    babelfish = pkgs.python3Package.babelfish;
  };
in
  jobs