let
  nixpkgsSHA = "07be0186f298e5b16897a168eae6ab01a5540fc4";
  pkgs = import (fetchTarball
    https://github.com/tbenst/nixpkgs/archive/${nixpkgsSHA}.tar.gz) {
      system = builtins.currentSystem;
    };

  jobs = rec {
    babelfish = pkgs.callPackage ./default.nix { };
  };
in
  jobs