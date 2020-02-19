let
  # now tracking https://github.com/tbenst/nixpkgs/tree/nix-data
  # when updating, replace all in project
  # nixpkgsSHA = "30510834d3232e324d96426d634b1c914fd5908f"; # failed
  nixpkgsSHA = "a21c2fa3ea2b88e698db6fc151d9c7259ae14d96";
  pkgs = import (fetchTarball
    "https://github.com/tbenst/nixpkgs/archive/${nixpkgsSHA}.tar.gz") {
      system = builtins.currentSystem;
      overlays = import ../nix/overlays.nix;
      config = with pkgs.stdenv; {
        whitelistedLicenses = with lib.licenses; [
          unfreeRedistributable
          issl
         ];
        allowUnfreePredicate = pkg: builtins.elem (lib.getName pkg) [
          "cudnn_cudatoolkit"
          "cudatoolkit"
        ];
      };
    };

in pkgs