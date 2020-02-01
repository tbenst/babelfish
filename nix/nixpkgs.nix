let
  # now tracking https://github.com/tbenst/nixpkgs/tree/nix-data
  # when updating, replace all in project
  # nixpkgsSHA = "30510834d3232e324d96426d634b1c914fd5908f"; # failed
  nixpkgsSHA = "ce0f8501d084d3d8c4231f26afe0eb4a7a7e7b9e";
  pkgs = import (fetchTarball
    "https://github.com/tbenst/nixpkgs/archive/${nixpkgsSHA}.tar.gz") {
      system = builtins.currentSystem;
      overlays = import ./overlays.nix;
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