{pkgs ? import ../nix/nixpkgs.nix }:

with pkgs;

singularity-tools.buildImage {
  name = "test";
  diskSize = 1024*25;
  contents = [ hello openjdk8 ];
  runScript = "${hello}/bin/hello";
}