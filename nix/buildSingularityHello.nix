{pkgs ? import <nixpkgs> {} }:

with pkgs;

singularity-tools.buildImage {
  name = "test";
  contents = [ hello ];
  runScript = "${hello}/bin/hello";
}