let
  pkgs = import <nixpkgs> {};
in
pkgs.singularity-tools.buildImage {
  name = "simpleBash"; 
  diskSize = 1024*25;
  contents = with pkgs; [ bash ]; 
  runScript = "/usr/bin/env bash";
}