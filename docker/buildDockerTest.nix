{pkgs ? import <nixpkgs> {} }:

with pkgs;

dockerTools.buildImage {
  name = "test";
  diskSize = 1024*25;
  contents = [ hello openjdk8 ];
  config = {
    Cmd = [ "${hello}/bin/hello" ];
  };
}
