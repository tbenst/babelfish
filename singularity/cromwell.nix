let
  pkgs = import ../nix/nixpkgs.nix;
  cromwell_version = "48";
  cromwell = builtins.fetchurl {
    url = "https://github.com/broadinstitute/cromwell/releases/download/${cromwell_version}/cromwell-${cromwell_version}.jar";
    sha256 = "0p0mfa0z41axcj84b7r0m7srrxy6sw5d4ci0lv8mn908y8s4yjy1";
  };
in
pkgs.singularity-tools.buildImage {
  name = "babelfish"; 
  diskSize = 1024*100;
  contents = with pkgs; [
    cromwell
    fd
    openjdk8
  ]; 
}