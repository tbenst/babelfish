let
  pkgs = import ./nixpkgs.nix;
  python-env = pkgs.python3.buildEnv.override {
    extraLibs = with pkgs.python3Packages; [
      pytorch
    ];
  };

  # generate with:
  # > nix-prefetch-docker 10.2-cudnn7-runtime-ubuntu18.04
  cudaImage = pkgs.dockerTools.pullImage {
    imageName = "nvidia/cuda";
    imageDigest = "sha256:a3c7a33d2cf7daddea491862499e47828808debf07f26cb18893a9410e3d72a0";
    sha256 = "1d2kbcbbj6a596b4kj6jg8yaqfyr1dcshz1dsq7kp09nggq4cc2n";
    finalImageName = "nvidia/cuda";
    finalImageTag = "10.2-cudnn7-runtime-ubuntu18.04";
  };

in
pkgs.dockerTools.buildImage {
  name = "pytorch"; 
  fromImage = cudaImage;
  contents = python-env;
  diskSize = 1024*20;
  runAsRoot = '' 
    #!${pkgs.runtimeShell}
    mkdir -p /data
  '';
  config = { 

    Cmd = [ "/bin/bash" ];
    WorkingDir = "/data";
    Volumes = {
      "/data" = {};
    };
  };
}