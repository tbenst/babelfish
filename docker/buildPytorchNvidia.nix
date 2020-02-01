let
  # pkgs = import ./nixpkgs.nix;
  pkgs = import <nixpkgs> {};

  # generate with:
  # > nix-prefetch-docker nvcr.io/nvidia/pytorch 20.01-py3
  cudaImage = pkgs.dockerTools.pullImage {
    imageName = "nvcr.io/nvidia/pytorch";
    imageDigest = "sha256:ab17c2521e4164e331976ca7d138614f8d20133ebfa4118032c48d897d96e052";
    sha256 = "1fm1mm7pw0zwi9ai8yxb5151lxzkrslyv8206vifvhp8866z6gsq";
    finalImageName = "nvcr.io/nvidia/pytorch";
    finalImageTag = "20.01-py3";
  };

in
pkgs.dockerTools.buildImage {
  name = "pytorch"; 
  fromImage = cudaImage;
  created = "now";
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