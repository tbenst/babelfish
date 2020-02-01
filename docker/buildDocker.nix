let
  pkgs = import ./nixpkgs.nix;
  python-env = pkgs.python3.buildEnv.override {
    extraLibs = with pkgs.python3Packages; [
      babelfish
      babelfish-models
      numba
    ];
    ignoreCollisions = true;
  };
  busybox = pkgs.dockerTools.pullImage {
    imageName = "busybox";
    imageDigest = "sha256:6915be4043561d64e0ab0f8f098dc2ac48e077fe23f488ac24b665166898115a";
    sha256 = "0wnflsl3pyrdf8cgmyf4aqk7pj9ca8yhns0lym6xdahx8i7rznc2";
    finalImageName = "busybox";
    finalImageTag = "latest";
  };
in
pkgs.dockerTools.buildImage {
  # /nix/store/4alzsklrvwi85l6bz2ah4lnbkvw39488-docker-image-babelfish.tar.gz
  # warning: takes ~20 minutes...
  # singularity build babelfish.sif docker-archive:/nix/store/4alzsklrvwi85l6bz2ah4lnbkvw39488-docker-image-babelfish.tar.gz
  name = "babelfish";
  fromImage = busybox;
  diskSize = 1024*25;
  contents = with pkgs; [
    python-env
    babelfish-scripts
  ]; 

  runAsRoot = '' 
    #!${pkgs.runtimeShell}
    mkdir -p /data
  '';
  config = { 

    Cmd = [ "/bin/sh" ];
    WorkingDir = "/data";
    Volumes = {
      "/data" = {};
    };
  };
}