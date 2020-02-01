let
  pkgs = import ../nix/nixpkgs.nix;
  debian = pkgs.dockerTools.pullImage {
    imageName = "debian";
    imageDigest = "sha256:bc426d777c448212e30bef92f695492b7ef2f10263d03491f03a5e4a9de6bcc2";
    sha256 = "0zacb7r20a0z66wky5ap51syxq60zm23jyam2ni7q9vza7h6cks3";
    finalImageName = "debian";
    finalImageTag = "stable-slim";
  };
  
  cromwell_version = "48";
  cromwell = builtins.fetchurl {
    url = "https://github.com/broadinstitute/cromwell/releases/download/${cromwell_version}/cromwell-${cromwell_version}.jar";
    sha256 = "0p0mfa0z41axcj84b7r0m7srrxy6sw5d4ci0lv8mn908y8s4yjy1";
  };
  kickoff = pkgs.writeScriptBin "kickoff" ''
    #!${pkgs.stdenv.shell}
    ${pkgs.openjdk8}/lib/openjdk/bin/java -jar /code/cromwell.jar
  '';
in
with pkgs;
dockerTools.buildImage {
  name = "cromwell";
  tag = "0.0.2";
  diskSize = 1024*20;
  fromImage = debian;
  contents =  [
    fd
    openjdk8
  ];
  runAsRoot = '' 
    #!${runtimeShell}
    mkdir -p /code
    mkdir -p /data
    cp ${cromwell} /code/cromwell.jar
  '';

  config = {
    # Cmd = [ "/bin/ash" ]; # busybox
    Cmd = [ "${kickoff}/bin/kickoff" ];
    WorkingDir = "/data";
  };
}