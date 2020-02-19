let
  pkgs = import ./nixpkgs.nix;
  secrets = import ../nix/secrets.nix;
  poetryEnv = pkgs.poetry2nix.mkPoetryEnv {
    poetrylock = ./poetry.lock;
    python = pkgs.python3;
  };
  python-env = pkgs.python3.buildEnv.override {
    extraLibs = with pkgs.python3Packages; [
      cython
    ];
    ignoreCollisions = true;
  };

in
with pkgs;
mkShell {
    buildInputs = [
      poetryEnv
      # python-env
      # fd
      # ffmpeg
      # fzf
      # strace
      # zstd
    ];
    # nativeBuildInputs = [ qt4.wrapQtAppsHook ];
}
