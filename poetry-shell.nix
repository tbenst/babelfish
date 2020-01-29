let
  pkgs = import ./nix/nixpkgs.nix;
  secrets = import ./nix/secrets.nix;
  poetryEnv = pkgs.poetry2nix.mkPoetryEnv {
    poetrylock = ./poetry.lock;
    python = pkgs.python3;
  };
in
with pkgs;
mkShell {
    inherit (secrets) MLFLOW_TRACKING_URI MLFLOW_TRACKING_USERNAME MLFLOW_TRACKING_PASSWORD;
    AIRFLOW_HOME=builtins.toString ./pipelines;
    # QTCOMPOSE="${xorg.libX11}/share/X11/locale";
    QT_PLUGIN_PATH = "${qt5.qtbase}/${qt5.qtbase.qtPluginPrefix}";
    # openGL workaround https://github.com/guibou/nixGL/blob/master/default.nix
    # LD_LIBRARY_PATH="${libglvnd}/lib:${nvidiaLibsOnly}/lib:\$LD_LIBRARY_PATH";
    buildInputs = [
      poetryEnv
      fd
      ffmpeg
      fzf
      strace
      zstd
    ];
    # nativeBuildInputs = [ qt4.wrapQtAppsHook ];
}
