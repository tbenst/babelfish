let
  pkgs = import ./nix/nixpkgs.nix;
  secrets = import ./nix/secrets.nix;
  nvidiaLibsOnly = pkgs.linuxPackages.nvidia_x11.override {
    libsOnly = true;
    kernel = null;
  };
  python-env = pkgs.python3.buildEnv.override {
    extraLibs = with pkgs.python3Packages; [
      apache-airflow
      babelfish
      babelfish-models
      jupyter
      mypy
      pyarrow
      pylint
      stytra
      seqnmf
    ];
    ignoreCollisions = true;
  };
in
# TODO how to hotload? don't install babelfish, just propagatedBuildInputs
# TODO avoid GC https://github.com/NixOS/nix/issues/2208
# TODO shell is broken in ssh or when using --pure
# due to Qt bug. 
# (`python -m stytra.offline.track_video` fails)
with pkgs;
mkShell {
    inherit (secrets) MLFLOW_TRACKING_URI MLFLOW_TRACKING_USERNAME MLFLOW_TRACKING_PASSWORD;
    AIRFLOW_HOME=builtins.toString ./pipelines;
    # QTCOMPOSE="${xorg.libX11}/share/X11/locale";
    QT_PLUGIN_PATH = "${qt5.qtbase}/${qt5.qtbase.qtPluginPrefix}";
    # openGL workaround https://github.com/guibou/nixGL/blob/master/default.nix
    # LD_LIBRARY_PATH="${libglvnd}/lib:${nvidiaLibsOnly}/lib:\$LD_LIBRARY_PATH";
    buildInputs = [
      babelfish-scripts
      python-env
      fd
      ffmpeg
      fzf
      strace
      zstd
    ];
    # nativeBuildInputs = [ qt4.wrapQtAppsHook ];
}
