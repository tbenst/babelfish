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
      jupyter_core
      mypy
      pylint
      stytra
    ];
    ignoreCollisions = true;
  };
in
# TODO how to hotload? don't install babelfish, just propagatedBuildInputs
with pkgs;
mkShell {
    inherit (secrets) MLFLOW_TRACKING_URI MLFLOW_TRACKING_USERNAME MLFLOW_TRACKING_PASSWORD;
    AIRFLOW_HOME=builtins.toString ./pipelines;
    # QTCOMPOSE="${xorg.libX11}/share/X11/locale";
    # QT_PLUGIN_PATH = "${qt4}/${qt4.qtPluginPrefix}";
    # openGL workaround https://github.com/guibou/nixGL/blob/master/default.nix
    # LD_LIBRARY_PATH="${libglvnd}/lib:${nvidiaLibsOnly}/lib:\$LD_LIBRARY_PATH";
    buildInputs = [ python-env ffmpeg ];
    # nativeBuildInputs = [ qt4.wrapQtAppsHook ];
}
