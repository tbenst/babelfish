{ pkgs ? import <nixpkgs> {}
, run ? "bash --rcfile <(echo '. ~/.bashrc; conda activate caiman')"
}:

let

  # Conda installs it's packages and environments under this directory
  installationPath = "~/.conda";

  # Downloaded Miniconda installer
  minicondaScript = pkgs.stdenv.mkDerivation rec {
    name = "miniconda-${version}";
    /* version = "4.5.12"; */
    version = "4.7.12.1";
    src = pkgs.fetchurl {
      url = "https://repo.continuum.io/miniconda/Miniconda3-${version}-Linux-x86_64.sh";
      /* sha256 = "0yhmi6sgrvbzkwdxnxnpb7f42wizfwi44lwmnfb0x3li5b6v9rg5"; */
      sha256 = "1lhmpvhxx60871vinqpmsmrmc9vjakx05z85mnkpavcdl8glxqxz";
    };
    # Nothing to unpack.
    unpackPhase = "true";
    # Rename the file so it's easier to use. The file needs to have .sh ending
    # because the installation script does some checks based on that assumption.
    # However, don't add it under $out/bin/ becase we don't really want to use
    # it within our environment. It is called by "conda-install" defined below.
    installPhase = ''
      mkdir -p $out
      cp $src $out/miniconda.sh
    '';
    # Add executable mode here after the fixup phase so that no patching will be
    # done by nix because we want to use this miniconda installer in the FHS
    # user env.
    fixupPhase = ''
      chmod +x $out/miniconda.sh
    '';
  };

  # Wrap miniconda installer so that it is non-interactive and installs into the
  # path specified by installationPath
  conda = pkgs.runCommand "conda-install"
    { buildInputs = [ pkgs.makeWrapper minicondaScript ]; }
    ''
      mkdir -p $out/bin
      makeWrapper                            \
        ${minicondaScript}/miniconda.sh      \
        $out/bin/conda-install               \
        --add-flags "-p ${installationPath}" \
        --add-flags "-b"
    '';

# need libjpeg.so.8 (could also potentially build against libjpeg-turbo
# via cmakeFlag "-DWITH_JPEG8=1")
libjpeg_original_8 = pkgs.libjpeg_original.overrideAttrs (oldAttrs: {
    src = pkgs.fetchurl{
      url = https://www.ijg.org/files/jpegsrc.v8d.tar.gz;
      sha256 = "1cz0dy05mgxqdgjf52p54yxpyy95rgl30cnazdrfmw7hfca9n0h0";
    };
  });

# gsettings_path = pkgs.glib.getSchemaPath pkgs.gsettings-desktop-schemas;
gsettings_path = pkgs.glib.getSchemaPath pkgs.gtk3;

in
(
  pkgs.buildFHSUserEnv {
    name = "conda";
    targetPkgs = pkgs: (
      with pkgs; [
        conda

        # Add here libraries that Conda packages require but aren't provided by
        # Conda because it assumes that the system has them.
        #
        # For instance, for IPython, these can be found using:
        # `LD_DEBUG=libs ipython --pylab`
        xorg.libSM
        xorg.libICE
        xorg.libXrender
        xorg.libX11
        xorg.libXinerama
        xorg.libXdamage
        xorg.libXcursor
        xorg.libXrender
        xorg.libXScrnSaver
        xorg.libXxf86vm
        xorg.libXi
        xorg.libXau 

        libGL
        libselinux

        # Just in case one installs a package with pip instead of conda and pip
        # needs to compile some C sources
        gcc
        stdenv.cc

        # Add any other packages here, for instance:
        fd
        vim
        git

        /* opencv3 */
        libsodium
        ffmpeg
        which

        # PsychoPy
        gtk3
        gtk2
        gdk_pixbuf
        /* ibus-engines.mozc */
        /* ibus-engines.uniemoji */
        /* gtkd */
        glib

        # WxPython
        libjpeg_original_8
        libtiff SDL
        gst-plugins-base libnotify freeglut ncurses
        libpng gstreamer
        (wxGTK.gtk)
        pango
        cairo
        gsettings_desktop_schemas
        #libjpeg-turbo

        # NVIDIA
        # TODO: test if these are causing build..?
        cudatoolkit_10_0
        cudnn_cudatoolkit_10_0
        linuxPackages.nvidia_x11

        # courtesy steam/chrootenv.nix
        gnome3.gtk
        dbus
        zlib
        glib
        atk
        cairo
        freetype
        pango
        fontconfig
        gdk-pixbuf
        gnome2.GConf
      ]
    );

    runScript = "${run}";
    
    profile = with pkgs; with stdenv.lib; ''
      # Add conda to PATH
      export PATH=${installationPath}/bin:$PATH
      # Paths for gcc if compiling some C sources with pip
      export NIX_CFLAGS_COMPILE="-I${installationPath}/include"
      export NIX_CFLAGS_LINK="-L${installationPath}lib"
      # Some other required environment variables
      export FONTCONFIG_FILE=/etc/fonts/fonts.conf
      export QTCOMPOSE=${pkgs.xorg.libX11}/share/X11/locale
      SOURCE_DATE_EPOCH=315532800
      export LD_LIBRARY_PATH=${pkgs.glib.out}/lib:$LD_LIBRARY_PATH
      export LIBARCHIVE=${libarchive.lib}/lib/libarchive.so
      export GSETTINGS_SCHEMA_DIR="${gsettings_path}"
      export GIO_EXTRA_MODULES="${pkgs.glib_networking.out}/lib/gio/modules:${pkgs.gnome3.dconf}/lib/gio/modules"
    '';
  }
).env
