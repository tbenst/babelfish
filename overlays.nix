# overlays for Continuos integration / shell scripts
[
  (self: super: {
    pythonOverrides = python-self: python-super: {
      opencv3 = python-super.opencv3.override {
          enableCuda = true;
          enableFfmpeg = true;
      };
      babelfish = python-super.callPackage ./default.nix {};
    };
    python37 = super.python37.override {packageOverrides = self.pythonOverrides;};
    ffmpeg = super.ffmpeg-full.override {
      nonfreeLicensing = true;
      nvenc = true; # nvidia support
    };
    ffmpeg-full = super.ffmpeg-full.override {
      nonfreeLicensing = true;
      nvenc = true; # nvidia support
    };
  })
]