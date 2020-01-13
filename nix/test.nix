{ pkgs ? import <nixpkgs> {}
, run ? "bash"
}:
(
  pkgs.buildFHSUserEnv {
    name = "test";
    targetPkgs = pkgs: (
      with pkgs; [
        (python3.withPackages (ps: [ ps.tifffile ]))
        gdb
      ]
    );
    runScript = "${run}";
    
  }
).env
