{ pkgs ? import ./nixpkgs.nix }:
with pkgs;
poetry2nix.mkPoetryEnv {
    poetrylock = ../poetry.lock;
    python = python3;
    overrides = [
      poetry2nix.defaultPoetryOverrides
      (self: super: {
      })
    ];
}