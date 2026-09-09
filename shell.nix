{ pkgs ? import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/nixos-26.05.tar.gz") {
    config.allowUnfree = true;
  }
}:

let
  unstable = import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/nixos-unstable.tar.gz") {
    config.allowUnfree = true;
  };
in
pkgs.mkShellNoCC {
  packages = with pkgs; [
    unstable.claude-code
    unstable.codex
    unstable.opencode
    unstable.pi-coding-agent
    nodejs
    uv
    ollama-cpu
  ]
  ++ (lib.optionals stdenv.isLinux [ inotify-tools ])
  ++ (lib.optionals stdenv.isDarwin [ git ]);

  LOCALE_ARCHIVE = if (pkgs.stdenv.isLinux) then "${pkgs.glibcLocales}/lib/locale/locale-archive" else "";
}
