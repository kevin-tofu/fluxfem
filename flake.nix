{
  description = "python dev environment with PETSc pinned via nixpkgs";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        lib = pkgs.lib;
        petsc = pkgs.petsc.overrideAttrs (old: {
          version = "3.23.6";
          src = pkgs.fetchurl {
            url = "https://web.cels.anl.gov/projects/petsc/download/release-snapshots/petsc-3.23.6.tar.gz";
            sha256 = "sha256-B+BJLFw40vxapt2YHEUAhvO4j4g03xEkeofUvs+4XHI=";
          };
        });
        fluxfemLdLibraryPath = lib.makeLibraryPath [
          pkgs.stdenv.cc.cc.lib
          pkgs.zlib
          pkgs.xorg.libX11
          pkgs.gfortran.cc.lib
        ];
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
            petsc
            pkgs.openmpi
            pkgs.pkg-config
            pkgs.python312
            pkgs.poetry
            pkgs.stdenv.cc.cc.lib
            pkgs.zlib
            pkgs.xorg.libX11
            pkgs.gfortran.cc.lib
          ];

          PETSC_DIR = "${petsc}";
          OMPI_MCA_btl = "self,vader,tcp";

          shellHook = ''
            export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
            export POETRY_VIRTUALENVS_CREATE=true
            export POETRY_VIRTUALENVS_IN_PROJECT=true
            export POETRY_VIRTUALENVS_PREFER_ACTIVE_PYTHON=true
            export FLUXFEM_LD_LIBRARY_PATH="${fluxfemLdLibraryPath}"
            export LD_LIBRARY_PATH="''${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$FLUXFEM_LD_LIBRARY_PATH"
            export JAX_ENABLE_X64=1

            poetry() {
              LD_LIBRARY_PATH="''${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$FLUXFEM_LD_LIBRARY_PATH" command poetry "$@"
            }

            python() {
              LD_LIBRARY_PATH="''${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$FLUXFEM_LD_LIBRARY_PATH" command python "$@"
            }
            echo "PETSc: ${petsc.version}"
            echo "Python: using Nix python at $(which python)"
            echo "Poetry: venv in project, prefer active python"
          '';
        };
      });
}
