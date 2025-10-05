# Building

Needs CMake 3.15+, a C++17 compiler, and the AFF3CT submodule.

```
git submodule update --init --recursive lib/aff3ct

cmake -S lib/aff3ct -B lib/aff3ct/build -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DAFF3CT_COMPILE_EXE=OFF -DAFF3CT_COMPILE_STATIC_LIB=ON
cmake --build lib/aff3ct/build -j
cmake --install lib/aff3ct/build --prefix .aff3ct/install
ln -sfn "$(basename .aff3ct/install/lib/cmake/aff3ct-*)" .aff3ct/install/lib/cmake/aff3ct

cmake -S . -B build -DCMAKE_PREFIX_PATH="$PWD/.aff3ct/install"
cmake --build build -j
```

AFF3CT installs its CMake files under a versioned directory, which
`find_package` does not resolve on its own; the symlink gives it the plain
name. The binary lands in `build/bin/msprs`.

Data paths are baked in at configure time, so the binary works from any
directory.

## Modes

```
msprs --mode uncoded-msprs --L0 3 --family balanced
msprs --mode coded-msprs   --L0 3 --family balanced --iters 7
msprs --mode ldpc-msprs    --L0 3 --family balanced --ldpc-h <matrix.alist>
msprs --mode exit          --L0 3 --family balanced
msprs --mode exit-decoder
msprs --mode bounds
msprs --mode alphabet      --L0 3 --family balanced
msprs --mode eye           --L0 3 --family balanced --ebn0 15.11
msprs --mode params
```

Add `--out results/ber` to a sweep to write JSON records instead of only
printing a table.

`modtest`, `demodtest`, `tdemodtest`, `enctest` and `dectest` read stdin and
print their output, so each block can be diffed against another implementation
on identical input.
