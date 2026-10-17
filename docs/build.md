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
msprs --mode timing        --L0 3 --family balanced --ebn0 4
msprs --mode params
```

Add `--out results/ber` to a sweep to write JSON records instead of only
printing a table.

`--bcjr map|log-map|max-log-map` picks the algorithm of both BCJRs, the
equalizer's and the outer decoder's, in every mode. `map` is exact and the
default; records of the other two carry a `_logmap` or `_maxlog` suffix.
`timing` runs all three on one thread and prints the time of each pass and of a
whole frame, with the speed-up over MAP.

## Learned equaliser

The trained network can stand in for the BCJR in `uncoded-msprs`,
`coded-msprs` and `exit`. Pass `--siso` with its weights; records then carry a
`_siso` suffix. Training needs PyTorch:

```
msprs --mode dataset --L0 3 --family balanced --frames 1000 \
  --ebn0-min 2 --ebn0-max 7 --out data/siso_L3_balanced
python -m siso_nn.train --data data/siso_L3_balanced --out siso_nn/weights_L3_balanced.txt

msprs --mode coded-msprs --L0 3 --family balanced --siso siso_nn/weights_L3_balanced.txt
```

## Quantised messages

`--llr-bits b --llr-step d` rounds the LLRs the turbo loop passes between the
equaliser and the decoder, both directions, to the codes
`-(2^(b-1)-1) .. 2^(b-1)-1` times `d`, in `coded-msprs`, `exit` and
`exit-decoder`. Records carry a `_q<b>_step<d>` suffix. The study in
`notebooks/llr_quant.ipynb` keeps its records under `results/quant`, its own
floating-point reference included. Each BER point runs as its own process,
seeded by its Eb/N0, so every quantiser sees the noise its reference saw:

```
for x in $(seq 2.6 0.2 5.8); do
  msprs --mode coded-msprs --L0 3 --family balanced --threads 12 --be 1000 --max-fra 10000 \
    --ebn0-min $x --ebn0-max 99 --ebn0-step 99 --seed $(echo $x | tr -d .) \
    --llr-bits 4 --llr-step 1 --out results/quant/ber
done

msprs --mode exit --L0 3 --family balanced --exit-bits 10000 --n-ia 40 --n-trials 32 \
  --exit-fresh-rx --threads 12 --ebn0-min 2 --ebn0-max 8.01 --ebn0-step 0.25 \
  --llr-bits 4 --llr-step 1 --out results/quant/exit
msprs --mode exit-decoder --n-ia 100 --n-trials 1000 --llr-bits 4 --llr-step 1 --out results/quant/exit
```

Leave out the two `--llr` flags for the reference.

`modtest`, `demodtest`, `tdemodtest`, `enctest` and `dectest` read stdin and
print their output, so each block can be diffed against another implementation
on identical input.
