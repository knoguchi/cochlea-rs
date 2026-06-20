# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Rust port of the Zilany et al. (2014) auditory periphery model. Converts sound to auditory nerve spike trains. The crate name is `cochlea` (lib name in Cargo.toml).

## Build Commands

```bash
cargo build                              # Build
cargo test                               # Run tests
cargo test test_tone_response            # Single test
cargo bench                              # Benchmarks (Criterion)
```

## Architecture

```
Sound → MiddleEar (3-stage IIR) → IHC (basilar membrane + transduction) → Synapse (adaptation) → SpikeGenerator (Poisson) → spikes
```

- **model.rs** — Top-level `run_zilany2014()`. Runs channels in parallel via Rayon.
- **middle_ear.rs** — Species-specific cascade filter (Cat, Human, HumanGlasberg).
- **ihc.rs** — C1/C2 chirp filters, gammatone, OHC nonlinearity, IHC transduction.
- **synapse.rs** — Three fiber types (`AnfType::{Lsr, Msr, Hsr}`). O(n) power-law adaptation via IIR.
- **spike_generator.rs** — Inhomogeneous Poisson with 0.75ms refractory period.
- **filters.rs**, **complex.rs** — Shared utilities.

## Testing

- Unit tests in each module, integration tests in `tests/validation.rs`.
- Benchmarks in `benches/benchmarks.rs` (Criterion).
- Use `approx` crate for float comparisons.
