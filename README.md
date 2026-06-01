# cochlea-rs

Rust port of the Zilany et al. (2014) auditory periphery model.

Converts sound pressure waveforms into auditory nerve spike trains through a biologically faithful pipeline:

```
Sound → Middle Ear → Inner Hair Cell → Synapse → Spike Generator → spikes
```

## Features

- **Species-specific** middle ear filters (cat, human, human Glasberg)
- **Three fiber types**: LSR, MSR, HSR (low/medium/high spontaneous rate)
- **O(n) power-law adaptation** — fixes O(n²) bug in original implementations
- **Parallel processing** across cochlear channels via Rayon
- **Reproducible** — optional seed for deterministic stochastic components
- **Python bindings** via PyO3

## Usage

### Rust

```rust
use cochlea::{run_zilany2014, ModelConfig, Species, AnfType, generate_cfs};

let signal: Vec<f64> = /* your audio samples at 100 kHz */;
let cfs = generate_cfs(200.0, 8000.0, 30, Species::Human);

let config = ModelConfig {
    fs: 100e3,
    species: Species::Human,
    anf_type: AnfType::Hsr,
    seed: Some(42),
    ..Default::default()
};

let output = run_zilany2014(&signal, &cfs, &config);

for channel in &output.channels {
    println!("CF {:.0} Hz: {} spikes", channel.cf, channel.spike_times.len());
}
```

### Python

```bash
pip install maturin
maturin develop --release
```

```python
import cochlea

cfs = cochlea.generate_cfs_py(200, 8000, 30, species="human")
result = cochlea.run_zilany2014_full(signal, fs=100000, cfs=cfs, species="human")
spike_times = result["spike_times"]
```

## Credits

Rust port of the [cochlea](https://github.com/mrkrd/cochlea) Python library by **Marek Rudnicki**, which implements:

> Zilany, M.S.A., Bruce, I.C., & Carney, L.H. (2014). "Updated parameters and expanded simulation options for a model of the auditory periphery." *J. Acoust. Soc. Am.* 135(1), 283-286.

## License

GPL-3.0
