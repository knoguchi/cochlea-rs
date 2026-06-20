# cochlea-rs

Rust port of the Zilany et al. (2014) auditory periphery model.

```
Sound → Middle Ear → Inner Hair Cell → Synapse → Spike Generator → spikes
```

## Usage

```rust
use cochlea::{Cochlea, Species, AnfType};

let signal: Vec<f64> = /* your audio samples at 100 kHz */;
let cfs = Species::Human.cfs(200.0, 8000.0, 30);

let cochlea = Cochlea::human()
    .sample_rate(100e3)
    .fiber_type(AnfType::Hsr)
    .seed(42);

let output = cochlea.simulate(&signal, &cfs);

for channel in &output.channels {
    println!("CF {:.0} Hz: {} spikes", channel.cf, channel.spike_times.len());
}
```

## Credits

Port of the [cochlea](https://github.com/mrkrd/cochlea) Python library by Marek Rudnicki, which implements:

> Zilany, Bruce & Carney (2014), *J. Acoust. Soc. Am.* 135(1), 283-286.

## License

GPL-3.0
