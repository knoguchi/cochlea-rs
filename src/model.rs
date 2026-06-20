//! Top-level Zilany 2014 cochlear model API.
//!
//! Provides both:
//! - The [`Cochlea`] struct with builder-style construction and `simulate*` methods
//!   (the canonical, Rust-native API).
//! - Free functions `run_zilany2014`, `run_channel`, `run_model_simple`, `generate_cfs`
//!   (preserved for backwards compatibility with the original port; they delegate into
//!   the canonical implementation).

use rand::Rng;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;

use crate::ihc::{run_ihc, SPECIES_CAT, SPECIES_HUMAN_GLASBERG, SPECIES_HUMAN_SHERA};
use crate::spike_generator::run_spike_generator;
use crate::synapse::{run_synapse, AnfType};

/// Model output for a single channel.
#[derive(Clone, Debug)]
pub struct ChannelOutput {
    /// Characteristic frequency in Hz
    pub cf: f64,
    /// IHC output (receptor potential)
    pub ihc_out: Vec<f64>,
    /// Synapse output (instantaneous firing rate)
    pub synapse_out: Vec<f64>,
    /// Spike times in seconds
    pub spike_times: Vec<f64>,
}

/// Model output for all channels.
#[derive(Clone, Debug)]
pub struct ModelOutput {
    /// Output for each channel
    pub channels: Vec<ChannelOutput>,
    /// Sampling frequency
    pub fs: f64,
    /// Signal duration in seconds
    pub duration: f64,
}

impl ModelOutput {
    /// Get all spike times across all channels as (channel_index, spike_time) pairs.
    pub fn all_spike_times(&self) -> Vec<(usize, f64)> {
        let mut spikes = Vec::new();
        for (i, ch) in self.channels.iter().enumerate() {
            for &t in &ch.spike_times {
                spikes.push((i, t));
            }
        }
        spikes.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        spikes
    }

    /// Get spike counts for each channel.
    pub fn spike_counts(&self) -> Vec<usize> {
        self.channels.iter().map(|ch| ch.spike_times.len()).collect()
    }

    /// Get firing rates (spikes/s) for each channel.
    pub fn firing_rates(&self) -> Vec<f64> {
        self.channels
            .iter()
            .map(|ch| ch.spike_times.len() as f64 / self.duration)
            .collect()
    }
}

/// Species enumeration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Species {
    Cat,
    /// Human (Shera et al. 2002 tuning)
    Human,
    /// Human (Glasberg & Moore 1990 tuning)
    HumanGlasberg,
}

impl Species {
    #[allow(dead_code)]
    fn to_code(&self) -> i32 {
        match self {
            Species::Cat => SPECIES_CAT,
            Species::Human => SPECIES_HUMAN_SHERA,
            Species::HumanGlasberg => SPECIES_HUMAN_GLASBERG,
        }
    }

    fn to_str(&self) -> &'static str {
        match self {
            Species::Cat => "cat",
            Species::Human => "human",
            Species::HumanGlasberg => "human_glasberg1990",
        }
    }

    /// Generate characteristic frequencies via the Greenwood function for this species.
    ///
    /// # Arguments
    /// * `min_hz` - Minimum frequency in Hz
    /// * `max_hz` - Maximum frequency in Hz
    /// * `count`  - Number of CFs to generate
    ///
    /// # Example
    /// ```
    /// use cochlea::Species;
    /// let cfs = Species::Human.cfs(200.0, 8000.0, 30);
    /// assert_eq!(cfs.len(), 30);
    /// ```
    pub fn cfs(self, min_hz: f64, max_hz: f64, count: usize) -> Vec<f64> {
        let (a_a, k, a) = match self {
            Species::Cat => (456.0, 0.8, 2.1), // Liberman (1982)
            Species::Human | Species::HumanGlasberg => (165.4, 0.88, 2.1),
        };

        let xmin = (min_hz / a_a + k).log10() / a;
        let xmax = (max_hz / a_a + k).log10() / a;

        (0..count)
            .map(|i| {
                let x = xmin + (xmax - xmin) * i as f64 / (count - 1).max(1) as f64;
                a_a * (10.0f64.powf(a * x) - k)
            })
            .collect()
    }
}

/// Model configuration.
#[derive(Clone, Debug)]
pub struct ModelConfig {
    /// Sampling frequency in Hz
    pub fs: f64,
    /// Species
    pub species: Species,
    /// OHC health (0-1)
    pub cohc: f64,
    /// IHC health (0-1)
    pub cihc: f64,
    /// ANF type
    pub anf_type: AnfType,
    /// Use fractional Gaussian noise
    pub use_ffgn: bool,
    /// Random seed (None for random)
    pub seed: Option<u64>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            fs: 100e3,
            species: Species::Cat,
            cohc: 1.0,
            cihc: 1.0,
            anf_type: AnfType::Hsr,
            use_ffgn: true,
            seed: None,
        }
    }
}

/// A configured cochlear model — one "ear" you can simulate signals against.
///
/// Use builder methods to configure species, sample rate, fiber type, etc.
/// Multiple `Cochlea` instances can coexist with independent parameters,
/// which is the natural way to model stereo, binaural, or asymmetric hearing.
///
/// # Example
///
/// ```no_run
/// use cochlea::{Cochlea, Species, AnfType};
///
/// let cochlea = Cochlea::human()
///     .sample_rate(100e3)
///     .fiber_type(AnfType::Hsr)
///     .seed(42);
///
/// let cfs = Species::Human.cfs(200.0, 8000.0, 30);
/// let signal: Vec<f64> = vec![0.0; 1000]; // your audio
/// let output = cochlea.simulate(&signal, &cfs);
///
/// for ch in &output.channels {
///     println!("CF {:.0} Hz: {} spikes", ch.cf, ch.spike_times.len());
/// }
/// ```
#[derive(Clone, Debug)]
pub struct Cochlea {
    config: ModelConfig,
}

impl Cochlea {
    /// Create a cochlea with default configuration (cat species, 100 kHz, HSR fibers, healthy).
    pub fn new() -> Self {
        Self { config: ModelConfig::default() }
    }

    /// Create from an explicit [`ModelConfig`].
    pub fn with_config(config: ModelConfig) -> Self {
        Self { config }
    }

    /// Cat cochlea preset.
    pub fn cat() -> Self {
        Self::new().species(Species::Cat)
    }

    /// Human cochlea preset (Shera et al. 2002 tuning).
    pub fn human() -> Self {
        Self::new().species(Species::Human)
    }

    /// Human cochlea preset (Glasberg & Moore 1990 tuning).
    pub fn human_glasberg() -> Self {
        Self::new().species(Species::HumanGlasberg)
    }

    /// Set the species.
    pub fn species(mut self, species: Species) -> Self {
        self.config.species = species;
        self
    }

    /// Set the sampling frequency in Hz. Zilany 2014 typically uses 100 kHz.
    pub fn sample_rate(mut self, fs: f64) -> Self {
        self.config.fs = fs;
        self
    }

    /// Set the auditory nerve fiber type (LSR/MSR/HSR).
    pub fn fiber_type(mut self, anf_type: AnfType) -> Self {
        self.config.anf_type = anf_type;
        self
    }

    /// Set outer hair cell health (0.0 = no OHC, 1.0 = healthy).
    pub fn ohc_health(mut self, cohc: f64) -> Self {
        self.config.cohc = cohc;
        self
    }

    /// Set inner hair cell health (0.0 = no IHC, 1.0 = healthy).
    pub fn ihc_health(mut self, cihc: f64) -> Self {
        self.config.cihc = cihc;
        self
    }

    /// Set the random seed for reproducibility. Without this, the cochlea uses entropy.
    pub fn seed(mut self, seed: u64) -> Self {
        self.config.seed = Some(seed);
        self
    }

    /// Toggle fractional Gaussian noise in the synapse model (default: on).
    pub fn fractional_gaussian_noise(mut self, on: bool) -> Self {
        self.config.use_ffgn = on;
        self
    }

    /// View the underlying configuration.
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Simulate the cochlear response to a sound signal across multiple CFs.
    ///
    /// Channels are processed in parallel via Rayon.
    pub fn simulate(&self, signal: &[f64], cfs: &[f64]) -> ModelOutput {
        let mut rng: StdRng = match self.config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        let duration = signal.len() as f64 / self.config.fs;

        // Per-channel seeds so parallel processing remains deterministic given a top-level seed.
        let channel_seeds: Vec<u64> = (0..cfs.len()).map(|_| rng.gen()).collect();

        let channels: Vec<ChannelOutput> = cfs
            .par_iter()
            .zip(channel_seeds.par_iter())
            .map(|(&cf, &seed)| {
                let mut channel_rng = StdRng::seed_from_u64(seed);
                self.simulate_channel(signal, cf, &mut channel_rng)
            })
            .collect();

        ModelOutput {
            channels,
            fs: self.config.fs,
            duration,
        }
    }

    /// Simulate one CF channel against the signal. Caller provides the RNG so
    /// parallel callers can shard reproducibly.
    pub fn simulate_channel<R: Rng>(&self, signal: &[f64], cf: f64, rng: &mut R) -> ChannelOutput {
        let tdres = 1.0 / self.config.fs;

        let ihc_out = run_ihc(
            signal,
            cf,
            self.config.fs,
            self.config.species.to_str(),
            self.config.cohc,
            self.config.cihc,
        );

        let synapse_out = run_synapse(
            &ihc_out,
            tdres,
            cf,
            self.config.anf_type,
            self.config.use_ffgn,
            rng,
        );

        let spike_times = run_spike_generator(&synapse_out, tdres, rng);

        ChannelOutput {
            cf,
            ihc_out,
            synapse_out,
            spike_times,
        }
    }

    /// Convenience: simulate and return just spike times per channel.
    pub fn spike_times(&self, signal: &[f64], cfs: &[f64]) -> Vec<Vec<f64>> {
        self.simulate(signal, cfs)
            .channels
            .into_iter()
            .map(|ch| ch.spike_times)
            .collect()
    }
}

impl Default for Cochlea {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Legacy free-function API. Preserved for backwards compatibility.
// These delegate INTO `Cochlea` rather than the reverse.
// ---------------------------------------------------------------------------

/// Run the complete Zilany 2014 model for a single CF.
///
/// **Legacy API** — prefer [`Cochlea::simulate_channel`].
pub fn run_channel<R: Rng>(signal: &[f64], cf: f64, config: &ModelConfig, rng: &mut R) -> ChannelOutput {
    Cochlea::with_config(config.clone()).simulate_channel(signal, cf, rng)
}

/// Run the complete Zilany 2014 model for multiple CFs.
///
/// **Legacy API** — prefer [`Cochlea::simulate`].
pub fn run_zilany2014(signal: &[f64], cfs: &[f64], config: &ModelConfig) -> ModelOutput {
    Cochlea::with_config(config.clone()).simulate(signal, cfs)
}

/// Generate characteristic frequencies spaced according to the Greenwood function.
///
/// **Legacy API** — prefer [`Species::cfs`].
pub fn generate_cfs(freq_min: f64, freq_max: f64, num_cfs: usize, species: Species) -> Vec<f64> {
    species.cfs(freq_min, freq_max, num_cfs)
}

/// Convenience function with simplified (string-based) parameters.
///
/// **Legacy API** — prefer [`Cochlea`] with builder methods.
pub fn run_model_simple(
    signal: &[f64],
    fs: f64,
    cfs: &[f64],
    species: &str,
    anf_type: &str,
) -> Vec<Vec<f64>> {
    let species = match species {
        "cat" => Species::Cat,
        "human" => Species::Human,
        "human_glasberg1990" => Species::HumanGlasberg,
        _ => Species::Cat,
    };

    let anf_type = match anf_type {
        "lsr" => AnfType::Lsr,
        "msr" => AnfType::Msr,
        "hsr" => AnfType::Hsr,
        _ => AnfType::Hsr,
    };

    Cochlea::new()
        .species(species)
        .sample_rate(fs)
        .fiber_type(anf_type)
        .spike_times(signal, cfs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_cfs() {
        let cfs = generate_cfs(125.0, 8000.0, 30, Species::Cat);

        assert_eq!(cfs.len(), 30);
        assert!(cfs[0] >= 125.0);
        assert!(cfs[29] <= 8000.0);

        for i in 1..cfs.len() {
            assert!(cfs[i] > cfs[i - 1]);
        }
    }

    #[test]
    fn test_species_cfs_matches_legacy() {
        let legacy = generate_cfs(200.0, 8000.0, 20, Species::Human);
        let new = Species::Human.cfs(200.0, 8000.0, 20);
        assert_eq!(legacy.len(), new.len());
        for (a, b) in legacy.iter().zip(new.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn test_run_zilany2014() {
        let fs = 100e3;
        let duration = 0.01; // 10 ms
        let n_samples = (duration * fs) as usize;
        let freq = 1000.0;

        let signal: Vec<f64> = (0..n_samples)
            .map(|i| 0.1 * (2.0 * std::f64::consts::PI * freq * i as f64 / fs).sin())
            .collect();

        let cfs = vec![500.0, 1000.0, 2000.0];
        let config = ModelConfig {
            fs,
            seed: Some(42),
            ..Default::default()
        };

        let output = run_zilany2014(&signal, &cfs, &config);

        assert_eq!(output.channels.len(), 3);
        assert_eq!(output.fs, fs);

        for ch in &output.channels {
            assert_eq!(ch.ihc_out.len(), n_samples);
            assert_eq!(ch.synapse_out.len(), n_samples);
        }
    }

    #[test]
    fn test_cochlea_struct_matches_legacy() {
        let fs = 100e3;
        let n_samples = 500;
        let signal: Vec<f64> = (0..n_samples)
            .map(|i| 0.1 * (2.0 * std::f64::consts::PI * 1000.0 * i as f64 / fs).sin())
            .collect();
        let cfs = vec![500.0, 1000.0, 2000.0];

        // Legacy path
        let config = ModelConfig {
            fs,
            seed: Some(42),
            ..Default::default()
        };
        let legacy = run_zilany2014(&signal, &cfs, &config);

        // New path
        let new = Cochlea::new()
            .sample_rate(fs)
            .seed(42)
            .simulate(&signal, &cfs);

        // Same channel count, same CFs, and (because of seeded RNG) identical spike trains.
        assert_eq!(legacy.channels.len(), new.channels.len());
        for (a, b) in legacy.channels.iter().zip(new.channels.iter()) {
            assert_eq!(a.cf, b.cf);
            assert_eq!(a.spike_times, b.spike_times);
        }
    }

    #[test]
    fn test_cochlea_builder_chains() {
        let c = Cochlea::human()
            .sample_rate(100e3)
            .fiber_type(AnfType::Lsr)
            .ohc_health(0.5)
            .ihc_health(0.8)
            .seed(7);

        assert_eq!(c.config().species, Species::Human);
        assert_eq!(c.config().fs, 100e3);
        assert_eq!(c.config().anf_type, AnfType::Lsr);
        assert_eq!(c.config().cohc, 0.5);
        assert_eq!(c.config().cihc, 0.8);
        assert_eq!(c.config().seed, Some(7));
    }

    #[test]
    fn test_two_independent_cochleae() {
        // Two ears with different damage profiles — the use case the struct enables.
        let fs = 100e3;
        let n = 500;
        let signal: Vec<f64> = (0..n)
            .map(|i| 0.1 * (2.0 * std::f64::consts::PI * 1000.0 * i as f64 / fs).sin())
            .collect();
        let cfs = vec![1000.0];

        let healthy = Cochlea::human().sample_rate(fs).seed(1);
        let damaged = Cochlea::human().sample_rate(fs).seed(1).ohc_health(0.2);

        let h = healthy.simulate(&signal, &cfs);
        let d = damaged.simulate(&signal, &cfs);

        // Different OHC health should produce different IHC outputs given identical seeds.
        assert_eq!(h.channels.len(), d.channels.len());
        let h_ihc = &h.channels[0].ihc_out;
        let d_ihc = &d.channels[0].ihc_out;
        let max_diff = h_ihc.iter().zip(d_ihc).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
        assert!(max_diff > 0.0, "OHC damage must change IHC output");
    }

    #[test]
    fn test_model_output_methods() {
        let output = ModelOutput {
            channels: vec![
                ChannelOutput {
                    cf: 1000.0,
                    ihc_out: vec![0.0; 100],
                    synapse_out: vec![0.0; 100],
                    spike_times: vec![0.001, 0.005, 0.008],
                },
                ChannelOutput {
                    cf: 2000.0,
                    ihc_out: vec![0.0; 100],
                    synapse_out: vec![0.0; 100],
                    spike_times: vec![0.002, 0.006],
                },
            ],
            fs: 100e3,
            duration: 0.01,
        };

        let counts = output.spike_counts();
        assert_eq!(counts, vec![3, 2]);

        let rates = output.firing_rates();
        assert_eq!(rates, vec![300.0, 200.0]);

        let all_spikes = output.all_spike_times();
        assert_eq!(all_spikes.len(), 5);
        assert_eq!(all_spikes[0], (0, 0.001));
        assert_eq!(all_spikes[1], (1, 0.002));
    }
}
