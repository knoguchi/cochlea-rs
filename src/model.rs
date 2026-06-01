//! Top-level Zilany 2014 cochlear model API.
//!
//! Provides the main entry point for running the complete auditory nerve model.

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

/// Run the complete Zilany 2014 model for a single CF.
///
/// # Arguments
/// * `signal` - Input signal (pressure in Pa)
/// * `cf` - Characteristic frequency in Hz
/// * `config` - Model configuration
///
/// # Returns
/// Channel output including IHC, synapse, and spike times
pub fn run_channel<R: Rng>(signal: &[f64], cf: f64, config: &ModelConfig, rng: &mut R) -> ChannelOutput {
    let tdres = 1.0 / config.fs;

    // Run IHC model
    let ihc_out = run_ihc(
        signal,
        cf,
        config.fs,
        config.species.to_str(),
        config.cohc,
        config.cihc,
    );

    // Run synapse model
    let synapse_out = run_synapse(
        &ihc_out,
        tdres,
        cf,
        config.anf_type,
        config.use_ffgn,
        rng,
    );

    // Run spike generator
    let spike_times = run_spike_generator(&synapse_out, tdres, rng);

    ChannelOutput {
        cf,
        ihc_out,
        synapse_out,
        spike_times,
    }
}

/// Run the complete Zilany 2014 model for multiple CFs.
///
/// # Arguments
/// * `signal` - Input signal (pressure in Pa)
/// * `cfs` - Vector of characteristic frequencies in Hz
/// * `config` - Model configuration
///
/// # Returns
/// Model output for all channels
pub fn run_zilany2014(signal: &[f64], cfs: &[f64], config: &ModelConfig) -> ModelOutput {
    let mut rng: StdRng = match config.seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_entropy(),
    };

    let duration = signal.len() as f64 / config.fs;

    // Generate per-channel seeds for reproducible parallel processing
    let channel_seeds: Vec<u64> = (0..cfs.len()).map(|_| rng.gen()).collect();

    // Process channels in parallel
    let channels: Vec<ChannelOutput> = cfs
        .par_iter()
        .zip(channel_seeds.par_iter())
        .map(|(&cf, &seed)| {
            let mut channel_rng = StdRng::seed_from_u64(seed);
            run_channel(signal, cf, config, &mut channel_rng)
        })
        .collect();

    ModelOutput {
        channels,
        fs: config.fs,
        duration,
    }
}

/// Generate characteristic frequencies spaced according to the Greenwood function.
///
/// # Arguments
/// * `freq_min` - Minimum frequency in Hz
/// * `freq_max` - Maximum frequency in Hz
/// * `num_cfs` - Number of CFs to generate
/// * `species` - Species
///
/// # Returns
/// Vector of characteristic frequencies
pub fn generate_cfs(freq_min: f64, freq_max: f64, num_cfs: usize, species: Species) -> Vec<f64> {
    let (a_a, k, a) = match species {
        Species::Cat => (456.0, 0.8, 2.1), // Liberman (1982)
        Species::Human | Species::HumanGlasberg => (165.4, 0.88, 2.1),
    };

    let xmin = (freq_min / a_a + k).log10() / a;
    let xmax = (freq_max / a_a + k).log10() / a;

    (0..num_cfs)
        .map(|i| {
            let x = xmin + (xmax - xmin) * i as f64 / (num_cfs - 1).max(1) as f64;
            a_a * (10.0f64.powf(a * x) - k)
        })
        .collect()
}

/// Convenience function with simplified parameters.
///
/// # Arguments
/// * `signal` - Input signal (pressure in Pa)
/// * `fs` - Sampling frequency in Hz
/// * `cf` - Single characteristic frequency or (min, max, num) tuple
/// * `species` - Species string ("cat", "human", "human_glasberg1990")
/// * `anf_type` - ANF type string ("lsr", "msr", "hsr")
///
/// # Returns
/// Vector of spike times for each CF
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

    let config = ModelConfig {
        fs,
        species,
        anf_type,
        ..Default::default()
    };

    let output = run_zilany2014(signal, cfs, &config);

    output.channels.into_iter().map(|ch| ch.spike_times).collect()
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

        // Check monotonically increasing
        for i in 1..cfs.len() {
            assert!(cfs[i] > cfs[i - 1]);
        }
    }

    #[test]
    fn test_run_zilany2014() {
        let fs = 100e3;
        let duration = 0.01; // 10 ms
        let n_samples = (duration * fs) as usize;
        let freq = 1000.0;

        // Generate a tone
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
        // Should be sorted by time
        assert_eq!(all_spikes[0], (0, 0.001));
        assert_eq!(all_spikes[1], (1, 0.002));
    }
}
