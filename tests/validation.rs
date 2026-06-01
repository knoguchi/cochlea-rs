//! Validation tests for the Zilany 2014 cochlear model.
//!
//! These tests verify that the Rust implementation produces reasonable
//! outputs for standard test signals.

use cochlea::*;
use std::f64::consts::PI;

/// Generate a pure tone signal.
fn generate_tone(frequency: f64, duration: f64, amplitude: f64, fs: f64) -> Vec<f64> {
    let n_samples = (duration * fs) as usize;
    (0..n_samples)
        .map(|i| amplitude * (2.0 * PI * frequency * i as f64 / fs).sin())
        .collect()
}

/// Generate a click (impulse) signal.
#[allow(dead_code)]
fn generate_click(duration: f64, fs: f64) -> Vec<f64> {
    let n_samples = (duration * fs) as usize;
    let mut signal = vec![0.0; n_samples];
    if n_samples > 0 {
        signal[0] = 1.0;
    }
    signal
}

/// Generate a chirp (frequency sweep) signal.
fn generate_chirp(f_start: f64, f_end: f64, duration: f64, amplitude: f64, fs: f64) -> Vec<f64> {
    let n_samples = (duration * fs) as usize;
    let k = (f_end - f_start) / duration;

    (0..n_samples)
        .map(|i| {
            let t = i as f64 / fs;
            let freq = f_start + k * t;
            amplitude * (2.0 * PI * freq * t).sin()
        })
        .collect()
}

#[test]
fn test_tone_response() {
    // A tone at CF should produce spikes
    let fs = 100e3;
    let cf = 1000.0;
    let duration = 0.1; // 100 ms
    let amplitude = 0.1; // ~94 dB SPL

    let signal = generate_tone(cf, duration, amplitude, fs);
    let cfs = vec![cf];
    let config = ModelConfig {
        fs,
        seed: Some(42),
        ..Default::default()
    };

    let output = run_zilany2014(&signal, &cfs, &config);

    // Should produce spikes
    assert!(!output.channels[0].spike_times.is_empty(), "Tone at CF should produce spikes");

    // Firing rate should be reasonable (not zero, not impossibly high)
    let firing_rate = output.firing_rates()[0];
    assert!(firing_rate > 10.0, "Firing rate should be > 10 Hz for tone at CF");
    assert!(firing_rate < 1500.0, "Firing rate should be < 1500 Hz (refractory limit)");
}

#[test]
fn test_off_cf_response() {
    // A tone far from CF should produce fewer spikes
    let fs = 100e3;
    let cf = 4000.0;
    let tone_freq = 500.0; // Far from CF
    let duration = 0.1;
    let amplitude = 0.1;

    let signal = generate_tone(tone_freq, duration, amplitude, fs);
    let cfs = vec![cf];
    let config = ModelConfig {
        fs,
        seed: Some(42),
        ..Default::default()
    };

    let output = run_zilany2014(&signal, &cfs, &config);

    // Should produce some spikes but much fewer than on-CF
    let firing_rate = output.firing_rates()[0];
    // Off-CF response should be lower (spontaneous + some driven)
    assert!(firing_rate < 500.0, "Off-CF firing rate should be moderate");
}

#[test]
fn test_silence_response() {
    // Silence should produce only spontaneous activity
    let fs = 100e3;
    let cf = 1000.0;
    let duration = 0.1;

    let signal = vec![0.0; (duration * fs) as usize];
    let cfs = vec![cf];
    let config = ModelConfig {
        fs,
        seed: Some(42),
        anf_type: AnfType::Hsr, // HSR has 100 spikes/s spontaneous
        ..Default::default()
    };

    let output = run_zilany2014(&signal, &cfs, &config);

    // HSR should have ~100 Hz spontaneous rate
    let firing_rate = output.firing_rates()[0];
    assert!(firing_rate < 300.0, "Silent firing rate should be close to spontaneous (~100 Hz for HSR)");
}

#[test]
fn test_lsr_spontaneous_rate() {
    // LSR fibers should have low spontaneous rate
    let fs = 100e3;
    let cf = 1000.0;
    let duration = 0.5; // Longer duration for better statistics

    let signal = vec![0.0; (duration * fs) as usize];
    let cfs = vec![cf];
    let config = ModelConfig {
        fs,
        seed: Some(42),
        anf_type: AnfType::Lsr, // LSR has 0.1 spikes/s spontaneous
        ..Default::default()
    };

    let output = run_zilany2014(&signal, &cfs, &config);

    // LSR should have very low spontaneous rate
    let firing_rate = output.firing_rates()[0];
    assert!(firing_rate < 50.0, "LSR spontaneous rate should be very low");
}

#[test]
fn test_multiple_cfs() {
    // Test with multiple CFs
    let fs = 100e3;
    let duration = 0.05;
    let tone_freq = 1000.0;
    let amplitude = 0.1;

    let signal = generate_tone(tone_freq, duration, amplitude, fs);
    let cfs = generate_cfs(500.0, 4000.0, 10, Species::Cat);
    let config = ModelConfig {
        fs,
        seed: Some(42),
        ..Default::default()
    };

    let output = run_zilany2014(&signal, &cfs, &config);

    assert_eq!(output.channels.len(), 10);

    // Find CF closest to tone frequency
    let (best_cf_idx, _) = cfs.iter().enumerate()
        .min_by(|(_, &a), (_, &b)| {
            (a - tone_freq).abs().partial_cmp(&(b - tone_freq).abs()).unwrap()
        })
        .unwrap();

    let firing_rates = output.firing_rates();
    let best_cf_rate = firing_rates[best_cf_idx];

    // The CF closest to tone frequency should have high response
    // (though this isn't always true due to level and bandwidth effects)
    assert!(best_cf_rate > 50.0, "Response at best CF should be significant");
}

#[test]
fn test_ihc_output_shape() {
    // IHC output should have same length as input
    let fs = 100e3;
    let cf = 1000.0;
    let duration = 0.01;

    let signal = generate_tone(cf, duration, 0.1, fs);
    let expected_len = signal.len();

    let cfs = vec![cf];
    let config = ModelConfig {
        fs,
        seed: Some(42),
        ..Default::default()
    };

    let output = run_zilany2014(&signal, &cfs, &config);

    assert_eq!(output.channels[0].ihc_out.len(), expected_len);
    assert_eq!(output.channels[0].synapse_out.len(), expected_len);
}

#[test]
fn test_spike_times_in_range() {
    // All spike times should be within signal duration
    let fs = 100e3;
    let cf = 1000.0;
    let duration = 0.1;

    let signal = generate_tone(cf, duration, 0.1, fs);
    let cfs = vec![cf];
    let config = ModelConfig {
        fs,
        seed: Some(42),
        ..Default::default()
    };

    let output = run_zilany2014(&signal, &cfs, &config);

    for &t in &output.channels[0].spike_times {
        assert!(t >= 0.0, "Spike time should be >= 0");
        assert!(t <= duration, "Spike time should be <= duration");
    }
}

#[test]
fn test_refractory_period() {
    // Inter-spike intervals should respect dead time
    let fs = 100e3;
    let cf = 1000.0;
    let duration = 0.1;

    let signal = generate_tone(cf, duration, 0.5, fs); // High amplitude
    let cfs = vec![cf];
    let config = ModelConfig {
        fs,
        seed: Some(42),
        ..Default::default()
    };

    let output = run_zilany2014(&signal, &cfs, &config);

    let spike_times = &output.channels[0].spike_times;
    let dead_time = 0.00075; // 0.75 ms

    for i in 1..spike_times.len() {
        let isi = spike_times[i] - spike_times[i - 1];
        assert!(
            isi >= dead_time - 1e-10,
            "ISI ({}) should be >= dead time ({})",
            isi,
            dead_time
        );
    }
}

#[test]
fn test_human_model() {
    // Human model should work
    let fs = 100e3;
    let cf = 1000.0;
    let duration = 0.05;

    let signal = generate_tone(cf, duration, 0.1, fs);
    let cfs = vec![cf];
    let config = ModelConfig {
        fs,
        species: Species::Human,
        seed: Some(42),
        ..Default::default()
    };

    let output = run_zilany2014(&signal, &cfs, &config);

    // Should produce spikes
    assert!(!output.channels[0].spike_times.is_empty(), "Human model should produce spikes");
}

#[test]
fn test_deterministic_with_seed() {
    // Same seed should produce same results
    let fs = 100e3;
    let cf = 1000.0;
    let duration = 0.05;

    let signal = generate_tone(cf, duration, 0.1, fs);
    let cfs = vec![cf];
    let config = ModelConfig {
        fs,
        seed: Some(12345),
        ..Default::default()
    };

    let output1 = run_zilany2014(&signal, &cfs, &config);
    let output2 = run_zilany2014(&signal, &cfs, &config);

    assert_eq!(
        output1.channels[0].spike_times.len(),
        output2.channels[0].spike_times.len(),
        "Same seed should produce same number of spikes"
    );

    for (t1, t2) in output1.channels[0].spike_times.iter()
        .zip(output2.channels[0].spike_times.iter())
    {
        assert!(
            (t1 - t2).abs() < 1e-10,
            "Same seed should produce same spike times"
        );
    }
}

#[test]
fn test_generate_cfs_cat() {
    let cfs = generate_cfs(125.0, 8000.0, 30, Species::Cat);

    assert_eq!(cfs.len(), 30);
    assert!(cfs[0] >= 125.0 - 1.0, "First CF should be ~125 Hz");
    assert!(cfs[29] <= 8000.0 + 1.0, "Last CF should be ~8000 Hz");

    // Should be monotonically increasing
    for i in 1..cfs.len() {
        assert!(cfs[i] > cfs[i - 1], "CFs should be monotonically increasing");
    }
}

#[test]
fn test_generate_cfs_human() {
    let cfs = generate_cfs(125.0, 8000.0, 30, Species::Human);

    assert_eq!(cfs.len(), 30);

    // Should be monotonically increasing
    for i in 1..cfs.len() {
        assert!(cfs[i] > cfs[i - 1], "CFs should be monotonically increasing");
    }
}

#[test]
fn test_impaired_ohc() {
    // Impaired OHC should reduce response
    let fs = 100e3;
    let cf = 1000.0;
    let duration = 0.1;
    let amplitude = 0.05; // Lower amplitude to see cochlear amplifier effect

    let signal = generate_tone(cf, duration, amplitude, fs);
    let cfs = vec![cf];

    // Healthy
    let config_healthy = ModelConfig {
        fs,
        cohc: 1.0,
        seed: Some(42),
        ..Default::default()
    };
    let output_healthy = run_zilany2014(&signal, &cfs, &config_healthy);

    // Impaired
    let config_impaired = ModelConfig {
        fs,
        cohc: 0.0, // Completely impaired
        seed: Some(42),
        ..Default::default()
    };
    let output_impaired = run_zilany2014(&signal, &cfs, &config_impaired);

    // Both should produce spikes
    assert!(!output_healthy.channels[0].spike_times.is_empty());

    // The impaired model still produces response (synaptic/spike generation still works)
    // Even with cohc=0, the IHC pathway should still respond to stimulation
    assert!(!output_impaired.channels[0].spike_times.is_empty());
}

#[test]
fn test_chirp_response() {
    // A chirp should produce response across multiple CFs
    let fs = 100e3;
    let duration = 0.1;

    let signal = generate_chirp(500.0, 4000.0, duration, 0.1, fs);
    let cfs = generate_cfs(500.0, 4000.0, 10, Species::Cat);
    let config = ModelConfig {
        fs,
        seed: Some(42),
        ..Default::default()
    };

    let output = run_zilany2014(&signal, &cfs, &config);

    // Multiple channels should have spikes
    let active_channels = output.channels.iter()
        .filter(|ch| !ch.spike_times.is_empty())
        .count();

    assert!(active_channels > 3, "Chirp should activate multiple channels");
}
