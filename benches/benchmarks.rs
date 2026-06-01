//! Benchmarks for the Zilany 2014 cochlear model.

use cochlea::*;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::f64::consts::PI;

fn generate_tone(frequency: f64, duration: f64, amplitude: f64, fs: f64) -> Vec<f64> {
    let n_samples = (duration * fs) as usize;
    (0..n_samples)
        .map(|i| amplitude * (2.0 * PI * frequency * i as f64 / fs).sin())
        .collect()
}

fn bench_ihc(c: &mut Criterion) {
    let fs = 100e3;
    let cf = 1000.0;
    let durations = [0.01, 0.1, 1.0];

    let mut group = c.benchmark_group("IHC");

    for duration in durations {
        let signal = generate_tone(cf, duration, 0.1, fs);

        group.bench_with_input(
            BenchmarkId::new("duration_s", duration),
            &signal,
            |b, signal| {
                b.iter(|| {
                    cochlea::ihc::run_ihc(
                        black_box(signal),
                        black_box(cf),
                        black_box(fs),
                        "cat",
                        1.0,
                        1.0,
                    )
                })
            },
        );
    }

    group.finish();
}

fn bench_synapse(c: &mut Criterion) {
    let fs = 100e3;
    let cf = 1000.0;
    let durations = [0.01, 0.1, 1.0];

    let mut group = c.benchmark_group("Synapse");

    for duration in durations {
        // Pre-compute IHC output
        let signal = generate_tone(cf, duration, 0.1, fs);
        let ihc_out = cochlea::ihc::run_ihc(&signal, cf, fs, "cat", 1.0, 1.0);

        group.bench_with_input(
            BenchmarkId::new("duration_s", duration),
            &ihc_out,
            |b, ihc_out| {
                b.iter(|| {
                    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
                    cochlea::synapse::run_synapse(
                        black_box(ihc_out),
                        black_box(1.0 / fs),
                        black_box(cf),
                        cochlea::AnfType::Hsr,
                        true,
                        &mut rng,
                    )
                })
            },
        );
    }

    group.finish();
}

fn bench_full_model(c: &mut Criterion) {
    let fs = 100e3;
    let cf = 1000.0;
    let durations = [0.01, 0.1, 1.0];

    let mut group = c.benchmark_group("Full Model (1 CF)");
    group.sample_size(20); // Reduce sample size for longer benchmarks

    for duration in durations {
        let signal = generate_tone(cf, duration, 0.1, fs);
        let cfs = vec![cf];
        let config = ModelConfig {
            fs,
            seed: Some(42),
            ..Default::default()
        };

        group.bench_with_input(
            BenchmarkId::new("duration_s", duration),
            &(&signal, &cfs, &config),
            |b, (signal, cfs, config)| {
                b.iter(|| run_zilany2014(black_box(signal), black_box(cfs), black_box(config)))
            },
        );
    }

    group.finish();
}

fn bench_multi_channel(c: &mut Criterion) {
    let fs = 100e3;
    let duration = 0.1;
    let signal = generate_tone(1000.0, duration, 0.1, fs);

    let mut group = c.benchmark_group("Full Model (100ms)");
    group.sample_size(10);

    let channel_counts = [1, 10, 30, 100];

    for n_channels in channel_counts {
        let cfs = generate_cfs(125.0, 8000.0, n_channels, Species::Cat);
        let config = ModelConfig {
            fs,
            seed: Some(42),
            ..Default::default()
        };

        group.bench_with_input(
            BenchmarkId::new("channels", n_channels),
            &(&signal, &cfs, &config),
            |b, (signal, cfs, config)| {
                b.iter(|| run_zilany2014(black_box(signal), black_box(cfs), black_box(config)))
            },
        );
    }

    group.finish();
}

fn bench_1s_100cfs(c: &mut Criterion) {
    // The target benchmark: 1s audio, 100 CFs
    let fs = 100e3;
    let duration = 1.0;
    let signal = generate_tone(1000.0, duration, 0.1, fs);
    let cfs = generate_cfs(125.0, 8000.0, 100, Species::Cat);
    let config = ModelConfig {
        fs,
        seed: Some(42),
        ..Default::default()
    };

    let mut group = c.benchmark_group("Target: 1s, 100 CFs");
    group.sample_size(10);

    group.bench_function("run", |b| {
        b.iter(|| run_zilany2014(black_box(&signal), black_box(&cfs), black_box(&config)))
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_ihc,
    bench_synapse,
    bench_full_model,
    bench_multi_channel,
    bench_1s_100cfs,
);

criterion_main!(benches);
