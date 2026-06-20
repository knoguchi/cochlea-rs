//! Place coding demo — Rust counterpart of `cochlea_demo.py`.
//!
//! Reproduces three classical observations using the Zilany 2014 model:
//!   1) Different stimulus frequencies excite different cochlear places.
//!   2) The neural firing rate is much slower than the basilar-membrane oscillation.
//!   3) Sweeping the stimulus frequency monotonically shifts the active place.
//!
//! Run with:
//!   cargo run --release --example place_code --no-default-features
//!
//! Writes three CSV files and one PNG (current working directory):
//!   place_code.csv  — CF (Hz), mean rate for low tone, mean rate for high tone
//!   slow_rate.csv   — time (ms), normalized IHC output, normalized synapse rate
//!   sweep.csv       — stimulus freq (Hz), peak-CF (Hz)
//!   place_code.png  — three-panel summary plot

use cochlea::{generate_cfs, run_zilany2014, AnfType, ModelConfig, Species};
use plotters::prelude::*;
use rustfft::{num_complex::Complex, FftPlanner};
use std::error::Error;
use std::f64::consts::PI;
use std::fs::File;
use std::io::{BufWriter, Write};

fn tone(freq: f64, dur: f64, fs: f64, amp_pa: f64) -> Vec<f64> {
    let n = (dur * fs) as usize;
    (0..n)
        .map(|i| amp_pa * (2.0 * PI * freq * i as f64 / fs).sin())
        .collect()
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn argmax(xs: &[f64]) -> usize {
    xs.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

fn nearest_cf_idx(cfs: &[f64], target: f64) -> usize {
    cfs.iter()
        .enumerate()
        .min_by(|a, b| (a.1 - target).abs().partial_cmp(&(b.1 - target).abs()).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

/// Compute DC (mean), AC RMS, and the dominant AC frequency of a steady-state
/// segment (skipping `skip` samples for onset/adaptation transient).
struct AcDcStats {
    dc_mean: f64,
    ac_rms: f64,
    dom_freq: f64,
}

fn ac_dc_stats(rate: &[f64], fs: f64, skip: usize) -> AcDcStats {
    let seg = &rate[skip..];
    let mu = mean(seg);
    let segf: Vec<f64> = seg.iter().map(|x| x - mu).collect();
    let ac_rms = (segf.iter().map(|x| x * x).sum::<f64>() / segf.len() as f64).sqrt();
    let dom = dominant_freq(&segf, fs);
    AcDcStats {
        dc_mean: mu,
        ac_rms,
        dom_freq: dom,
    }
}

fn dominant_freq(x: &[f64], fs: f64) -> f64 {
    let n = x.len();
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);
    let mut buf: Vec<Complex<f64>> = x.iter().map(|&v| Complex::new(v, 0.0)).collect();
    fft.process(&mut buf);
    let half = n / 2;
    let mut best_i = 1usize;
    let mut best_mag = 0.0f64;
    for i in 1..half {
        let m = buf[i].norm();
        if m > best_mag {
            best_mag = m;
            best_i = i;
        }
    }
    best_i as f64 * fs / n as f64
}

fn main() -> Result<(), Box<dyn Error>> {
    let fs = 100_000.0;
    let duration = 0.2;
    let amp_pa = 0.02; // ~60 dB SPL re 20 µPa
    let n_ch = 40;
    let cfs = generate_cfs(125.0, 8000.0, n_ch, Species::Human);

    let config = ModelConfig {
        fs,
        species: Species::Human,
        anf_type: AnfType::Hsr,
        use_ffgn: false,
        seed: Some(42),
        ..Default::default()
    };

    let skip = (0.05 * fs) as usize; // drop onset/adaptation transient

    // ---- 1) place coding: low vs high tone ----
    let f_low = 200.0;
    let f_high = 4000.0;

    println!("Running {} Hz tone...", f_low as i32);
    let out_low = run_zilany2014(&tone(f_low, duration, fs, amp_pa), &cfs, &config);
    println!("Running {} Hz tone...", f_high as i32);
    let out_high = run_zilany2014(&tone(f_high, duration, fs, amp_pa), &cfs, &config);

    let rate_low: Vec<f64> = out_low
        .channels
        .iter()
        .map(|ch| mean(&ch.synapse_out[skip..]))
        .collect();
    let rate_high: Vec<f64> = out_high
        .channels
        .iter()
        .map(|ch| mean(&ch.synapse_out[skip..]))
        .collect();
    let peak_low = cfs[argmax(&rate_low)];
    let peak_high = cfs[argmax(&rate_high)];

    {
        let mut f = BufWriter::new(File::create("place_code.csv")?);
        writeln!(f, "cf_hz,rate_low_sps,rate_high_sps")?;
        for i in 0..n_ch {
            writeln!(f, "{:.3},{:.6},{:.6}", cfs[i], rate_low[i], rate_high[i])?;
        }
    }

    // ---- 2) phase locking vs. place coding ----
    //
    // Note: a literal "the firing rate is much slower than the stimulus" is only
    // true at high CFs.  Real auditory nerve fibres phase-lock up to ~3-4 kHz, so
    // at CF = 4 kHz the firing rate still oscillates at the stimulus frequency.
    // We illustrate both regimes: 4 kHz (phase locking preserved) and 8 kHz
    // (phase locking has decayed; only place coding remains).
    let ch_4k = nearest_cf_idx(&cfs, 4000.0);
    let ch_8k = nearest_cf_idx(&cfs, 8000.0);

    let sig_8k = tone(8000.0, duration, fs, amp_pa);
    println!("Running 8000 Hz tone for phase-locking comparison...");
    let out_8k = run_zilany2014(&sig_8k, &cfs, &config);

    let stats_4k = ac_dc_stats(&out_high.channels[ch_4k].synapse_out, fs, skip);
    let stats_8k = ac_dc_stats(&out_8k.channels[ch_8k].synapse_out, fs, skip);

    // CSV of IHC vs synapse waveforms at the 4 kHz channel (5 ms window)
    let ihc_h = &out_high.channels[ch_4k].ihc_out;
    let syn_h = &out_high.channels[ch_4k].synapse_out;
    let t_start = (0.100 * fs) as usize;
    let t_end = (0.105 * fs) as usize;
    let ihc_win = &ihc_h[t_start..t_end];
    let syn_win = &syn_h[t_start..t_end];
    let ihc_max = ihc_win
        .iter()
        .map(|x| x.abs())
        .fold(0.0f64, f64::max)
        .max(1e-12);
    let syn_max = syn_win
        .iter()
        .cloned()
        .fold(0.0f64, f64::max)
        .max(1e-12);

    {
        let mut f = BufWriter::new(File::create("slow_rate.csv")?);
        writeln!(f, "time_ms,ihc_norm,rate_norm")?;
        for (i, (&a, &b)) in ihc_win.iter().zip(syn_win.iter()).enumerate() {
            let t_ms = (t_start + i) as f64 / fs * 1000.0;
            writeln!(f, "{:.4},{:.6},{:.6}", t_ms, a / ihc_max, b / syn_max)?;
        }
    }

    // ---- 3) sweep ----
    let sweep_n = 16;
    let f_min: f64 = 200.0;
    let f_max: f64 = 7000.0;
    let sweep: Vec<f64> = (0..sweep_n)
        .map(|i| {
            let frac = i as f64 / (sweep_n - 1) as f64;
            f_min * (f_max / f_min).powf(frac as f64)
        })
        .collect();

    println!("Running sweep ({} tones)...", sweep_n);
    let peak_cfs: Vec<f64> = sweep
        .iter()
        .map(|&f| {
            let s = tone(f, duration, fs, amp_pa);
            let out = run_zilany2014(&s, &cfs, &config);
            let r: Vec<f64> = out
                .channels
                .iter()
                .map(|ch| mean(&ch.synapse_out[skip..]))
                .collect();
            cfs[argmax(&r)]
        })
        .collect();

    {
        let mut f = BufWriter::new(File::create("sweep.csv")?);
        writeln!(f, "stim_hz,peak_cf_hz")?;
        for i in 0..sweep_n {
            writeln!(f, "{:.3},{:.3}", sweep[i], peak_cfs[i])?;
        }
    }

    println!();
    println!("1) place coding");
    println!("   low tone  {:>5.0} Hz -> peak-CF = {:>5.0} Hz", f_low, peak_low);
    println!("   high tone {:>5.0} Hz -> peak-CF = {:>5.0} Hz", f_high, peak_high);
    println!();
    println!("2) phase locking vs. place coding");
    println!(
        "   CF = {:>5.0} Hz, 4 kHz tone : DC = {:>6.1} sp/s, AC RMS = {:>6.1}, dominant AC ≈ {:>5.0} Hz",
        cfs[ch_4k], stats_4k.dc_mean, stats_4k.ac_rms, stats_4k.dom_freq
    );
    println!(
        "   CF = {:>5.0} Hz, 8 kHz tone : DC = {:>6.1} sp/s, AC RMS = {:>6.1}, dominant AC ≈ {:>5.0} Hz",
        cfs[ch_8k], stats_8k.dc_mean, stats_8k.ac_rms, stats_8k.dom_freq
    );
    println!(
        "   -> at 4 kHz the firing rate still phase-locks to the stimulus;"
    );
    println!(
        "      at 8 kHz the AC/DC ratio collapses ({:.2} -> {:.2}) — only place coding remains.",
        stats_4k.ac_rms / stats_4k.dc_mean.max(1e-9),
        stats_8k.ac_rms / stats_8k.dc_mean.max(1e-9)
    );
    println!();
    println!("3) sweep tracked from {:.0} Hz to {:.0} Hz", sweep[0], *sweep.last().unwrap());
    println!(
        "   peak-CF from {:.0} Hz to {:.0} Hz",
        peak_cfs[0],
        peak_cfs.last().unwrap()
    );
    // ---- PNG plot ----
    plot_png(
        "place_code.png",
        &cfs,
        &rate_low,
        &rate_high,
        f_low,
        f_high,
        cfs[ch_4k],
        t_start,
        fs,
        ihc_win,
        syn_win,
        ihc_max,
        syn_max,
        &sweep,
        &peak_cfs,
    )?;

    println!();
    println!("Wrote place_code.csv, slow_rate.csv, sweep.csv, place_code.png");

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn plot_png(
    path: &str,
    cfs: &[f64],
    rate_low: &[f64],
    rate_high: &[f64],
    f_low: f64,
    f_high: f64,
    cf_4k: f64,
    t_start: usize,
    fs: f64,
    ihc_win: &[f64],
    syn_win: &[f64],
    ihc_max: f64,
    syn_max: f64,
    sweep: &[f64],
    peak_cfs: &[f64],
) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(path, (1700, 520)).into_drawing_area();
    root.fill(&WHITE)?;
    let panels = root.split_evenly((1, 3));

    let cf_min = cfs.first().copied().unwrap_or(100.0);
    let cf_max = cfs.last().copied().unwrap_or(10_000.0);

    // ---- panel 1: tuning curves ----
    let max_rate = rate_low
        .iter()
        .chain(rate_high.iter())
        .cloned()
        .fold(0.0f64, f64::max)
        * 1.1;
    let mut p1 = ChartBuilder::on(&panels[0])
        .caption("1) Place coding", ("sans-serif", 22))
        .margin(15)
        .x_label_area_size(45)
        .y_label_area_size(60)
        .build_cartesian_2d((cf_min..cf_max).log_scale(), 0f64..max_rate)?;
    p1.configure_mesh()
        .x_desc("CF (Hz) — place on basilar membrane")
        .y_desc("Mean firing rate (sp/s)")
        .draw()?;
    p1.draw_series(LineSeries::new(
        cfs.iter().zip(rate_low.iter()).map(|(&c, &r)| (c, r)),
        BLUE.stroke_width(2),
    ))?
    .label(format!("{:.0} Hz tone", f_low))
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 18, y)], BLUE.stroke_width(2)));
    p1.draw_series(LineSeries::new(
        cfs.iter().zip(rate_high.iter()).map(|(&c, &r)| (c, r)),
        RED.stroke_width(2),
    ))?
    .label(format!("{:.0} Hz tone", f_high))
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 18, y)], RED.stroke_width(2)));
    p1.configure_series_labels()
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK)
        .draw()?;

    // ---- panel 2: IHC vs firing rate at the 4 kHz channel ----
    let t_data: Vec<(f64, f64, f64)> = (0..ihc_win.len())
        .map(|i| {
            let t_ms = (t_start + i) as f64 / fs * 1000.0;
            (t_ms, ihc_win[i] / ihc_max, syn_win[i] / syn_max)
        })
        .collect();
    let t_min = t_data.first().unwrap().0;
    let t_max = t_data.last().unwrap().0;
    let mut p2 = ChartBuilder::on(&panels[1])
        .caption(
            format!("2) Phase locking at CF ≈ {:.0} Hz, 4 kHz tone", cf_4k),
            ("sans-serif", 22),
        )
        .margin(15)
        .x_label_area_size(45)
        .y_label_area_size(60)
        .build_cartesian_2d(t_min..t_max, -1.1f64..1.1f64)?;
    p2.configure_mesh()
        .x_desc("Time (ms)")
        .y_desc("Normalized")
        .draw()?;
    let gray = RGBColor(120, 120, 120);
    p2.draw_series(LineSeries::new(
        t_data.iter().map(|&(t, a, _)| (t, a)),
        gray.stroke_width(1),
    ))?
    .label("IHC potential")
    .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 18, y)], gray.stroke_width(1)));
    p2.draw_series(LineSeries::new(
        t_data.iter().map(|&(t, _, b)| (t, b)),
        RED.stroke_width(2),
    ))?
    .label("Firing rate")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 18, y)], RED.stroke_width(2)));
    p2.configure_series_labels()
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK)
        .draw()?;

    // ---- panel 3: sweep tracking ----
    let mut p3 = ChartBuilder::on(&panels[2])
        .caption("3) Active place tracks stimulus", ("sans-serif", 22))
        .margin(15)
        .x_label_area_size(45)
        .y_label_area_size(60)
        .build_cartesian_2d((100f64..10_000f64).log_scale(), (100f64..10_000f64).log_scale())?;
    p3.configure_mesh()
        .x_desc("Stimulus frequency (Hz)")
        .y_desc("Peak-CF of most-active place (Hz)")
        .draw()?;
    p3.draw_series(LineSeries::new(
        [(100.0, 100.0), (10_000.0, 10_000.0)],
        BLACK.stroke_width(1),
    ))?
    .label("y = x")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 18, y)], BLACK.stroke_width(1)));
    p3.draw_series(LineSeries::new(
        sweep.iter().zip(peak_cfs.iter()).map(|(&s, &p)| (s, p)),
        RED.stroke_width(2),
    ))?
    .label("model")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 18, y)], RED.stroke_width(2)));
    p3.draw_series(
        sweep
            .iter()
            .zip(peak_cfs.iter())
            .map(|(&s, &p)| Circle::new((s, p), 4, RED.filled())),
    )?;
    p3.configure_series_labels()
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}
