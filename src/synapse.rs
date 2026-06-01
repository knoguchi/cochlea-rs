//! Synapse model implementation.
//!
//! Ported from model_Synapse.c.
//! Implements double exponential adaptation and power-law adaptation.
//!
//! IMPORTANT: This implementation uses the O(n) approximate IIR version
//! of the power-law adaptation, fixing the O(n²) performance bug in the
//! "actual" implementation.

use rand::Rng;
use rand_distr::{Distribution, StandardNormal};
use rustfft::num_complex::Complex as FftComplex;
use rustfft::FftPlanner;

/// Auditory nerve fiber type.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AnfType {
    /// Low spontaneous rate (0.1 spikes/s)
    Lsr,
    /// Medium spontaneous rate (4 spikes/s)
    Msr,
    /// High spontaneous rate (100 spikes/s)
    Hsr,
}

impl AnfType {
    /// Get the spontaneous rate for this fiber type.
    pub fn spont_rate(&self) -> f64 {
        match self {
            AnfType::Lsr => 0.1,
            AnfType::Msr => 4.0,
            AnfType::Hsr => 100.0,
        }
    }
}

/// Power-law adaptation filter state.
/// Uses IIR approximation to avoid O(n²) complexity.
#[derive(Clone, Debug, Default)]
pub struct PowerLawState {
    // For slow adaptation (sout2 -> I2)
    pub n1: [f64; 3],
    pub n2: [f64; 3],
    pub n3: [f64; 3],
    // For fast adaptation (sout1 -> I1)
    pub m1: [f64; 3],
    pub m2: [f64; 3],
    pub m3: [f64; 3],
    pub m4: [f64; 3],
    pub m5: [f64; 3],
}

impl PowerLawState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Process fast adaptation (sout1 -> I1).
    /// Returns I1 at the current time step.
    #[inline]
    pub fn process_fast(&mut self, k: usize, sout1_k: f64, sout1_prev: &[f64]) -> f64 {
        if k == 0 {
            self.m1[0] = 0.2 * sout1_k;
            self.m2[0] = self.m1[0];
            self.m3[0] = self.m2[0];
            self.m4[0] = self.m3[0];
            self.m5[0] = self.m4[0];
        } else if k == 1 {
            let sout1_km1 = sout1_prev[0];
            self.m1[1] = 0.491115852967412 * self.m1[0]
                + 0.2 * (sout1_k - 0.173492003319319 * sout1_km1);
            self.m2[1] = 1.084520302502860 * self.m2[0] + self.m1[1] - 0.803462163297112 * self.m1[0];
            self.m3[1] = 1.588427084535629 * self.m3[0] + self.m2[1] - 1.416084732997016 * self.m2[0];
            self.m4[1] = 1.886287488516458 * self.m4[0] + self.m3[1] - 1.830362725074550 * self.m3[0];
            self.m5[1] = 1.989549282714008 * self.m5[0] + self.m4[1] - 1.983165053215032 * self.m4[0];

            // Shift history
            self.m1[2] = self.m1[1];
            self.m1[1] = self.m1[0];
            self.m2[2] = self.m2[1];
            self.m2[1] = self.m2[0];
            self.m3[2] = self.m3[1];
            self.m3[1] = self.m3[0];
            self.m4[2] = self.m4[1];
            self.m4[1] = self.m4[0];
            self.m5[2] = self.m5[1];
            self.m5[1] = self.m5[0];
        } else {
            let sout1_km1 = sout1_prev[0];
            let sout1_km2 = sout1_prev[1];

            let m1_new = 0.491115852967412 * self.m1[0] - 0.055050209956838 * self.m1[1]
                + 0.2 * (sout1_k - 0.173492003319319 * sout1_km1 + 0.000000172983796 * sout1_km2);
            let m2_new = 1.084520302502860 * self.m2[0] - 0.288760329320566 * self.m2[1]
                + m1_new - 0.803462163297112 * self.m1[0] + 0.154962026341513 * self.m1[1];
            let m3_new = 1.588427084535629 * self.m3[0] - 0.628138993662508 * self.m3[1]
                + m2_new - 1.416084732997016 * self.m2[0] + 0.496615555008723 * self.m2[1];
            let m4_new = 1.886287488516458 * self.m4[0] - 0.888972875389923 * self.m4[1]
                + m3_new - 1.830362725074550 * self.m3[0] + 0.836399964176882 * self.m3[1];
            let m5_new = 1.989549282714008 * self.m5[0] - 0.989558985673023 * self.m5[1]
                + m4_new - 1.983165053215032 * self.m4[0] + 0.983193027347456 * self.m4[1];

            // Shift history
            self.m1[1] = self.m1[0];
            self.m1[0] = m1_new;
            self.m2[1] = self.m2[0];
            self.m2[0] = m2_new;
            self.m3[1] = self.m3[0];
            self.m3[0] = m3_new;
            self.m4[1] = self.m4[0];
            self.m4[0] = m4_new;
            self.m5[1] = self.m5[0];
            self.m5[0] = m5_new;
        }

        self.m5[0]
    }

    /// Process slow adaptation (sout2 -> I2).
    /// Returns I2 at the current time step.
    #[inline]
    pub fn process_slow(&mut self, k: usize, sout2_k: f64, sout2_prev: &[f64]) -> f64 {
        if k == 0 {
            self.n1[0] = 1.0e-3 * sout2_k;
            self.n2[0] = self.n1[0];
            self.n3[0] = self.n2[0];
        } else if k == 1 {
            let sout2_km1 = sout2_prev[0];
            self.n1[1] = 1.992127932802320 * self.n1[0]
                + 1.0e-3 * (sout2_k - 0.994466986569624 * sout2_km1);
            self.n2[1] = 1.999195329360981 * self.n2[0] + self.n1[1] - 1.997855276593802 * self.n1[0];
            self.n3[1] = -0.798261718183851 * self.n3[0] + self.n2[1] + 0.798261718184977 * self.n2[0];

            // Shift history
            self.n1[2] = self.n1[1];
            self.n1[1] = self.n1[0];
            self.n2[2] = self.n2[1];
            self.n2[1] = self.n2[0];
            self.n3[2] = self.n3[1];
            self.n3[1] = self.n3[0];
        } else {
            let sout2_km1 = sout2_prev[0];
            let sout2_km2 = sout2_prev[1];

            let n1_new = 1.992127932802320 * self.n1[0] - 0.992140616993846 * self.n1[1]
                + 1.0e-3 * (sout2_k - 0.994466986569624 * sout2_km1 + 0.000000000002347 * sout2_km2);
            let n2_new = 1.999195329360981 * self.n2[0] - 0.999195402928777 * self.n2[1]
                + n1_new - 1.997855276593802 * self.n1[0] + 0.997855827934345 * self.n1[1];
            let n3_new = -0.798261718183851 * self.n3[0] - 0.199131619873480 * self.n3[1]
                + n2_new + 0.798261718184977 * self.n2[0] + 0.199131619874064 * self.n2[1];

            // Shift history
            self.n1[1] = self.n1[0];
            self.n1[0] = n1_new;
            self.n2[1] = self.n2[0];
            self.n2[0] = n2_new;
            self.n3[1] = self.n3[0];
            self.n3[0] = n3_new;
        }

        self.n3[0]
    }
}

/// Generate fractional Gaussian noise (fGn).
///
/// # Arguments
/// * `n` - Number of samples
/// * `tdres` - Time resolution (1/fs)
/// * `h_input` - Hurst parameter (0-2, typically 0.9)
/// * `mu` - Spontaneous rate (affects sigma)
/// * `rng` - Random number generator
pub fn ffgn<R: Rng>(n: usize, tdres: f64, h_input: f64, mu: f64, rng: &mut R) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }

    // Downsampling factor to match Scott Jackson's implementation (tau 1e-1)
    let resamp = (1e-1 / tdres).ceil() as usize;
    let mut n_internal = (n as f64 / resamp as f64).ceil() as usize + 1;
    if n_internal < 10 {
        n_internal = 10;
    }

    // Determine whether fGn or fBn should be produced
    let (h, fbn) = if h_input <= 1.0 {
        (h_input, false)
    } else {
        (h_input - 1.0, true)
    };

    let mut y = if (h - 0.5).abs() < 1e-10 {
        // H = 0.5 is white Gaussian noise
        (0..n_internal)
            .map(|_| StandardNormal.sample(rng))
            .collect::<Vec<f64>>()
    } else {
        // Generate fGn using FFT (Davies-Harte method)
        let nfft = (2 * (n_internal - 1)).next_power_of_two();
        let nfft_half = nfft / 2;

        // Create autocorrelation sequence (symmetric)
        let mut autocov: Vec<f64> = vec![0.0; nfft];
        for i in 0..=nfft_half {
            let ki = i as f64;
            autocov[i] = 0.5 * ((ki + 1.0).powf(2.0 * h) - 2.0 * ki.powf(2.0 * h)
                + (ki - 1.0).abs().powf(2.0 * h));
        }
        // Mirror for symmetric autocorrelation
        for i in (nfft_half + 1)..nfft {
            autocov[i] = autocov[nfft - i];
        }

        // Take FFT of autocorrelation to get eigenvalues
        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(nfft);

        let mut spectrum: Vec<FftComplex<f64>> = autocov
            .iter()
            .map(|&x| FftComplex::new(x, 0.0))
            .collect();
        fft.process(&mut spectrum);

        // Take sqrt of real parts (eigenvalues should be real and non-negative)
        let eigenval_sqrt: Vec<f64> = spectrum
            .iter()
            .map(|c| c.re.max(0.0).sqrt())
            .collect();

        // Generate random complex numbers weighted by sqrt(eigenvalues)
        let mut z: Vec<FftComplex<f64>> = eigenval_sqrt
            .iter()
            .map(|&ev| {
                let re: f64 = StandardNormal.sample(rng);
                let im: f64 = StandardNormal.sample(rng);
                FftComplex::new(ev * re, ev * im)
            })
            .collect();

        // Inverse FFT
        let ifft = planner.plan_fft_inverse(nfft);
        ifft.process(&mut z);

        // Normalize and take first n_internal samples (take real parts)
        let scale = 1.0 / (nfft as f64).sqrt();
        z.iter()
            .take(n_internal)
            .map(|c| c.re * scale)
            .collect()
    };

    // Convert fGn to fBn if necessary
    if fbn {
        let mut cumsum = 0.0;
        for yi in y.iter_mut() {
            cumsum += *yi;
            *yi = cumsum;
        }
    }

    // Resample to match the AN model sampling rate
    let y_resampled = resample(&y, resamp);

    // Determine sigma based on spontaneous rate
    let sigma = if mu < 0.5 {
        3.0
    } else if mu < 18.0 {
        30.0
    } else {
        200.0
    };

    // Scale and return
    y_resampled.iter().take(n).map(|&yi| yi * sigma).collect()
}

/// Simple linear interpolation resampling.
fn resample(signal: &[f64], factor: usize) -> Vec<f64> {
    let n = signal.len();
    let out_len = n * factor;
    let mut result = Vec::with_capacity(out_len);

    for i in 0..n - 1 {
        let start = signal[i];
        let end = signal[i + 1];
        for j in 0..factor {
            let t = j as f64 / factor as f64;
            result.push(start + t * (end - start));
        }
    }

    // Last sample
    for _ in 0..factor {
        result.push(signal[n - 1]);
    }

    result
}

/// Simple decimation (FIR lowpass + downsampling).
fn decimate(signal: &[f64], factor: usize) -> Vec<f64> {
    if factor <= 1 {
        return signal.to_vec();
    }

    // Simple averaging filter for decimation
    let mut result = Vec::with_capacity(signal.len() / factor + 1);
    let mut i = 0;

    while i < signal.len() {
        let end = (i + factor).min(signal.len());
        let sum: f64 = signal[i..end].iter().sum();
        result.push(sum / (end - i) as f64);
        i += factor;
    }

    result
}

/// Synapse processor state.
#[allow(dead_code)]
pub struct SynapseProcessor {
    /// Power-law filter state
    power_law: PowerLawState,
    /// Spontaneous rate
    spont: f64,
    /// Time resolution
    tdres: f64,
    /// Characteristic frequency
    cf: f64,
    /// Sampling frequency for power-law (typically 10 kHz)
    samp_freq: f64,
    /// Resampling factor
    resamp: usize,
    /// Double exponential parameters
    pi_max: f64,
    kslope: f64,
    synstrength: f64,
    synslope: f64,
    ci: f64,
    cl: f64,
    vi: f64,
    vl: f64,
    pg: f64,
    pl: f64,
    cg: f64,
    /// Power-law parameters
    alpha1: f64,
    beta1: f64,
    alpha2: f64,
    beta2: f64,
    /// History for power-law IIR
    sout1_history: [f64; 2],
    sout2_history: [f64; 2],
    /// Integral accumulators (for approximate implementation)
    i1: f64,
    i2: f64,
}

impl SynapseProcessor {
    /// Create a new synapse processor.
    ///
    /// # Arguments
    /// * `spont` - Spontaneous rate
    /// * `cf` - Characteristic frequency
    /// * `tdres` - Time resolution
    /// * `samp_freq` - Sampling frequency for power-law (typically 10 kHz)
    pub fn new(spont: f64, cf: f64, tdres: f64, samp_freq: f64) -> Self {
        // Calculate CF factor
        let cf_factor = if (spont - 100.0).abs() < 1e-10 {
            (10.0f64.powf(0.29 * cf / 1e3 + 0.7)).min(800.0)
        } else if (spont - 4.0).abs() < 1e-10 {
            (2.5e-4 * cf * 4.0 + 0.2).min(50.0)
        } else {
            // spont = 0.1
            (2.5e-4 * cf * 0.1 + 0.15).min(1.0)
        };

        let pi_max = 0.6;
        let kslope = (1.0 + 50.0) / (5.0 + 50.0) * cf_factor * 20.0 * pi_max;
        let ass = 800.0 * (1.0 + cf / 100e3);

        // Use approximate implementation spontaneous rate
        let asp = spont * 2.75;

        let tau_r = 2e-3;
        let tau_st = 60e-3;
        let ar_ast = 6.0;
        let pts = 3.0;

        let aon = pts * ass;
        let ar = (aon - ass) * ar_ast / (1.0 + ar_ast);
        let ast = aon - ass - ar;
        let prest = pi_max / aon * asp;
        let cg = (asp * (aon - asp)) / (aon * prest * (1.0 - asp / ass));
        let gamma1 = cg / asp;
        let gamma2 = cg / ass;
        let k1 = -1.0 / tau_r;
        let k2 = -1.0 / tau_st;

        let vi0 = (1.0 - pi_max / prest)
            / (gamma1 * (ar * (k1 - k2) / cg / pi_max + k2 / prest / gamma1 - k2 / pi_max / gamma2));
        let vi1 = (1.0 - pi_max / prest)
            / (gamma1 * (ast * (k2 - k1) / cg / pi_max + k1 / prest / gamma1 - k1 / pi_max / gamma2));
        let vi = (vi0 + vi1) / 2.0;
        let alpha = gamma2 / k1 / k2;
        let beta = -(k1 + k2) * alpha;
        let theta1 = alpha * pi_max / vi;
        let theta2 = vi / pi_max;
        let theta3 = gamma2 - 1.0 / pi_max;

        let pl = ((beta - theta2 * theta3) / theta1 - 1.0) * pi_max;
        let pg = 1.0 / (theta3 - 1.0 / pl);
        let vl = theta1 * pl * pg;
        let ci = asp / prest;
        let cl = ci * (prest + pl) / pl;

        let vsat = if kslope >= 0.0 { kslope + prest } else { prest };
        let tmpst = vsat / prest * 2.0_f64.ln();
        let synstrength = if tmpst < 400.0 {
            (tmpst.exp() - 1.0).ln()
        } else {
            tmpst
        };
        let synslope = prest / 2.0_f64.ln() * synstrength;

        // Power-law parameters
        let alpha1 = 2.5e-6 * 100e3;
        let beta1 = 5e-4;
        let alpha2 = 1e-2 * 100e3;
        let beta2 = 1e-1;

        Self {
            power_law: PowerLawState::new(),
            spont,
            tdres,
            cf,
            samp_freq,
            resamp: (1.0 / (tdres * samp_freq)).ceil() as usize,
            pi_max,
            kslope,
            synstrength,
            synslope,
            ci,
            cl,
            vi,
            vl,
            pg,
            pl,
            cg,
            alpha1,
            beta1,
            alpha2,
            beta2,
            sout1_history: [0.0; 2],
            sout2_history: [0.0; 2],
            i1: 0.0,
            i2: 0.0,
        }
    }

    pub fn reset(&mut self) {
        self.power_law.reset();
        self.sout1_history = [0.0; 2];
        self.sout2_history = [0.0; 2];
        self.i1 = 0.0;
        self.i2 = 0.0;

        // Reset CI and CL to initial values
        let asp = self.spont * 2.75;
        let ass = 800.0 * (1.0 + self.cf / 100e3);
        let aon = 3.0 * ass;
        let prest = self.pi_max / aon * asp;
        self.ci = asp / prest;
        self.cl = self.ci * (prest + self.pl) / self.pl;
    }
}

/// Run the synapse model on IHC output.
///
/// # Arguments
/// * `ihcout` - IHC output signal
/// * `tdres` - Time resolution (1/fs)
/// * `cf` - Characteristic frequency
/// * `anf_type` - Auditory nerve fiber type
/// * `use_ffgn` - Whether to use fractional Gaussian noise
///
/// # Returns
/// Synapse output (instantaneous firing rate)
pub fn run_synapse<R: Rng>(
    ihcout: &[f64],
    tdres: f64,
    cf: f64,
    anf_type: AnfType,
    use_ffgn: bool,
    rng: &mut R,
) -> Vec<f64> {
    let totalstim = ihcout.len();
    let spont = anf_type.spont_rate();
    let samp_freq = 10e3;
    let delaypoint = (7500.0 / (cf / 1e3)).floor() as usize;

    let processor = SynapseProcessor::new(spont, cf, tdres, samp_freq);

    // Generate random noise
    let resamp = processor.resamp;
    let noise_len = ((totalstim + 2 * delaypoint) as f64 * tdres * samp_freq).ceil() as usize;
    let rand_nums = if use_ffgn {
        ffgn(noise_len, 1.0 / samp_freq, 0.9, spont, rng)
    } else {
        vec![0.0; noise_len]
    };

    // Double exponential adaptation
    let mut expon_out = vec![0.0; totalstim];
    let mut ci = processor.ci;
    let mut cl = processor.cl;

    for (indx, &ihc) in ihcout.iter().enumerate() {
        let tmp = processor.synstrength * ihc;
        let tmp = if tmp < 400.0 {
            (1.0 + tmp.exp()).ln()
        } else {
            tmp
        };
        let ppi = processor.synslope / processor.synstrength * tmp;

        let ci_last = ci;
        ci += (tdres / processor.vi) * (-ppi * ci + processor.pl * (cl - ci));
        cl += (tdres / processor.vl) * (-processor.pl * (cl - ci_last) + processor.pg * (processor.cg - cl));

        if ci < 0.0 {
            let temp = 1.0 / processor.pg + 1.0 / processor.pl + 1.0 / ppi;
            ci = processor.cg / (ppi * temp);
            cl = ci * (ppi + processor.pl) / processor.pl;
        }

        expon_out[indx] = ci * ppi;
    }

    // Add delay padding for power-law
    let power_law_in_len = totalstim + 3 * delaypoint;
    let mut power_law_in = vec![0.0; power_law_in_len];
    for k in 0..delaypoint {
        power_law_in[k] = expon_out[0];
    }
    for k in delaypoint..(totalstim + delaypoint) {
        power_law_in[k] = expon_out[k - delaypoint];
    }
    for k in (totalstim + delaypoint)..power_law_in_len {
        power_law_in[k] = power_law_in[k - 1];
    }

    // Downsample for power-law processing
    let samp_ihc = decimate(&power_law_in, resamp);

    // Power-law adaptation (O(n) approximate implementation)
    let synapse_len = ((totalstim + 2 * delaypoint) as f64 * tdres * samp_freq).floor() as usize;
    let mut syn_samp_out = vec![0.0; synapse_len];

    let _binwidth = 1.0 / samp_freq;
    let alpha1 = processor.alpha1;
    let alpha2 = processor.alpha2;

    let mut power_law = PowerLawState::new();
    let mut sout1_hist = [0.0; 2];
    let mut sout2_hist = [0.0; 2];
    let mut i1 = 0.0;
    let mut i2 = 0.0;

    for k in 0..synapse_len.min(samp_ihc.len()).min(rand_nums.len()) {
        let samp = samp_ihc[k];
        let noise = rand_nums[k];

        let sout1 = (samp + noise - alpha1 * i1).max(0.0);
        let sout2 = (samp - alpha2 * i2).max(0.0);

        // Process through IIR filters (approximate power-law)
        i1 = power_law.process_fast(k, sout1, &sout1_hist);
        i2 = power_law.process_slow(k, sout2, &sout2_hist);

        // Update history
        sout1_hist[1] = sout1_hist[0];
        sout1_hist[0] = sout1;
        sout2_hist[1] = sout2_hist[0];
        sout2_hist[0] = sout2;

        syn_samp_out[k] = sout1 + sout2;
    }

    // Upsample back to original rate
    let mut tmp_syn = vec![0.0; totalstim + 2 * delaypoint];
    for z in 0..(synapse_len - 1).min(syn_samp_out.len() - 1) {
        let incr = (syn_samp_out[z + 1] - syn_samp_out[z]) / resamp as f64;
        for b in 0..resamp {
            let idx = z * resamp + b;
            if idx < tmp_syn.len() {
                tmp_syn[idx] = syn_samp_out[z] + b as f64 * incr;
            }
        }
    }

    // Extract output with delay correction
    let mut synout = vec![0.0; totalstim];
    for i in 0..totalstim {
        if i + delaypoint < tmp_syn.len() {
            synout[i] = tmp_syn[i + delaypoint];
        }
    }

    synout
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_ffgn() {
        let mut rng = StdRng::seed_from_u64(42);
        let noise = ffgn(1000, 1.0 / 100e3, 0.9, 100.0, &mut rng);

        assert_eq!(noise.len(), 1000);
        // Check that noise has reasonable variance
        let mean: f64 = noise.iter().sum::<f64>() / noise.len() as f64;
        let variance: f64 = noise.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / noise.len() as f64;
        assert!(variance > 0.0);
    }

    #[test]
    fn test_power_law_state() {
        let mut state = PowerLawState::new();
        let sout_hist = [0.0, 0.0];

        // Process a few samples
        for k in 0..10 {
            let _ = state.process_fast(k, 1.0, &sout_hist);
            let _ = state.process_slow(k, 1.0, &sout_hist);
        }
    }

    #[test]
    fn test_synapse() {
        let mut rng = StdRng::seed_from_u64(42);
        let fs = 100e3;
        let tdres = 1.0 / fs;
        let cf = 1000.0;

        // Create a simple IHC-like signal
        let n_samples = 1000;
        let ihcout: Vec<f64> = (0..n_samples)
            .map(|i| 0.5 + 0.5 * (2.0 * std::f64::consts::PI * 100.0 * i as f64 / fs).sin())
            .collect();

        let synout = run_synapse(&ihcout, tdres, cf, AnfType::Hsr, false, &mut rng);

        assert_eq!(synout.len(), n_samples);
        // Check that output is non-negative
        assert!(synout.iter().all(|&x| x >= 0.0));
    }
}
