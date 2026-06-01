//! IIR filter primitives for cochlear model.
//!
//! Provides state structures and processing functions for various filter types
//! used in the Zilany 2014 model.

use crate::complex::Complex;

/// Two pi constant.
pub const TWOPI: f64 = std::f64::consts::TAU;

/// Generic IIR filter state with configurable order.
#[derive(Clone, Debug)]
pub struct IirState<const ORDER: usize> {
    /// Input history (x[n-1], x[n-2], ...)
    pub x: [f64; ORDER],
    /// Output history (y[n-1], y[n-2], ...)
    pub y: [f64; ORDER],
}

impl<const ORDER: usize> Default for IirState<ORDER> {
    fn default() -> Self {
        Self {
            x: [0.0; ORDER],
            y: [0.0; ORDER],
        }
    }
}

impl<const ORDER: usize> IirState<ORDER> {
    /// Create a new zeroed filter state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset the filter state to zero.
    pub fn reset(&mut self) {
        self.x = [0.0; ORDER];
        self.y = [0.0; ORDER];
    }
}

/// Second-order section (biquad) filter coefficients.
#[derive(Clone, Debug)]
pub struct BiquadCoeffs {
    pub b0: f64,
    pub b1: f64,
    pub b2: f64,
    pub a1: f64,
    pub a2: f64,
}

/// Second-order section (biquad) filter state.
#[derive(Clone, Debug, Default)]
pub struct BiquadState {
    pub x1: f64,
    pub x2: f64,
    pub y1: f64,
    pub y2: f64,
}

impl BiquadState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Process a single sample through the biquad filter.
    #[inline]
    pub fn process(&mut self, x: f64, coeffs: &BiquadCoeffs) -> f64 {
        let y = coeffs.b0 * x + coeffs.b1 * self.x1 + coeffs.b2 * self.x2
            - coeffs.a1 * self.y1 - coeffs.a2 * self.y2;

        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;

        y
    }
}

/// Cascaded lowpass filter state (up to 8 sections).
#[derive(Clone, Debug)]
pub struct CascadeLowpassState {
    /// Filter values at each stage.
    pub stages: [f64; 8],
    /// Previous values at each stage.
    pub stages_prev: [f64; 8],
}

impl Default for CascadeLowpassState {
    fn default() -> Self {
        Self {
            stages: [0.0; 8],
            stages_prev: [0.0; 8],
        }
    }
}

impl CascadeLowpassState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        self.stages = [0.0; 8];
        self.stages_prev = [0.0; 8];
    }

    /// Process a sample through cascaded first-order lowpass sections.
    ///
    /// # Arguments
    /// * `x` - Input sample
    /// * `tdres` - Time resolution (1/fs)
    /// * `fc` - Cutoff frequency in Hz
    /// * `gain` - Input gain
    /// * `order` - Number of cascade stages (1-7)
    #[inline]
    pub fn process(&mut self, x: f64, tdres: f64, fc: f64, gain: f64, order: usize) -> f64 {
        let c = 2.0 / tdres;
        let c1lp = (c - TWOPI * fc) / (c + TWOPI * fc);
        let c2lp = TWOPI * fc / (TWOPI * fc + c);

        self.stages[0] = x * gain;

        for i in 0..order {
            self.stages[i + 1] = c1lp * self.stages_prev[i + 1]
                + c2lp * (self.stages[i] + self.stages_prev[i]);
        }

        // Update previous values
        for i in 0..=order {
            self.stages_prev[i] = self.stages[i];
        }

        self.stages[order]
    }
}

/// Complex gammatone filter state for wideband filter.
#[derive(Clone, Debug)]
pub struct WbGammatoneState {
    /// Phase accumulator.
    pub phase: f64,
    /// Current filter state for each order.
    pub gtf: [Complex; 4],
    /// Previous filter state for each order.
    pub gtf_prev: [Complex; 4],
}

impl Default for WbGammatoneState {
    fn default() -> Self {
        Self {
            phase: 0.0,
            gtf: [Complex::default(); 4],
            gtf_prev: [Complex::default(); 4],
        }
    }
}

impl WbGammatoneState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        self.phase = 0.0;
        self.gtf = [Complex::default(); 4];
        self.gtf_prev = [Complex::default(); 4];
    }

    /// Process a sample through the wideband gammatone filter.
    ///
    /// # Arguments
    /// * `x` - Input sample
    /// * `tdres` - Time resolution (1/fs)
    /// * `centerfreq` - Center frequency in Hz
    /// * `tau` - Time constant
    /// * `gain` - Filter gain
    /// * `order` - Filter order (typically 3)
    #[inline]
    pub fn process(
        &mut self,
        x: f64,
        tdres: f64,
        centerfreq: f64,
        tau: f64,
        gain: f64,
        order: usize,
    ) -> f64 {
        let delta_phase = -TWOPI * centerfreq * tdres;
        self.phase += delta_phase;

        let dtmp = tau * 2.0 / tdres;
        let c1lp = (dtmp - 1.0) / (dtmp + 1.0);
        let c2lp = 1.0 / (dtmp + 1.0);

        // Frequency shift input
        self.gtf[0] = Complex::exp_i(self.phase).scale(x);

        // IIR bilinear transformation LPF cascade
        for j in 1..=order {
            let sum = self.gtf[j - 1] + self.gtf_prev[j - 1];
            let scaled_sum = sum.scale(c2lp * gain);
            let feedback = self.gtf_prev[j].scale(c1lp);
            self.gtf[j] = scaled_sum + feedback;
        }

        // Frequency shift back
        let out = (Complex::exp_i(-self.phase) * self.gtf[order]).real();

        // Update previous state
        for i in 0..=order {
            self.gtf_prev[i] = self.gtf[i];
        }

        out
    }
}

/// Chirp filter state for C1 and C2 signal-path filters.
/// These are 10th order filters with 5 pole pairs.
#[derive(Clone, Debug)]
pub struct ChirpFilterState {
    /// Input history for each biquad section (6 sections, 3 samples each).
    pub input: [[f64; 4]; 12],
    /// Output history for each biquad section.
    pub output: [[f64; 4]; 12],
    /// Initial phase (computed once at n=0).
    pub init_phase: f64,
    /// Gain normalization factor.
    pub gain_norm: f64,
    /// Whether the filter has been initialized.
    pub initialized: bool,
}

impl Default for ChirpFilterState {
    fn default() -> Self {
        Self {
            input: [[0.0; 4]; 12],
            output: [[0.0; 4]; 12],
            init_phase: 0.0,
            gain_norm: 1.0,
            initialized: false,
        }
    }
}

impl ChirpFilterState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        self.input = [[0.0; 4]; 12];
        self.output = [[0.0; 4]; 12];
        self.init_phase = 0.0;
        self.gain_norm = 1.0;
        self.initialized = false;
    }
}

/// Calculate gain and group delay for the control-path wideband filter.
///
/// # Arguments
/// * `tdres` - Time resolution (1/fs)
/// * `centerfreq` - Center frequency in Hz
/// * `cf` - Characteristic frequency in Hz
/// * `tau` - Time constant
///
/// # Returns
/// Tuple of (gain, group_delay_samples)
#[inline]
pub fn gain_groupdelay(tdres: f64, centerfreq: f64, cf: f64, tau: f64) -> (f64, i32) {
    let tmpcos = (TWOPI * (centerfreq - cf) * tdres).cos();
    let dtmp2 = tau * 2.0 / tdres;
    let c1lp = (dtmp2 - 1.0) / (dtmp2 + 1.0);
    let c2lp = 1.0 / (dtmp2 + 1.0);

    let tmp1 = 1.0 + c1lp * c1lp - 2.0 * c1lp * tmpcos;
    let tmp2 = 2.0 * c2lp * c2lp * (1.0 + tmpcos);

    let wb_gain = (tmp1 / tmp2).sqrt();
    let grdelay = (0.5 - (c1lp * c1lp - c1lp * tmpcos) / (1.0 + c1lp * c1lp - 2.0 * c1lp * tmpcos)).floor() as i32;

    (wb_gain, grdelay)
}

/// Calculate the delay for cat species.
#[inline]
pub fn delay_cat(cf: f64) -> f64 {
    let a0 = 3.0;
    let a1 = 12.5;
    let x = 11.9 * (0.80 + cf / 456.0).log10();
    a0 * (-x / a1).exp() * 1e-3
}

/// Calculate the delay for human species (based on Harte et al., JASA 2009).
#[inline]
pub fn delay_human(cf: f64) -> f64 {
    let a = -0.37;
    let b = 11.09 / 2.0;
    b * (cf * 1e-3).powf(a) * 1e-3
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cascade_lowpass() {
        let mut state = CascadeLowpassState::new();
        let tdres = 1.0 / 100e3;
        let fc = 3000.0;

        // Process some samples
        for i in 0..100 {
            let x = if i < 50 { 1.0 } else { 0.0 };
            let _ = state.process(x, tdres, fc, 1.0, 7);
        }

        // Output should be between 0 and 1
        assert!(state.stages[7] >= 0.0);
        assert!(state.stages[7] <= 1.0);
    }

    #[test]
    fn test_delay_cat() {
        let delay = delay_cat(1000.0);
        assert!(delay > 0.0);
        assert!(delay < 0.01); // Should be a few ms
    }
}
