//! Middle ear filter implementation.
//!
//! Ported from model_IHC.c lines 196-241.
//! Implements a 3-stage cascade IIR filter using bilinear transform.

use crate::filters::TWOPI;

/// Species-specific middle ear filter coefficients.
#[derive(Clone, Debug)]
pub struct MiddleEarCoeffs {
    /// Stage 1 coefficients
    pub m11: f64,
    pub m12: f64,
    pub m13: f64,
    pub m14: f64,
    pub m15: f64,
    pub m16: f64,
    /// Stage 2 coefficients
    pub m21: f64,
    pub m22: f64,
    pub m23: f64,
    pub m24: f64,
    pub m25: f64,
    pub m26: f64,
    /// Stage 3 coefficients
    pub m31: f64,
    pub m32: f64,
    pub m33: f64,
    pub m34: f64,
    pub m35: f64,
    pub m36: f64,
    /// Gain normalization
    pub megainmax: f64,
    /// Species flag (1=cat, 2+=human)
    pub species: i32,
}

impl MiddleEarCoeffs {
    /// Calculate middle ear filter coefficients for given species and sample rate.
    ///
    /// # Arguments
    /// * `species` - 1 for cat, 2 for human (Shera), 3 for human (Glasberg & Moore)
    /// * `tdres` - Time resolution (1/fs)
    pub fn new(species: i32, tdres: f64) -> Self {
        // Prewarping frequency 1 kHz
        let fp = 1e3;
        let c = TWOPI * fp / (TWOPI / 2.0 * fp * tdres).tan();

        if species == 1 {
            // Cat middle-ear filter - simplified version from Bruce et al. (JASA 2003)
            Self {
                m11: c / (c + 693.48),
                m12: (693.48 - c) / c,
                m13: 0.0,
                m14: 1.0,
                m15: -1.0,
                m16: 0.0,
                m21: 1.0 / (c.powi(2) + 11053.0 * c + 1.163e8),
                m22: -2.0 * c.powi(2) + 2.326e8,
                m23: c.powi(2) - 11053.0 * c + 1.163e8,
                m24: c.powi(2) + 1356.3 * c + 7.4417e8,
                m25: -2.0 * c.powi(2) + 14.8834e8,
                m26: c.powi(2) - 1356.3 * c + 7.4417e8,
                m31: 1.0 / (c.powi(2) + 4620.0 * c + 909059944.0),
                m32: -2.0 * c.powi(2) + 2.0 * 909059944.0,
                m33: c.powi(2) - 4620.0 * c + 909059944.0,
                m34: 5.7585e5 * c + 7.1665e7,
                m35: 14.333e7,
                m36: 7.1665e7 - 5.7585e5 * c,
                megainmax: 41.1405,
                species,
            }
        } else {
            // Human middle-ear filter - based on Pascal et al. (JASA 1998)
            Self {
                m11: 1.0 / (c.powi(2) + 5.9761e3 * c + 2.5255e7),
                m12: -2.0 * c.powi(2) + 2.0 * 2.5255e7,
                m13: c.powi(2) - 5.9761e3 * c + 2.5255e7,
                m14: c.powi(2) + 5.6665e3 * c,
                m15: -2.0 * c.powi(2),
                m16: c.powi(2) - 5.6665e3 * c,
                m21: 1.0 / (c.powi(2) + 6.4255e3 * c + 1.3975e8),
                m22: -2.0 * c.powi(2) + 2.0 * 1.3975e8,
                m23: c.powi(2) - 6.4255e3 * c + 1.3975e8,
                m24: c.powi(2) + 5.8934e3 * c + 1.7926e8,
                m25: -2.0 * c.powi(2) + 2.0 * 1.7926e8,
                m26: c.powi(2) - 5.8934e3 * c + 1.7926e8,
                m31: 1.0 / (c.powi(2) + 2.4891e4 * c + 1.27e9),
                m32: -2.0 * c.powi(2) + 2.0 * 1.27e9,
                m33: c.powi(2) - 2.4891e4 * c + 1.27e9,
                m34: 3.1137e3 * c + 6.9768e8,
                m35: 2.0 * 6.9768e8,
                m36: -3.1137e3 * c + 6.9768e8,
                megainmax: 2.0,
                species,
            }
        }
    }
}

/// Middle ear filter state.
#[derive(Clone, Debug)]
pub struct MiddleEarState {
    /// Stage 1 output history (needs 2 samples)
    pub mey1: [f64; 3],
    /// Stage 2 output history
    pub mey2: [f64; 3],
    /// Stage 3 output history
    pub mey3: [f64; 3],
    /// Input history (for human filter)
    pub px: [f64; 3],
    /// Current sample index
    pub n: usize,
}

impl Default for MiddleEarState {
    fn default() -> Self {
        Self {
            mey1: [0.0; 3],
            mey2: [0.0; 3],
            mey3: [0.0; 3],
            px: [0.0; 3],
            n: 0,
        }
    }
}

impl MiddleEarState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Process a single sample through the middle ear filter.
    ///
    /// # Arguments
    /// * `x` - Input sample (pressure)
    /// * `coeffs` - Filter coefficients
    ///
    /// # Returns
    /// Filtered output sample
    #[inline]
    pub fn process(&mut self, x: f64, coeffs: &MiddleEarCoeffs) -> f64 {
        let n = self.n;

        // Shift input history
        self.px[2] = self.px[1];
        self.px[1] = self.px[0];
        self.px[0] = x;

        let meout = if n == 0 {
            // First sample
            self.mey1[0] = if coeffs.species > 1 {
                coeffs.m11 * coeffs.m14 * x
            } else {
                coeffs.m11 * x
            };
            self.mey2[0] = self.mey1[0] * coeffs.m24 * coeffs.m21;
            self.mey3[0] = self.mey2[0] * coeffs.m34 * coeffs.m31;
            self.mey3[0] / coeffs.megainmax
        } else if n == 1 {
            // Second sample
            self.mey1[1] = if coeffs.species > 1 {
                coeffs.m11 * (-coeffs.m12 * self.mey1[0]
                    + coeffs.m14 * self.px[0]
                    + coeffs.m15 * self.px[1])
            } else {
                coeffs.m11 * (-coeffs.m12 * self.mey1[0] + self.px[0] - self.px[1])
            };
            self.mey2[1] = coeffs.m21
                * (-coeffs.m22 * self.mey2[0]
                    + coeffs.m24 * self.mey1[1]
                    + coeffs.m25 * self.mey1[0]);
            self.mey3[1] = coeffs.m31
                * (-coeffs.m32 * self.mey3[0]
                    + coeffs.m34 * self.mey2[1]
                    + coeffs.m35 * self.mey2[0]);
            self.mey3[1] / coeffs.megainmax
        } else {
            // General case (n >= 2)
            // Rotate history
            self.mey1[2] = self.mey1[1];
            self.mey1[1] = self.mey1[0];
            self.mey2[2] = self.mey2[1];
            self.mey2[1] = self.mey2[0];
            self.mey3[2] = self.mey3[1];
            self.mey3[1] = self.mey3[0];

            self.mey1[0] = if coeffs.species > 1 {
                coeffs.m11
                    * (-coeffs.m12 * self.mey1[1]
                        - coeffs.m13 * self.mey1[2]
                        + coeffs.m14 * self.px[0]
                        + coeffs.m15 * self.px[1]
                        + coeffs.m16 * self.px[2])
            } else {
                coeffs.m11 * (-coeffs.m12 * self.mey1[1] + self.px[0] - self.px[1])
            };

            self.mey2[0] = coeffs.m21
                * (-coeffs.m22 * self.mey2[1]
                    - coeffs.m23 * self.mey2[2]
                    + coeffs.m24 * self.mey1[0]
                    + coeffs.m25 * self.mey1[1]
                    + coeffs.m26 * self.mey1[2]);

            self.mey3[0] = coeffs.m31
                * (-coeffs.m32 * self.mey3[1]
                    - coeffs.m33 * self.mey3[2]
                    + coeffs.m34 * self.mey2[0]
                    + coeffs.m35 * self.mey2[1]
                    + coeffs.m36 * self.mey2[2]);

            self.mey3[0] / coeffs.megainmax
        };

        self.n += 1;
        meout
    }
}

/// Convenience function to process entire signal through middle ear filter.
pub fn process_middle_ear(signal: &[f64], species: i32, tdres: f64) -> Vec<f64> {
    let coeffs = MiddleEarCoeffs::new(species, tdres);
    let mut state = MiddleEarState::new();

    signal.iter().map(|&x| state.process(x, &coeffs)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_middle_ear_cat() {
        let tdres = 1.0 / 100e3;
        let coeffs = MiddleEarCoeffs::new(1, tdres);
        let mut state = MiddleEarState::new();

        // Process an impulse
        let impulse_response: Vec<f64> = (0..1000)
            .map(|i| {
                let x = if i == 0 { 1.0 } else { 0.0 };
                state.process(x, &coeffs)
            })
            .collect();

        // Check that we get some response
        let max_response = impulse_response.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        assert!(max_response > 0.0);
    }

    #[test]
    fn test_middle_ear_human() {
        let tdres = 1.0 / 100e3;
        let coeffs = MiddleEarCoeffs::new(2, tdres);
        let mut state = MiddleEarState::new();

        // Process an impulse
        let impulse_response: Vec<f64> = (0..1000)
            .map(|i| {
                let x = if i == 0 { 1.0 } else { 0.0 };
                state.process(x, &coeffs)
            })
            .collect();

        // Check that we get some response
        let max_response = impulse_response.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        assert!(max_response > 0.0);
    }
}
