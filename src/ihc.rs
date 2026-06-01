//! Inner Hair Cell (IHC) model implementation.
//!
//! Ported from model_IHC.c.
//! Includes C1/C2 chirp filters, wideband gammatone, OHC nonlinearity, and IHC transduction.

use crate::complex::Complex;
use crate::filters::{
    delay_cat, gain_groupdelay, CascadeLowpassState, ChirpFilterState, WbGammatoneState, TWOPI,
};
use crate::middle_ear::{MiddleEarCoeffs, MiddleEarState};

/// Species constants.
pub const SPECIES_CAT: i32 = 1;
pub const SPECIES_HUMAN_SHERA: i32 = 2;
pub const SPECIES_HUMAN_GLASBERG: i32 = 3;

/// Calculate center frequency for control-path wideband filter.
///
/// Based on Greenwood (JASA 1990).
fn calculate_centerfreq(cf: f64, species: i32) -> f64 {
    if species == SPECIES_CAT {
        // Cat frequency shift corresponding to 1.2 mm
        let bmplace = 11.9 * (0.80 + cf / 456.0).log10();
        456.0 * (10.0f64.powf((bmplace + 1.2) / 11.9) - 0.80)
    } else {
        // Human frequency shift corresponding to 1.2 mm
        let bmplace = (35.0 / 2.1) * (1.0 + cf / 165.4).log10();
        165.4 * (10.0f64.powf((bmplace + 1.2) / (35.0 / 2.1)) - 1.0)
    }
}

/// Calculate gain based on CF.
fn calculate_gain(cf: f64) -> f64 {
    let gain = 52.0 / 2.0 * ((2.2 * (cf / 0.6e3).log10() + 0.15).tanh() + 1.0);
    gain.clamp(15.0, 60.0)
}

/// Get tau parameters for wideband filter.
fn get_tauwb(cf: f64, species: i32, order: i32) -> (f64, f64) {
    let gain = calculate_gain(cf);
    let ratio = 10.0f64.powf(-gain / (20.0 * order as f64));

    let q10 = match species {
        SPECIES_CAT => 10.0f64.powf(0.4708 * (cf / 1e3).log10() + 0.4664),
        SPECIES_HUMAN_SHERA => (cf / 1000.0).powf(0.3) * 12.7 * 0.505 + 0.2085,
        _ => cf / 24.7 / (4.37 * (cf / 1000.0) + 1.0) * 0.505 + 0.2085, // Glasberg & Moore
    };

    let bw = cf / q10;
    let taumax = 2.0 / (TWOPI * bw);
    let taumin = taumax * ratio;

    (taumax, taumin)
}

/// Get tau parameters for BM filter.
fn get_taubm(cf: f64, taumax: f64) -> (f64, f64, f64) {
    let gain = calculate_gain(cf);
    let bwfactor = 0.7;
    let factor = 2.5;
    let ratio = 10.0f64.powf(-gain / (20.0 * factor));

    let bmtaumax = taumax / bwfactor;
    let bmtaumin = bmtaumax * ratio;

    (bmtaumax, bmtaumin, ratio)
}

/// Boltzman function - OHC nonlinearity.
///
/// Output is normalized with maximum value of 1.
#[inline]
fn boltzman(x: f64, asym: f64, s0: f64, s1: f64, x1: f64) -> f64 {
    let shift = 1.0 / (1.0 + asym);
    let x0 = s0 * ((1.0 / shift - 1.0) / (1.0 + (x1 / s1).exp())).ln();
    let out1 = 1.0 / (1.0 + (-(x - x0) / s0).exp() * (1.0 + (-(x - x1) / s1).exp())) - shift;
    out1 / (1.0 - shift)
}

/// Nonlinear function after OHC low-pass filter.
#[inline]
fn nl_after_ohc(x: f64, taumin: f64, taumax: f64, asym: f64) -> f64 {
    let mut minr = 0.05;
    let r = taumin / taumax;

    if r < minr {
        minr = 0.5 * r;
    }

    let dc = (asym - 1.0) / (asym + 1.0) / 2.0 - minr;
    let r1 = r - minr;
    let s0 = -dc / (r1 / (1.0 - minr)).ln();
    let x1 = x.abs();

    let out = taumax * (minr + (1.0 - minr) * (-x1 / s0).exp());
    out.clamp(taumin, taumax)
}

/// IHC Nonlinear Function (Logarithmic Transduction).
#[inline]
fn nlogarithm(x: f64, slope: f64, asym: f64, _cf: f64) -> f64 {
    let corner = 80.0;
    let strength = 20.0e6 / 10.0f64.powf(corner / 20.0);

    let mut xx = (1.0 + strength * x.abs()).ln() * slope;

    if x < 0.0 {
        let splx = 20.0 * (-x / 20e-6).log10();
        let asym_t = asym - (asym - 1.0) / (1.0 + (splx / 5.0).exp());
        xx = -1.0 / asym_t * xx;
    }

    xx
}

/// C1 Chirp Filter State and processing.
#[derive(Clone, Debug)]
pub struct C1ChirpFilter {
    state: ChirpFilterState,
}

impl Default for C1ChirpFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl C1ChirpFilter {
    pub fn new() -> Self {
        Self {
            state: ChirpFilterState::new(),
        }
    }

    pub fn reset(&mut self) {
        self.state.reset();
    }

    /// Process a single sample through the C1 chirp filter.
    ///
    /// # Arguments
    /// * `x` - Input sample
    /// * `tdres` - Time resolution (1/fs)
    /// * `cf` - Characteristic frequency
    /// * `taumax` - Maximum time constant
    /// * `rsigma` - Shift of pole locations
    #[inline]
    pub fn process(&mut self, x: f64, tdres: f64, cf: f64, taumax: f64, rsigma: f64) -> f64 {
        let sigma0 = 1.0 / taumax;
        let ipw = 1.01 * cf * TWOPI - 50.0;
        let ipb = 0.2343 * TWOPI * cf - 1104.0;
        let rpa = 10.0f64.powf((cf).log10() * 0.9 + 0.55) + 2000.0;
        let pzero = 10.0f64.powf((cf).log10() * 0.7 + 1.6) + 500.0;

        let order_of_pole = 10;
        let half_order_pole = order_of_pole / 2;
        let order_of_zero = half_order_pole;

        let fs_bilinear = TWOPI * cf / (TWOPI * cf * tdres / 2.0).tan();
        let cf_rad = TWOPI * cf;

        // Setup pole locations
        let mut p = [Complex::default(); 11];

        if !self.state.initialized {
            // Initialize on first sample
            p[1] = Complex::new(-sigma0, ipw);
            p[5] = Complex::new(p[1].re - rpa, p[1].im - ipb);
            p[3] = Complex::new((p[1].re + p[5].re) * 0.5, (p[1].im + p[5].im) * 0.5);
            p[2] = p[1].conj();
            p[4] = p[3].conj();
            p[6] = p[5].conj();
            p[7] = p[1];
            p[8] = p[2];
            p[9] = p[5];
            p[10] = p[6];

            let rzero_init = -pzero;
            self.state.init_phase = 0.0;
            for i in 1..=half_order_pole {
                let preal = p[i * 2 - 1].re;
                let pimg = p[i * 2 - 1].im;
                self.state.init_phase += (cf_rad / (-rzero_init)).atan()
                    - ((cf_rad - pimg) / (-preal)).atan()
                    - ((cf_rad + pimg) / (-preal)).atan();
            }

            // Initialize input/output arrays
            for i in 1..=(half_order_pole + 1) {
                self.state.input[i][1] = 0.0;
                self.state.input[i][2] = 0.0;
                self.state.input[i][3] = 0.0;
                self.state.output[i][1] = 0.0;
                self.state.output[i][2] = 0.0;
                self.state.output[i][3] = 0.0;
            }

            // Normalize gain
            self.state.gain_norm = 1.0;
            for r in 1..=order_of_pole {
                self.state.gain_norm *=
                    (cf_rad - p[r].im).powi(2) + p[r].re.powi(2);
            }

            self.state.initialized = true;
        }

        // Calculate norm gain
        let rzero_init = -pzero;
        let norm_gain = self.state.gain_norm.sqrt()
            / (cf_rad * cf_rad + rzero_init * rzero_init).sqrt().powi(order_of_zero as i32);

        // Update poles with rsigma
        p[1] = Complex::new(-sigma0 - rsigma, ipw);
        p[5] = Complex::new(p[1].re - rpa, p[1].im - ipb);
        p[3] = Complex::new((p[1].re + p[5].re) * 0.5, (p[1].im + p[5].im) * 0.5);
        p[2] = p[1].conj();
        p[4] = p[3].conj();
        p[6] = p[5].conj();
        p[7] = p[1];
        p[8] = p[2];
        p[9] = p[5];
        p[10] = p[6];

        // Calculate current phase
        let mut phase = 0.0;
        for i in 1..=half_order_pole {
            let preal = p[i * 2 - 1].re;
            let pimg = p[i * 2 - 1].im;
            phase -= ((cf_rad - pimg) / (-preal)).atan() + ((cf_rad + pimg) / (-preal)).atan();
        }

        let rzero = -cf_rad / ((self.state.init_phase - phase) / order_of_zero as f64).tan();

        // Process through filter stages
        self.state.input[1][3] = self.state.input[1][2];
        self.state.input[1][2] = self.state.input[1][1];
        self.state.input[1][1] = x;

        for i in 1..=half_order_pole {
            let preal = p[i * 2 - 1].re;
            let pimg = p[i * 2 - 1].im;
            let temp = (fs_bilinear - preal).powi(2) + pimg.powi(2);

            let dy = self.state.input[i][1] * (fs_bilinear - rzero)
                - 2.0 * rzero * self.state.input[i][2]
                - (fs_bilinear + rzero) * self.state.input[i][3]
                + 2.0
                    * self.state.output[i][1]
                    * (fs_bilinear * fs_bilinear - preal * preal - pimg * pimg)
                - self.state.output[i][2]
                    * ((fs_bilinear + preal).powi(2) + pimg.powi(2));

            let dy = dy / temp;

            self.state.input[i + 1][3] = self.state.output[i][2];
            self.state.input[i + 1][2] = self.state.output[i][1];
            self.state.input[i + 1][1] = dy;

            self.state.output[i][2] = self.state.output[i][1];
            self.state.output[i][1] = dy;
        }

        let dy = self.state.output[half_order_pole][1] * norm_gain;
        dy / 4.0 // Signal path output is divided by 4 to give correct C1 filter gain
    }
}

/// C2 Chirp Filter State and processing.
/// Same structure as C1 but with different pole calculation.
#[derive(Clone, Debug)]
pub struct C2ChirpFilter {
    state: ChirpFilterState,
}

impl Default for C2ChirpFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl C2ChirpFilter {
    pub fn new() -> Self {
        Self {
            state: ChirpFilterState::new(),
        }
    }

    pub fn reset(&mut self) {
        self.state.reset();
    }

    /// Process a single sample through the C2 chirp filter.
    ///
    /// # Arguments
    /// * `x` - Input sample
    /// * `tdres` - Time resolution (1/fs)
    /// * `cf` - Characteristic frequency
    /// * `taumax` - Maximum time constant
    /// * `fcohc` - OHC impairment factor
    #[inline]
    pub fn process(&mut self, x: f64, tdres: f64, cf: f64, taumax: f64, fcohc: f64) -> f64 {
        let sigma0 = 1.0 / taumax;
        let ipw = 1.01 * cf * TWOPI - 50.0;
        let ipb = 0.2343 * TWOPI * cf - 1104.0;
        let rpa = 10.0f64.powf((cf).log10() * 0.9 + 0.55) + 2000.0;
        let pzero = 10.0f64.powf((cf).log10() * 0.7 + 1.6) + 500.0;

        let order_of_pole = 10;
        let half_order_pole = order_of_pole / 2;
        let order_of_zero = half_order_pole;

        let fs_bilinear = TWOPI * cf / (TWOPI * cf * tdres / 2.0).tan();
        let cf_rad = TWOPI * cf;

        let mut p = [Complex::default(); 11];

        if !self.state.initialized {
            p[1] = Complex::new(-sigma0, ipw);
            p[5] = Complex::new(p[1].re - rpa, p[1].im - ipb);
            p[3] = Complex::new((p[1].re + p[5].re) * 0.5, (p[1].im + p[5].im) * 0.5);
            p[2] = p[1].conj();
            p[4] = p[3].conj();
            p[6] = p[5].conj();
            p[7] = p[1];
            p[8] = p[2];
            p[9] = p[5];
            p[10] = p[6];

            let rzero_init = -pzero;
            self.state.init_phase = 0.0;
            for i in 1..=half_order_pole {
                let preal = p[i * 2 - 1].re;
                let pimg = p[i * 2 - 1].im;
                self.state.init_phase += (cf_rad / (-rzero_init)).atan()
                    - ((cf_rad - pimg) / (-preal)).atan()
                    - ((cf_rad + pimg) / (-preal)).atan();
            }

            for i in 1..=(half_order_pole + 1) {
                self.state.input[i][1] = 0.0;
                self.state.input[i][2] = 0.0;
                self.state.input[i][3] = 0.0;
                self.state.output[i][1] = 0.0;
                self.state.output[i][2] = 0.0;
                self.state.output[i][3] = 0.0;
            }

            self.state.gain_norm = 1.0;
            for r in 1..=order_of_pole {
                self.state.gain_norm *=
                    (cf_rad - p[r].im).powi(2) + p[r].re.powi(2);
            }

            self.state.initialized = true;
        }

        let rzero_init = -pzero;
        let norm_gain = self.state.gain_norm.sqrt()
            / (cf_rad * cf_rad + rzero_init * rzero_init).sqrt().powi(order_of_zero as i32);

        // For C2, pole is scaled by fcohc
        p[1] = Complex::new(-sigma0 * fcohc, ipw);
        p[5] = Complex::new(p[1].re - rpa, p[1].im - ipb);
        p[3] = Complex::new((p[1].re + p[5].re) * 0.5, (p[1].im + p[5].im) * 0.5);
        p[2] = p[1].conj();
        p[4] = p[3].conj();
        p[6] = p[5].conj();
        p[7] = p[1];
        p[8] = p[2];
        p[9] = p[5];
        p[10] = p[6];

        let mut phase = 0.0;
        for i in 1..=half_order_pole {
            let preal = p[i * 2 - 1].re;
            let pimg = p[i * 2 - 1].im;
            phase -= ((cf_rad - pimg) / (-preal)).atan() + ((cf_rad + pimg) / (-preal)).atan();
        }

        let rzero = -cf_rad / ((self.state.init_phase - phase) / order_of_zero as f64).tan();

        self.state.input[1][3] = self.state.input[1][2];
        self.state.input[1][2] = self.state.input[1][1];
        self.state.input[1][1] = x;

        for i in 1..=half_order_pole {
            let preal = p[i * 2 - 1].re;
            let pimg = p[i * 2 - 1].im;
            let temp = (fs_bilinear - preal).powi(2) + pimg.powi(2);

            let dy = self.state.input[i][1] * (fs_bilinear - rzero)
                - 2.0 * rzero * self.state.input[i][2]
                - (fs_bilinear + rzero) * self.state.input[i][3]
                + 2.0
                    * self.state.output[i][1]
                    * (fs_bilinear * fs_bilinear - preal * preal - pimg * pimg)
                - self.state.output[i][2]
                    * ((fs_bilinear + preal).powi(2) + pimg.powi(2));

            let dy = dy / temp;

            self.state.input[i + 1][3] = self.state.output[i][2];
            self.state.input[i + 1][2] = self.state.output[i][1];
            self.state.input[i + 1][1] = dy;

            self.state.output[i][2] = self.state.output[i][1];
            self.state.output[i][1] = dy;
        }

        let dy = self.state.output[half_order_pole][1] * norm_gain;
        dy / 4.0
    }
}

/// Complete IHC processor state.
#[derive(Clone)]
pub struct IhcProcessor {
    /// Middle ear filter coefficients
    pub me_coeffs: MiddleEarCoeffs,
    /// Middle ear filter state
    pub me_state: MiddleEarState,
    /// Wideband gammatone filter state
    pub wb_state: WbGammatoneState,
    /// OHC lowpass filter state
    pub ohc_lp_state: CascadeLowpassState,
    /// IHC lowpass filter state
    pub ihc_lp_state: CascadeLowpassState,
    /// C1 chirp filter
    pub c1_filter: C1ChirpFilter,
    /// C2 chirp filter
    pub c2_filter: C2ChirpFilter,
    /// Temporary gain values (for group delay)
    pub tmpgain: Vec<f64>,
    /// Last tmpgain value
    pub lasttmpgain: f64,
    /// Current wideband gain
    pub wbgain: f64,
    /// Time resolution
    pub tdres: f64,
    /// Characteristic frequency
    pub cf: f64,
    /// Center frequency for wideband filter
    pub centerfreq: f64,
    /// Species (1=cat, 2+=human)
    pub species: i32,
    /// OHC health (0-1)
    pub cohc: f64,
    /// IHC health (0-1)
    pub cihc: f64,
    /// Maximum tau for wideband
    pub taumax: f64,
    /// Minimum tau for wideband
    pub taumin: f64,
    /// Maximum tau for BM
    pub bmtaumax: f64,
    /// Minimum tau for BM
    pub bmtaumin: f64,
    /// BM tau ratio
    pub ratiobm: f64,
    /// Maximum tau for wideband control path
    pub tauwbmax: f64,
    /// Minimum tau for wideband control path
    pub tauwbmin: f64,
    /// Current wideband tau
    pub tauwb: f64,
    /// OHC asymmetry
    pub ohcasym: f64,
    /// IHC asymmetry
    pub ihcasym: f64,
    /// Sample counter
    pub n: usize,
}

impl IhcProcessor {
    /// Create a new IHC processor.
    ///
    /// # Arguments
    /// * `cf` - Characteristic frequency in Hz
    /// * `fs` - Sampling frequency in Hz
    /// * `species` - Species (1=cat, 2=human Shera, 3=human Glasberg)
    /// * `cohc` - OHC health (0-1)
    /// * `cihc` - IHC health (0-1)
    /// * `totalstim` - Total stimulus length for tmpgain buffer
    pub fn new(cf: f64, fs: f64, species: i32, cohc: f64, cihc: f64, totalstim: usize) -> Self {
        let tdres = 1.0 / fs;
        let centerfreq = calculate_centerfreq(cf, species);

        // Get tau parameters
        let bmorder = 3;
        let (taumax, taumin) = get_tauwb(cf, species, bmorder);
        let _taubm = cohc * (taumax - taumin) + taumin;
        let (bmtaumax, bmtaumin, ratiobm) = get_taubm(cf, taumax);
        let bmtaubm = cohc * (bmtaumax - bmtaumin) + bmtaumin;

        // Control-path parameters
        let tauwbmax = taumin + 0.2 * (taumax - taumin);
        let tauwbmin = tauwbmax / taumax * taumin;
        let tauwb = tauwbmax + (bmtaubm - bmtaumax) * (tauwbmax - tauwbmin) / (bmtaumax - bmtaumin);

        let (wbgain, _) = gain_groupdelay(tdres, centerfreq, cf, tauwb);

        Self {
            me_coeffs: MiddleEarCoeffs::new(species, tdres),
            me_state: MiddleEarState::new(),
            wb_state: WbGammatoneState::new(),
            ohc_lp_state: CascadeLowpassState::new(),
            ihc_lp_state: CascadeLowpassState::new(),
            c1_filter: C1ChirpFilter::new(),
            c2_filter: C2ChirpFilter::new(),
            tmpgain: vec![0.0; totalstim],
            lasttmpgain: wbgain,
            wbgain,
            tdres,
            cf,
            centerfreq,
            species,
            cohc,
            cihc,
            taumax,
            taumin,
            bmtaumax,
            bmtaumin,
            ratiobm,
            tauwbmax,
            tauwbmin,
            tauwb,
            ohcasym: 7.0,
            ihcasym: 3.0,
            n: 0,
        }
    }

    /// Process a single sample through the IHC model.
    ///
    /// # Arguments
    /// * `px` - Input sample (pressure)
    ///
    /// # Returns
    /// IHC output voltage
    #[inline]
    pub fn process(&mut self, px: f64) -> f64 {
        let n = self.n;

        // Middle ear filter
        let meout = self.me_state.process(px, &self.me_coeffs);

        // Control-path filter (wideband gammatone)
        let wbout1 = self.wb_state.process(
            meout,
            self.tdres,
            self.centerfreq,
            self.tauwb,
            self.wbgain,
            3,
        );
        let wbout = (self.tauwb / self.tauwbmax).powi(3) * wbout1 * 10e3 * self.cf.max(5e3) / 5e3;

        // OHC nonlinearity
        let ohcnonlinout = boltzman(wbout, self.ohcasym, 12.0, 5.0, 5.0);

        // OHC lowpass (2nd order, 600 Hz)
        let ohcout = self.ohc_lp_state.process(ohcnonlinout, self.tdres, 600.0, 1.0, 2);

        // Nonlinear function after OHC
        let tmptauc1 = nl_after_ohc(ohcout, self.bmtaumin, self.bmtaumax, self.ohcasym);
        let tauc1 = self.cohc * (tmptauc1 - self.bmtaumin) + self.bmtaumin;
        let rsigma = 1.0 / tauc1 - 1.0 / self.bmtaumax;

        // Update wideband tau
        self.tauwb = self.tauwbmax
            + (tauc1 - self.bmtaumax) * (self.tauwbmax - self.tauwbmin)
                / (self.bmtaumax - self.bmtaumin);

        let (wb_gain, grdelay) = gain_groupdelay(self.tdres, self.centerfreq, self.cf, self.tauwb);

        // Store gain for group delay compensation
        let grd = grdelay as usize;
        if grd + n < self.tmpgain.len() {
            self.tmpgain[grd + n] = wb_gain;
        }

        if n < self.tmpgain.len() && self.tmpgain[n] == 0.0 {
            self.tmpgain[n] = self.lasttmpgain;
        }

        if n < self.tmpgain.len() {
            self.wbgain = self.tmpgain[n];
            self.lasttmpgain = self.wbgain;
        }

        // C1 filter (signal path)
        let c1filterout = self.c1_filter.process(meout, self.tdres, self.cf, self.bmtaumax, rsigma);

        // C2 filter (parallel path)
        let c2filterout = self.c2_filter.process(
            meout,
            self.tdres,
            self.cf,
            self.bmtaumax,
            1.0 / self.ratiobm,
        );

        // IHC transduction
        let c1vihc = nlogarithm(self.cihc * c1filterout, 0.1, self.ihcasym, self.cf);
        let c2vihc =
            -nlogarithm(c2filterout * c2filterout.abs() * self.cf / 10.0 * self.cf / 2e3, 0.2, 1.0, self.cf);

        // IHC lowpass (7th order, 3000 Hz)
        let ihcout = self.ihc_lp_state.process(c1vihc + c2vihc, self.tdres, 3000.0, 1.0, 7);

        self.n += 1;
        ihcout
    }

    /// Reset the processor state for a new run.
    pub fn reset(&mut self, totalstim: usize) {
        self.me_state.reset();
        self.wb_state.reset();
        self.ohc_lp_state.reset();
        self.ihc_lp_state.reset();
        self.c1_filter.reset();
        self.c2_filter.reset();
        self.tmpgain = vec![0.0; totalstim];

        // Recalculate initial wideband gain
        let bmtaubm = self.cohc * (self.bmtaumax - self.bmtaumin) + self.bmtaumin;
        self.tauwb = self.tauwbmax
            + (bmtaubm - self.bmtaumax) * (self.tauwbmax - self.tauwbmin)
                / (self.bmtaumax - self.bmtaumin);

        let (wbgain, _) = gain_groupdelay(self.tdres, self.centerfreq, self.cf, self.tauwb);
        self.wbgain = wbgain;
        self.lasttmpgain = wbgain;
        self.n = 0;
    }
}

/// Run the complete IHC model on a signal.
///
/// # Arguments
/// * `signal` - Input signal (pressure)
/// * `cf` - Characteristic frequency in Hz
/// * `fs` - Sampling frequency in Hz
/// * `species` - Species string ("cat", "human", "human_glasberg1990")
/// * `cohc` - OHC health (0-1)
/// * `cihc` - IHC health (0-1)
///
/// # Returns
/// IHC output signal
pub fn run_ihc(
    signal: &[f64],
    cf: f64,
    fs: f64,
    species: &str,
    cohc: f64,
    cihc: f64,
) -> Vec<f64> {
    let species_code = match species {
        "cat" => SPECIES_CAT,
        "human" => SPECIES_HUMAN_SHERA,
        "human_glasberg1990" => SPECIES_HUMAN_GLASBERG,
        _ => SPECIES_CAT,
    };

    let mut processor = IhcProcessor::new(cf, fs, species_code, cohc, cihc, signal.len());

    // Process signal
    let mut ihcout = vec![0.0; signal.len()];
    for (i, &px) in signal.iter().enumerate() {
        ihcout[i] = processor.process(px);
    }

    // Apply delay
    let delay = if species_code == SPECIES_CAT {
        delay_cat(cf)
    } else {
        delay_cat(cf) // Version 5.2 uses cat delay for humans too
    };
    let delaypoint = (delay / processor.tdres).ceil().max(0.0) as usize;

    // Shift output by delay
    let mut result = vec![0.0; signal.len()];
    for i in delaypoint..signal.len() {
        result[i] = ihcout[i - delaypoint];
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boltzman() {
        let out = boltzman(0.0, 7.0, 12.0, 5.0, 5.0);
        assert!(out.abs() < 1.0);

        let out_pos = boltzman(100.0, 7.0, 12.0, 5.0, 5.0);
        let out_neg = boltzman(-100.0, 7.0, 12.0, 5.0, 5.0);
        assert!(out_pos > out_neg);
    }

    #[test]
    fn test_ihc_processor() {
        let fs = 100e3;
        let cf = 1000.0;
        let duration = 0.01; // 10 ms
        let n_samples = (duration * fs) as usize;

        // Generate a tone at CF
        let signal: Vec<f64> = (0..n_samples)
            .map(|i| 0.1 * (2.0 * std::f64::consts::PI * cf * i as f64 / fs).sin())
            .collect();

        let ihcout = run_ihc(&signal, cf, fs, "cat", 1.0, 1.0);

        // Check that output is non-zero
        let max_out = ihcout.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        assert!(max_out > 0.0);
    }
}
