//! Cochlea-rs: Rust port of the Zilany et al. (2014) auditory periphery model.
//!
//! Converts sound pressure waveforms to auditory nerve spike trains via:
//! - Middle ear filtering (species-specific)
//! - Inner Hair Cell (IHC) mechanoelectrical transduction
//! - Auditory nerve synapse adaptation (power-law, O(n))
//! - Stochastic spike generation (inhomogeneous Poisson)
//!
//! Based on cochlea Python library by Marek Rudnicki, which implements:
//! Zilany, M.S.A., Bruce, I.C., & Carney, L.H. (2014)
//! "Updated parameters and expanded simulation options for a model of the auditory periphery"
//! J. Acoust. Soc. Am. 135(1), 283-286

pub mod complex;
pub mod filters;
pub mod ihc;
pub mod middle_ear;
pub mod model;
pub mod spike_generator;
pub mod synapse;

// Re-exports for convenience
pub use model::{generate_cfs, run_channel, run_model_simple, run_zilany2014, ModelConfig, ModelOutput, Species};
pub use synapse::AnfType;
