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

// Python bindings (only when "python" feature is enabled)
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
use numpy::{PyArray1, PyReadonlyArray1, IntoPyArray};

#[cfg(feature = "python")]
use rand::rngs::StdRng;

#[cfg(feature = "python")]
use rand::SeedableRng;

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (signal, cf, fs, species="cat", cohc=1.0, cihc=1.0))]
fn run_ihc_py<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<'py, f64>,
    cf: f64,
    fs: f64,
    species: &str,
    cohc: f64,
    cihc: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let signal_slice = signal.as_slice()?;
    let ihc_out = ihc::run_ihc(signal_slice, cf, fs, species, cohc, cihc);
    Ok(ihc_out.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (vihc, fs, cf, anf_type="hsr", ffGn=true))]
#[allow(non_snake_case)]
fn run_synapse_py<'py>(
    py: Python<'py>,
    vihc: PyReadonlyArray1<'py, f64>,
    fs: f64,
    cf: f64,
    anf_type: &str,
    ffGn: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let vihc_slice = vihc.as_slice()?;
    let tdres = 1.0 / fs;

    let anf = match anf_type {
        "lsr" => synapse::AnfType::Lsr,
        "msr" => synapse::AnfType::Msr,
        "hsr" => synapse::AnfType::Hsr,
        _ => synapse::AnfType::Hsr,
    };

    let mut rng = StdRng::from_entropy();
    let synapse_out = synapse::run_synapse(vihc_slice, tdres, cf, anf, ffGn, &mut rng);
    Ok(synapse_out.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (synout, fs))]
fn run_spike_generator_py<'py>(
    py: Python<'py>,
    synout: PyReadonlyArray1<'py, f64>,
    fs: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let synout_slice = synout.as_slice()?;
    let tdres = 1.0 / fs;

    let mut rng = StdRng::from_entropy();
    let spike_times = spike_generator::run_spike_generator(synout_slice, tdres, &mut rng);
    Ok(spike_times.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (signal, fs, cfs, species="cat", anf_type="hsr", cohc=1.0, cihc=1.0, ffGn=true, seed=None))]
#[allow(non_snake_case)]
fn run_zilany2014_py<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<'py, f64>,
    fs: f64,
    cfs: Vec<f64>,
    species: &str,
    anf_type: &str,
    cohc: f64,
    cihc: f64,
    ffGn: bool,
    seed: Option<u64>,
) -> PyResult<Vec<Bound<'py, PyArray1<f64>>>> {
    let signal_slice = signal.as_slice()?;

    let species_enum = match species {
        "cat" => model::Species::Cat,
        "human" => model::Species::Human,
        "human_glasberg1990" => model::Species::HumanGlasberg,
        _ => model::Species::Cat,
    };

    let anf = match anf_type {
        "lsr" => synapse::AnfType::Lsr,
        "msr" => synapse::AnfType::Msr,
        "hsr" => synapse::AnfType::Hsr,
        _ => synapse::AnfType::Hsr,
    };

    let config = model::ModelConfig {
        fs,
        species: species_enum,
        cohc,
        cihc,
        anf_type: anf,
        use_ffgn: ffGn,
        seed,
    };

    let output = model::run_zilany2014(signal_slice, &cfs, &config);

    let spike_times: Vec<Bound<'py, PyArray1<f64>>> = output
        .channels
        .into_iter()
        .map(|ch| ch.spike_times.into_pyarray(py))
        .collect();

    Ok(spike_times)
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (signal, fs, cfs, species="cat", anf_type="hsr", cohc=1.0, cihc=1.0, ffGn=true, seed=None))]
#[allow(non_snake_case)]
fn run_zilany2014_full<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<'py, f64>,
    fs: f64,
    cfs: Vec<f64>,
    species: &str,
    anf_type: &str,
    cohc: f64,
    cihc: f64,
    ffGn: bool,
    seed: Option<u64>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let signal_slice = signal.as_slice()?;

    let species_enum = match species {
        "cat" => model::Species::Cat,
        "human" => model::Species::Human,
        "human_glasberg1990" => model::Species::HumanGlasberg,
        _ => model::Species::Cat,
    };

    let anf = match anf_type {
        "lsr" => synapse::AnfType::Lsr,
        "msr" => synapse::AnfType::Msr,
        "hsr" => synapse::AnfType::Hsr,
        _ => synapse::AnfType::Hsr,
    };

    let config = model::ModelConfig {
        fs,
        species: species_enum,
        cohc,
        cihc,
        anf_type: anf,
        use_ffgn: ffGn,
        seed,
    };

    let output = model::run_zilany2014(signal_slice, &cfs, &config);

    let dict = pyo3::types::PyDict::new(py);

    let ihc_out: Vec<Bound<'py, PyArray1<f64>>> = output
        .channels
        .iter()
        .map(|ch| ch.ihc_out.clone().into_pyarray(py))
        .collect();

    let synapse_out: Vec<Bound<'py, PyArray1<f64>>> = output
        .channels
        .iter()
        .map(|ch| ch.synapse_out.clone().into_pyarray(py))
        .collect();

    let spike_times: Vec<Bound<'py, PyArray1<f64>>> = output
        .channels
        .into_iter()
        .map(|ch| ch.spike_times.into_pyarray(py))
        .collect();

    dict.set_item("ihc_out", ihc_out)?;
    dict.set_item("synapse_out", synapse_out)?;
    dict.set_item("spike_times", spike_times)?;
    dict.set_item("cfs", cfs)?;

    Ok(dict)
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (freq_min, freq_max, num_cfs, species="cat"))]
fn generate_cfs_py<'py>(
    py: Python<'py>,
    freq_min: f64,
    freq_max: f64,
    num_cfs: usize,
    species: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let species_enum = match species {
        "cat" => model::Species::Cat,
        "human" => model::Species::Human,
        "human_glasberg1990" => model::Species::HumanGlasberg,
        _ => model::Species::Cat,
    };

    let cfs = model::generate_cfs(freq_min, freq_max, num_cfs, species_enum);
    Ok(cfs.into_pyarray(py))
}

/// Python module for cochlea-rs.
#[cfg(feature = "python")]
#[pymodule]
fn cochlea(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_ihc_py, m)?)?;
    m.add_function(wrap_pyfunction!(run_synapse_py, m)?)?;
    m.add_function(wrap_pyfunction!(run_spike_generator_py, m)?)?;
    m.add_function(wrap_pyfunction!(run_zilany2014_py, m)?)?;
    m.add_function(wrap_pyfunction!(run_zilany2014_full, m)?)?;
    m.add_function(wrap_pyfunction!(generate_cfs_py, m)?)?;
    Ok(())
}
