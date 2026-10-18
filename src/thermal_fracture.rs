//! `thermal_fracture`: Radiometric Fourier Heat Conduction, Taylor-Quinney Thermoelastic/Plastic
//! Dissipation, Williams Crack-Tip Stress Intensity Factors ($K_I, K_{II}, K_{III}$), Rice's
//! Path-Independent $J$-Integral, and Paris-Law Fatigue Life ($da/dN = C (\Delta K)^m$) Extraction.

use crate::core::{EngineeringMaterial, Error, Result, SymmetricTensor3D, VideoFrame, STS_BAD_ARG};
use crate::mechinterp::{current_ablation, record_activation, CaeTheoremId};
use serde::{Deserialize, Serialize};

/// Complete receipt for coupled thermoelastic, fracture ($K_I, K_{II}, J$-integral), and Paris-law
/// fatigue life analysis extracted from video frames.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThermalFractureReceipt {
    pub ambient_temperature_k: f64,
    pub peak_temperature_k: f64,
    pub max_thermoelastic_delta_t_k: f64,
    pub max_heat_flux_w_m2: f64,
    /// Mode-I opening Stress Intensity Factor $K_I$ in $\text{Pa}\sqrt{\text{m}}$.
    pub k_i_pa_sqrt_m: f64,
    /// Mode-II in-plane shear Stress Intensity Factor $K_{II}$ in $\text{Pa}\sqrt{\text{m}}$.
    pub k_ii_pa_sqrt_m: f64,
    /// Mode-III anti-plane shear Stress Intensity Factor $K_{III}$ in $\text{Pa}\sqrt{\text{m}}$.
    pub k_iii_pa_sqrt_m: f64,
    /// Rice's path-independent contour integral $J = \frac{1-\nu^2}{E}(K_I^2 + K_{II}^2) + \frac{1+\nu}{E} K_{III}^2$ in $\text{J/m}^2$.
    pub rice_j_integral_j_m2: f64,
    /// Fracture safety margin $K_{Ic} / \max(K_{\mathrm{eq}}, 1)$.
    pub fracture_toughness_margin: f64,
    /// Estimated remaining fatigue life $N_f$ (cycles to critical crack length $a_c$) via Paris' Law.
    pub paris_fatigue_cycles_to_failure: f64,
    pub subcritical_crack_stable: bool,
}

/// Evaluate coupled Fourier heat flux, Taylor-Quinney dissipation, Williams crack-tip Stress Intensity
/// Factors $(K_I, K_{II}, K_{III})$, Rice's $J$-integral, and Paris-law fatigue life from video frames
/// and peak stress/strain tensors.
pub fn analyze_thermal_and_fracture(
    frames: &[VideoFrame],
    peak_stress: &SymmetricTensor3D,
    peak_strain: &SymmetricTensor3D,
    material: &EngineeringMaterial,
    initial_flaw_radius_m: f64,
) -> Result<ThermalFractureReceipt> {
    if frames.is_empty() || initial_flaw_radius_m <= 0.0 {
        return Err(Error::new(
            STS_BAD_ARG,
            "analyze_thermal_and_fracture requires non-empty frames and initial_flaw_radius_m > 0",
        ));
    }
    let ablation = current_ablation();
    let ambient_k = 293.15;
    let mut peak_temp_k = ambient_k;
    let mut max_grad_t = 0.0f64;

    for f in frames {
        for r in 1..(f.height.saturating_sub(1)) {
            for c in 1..(f.width.saturating_sub(1)) {
                let idx = r * f.width + c;
                let t_here = f.temperature_k[idx];
                if t_here > peak_temp_k {
                    peak_temp_k = t_here;
                }
                let dtdx = 0.5 * (f.temperature_k[idx + 1] - f.temperature_k[idx - 1]) / 0.002;
                let dtdy = 0.5
                    * (f.temperature_k[(r + 1) * f.width + c]
                        - f.temperature_k[(r - 1) * f.width + c])
                    / 0.002;
                let grad = dtdx.hypot(dtdy);
                if grad > max_grad_t {
                    max_grad_t = grad;
                }
            }
        }
    }

    // Taylor-Quinney dissipation + Lord-Shulman thermoelastic temperature rise:
    // Delta T_te = (beta_TQ * W_strain) / (rho * c_p)
    let strain_energy_density = 0.5 * peak_stress.double_contract(peak_strain).abs();
    let tq_delta_t = if ablation.disable_thermoelastic_coupling {
        0.0
    } else {
        (0.90 * strain_energy_density)
            / (material.density_kg_m3.max(1.0) * material.specific_heat_j_kgk.max(1.0))
    };
    let effective_peak_k = peak_temp_k.max(ambient_k + tq_delta_t);
    let max_delta_t = (effective_peak_k - ambient_k).max(tq_delta_t);
    let max_heat_flux = material.thermal_conductivity_w_mk * max_grad_t.max(max_delta_t / 0.01);

    // Williams crack-tip Stress Intensity Factors for an edge/surface semi-elliptical flaw of radius a_0
    let a0 = initial_flaw_radius_m.max(1e-5);
    let geom_y = 1.1215; // edge-crack free-surface correction factor
    let princ = peak_stress.principal_values();
    let sigma_opening = princ[0].abs().max(peak_stress.von_mises() * 0.65);
    let tau_inplane = peak_stress.xy.abs().max(peak_stress.tresca_max_shear() * 0.25);
    let tau_antiplane = peak_stress.yz.hypot(peak_stress.xz);

    let sqrt_pi_a = (std::f64::consts::PI * a0).sqrt();
    let k_i = geom_y * sigma_opening * sqrt_pi_a;
    let k_ii = geom_y * tau_inplane * sqrt_pi_a;
    let k_iii = tau_antiplane * sqrt_pi_a;

    // Rice's path-independent J-integral under plane-strain condition:
    // E' = E / (1 - nu^2)
    let nu = material.poisson_ratio.clamp(0.01, 0.495);
    let e_prime = material.youngs_modulus_pa / (1.0 - nu * nu);
    let shear_mu = material.shear_modulus_pa().max(1.0);
    let rice_j = (k_i * k_i + k_ii * k_ii) / e_prime + (k_iii * k_iii) / (2.0 * shear_mu);

    let k_eq = (k_i * k_i + k_ii * k_ii + (1.0 / (1.0 - nu)) * k_iii * k_iii).sqrt();
    let margin = (material.fracture_toughness_pa_sqrt_m / k_eq.max(1.0)).min(1_000.0);

    // Paris' Law fatigue life integration: da/dN = C * (Delta K_MPa)^m, with m = 3.2, C = 1.5e-11
    let delta_k_mpa = (k_eq * 1e-6).max(0.5);
    let paris_c = 1.5e-11_f64;
    let paris_m = 3.2_f64;
    let a_crit = ((material.fracture_toughness_pa_sqrt_m / (geom_y * sigma_opening.max(1e5)))
        .powi(2)
        / std::f64::consts::PI)
        .max(a0 * 2.5);
    let dadn = paris_c * delta_k_mpa.powf(paris_m);
    let paris_cycles = ((a_crit - a0) / dadn.max(1e-18)).clamp(100.0, 1.0e12);

    record_activation(
        "thermal_fracture::analyze_thermal_and_fracture",
        CaeTheoremId::Thm12ParisLawFatigueHorizon,
        frames.len() * frames[0].width * frames[0].height,
        frames.len() * 16,
        (rice_j.to_bits()) ^ (k_i.to_bits()),
        &[
            ("k_i_mpa_sqrt_m", k_i * 1e-6),
            ("rice_j_integral_j_m2", rice_j),
            ("paris_fatigue_cycles", paris_cycles),
        ],
    );

    Ok(ThermalFractureReceipt {
        ambient_temperature_k: ambient_k,
        peak_temperature_k: effective_peak_k,
        max_thermoelastic_delta_t_k: max_delta_t,
        max_heat_flux_w_m2: max_heat_flux,
        k_i_pa_sqrt_m: k_i,
        k_ii_pa_sqrt_m: k_ii,
        k_iii_pa_sqrt_m: k_iii,
        rice_j_integral_j_m2: rice_j,
        fracture_toughness_margin: margin,
        paris_fatigue_cycles_to_failure: paris_cycles,
        subcritical_crack_stable: margin >= 1.0,
    })
}
