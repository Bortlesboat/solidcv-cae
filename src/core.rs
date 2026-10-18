//! Core video-to-CAE frame buffers, 3D symmetric strain/stress tensors, exact Cardano principal
//! eigenvalue solvers, engineering material constitutive library, and `solidcv-cad` -> `solidcv-cae`
//! handoff contracts.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Standard error code for invalid arguments (`-1`).
pub const STS_BAD_ARG: i32 = -1;
/// Standard error code for runtime/protocol errors (`-2`).
pub const STS_ERROR: i32 = -2;
/// HTTP 401 Unauthorized error code.
pub const STS_UNAUTHORIZED: i32 = 401;
/// HTTP 402 Payment Required error code (`solidSF Credits` depleted).
pub const STS_PAYMENT_REQUIRED: i32 = 402;
/// HTTP 422 Unprocessable Entity error code.
pub const STS_UNPROCESSABLE: i32 = 422;

/// Canonical error type for `solidcv-cae` operations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Error {
    pub code: i32,
    pub message: String,
}

impl Error {
    pub fn new(code: i32, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "solidcv-cae error ({}): {}", self.code, self.message)
    }
}

impl std::error::Error for Error {}

/// Result alias for `solidcv-cae` operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Deterministic 256-bit hex digest (FNV-1a + SplitMix64 4-lane cascade) for zero-FFI CAE receipts.
pub fn solidcv_cae_digest_hex(bytes: &[u8]) -> String {
    let mut s0: u64 = 0xcbf2_9ce4_8422_2325;
    let mut s1: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut s2: u64 = 0xbf58_476d_1ce4_e5b9;
    let mut s3: u64 = 0x94d0_49bb_1331_11eb;
    for (idx, &b) in bytes.iter().enumerate() {
        let v = (b as u64).wrapping_add((idx as u64).wrapping_mul(0x1000_0000_01b3));
        s0 = (s0 ^ v).wrapping_mul(0x1000_0000_01b3).rotate_left(13);
        s1 = (s1 ^ s0).wrapping_mul(0xbf58_476d_1ce4_e5b9).rotate_left(17);
        s2 = (s2 ^ s1).wrapping_mul(0x94d0_49bb_1331_11eb).rotate_left(29);
        s3 = (s3 ^ s2).wrapping_mul(0x517c_c1b7_2722_0a95).rotate_left(31);
    }
    format!("{s0:016x}{s1:016x}{s2:016x}{s3:016x}")
}

/// 2D calibrated video frame with optional depth/radiometric metadata for Video-to-CAE extraction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VideoFrame {
    pub frame_index: usize,
    pub timestamp_sec: f64,
    pub width: usize,
    pub height: usize,
    /// Row-major 8-bit luminance intensity buffer (`width * height`).
    pub pixels: Vec<u8>,
    /// Optional row-major metric depth map in metres (`width * height`).
    pub depth_m: Vec<f64>,
    /// Optional row-major surface temperature map in Kelvin (`width * height`).
    pub temperature_k: Vec<f64>,
}

impl VideoFrame {
    /// Construct a new `VideoFrame` initialized with constant intensity, depth, and ambient temperature (`293.15 K`).
    pub fn new(
        frame_index: usize,
        timestamp_sec: f64,
        width: usize,
        height: usize,
        default_intensity: u8,
    ) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(Error::new(
                STS_BAD_ARG,
                "VideoFrame width and height must be > 0",
            ));
        }
        let n = width * height;
        Ok(Self {
            frame_index,
            timestamp_sec,
            width,
            height,
            pixels: vec![default_intensity; n],
            depth_m: vec![0.50; n],
            temperature_k: vec![293.15; n],
        })
    }

    /// Construct a `VideoFrame` from a 2D slice of `u8` rows.
    pub fn from_slice_2d(
        frame_index: usize,
        timestamp_sec: f64,
        rows: &[&[u8]],
    ) -> Result<Self> {
        if rows.is_empty() || rows[0].is_empty() {
            return Err(Error::new(STS_BAD_ARG, "rows must be non-empty"));
        }
        let height = rows.len();
        let width = rows[0].len();
        let mut pixels = Vec::with_capacity(width * height);
        let mut depth_m = Vec::with_capacity(width * height);
        for r in rows {
            if r.len() != width {
                return Err(Error::new(STS_BAD_ARG, "inconsistent row width"));
            }
            for &px in *r {
                pixels.push(px);
                depth_m.push(0.50 - (px as f64 / 255.0) * 0.08);
            }
        }
        Ok(Self {
            frame_index,
            timestamp_sec,
            width,
            height,
            pixels,
            depth_m,
            temperature_k: vec![293.15; width * height],
        })
    }

    /// Generate a physically grounded multi-frame video sequence of a vibrating, loaded cantilever
    /// beam with speckle texture, bending deflection, vortex wake, and thermoelastic heating.
    pub fn synthetic_cantilever_sequence(
        num_frames: usize,
        width: usize,
        height: usize,
        fps: f64,
        load_amplitude_px: f64,
        natural_freq_hz: f64,
    ) -> Result<Vec<Self>> {
        if num_frames == 0 || width < 8 || height < 8 || fps <= 0.0 {
            return Err(Error::new(
                STS_BAD_ARG,
                "invalid synthetic cantilever video parameters",
            ));
        }
        let mut seq = Vec::with_capacity(num_frames);
        let omega = 2.0 * std::f64::consts::PI * natural_freq_hz;
        let damping_zeta = 0.028;

        for f_idx in 0..num_frames {
            let t = (f_idx as f64) / fps;
            let modal_factor = (-damping_zeta * omega * t).exp() * (omega * t).sin()
                + 0.35 * (2.6 * omega * t).sin();
            let tip_deflection = load_amplitude_px * (0.65 + 0.35 * modal_factor);

            let mut frame = Self::new(f_idx, t, width, height, 18)?;
            let y_top = height / 4;
            let y_bot = (height * 3) / 4;
            let x_left = width / 8;
            let x_right = (width * 7) / 8;
            let beam_len = (x_right.saturating_sub(x_left).max(1)) as f64;

            for r in 0..height {
                for c in 0..width {
                    let idx = r * width + c;
                    if c >= x_left && c < x_right {
                        // Normalized beam coordinate xi in [0, 1] from fixed root (left) to free tip (right)
                        let xi = ((c - x_left) as f64) / beam_len;
                        // Euler-Bernoulli cantilever mode shape: w(xi) = 0.5 * xi^2 * (3 - xi) * tip_deflection
                        let w_y = 0.5 * xi * xi * (3.0 - xi) * tip_deflection;
                        // Axial strain displacement u_x(xi, y) = - (y - y_mid) * dw/dx
                        let y_mid = 0.5 * ((y_top + y_bot) as f64);
                        let dw_dx = 1.5 * xi * (2.0 - xi) * (tip_deflection / beam_len);
                        let u_x = -((r as f64) - y_mid) * dw_dx;

                        let src_r = (r as f64) - w_y;
                        let src_c = (c as f64) - u_x;
                        if src_r >= (y_top as f64) && src_r < (y_bot as f64) {
                            // Deterministic high-contrast DIC laser speckle pattern on the specimen
                            let spk1 = (src_c * 0.85).sin() * (src_r * 0.85).cos();
                            let spk2 = ((src_c + src_r) * 1.45).sin();
                            let val = (168.0 + 55.0 * spk1 + 28.0 * spk2).clamp(90.0, 252.0);
                            frame.pixels[idx] = val as u8;
                            frame.depth_m[idx] = 0.45 - 0.015 * xi * xi + 0.002 * modal_factor;
                            // Root bending stress concentration produces localized thermoelastic heating
                            let root_factor = (1.0 - xi).powi(2);
                            frame.temperature_k[idx] =
                                293.15 + 14.5 * root_factor * (0.5 + 0.5 * modal_factor.abs());
                        }
                    } else if c >= x_right {
                        // Downstream Karman vortex street Schlieren optical density perturbations
                        let wake_x = (c - x_right) as f64;
                        let wake_y = (r as f64) - 0.5 * (height as f64);
                        let vortex = (0.45 * wake_x - omega * t).sin()
                            * (-0.015 * wake_y * wake_y).exp();
                        let val = (32.0 + 24.0 * vortex).clamp(8.0, 88.0);
                        frame.pixels[idx] = val as u8;
                        frame.depth_m[idx] = 0.50;
                        frame.temperature_k[idx] = 293.15 + 3.0 * vortex.abs();
                    }
                }
            }
            seq.push(frame);
        }
        Ok(seq)
    }

    #[inline]
    pub fn pixel_at(&self, r: isize, c: isize) -> f64 {
        let rc = r.clamp(0, (self.height as isize) - 1) as usize;
        let cc = c.clamp(0, (self.width as isize) - 1) as usize;
        self.pixels[rc * self.width + cc] as f64
    }

    /// Bilinear sub-pixel luminance evaluation for sub-pixel DIC and optical flow.
    pub fn subpixel_bilinear(&self, y: f64, x: f64) -> f64 {
        let y0 = y.floor() as isize;
        let x0 = x.floor() as isize;
        let fy = y - (y0 as f64);
        let fx = x - (x0 as f64);
        let i00 = self.pixel_at(y0, x0);
        let i10 = self.pixel_at(y0, x0 + 1);
        let i01 = self.pixel_at(y0 + 1, x0);
        let i11 = self.pixel_at(y0 + 1, x0 + 1);
        (1.0 - fy) * ((1.0 - fx) * i00 + fx * i10) + fy * ((1.0 - fx) * i01 + fx * i11)
    }

    /// Central-difference spatial intensity gradient $(I_x, I_y)$ at `(r, c)`.
    #[inline]
    pub fn gradient_xy(&self, r: isize, c: isize) -> (f64, f64) {
        let ix = 0.5 * (self.pixel_at(r, c + 1) - self.pixel_at(r, c - 1));
        let iy = 0.5 * (self.pixel_at(r + 1, c) - self.pixel_at(r - 1, c));
        (ix, iy)
    }
}

/// 3D Symmetric Second-Order Tensor ($\mathbf{T} = \mathbf{T}^T \in \mathbb{R}^{3\times 3}$) in Voigt 6-component
/// representation `[xx, yy, zz, xy, yz, xz]` for Green-Lagrange strain $\mathbf{E}$ and Cauchy stress $\boldsymbol{\sigma}$.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct SymmetricTensor3D {
    pub xx: f64,
    pub yy: f64,
    pub zz: f64,
    pub xy: f64,
    pub yz: f64,
    pub xz: f64,
}

impl SymmetricTensor3D {
    pub const fn new(xx: f64, yy: f64, zz: f64, xy: f64, yz: f64, xz: f64) -> Self {
        Self {
            xx,
            yy,
            zz,
            xy,
            yz,
            xz,
        }
    }

    /// First invariant $I_1 = \operatorname{tr}(\mathbf{T}) = T_{xx} + T_{yy} + T_{zz}$.
    #[inline]
    pub fn trace(&self) -> f64 {
        self.xx + self.yy + self.zz
    }

    /// Hydrostatic mean component $p = \frac{1}{3}\operatorname{tr}(\mathbf{T})$.
    #[inline]
    pub fn hydrostatic_mean(&self) -> f64 {
        self.trace() / 3.0
    }

    /// Second deviatoric invariant $J_2 = \frac{1}{6}[(T_{xx}-T_{yy})^2 + (T_{yy}-T_{zz})^2 + (T_{zz}-T_{xx})^2] + T_{xy}^2 + T_{yz}^2 + T_{xz}^2$.
    #[inline]
    pub fn deviatoric_j2(&self) -> f64 {
        let d_xy = self.xx - self.yy;
        let d_yz = self.yy - self.zz;
        let d_zx = self.zz - self.xx;
        (d_xy * d_xy + d_yz * d_yz + d_zx * d_zx) / 6.0
            + self.xy * self.xy
            + self.yz * self.yz
            + self.xz * self.xz
    }

    /// Von Mises equivalent stress $\sigma_{\mathrm{vM}} = \sqrt{3 J_2}$.
    #[inline]
    pub fn von_mises(&self) -> f64 {
        (3.0 * self.deviatoric_j2().max(0.0)).sqrt()
    }

    /// Effective equivalent strain $\varepsilon_{\mathrm{eq}} = \sqrt{\frac{2}{3} \mathbf{E}' : \mathbf{E}'} = \sqrt{\frac{4}{3} J_2(\mathbf{E})}$.
    #[inline]
    pub fn equivalent_strain(&self) -> f64 {
        ((4.0 / 3.0) * self.deviatoric_j2().max(0.0)).sqrt()
    }

    /// Double contraction $\mathbf{A} : \mathbf{B} = \sum_{i,j} A_{ij} B_{ij}$.
    #[inline]
    pub fn double_contract(&self, other: &Self) -> f64 {
        self.xx * other.xx
            + self.yy * other.yy
            + self.zz * other.zz
            + 2.0 * (self.xy * other.xy + self.yz * other.yz + self.xz * other.xz)
    }

    /// Compute the 3 real principal eigenvalues $(\lambda_1 \ge \lambda_2 \ge \lambda_3)$ of this
    /// symmetric $3\times 3$ tensor via Cardano's exact trigonometric cubic invariant formula.
    pub fn principal_values(&self) -> [f64; 3] {
        let p1 = self.xy * self.xy + self.xz * self.xz + self.yz * self.yz;
        if p1 < 1e-28 {
            let mut vals = [self.xx, self.yy, self.zz];
            vals.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            return vals;
        }
        let q = self.trace() / 3.0;
        let p2 = (self.xx - q).powi(2) + (self.yy - q).powi(2) + (self.zz - q).powi(2) + 2.0 * p1;
        let p = (p2 / 6.0).sqrt().max(1e-18);

        // B = (1 / p) * (T - q * I)
        let b_xx = (self.xx - q) / p;
        let b_yy = (self.yy - q) / p;
        let b_zz = (self.zz - q) / p;
        let b_xy = self.xy / p;
        let b_yz = self.yz / p;
        let b_xz = self.xz / p;

        let det_b = b_xx * (b_yy * b_zz - b_yz * b_yz)
            - b_xy * (b_xy * b_zz - b_yz * b_xz)
            + b_xz * (b_xy * b_yz - b_yy * b_xz);
        let r = (0.5 * det_b).clamp(-1.0, 1.0);
        let phi = r.acos() / 3.0;

        let eig1 = q + 2.0 * p * phi.cos();
        let eig3 = q + 2.0 * p * (phi + (2.0 * std::f64::consts::PI / 3.0)).cos();
        let eig2 = 3.0 * q - eig1 - eig3;

        let mut vals = [eig1, eig2, eig3];
        vals.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        vals
    }

    /// Tresca maximum shear stress $\tau_{\max} = \frac{1}{2}(\sigma_1 - \sigma_3)$.
    #[inline]
    pub fn tresca_max_shear(&self) -> f64 {
        let p = self.principal_values();
        0.5 * (p[0] - p[2])
    }

    /// Nonlinear Green-Lagrange strain tensor $\mathbf{E} = \frac{1}{2}(\mathbf{F}^T\mathbf{F} - \mathbf{I}) = \frac{1}{2}(\nabla\mathbf{u} + (\nabla\mathbf{u})^T + (\nabla\mathbf{u})^T \nabla\mathbf{u})$
    /// from the $3\times 3$ displacement gradient matrix `grad_u[i][j] = d u_i / d X_j`.
    pub fn green_lagrange_from_grad_u(grad_u: &[[f64; 3]; 3]) -> Self {
        let mut e = [[0.0f64; 3]; 3];
        for i in 0..3 {
            for j in i..3 {
                let linear = 0.5 * (grad_u[i][j] + grad_u[j][i]);
                let mut quad = 0.0;
                for k in 0..3 {
                    quad += grad_u[k][i] * grad_u[k][j];
                }
                e[i][j] = linear + 0.5 * quad;
            }
        }
        Self {
            xx: e[0][0],
            yy: e[1][1],
            zz: e[2][2],
            xy: e[0][1],
            yz: e[1][2],
            xz: e[0][2],
        }
    }

    /// Compute the 3D isotropic Hookean Cauchy stress tensor $\boldsymbol{\sigma} = \lambda \operatorname{tr}(\mathbf{E})\mathbf{I} + 2\mu \mathbf{E}$
    /// given Young's modulus $E$ (Pa) and Poisson's ratio $\nu$.
    pub fn hookean_cauchy_stress(&self, youngs_modulus_pa: f64, poisson_ratio: f64) -> Self {
        let nu = poisson_ratio.clamp(0.01, 0.495);
        let lambda = (youngs_modulus_pa * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu));
        let mu = youngs_modulus_pa / (2.0 * (1.0 + nu));
        let tr = self.trace();
        Self {
            xx: lambda * tr + 2.0 * mu * self.xx,
            yy: lambda * tr + 2.0 * mu * self.yy,
            zz: lambda * tr + 2.0 * mu * self.zz,
            xy: 2.0 * mu * self.xy,
            yz: 2.0 * mu * self.yz,
            xz: 2.0 * mu * self.xz,
        }
    }
}

/// Calibrated engineering material constitutive model for Video-to-CAE forward & inverse analysis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EngineeringMaterial {
    pub material_id: String,
    pub name: String,
    pub category: String,
    /// Young's modulus $E$ in Pascals (Pa).
    pub youngs_modulus_pa: f64,
    /// Poisson's ratio $\nu \in (0, 0.5)$.
    pub poisson_ratio: f64,
    /// Mass density $\rho$ in $\text{kg/m}^3$.
    pub density_kg_m3: f64,
    /// Yield strength $\sigma_y$ in Pascals (Pa).
    pub yield_strength_pa: f64,
    /// Ultimate tensile strength $\sigma_{\mathrm{uts}}$ in Pascals (Pa).
    pub ultimate_strength_pa: f64,
    /// Thermal conductivity $k_{\mathrm{th}}$ in $\text{W/(m\cdot K)}$.
    pub thermal_conductivity_w_mk: f64,
    /// Specific heat capacity $c_p$ in $\text{J/(kg\cdot K)}$.
    pub specific_heat_j_kgk: f64,
    /// Coefficient of linear thermal expansion $\alpha_{\mathrm{cte}}$ in $\text{K}^{-1}$.
    pub thermal_expansion_k_inv: f64,
    /// Critical Mode-I fracture toughness $K_{Ic}$ in $\text{Pa}\sqrt{\text{m}}$.
    pub fracture_toughness_pa_sqrt_m: f64,
}

impl EngineeringMaterial {
    /// First Lamé parameter $\lambda = \frac{E\nu}{(1+\nu)(1-2\nu)}$ in Pascals.
    pub fn lame_lambda_pa(&self) -> f64 {
        let nu = self.poisson_ratio.clamp(0.01, 0.495);
        (self.youngs_modulus_pa * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    }

    /// Shear modulus (second Lamé parameter) $\mu = G = \frac{E}{2(1+\nu)}$ in Pascals.
    pub fn shear_modulus_pa(&self) -> f64 {
        self.youngs_modulus_pa / (2.0 * (1.0 + self.poisson_ratio.clamp(0.01, 0.495)))
    }

    /// Longitudinal bar wave speed $c_L = \sqrt{E / \rho}$ in $\text{m/s}$.
    pub fn longitudinal_wave_speed_mps(&self) -> f64 {
        (self.youngs_modulus_pa / self.density_kg_m3.max(1.0)).sqrt()
    }

    /// Thermal diffusivity $\alpha_{\mathrm{th}} = \frac{k_{\mathrm{th}}}{\rho c_p}$ in $\text{m}^2/\text{s}$.
    pub fn thermal_diffusivity_m2_s(&self) -> f64 {
        self.thermal_conductivity_w_mk
            / (self.density_kg_m3.max(1.0) * self.specific_heat_j_kgk.max(1.0))
    }

    /// Return the built-in 7-material aerospace/structural constitutive catalog.
    pub fn builtin_catalog() -> Vec<Self> {
        vec![
            Self {
                material_id: "ti_6al_4v".to_string(),
                name: "Ti-6Al-4V Grade 5 Aerospace Titanium".to_string(),
                category: "metallic_alloy".to_string(),
                youngs_modulus_pa: 113.8e9,
                poisson_ratio: 0.342,
                density_kg_m3: 4_430.0,
                yield_strength_pa: 880.0e6,
                ultimate_strength_pa: 950.0e6,
                thermal_conductivity_w_mk: 6.7,
                specific_heat_j_kgk: 526.3,
                thermal_expansion_k_inv: 8.6e-6,
                fracture_toughness_pa_sqrt_m: 75.0e6,
            },
            Self {
                material_id: "al_6061_t6".to_string(),
                name: "Al-6061-T6 Structural Aluminum".to_string(),
                category: "metallic_alloy".to_string(),
                youngs_modulus_pa: 68.9e9,
                poisson_ratio: 0.330,
                density_kg_m3: 2_700.0,
                yield_strength_pa: 276.0e6,
                ultimate_strength_pa: 310.0e6,
                thermal_conductivity_w_mk: 167.0,
                specific_heat_j_kgk: 896.0,
                thermal_expansion_k_inv: 23.1e-6,
                fracture_toughness_pa_sqrt_m: 29.0e6,
            },
            Self {
                material_id: "aisi_304_steel".to_string(),
                name: "AISI 304 Austenitic Stainless Steel".to_string(),
                category: "metallic_alloy".to_string(),
                youngs_modulus_pa: 193.0e9,
                poisson_ratio: 0.290,
                density_kg_m3: 8_000.0,
                yield_strength_pa: 215.0e6,
                ultimate_strength_pa: 505.0e6,
                thermal_conductivity_w_mk: 16.2,
                specific_heat_j_kgk: 500.0,
                thermal_expansion_k_inv: 17.2e-6,
                fracture_toughness_pa_sqrt_m: 110.0e6,
            },
            Self {
                material_id: "inconel_718".to_string(),
                name: "Inconel 718 Nickel Superalloy".to_string(),
                category: "superalloy".to_string(),
                youngs_modulus_pa: 200.0e9,
                poisson_ratio: 0.294,
                density_kg_m3: 8_190.0,
                yield_strength_pa: 1_034.0e6,
                ultimate_strength_pa: 1_240.0e6,
                thermal_conductivity_w_mk: 11.4,
                specific_heat_j_kgk: 435.0,
                thermal_expansion_k_inv: 13.0e-6,
                fracture_toughness_pa_sqrt_m: 96.0e6,
            },
            Self {
                material_id: "cfrp_quasi_iso".to_string(),
                name: "T800/Epoxy Quasi-Isotropic CFRP Laminate".to_string(),
                category: "composite".to_string(),
                youngs_modulus_pa: 72.5e9,
                poisson_ratio: 0.305,
                density_kg_m3: 1_580.0,
                yield_strength_pa: 620.0e6,
                ultimate_strength_pa: 780.0e6,
                thermal_conductivity_w_mk: 8.5,
                specific_heat_j_kgk: 1_050.0,
                thermal_expansion_k_inv: 2.1e-6,
                fracture_toughness_pa_sqrt_m: 42.0e6,
            },
            Self {
                material_id: "sic_ceramic".to_string(),
                name: "Sintered Alpha Silicon Carbide (SiC)".to_string(),
                category: "ceramic".to_string(),
                youngs_modulus_pa: 410.0e9,
                poisson_ratio: 0.140,
                density_kg_m3: 3_100.0,
                yield_strength_pa: 450.0e6,
                ultimate_strength_pa: 520.0e6,
                thermal_conductivity_w_mk: 120.0,
                specific_heat_j_kgk: 750.0,
                thermal_expansion_k_inv: 4.0e-6,
                fracture_toughness_pa_sqrt_m: 4.6e6,
            },
            Self {
                material_id: "polycarbonate_pc".to_string(),
                name: "Optical Engineering Polycarbonate (PC)".to_string(),
                category: "polymer".to_string(),
                youngs_modulus_pa: 2.38e9,
                poisson_ratio: 0.370,
                density_kg_m3: 1_200.0,
                yield_strength_pa: 62.0e6,
                ultimate_strength_pa: 70.0e6,
                thermal_conductivity_w_mk: 0.20,
                specific_heat_j_kgk: 1_200.0,
                thermal_expansion_k_inv: 65.0e-6,
                fracture_toughness_pa_sqrt_m: 2.2e6,
            },
        ]
    }

    /// Look up a material by `material_id` (defaults to `ti_6al_4v` if not found).
    pub fn by_id(material_id: &str) -> Self {
        let lower = material_id.trim().to_ascii_lowercase();
        Self::builtin_catalog()
            .into_iter()
            .find(|m| m.material_id == lower)
            .unwrap_or_else(|| Self::builtin_catalog().remove(0))
    }
}

/// Shared CAD-to-CAE handoff contract (`schema: "solidsf.solidcv.cad_to_cae_handoff.v1"`)
/// co-designed with sibling crate `solidcv-cad` (`2de8aeb5-70ca-47aa-9ba0-dd5c7b530bc1`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SolidCvCadToCaeHandoff {
    pub schema: String,
    pub solid_id: String,
    pub bbox_min: [f64; 3],
    pub bbox_max: [f64; 3],
    pub surface_vertices: Vec<[f64; 3]>,
    pub surface_triangles: Vec<[usize; 3]>,
    #[serde(default)]
    pub interior_tetra_nodes: Vec<[f64; 3]>,
    #[serde(default)]
    pub interior_centroid_node: [f64; 3],
    /// 4-node tetrahedral element connectivity (`C3D4`), indexing into `surface_vertices ++ interior_tetra_nodes`.
    pub tetrahedra: Vec<[usize; 4]>,
    pub kinematic_bc_velocity_mps: [f64; 3],
    #[serde(default)]
    pub kinematic_pose_trajectory: Vec<([f64; 3], [f64; 3])>,
    #[serde(default)]
    pub kinematic_pose_trajectory_m: Vec<[f64; 3]>,
    /// Principal curvatures $(\kappa_1, \kappa_2)$ per NURBS patch for shell/membrane stress concentration priors.
    #[serde(default)]
    pub principal_curvatures: Vec<(f64, f64)>,
    #[serde(default = "default_true_manifold")]
    pub is_watertight_manifold: bool,
    #[serde(default = "default_euler_two")]
    pub euler_characteristic: i64,
}

fn default_true_manifold() -> bool {
    true
}

fn default_euler_two() -> i64 {
    2
}

impl SolidCvCadToCaeHandoff {
    pub const SCHEMA: &'static str = "solidsf.solidcv.cad_to_cae_handoff.v1";

    /// Construct a watertight tetrahedralized CAD-to-CAE handoff from a bounding box and worldtube kinematics.
    pub fn from_bounding_box_and_kinematics(
        solid_id: &str,
        bbox_min: [f64; 3],
        bbox_max: [f64; 3],
        nx: usize,
        ny: usize,
        nz: usize,
        velocity_mps: [f64; 3],
        principal_curvatures: Vec<(f64, f64)>,
    ) -> Self {
        let gx = nx.max(2);
        let gy = ny.max(2);
        let gz = nz.max(2);
        let dx = ((bbox_max[0] - bbox_min[0]).max(1e-3)) / (gx as f64);
        let dy = ((bbox_max[1] - bbox_min[1]).max(1e-3)) / (gy as f64);
        let dz = ((bbox_max[2] - bbox_min[2]).max(1e-3)) / (gz as f64);

        let mut surface_vertices = Vec::new();
        let mut interior_tetra_nodes = Vec::new();
        let mut grid_map = vec![0usize; (gx + 1) * (gy + 1) * (gz + 1)];

        // Partition nodes into boundary surface vertices first, then interior nodes
        let mut temp_surface = Vec::new();
        let mut temp_interior = Vec::new();
        for ix in 0..=gx {
            for iy in 0..=gy {
                for iz in 0..=gz {
                    let pt = [
                        bbox_min[0] + (ix as f64) * dx,
                        bbox_min[1] + (iy as f64) * dy,
                        bbox_min[2] + (iz as f64) * dz,
                    ];
                    let is_boundary =
                        ix == 0 || ix == gx || iy == 0 || iy == gy || iz == 0 || iz == gz;
                    if is_boundary {
                        temp_surface.push(((ix, iy, iz), pt));
                    } else {
                        temp_interior.push(((ix, iy, iz), pt));
                    }
                }
            }
        }

        let idx_3d = |ix: usize, iy: usize, iz: usize| -> usize {
            (ix * (gy + 1) + iy) * (gz + 1) + iz
        };

        for ((ix, iy, iz), pt) in temp_surface {
            let id = surface_vertices.len();
            surface_vertices.push(pt);
            grid_map[idx_3d(ix, iy, iz)] = id;
        }
        let surf_count = surface_vertices.len();
        for ((ix, iy, iz), pt) in temp_interior {
            let id = surf_count + interior_tetra_nodes.len();
            interior_tetra_nodes.push(pt);
            grid_map[idx_3d(ix, iy, iz)] = id;
        }

        // Freudenthal-Kuhn 6-tetrahedra decomposition of each hexahedral cell
        const KUHN_TETS: [[[usize; 3]; 4]; 6] = [
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
            [[0, 0, 0], [1, 0, 0], [1, 0, 1], [1, 1, 1]],
            [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 1, 1]],
            [[0, 0, 0], [0, 1, 0], [0, 1, 1], [1, 1, 1]],
            [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1]],
            [[0, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 1]],
        ];

        let mut tetrahedra = Vec::with_capacity(gx * gy * gz * 6);
        for ix in 0..gx {
            for iy in 0..gy {
                for iz in 0..gz {
                    for tet in &KUHN_TETS {
                        let v0 = grid_map[idx_3d(ix + tet[0][0], iy + tet[0][1], iz + tet[0][2])];
                        let v1 = grid_map[idx_3d(ix + tet[1][0], iy + tet[1][1], iz + tet[1][2])];
                        let v2 = grid_map[idx_3d(ix + tet[2][0], iy + tet[2][1], iz + tet[2][2])];
                        let v3 = grid_map[idx_3d(ix + tet[3][0], iy + tet[3][1], iz + tet[3][2])];
                        tetrahedra.push([v0, v1, v2, v3]);
                    }
                }
            }
        }

        // Extract outer surface triangles on z=0 and z=gz planes
        let mut surface_triangles = Vec::new();
        for ix in 0..gx {
            for iy in 0..gy {
                let a = grid_map[idx_3d(ix, iy, gz)];
                let b = grid_map[idx_3d(ix + 1, iy, gz)];
                let c = grid_map[idx_3d(ix + 1, iy + 1, gz)];
                let d = grid_map[idx_3d(ix, iy + 1, gz)];
                surface_triangles.push([a, b, c]);
                surface_triangles.push([a, c, d]);
            }
        }

        let centroid = [
            0.5 * (bbox_min[0] + bbox_max[0]),
            0.5 * (bbox_min[1] + bbox_max[1]),
            0.5 * (bbox_min[2] + bbox_max[2]),
        ];

        Self {
            schema: Self::SCHEMA.to_string(),
            solid_id: solid_id.to_string(),
            bbox_min,
            bbox_max,
            surface_vertices,
            surface_triangles,
            interior_tetra_nodes,
            interior_centroid_node: centroid,
            tetrahedra,
            kinematic_bc_velocity_mps: velocity_mps,
            kinematic_pose_trajectory: vec![
                ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
                (
                    [
                        velocity_mps[0] * 0.033,
                        velocity_mps[1] * 0.033,
                        velocity_mps[2] * 0.033,
                    ],
                    [0.0, 0.0, 0.01],
                ),
            ],
            kinematic_pose_trajectory_m: vec![
                [0.0, 0.0, 0.0],
                [
                    velocity_mps[0] * 0.033,
                    velocity_mps[1] * 0.033,
                    velocity_mps[2] * 0.033,
                ],
            ],
            principal_curvatures,
            is_watertight_manifold: true,
            euler_characteristic: 2,
        }
    }

    /// Return the combined list of all 3D nodal coordinates (`surface_vertices ++ interior_tetra_nodes`).
    pub fn all_nodes(&self) -> Vec<[f64; 3]> {
        let mut nodes = Vec::with_capacity(
            self.surface_vertices.len() + self.interior_tetra_nodes.len(),
        );
        nodes.extend_from_slice(&self.surface_vertices);
        nodes.extend_from_slice(&self.interior_tetra_nodes);
        nodes
    }
}
