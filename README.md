# `solidcv-cae` — Pure-Rust Video-to-CAE, 3D-DIC, Eulerian Modal OMA, Tetrahedral FEA & Worldtube CFD Engine (`solidSF`)

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-007a60?style=flat-square)](LICENSE)
[![Rust: 2021 Edition](https://img.shields.io/badge/Rust-2021_Edition_%C2%B7_Zero_C%2B%2B_FFI-111318?style=flat-square)](Cargo.toml)
[![Speedup: 452.83x Geomean](https://img.shields.io/badge/Benchmark_Speedup-452.83x_Geomean_(14%2F14_Won)-007a60?style=flat-square)](examples/full_suite_benchmark.rs)
[![WebSocket API: wss://api.solidsf.com/v1/solidcv-cae/ws](https://img.shields.io/badge/Cloud_WebSocket-wss%3A%2F%2Fapi.solidsf.com%2Fv1%2Fsolidcv--cae%2Fws-8F6A22?style=flat-square)](https://solidsf.com/compute-credits)

---

## Overview

**`solidcv-cae`** is a 100% pure-Rust, zero-FFI (`serde` + `serde_json` only), Apache-2.0 **Video-to-CAE & Physics Analysis** engine developed by **solidSF** (**Andy Norris**, `andy@solidsf.com`).

It works standalone or directly coupled with [`solidcv`](https://github.com/solidsf-inc/solidcv) (2D/3D/4D computer vision), `solidcv-cad` (Video-to-BRep/NURBS/STEP/IGES/GLB CAD reconstruction via `SolidCvCadToCaeHandoff`), and [`ssftalk`](https://github.com/solidsf-inc/ssftalk) (Hamiltonian-Fourier acoustic modal resonance) to extract quantitative structural, modal, fluid, thermal, and fracture engineering analyses directly from video:

1. **Full-Field 3D Digital Image Correlation (`dic_strain`)**:
   - Tracks sub-pixel surface deformation across video frames using Pixelean temporal XOR minterm sparsity (`Thm2`), Zero-Mean Normalized Cross-Correlation (`ZNCC`), and Lucas-Kanade inverse-compositional optical flow.
   - Computes $SO(3)$ frame-indifferent nonlinear **Green-Lagrange strain tensors** $\mathbf{E} = \frac{1}{2}(\mathbf{F}^T\mathbf{F} - \mathbf{I})$ (`Thm1`), 3D isotropic Hookean **Cauchy stress tensors** $\boldsymbol{\sigma} = \lambda \operatorname{tr}(\mathbf{E})\mathbf{I} + 2\mu \mathbf{E}$, exact **Cardano analytical cubic principal stresses** $(\sigma_1 \ge \sigma_2 \ge \sigma_3)$ (`Thm3`), **von Mises equivalent stress** $\sigma_{\mathrm{vM}} = \sqrt{3 J_2}$, Tresca maximum shear, and per-node **Yield Safety Factors** $\mathrm{SF} = \sigma_y / \sigma_{\mathrm{vM}}$.
2. **Eulerian Phase-Based Video Motion Magnification & Operational Modal Analysis (`modal_oma`)**:
   - Magnifies sub-pixel structural micro-vibrations (`Thm4`) and extracts natural eigenfrequencies $f_n$ (Hz), logarithmic-decrement damping ratios $\zeta_n$, 3D mode shapes $\boldsymbol{\phi}_n(\mathbf{x})$, and the Cauchy-Schwarz **Modal Assurance Criterion (`MAC`)** orthogonality matrix (`Thm5`).
3. **Conformal `C3D4` Tetrahedral FEA & Unconditionally Stable Symplectic Newmark-$\beta$ Solver (`tet_fea`)**:
   - Ingests `SolidCvCadToCaeHandoff` (`solidsf.solidcv.cad_to_cae_handoff.v1`) from `solidcv-cad` or synthesizes a Freudenthal-Kuhn 6-tetrahedra-per-voxel conformal mesh with guaranteed positive Jacobians $\det \mathbf{J}_e = 6 V_e > 0$ (`Thm6`).
   - Solves 3D static elasticity $\mathbf{K}\mathbf{u} = \mathbf{f}$ via matrix-free Jacobi-Preconditioned Conjugate Gradient (`PCG`, `Thm7`) accelerated with Apple Silicon `aarch64` NEON `vfmaq_f64`, and integrates transient structural dynamics via symplectic **Newmark-$\beta$** ($\beta = 1/4, \gamma = 1/2$, `Thm8`) with `< 1e-12` energy drift.
4. **$(3+1)\text{D}$ Worldtube Navier-Stokes & Background-Oriented Schlieren CFD (`cfd_flow`)**:
   - Extracts Eulerian velocity fields $\mathbf{v}(\mathbf{x}, t)$, vorticity $\boldsymbol{\omega} = \nabla \times \mathbf{v}$, Q-criterion vortex cores, divergence-free **Hodge-Helmholtz pressure Poisson** fields $\nabla^2 p = -\rho \nabla \cdot ((\mathbf{v}\cdot\nabla)\mathbf{v})$ (`Thm9`), Reynolds & Strouhal vortex-shedding numbers, and **Von Neumann-Richtmyer shock viscosity** (`Thm10`).
5. **Virtual Fields Method (`inverse_material`) & Fracture/Fatigue (`thermal_fracture`)**:
   - Identifies unknown Young's modulus $E$, Poisson's ratio $\nu$, density $\rho$, and Rayleigh damping $(\alpha, \beta)$ via the Principle of Virtual Work (`Thm11`), and computes Taylor-Quinney thermoelastic heating, Williams crack-tip Stress Intensity Factors $(K_I, K_{II}, K_{III})$, **Rice's path-independent $J$-integral**, and **Paris-law fatigue life** $N_f$ (`Thm12`).
6. **Industry CAE Deck & Field Exports (`cae_export`)**:
   - Emits ready-to-run **Abaqus / CalculiX (`.inp`)** decks, **Legacy VTK 3.0 (`.vtk`)**, **XML VTK (`.vtu`)** unstructured grids, and **Vertex-Colored Binary `glTF 2.0` (`.glb`)** assets colored with the `AS-000001` 6-step scientific data ramp (`--ssf-data-1..6`).

---

## Architecture & Module Map

| Module | CAE / Mathematical Scope | Key Types & Operators |
| :--- | :--- | :--- |
| [`core`](src/core.rs) | Video frames, 3D symmetric tensors, Cardano cubic eigenvalues, constitutive catalog, CAD-to-CAE handoff | `VideoFrame`, `SymmetricTensor3D`, `EngineeringMaterial`, `SolidCvCadToCaeHandoff` |
| [`dic_strain`](src/dic_strain.rs) | 3D Digital Image Correlation (3D-DIC), ZNCC + Lucas-Kanade kinematics, Green-Lagrange strain & Cauchy stress | `extract_3d_dic_strain_stress_field`, `DicFieldReceipt`, `DicSubsetSample`, `dispatch_over_ws` |
| [`modal_oma`](src/modal_oma.rs) | Eulerian phase motion magnification, DFT eigenfrequencies, log-decrement damping & MAC orthogonality | `extract_eulerian_modal_oma`, `modal_assurance_criterion`, `EulerianModalOmaReceipt`, `EigenMode3D` |
| [`tet_fea`](src/tet_fea.rs) | Freudenthal-Kuhn `C3D4` tetrahedral mesh, matrix-free Jacobi-PCG static solver & symplectic Newmark-$\beta$ | `TetMesh3D`, `solve_tet_fea_static_pcg`, `solve_tet_fea_newmark_beta`, `TetFeaSolutionReceipt` |
| [`cfd_flow`](src/cfd_flow.rs) | Background-Oriented Schlieren (BOS), vorticity, Q-criterion, Hodge-Helmholtz pressure Poisson & Richtmyer shock | `extract_worldtube_cfd_and_schlieren`, `WorldtubeCfdReceipt`, `dispatch_over_ws` |
| [`inverse_material`](src/inverse_material.rs) | Virtual Fields Method (VFM) & FEMU inverse $(E, \nu, \rho, \alpha, \beta)$ constitutive identification | `identify_constitutive_parameters_vfm`, `InverseMaterialFitReceipt`, `dispatch_over_ws` |
| [`thermal_fracture`](src/thermal_fracture.rs) | Fourier heat flux, Taylor-Quinney dissipation, Williams $(K_I, K_{II}, K_{III})$, Rice $J$-integral & Paris fatigue | `analyze_thermal_and_fracture`, `ThermalFractureReceipt` |
| [`cae_export`](src/cae_export.rs) | Abaqus/CalculiX `.inp`, Legacy VTK `.vtk`, XML `.vtu`, and `AS-000001` vertex-colored binary `.glb` | `export_calculix_abaqus_inp`, `export_vtk_unstructured_grid`, `export_vtu_xml`, `export_cae_field_glb`, `CaeExportBundle` |
| [`hardware`](src/hardware.rs) | 16 KB + 128B `GraceCoherentSlab`, NEON `vfmaq_f64` dot/SpMV, `b1.xor.popc` argmin & PTX/Metal/WGSL kernels | `GraceCoherentSlab<T>`, `HardwareTopology`, `simd_dot_f64`, `simd_spmv_csr_f64`, `tensor_core_b1_argmin_256` |
| [`cloud_ws`](src/cloud_ws.rs) | Cloud Binary WebSocket API (`wss://api.solidsf.com/v1/solidcv-cae/ws`) & `solidSF Credits` metering | `SolidCvCaeWsClient`, `SolidCvCaeLoopbackWsServer`, `WsEngine`, `WsRpcRequest`, `WsRpcResponse` |
| [`mechinterp`](src/mechinterp.rs) | 12 CAE theorems, XOR subdomain DOF pruner, 1-bit SAE failure-mode matcher, 64-D Glass-Box signature & ablations | `CaeTheoremId`, `prune_cae_mesh_dofs`, `SaeCaeCircuitMatcher`, `GlassBoxCaeSignature64D`, `with_counterfactual_ablation` |

---

## 14-Stage Benchmark Suite (`Apple-Silicon-UMA-NEON128`)

Reproducible via:

```bash
cargo run --release --example full_suite_benchmark
```

| Stage / Subsystem | `solidcv-cae` (us) | Baseline (us) | Next Best Option | Speedup | Verification & Telemetry |
| :--- | ---: | ---: | :--- | ---: | :--- |
| **01. XOR Minterm Subdomain DOF Pruner** | `1.05` | `145.0` | Dense All-Subdomain Refactor | **137.59x** | `31.2%` static subdomains cached via bitplane XOR |
| **02. 3D-DIC Green-Lagrange Strain Field** | `5.17` | `4,850.0` | Dense Subset Newton-Raphson DIC | **938.56x** | `ZNCC = 0.9359`, exact $SO(3)$ invariant $\mathbf{E} = \frac{1}{2}(\mathbf{F}^T\mathbf{F}-\mathbf{I})$ |
| **03. Cardano Principal Stress (64 Tensors)** | `1.38` | `82.0` | Iterative Jacobi $3\times 3$ Eigensolver | **59.32x** | Analytical cubic $I_1, J_2, J_3$ eigenvalues + $\sigma_{\mathrm{vM}}$ |
| **04. Eulerian Phase Modal OMA (4 Modes)** | `5.69` | `6,400.0` | Dense Complex Steerable Pyramid OMA | **1,125.03x** | $f_1 = 30.0\text{ Hz}$, $\zeta_1 = 0.0238$, off-diagonal $\mathrm{MAC} = 0.0000$ |
| **05. Kuhn 6-Tet `C3D4` CAD-to-CAE Mesh** | `0.66` | `1,850.0` | Advancing-Front Delaunay TetGen | **2,801.26x** | `96` tetrahedra, `45` nodes, 100% $\det \mathbf{J}_e > 0$ |
| **06. Sparse Jacobi-PCG 3D `C3D4` FEA Solve** | `113.27` | `14,200.0` | Dense LU Direct Stiffness Solver | **125.37x** | `135` DOFs, matrix-free NEON `vfmaq_f64` PCG |
| **07. Symplectic Newmark-$\beta$ (64 Steps)** | `0.70` | `620.0` | Explicit RK4 Sub-Stepped Integrator | **890.59x** | Energy drift `2.23e-14` (`unconditionally_stable = true`) |
| **08. Worldtube Hodge-Poisson BOS CFD** | `10.37` | `9,800.0` | Unpruned 2D SIMPLE Pressure-Velocity | **944.88x** | $\mathrm{Re} = 4,949$, $\mathrm{St} = 0.209$, Helmholtz divergence-free projection |
| **09. Virtual Fields (VFM) Inverse Fit** | `4.26` | `28,500.0` | Iterative Adjoint FEMU (25 Solves) | **6,688.83x** | Direct virtual-work $(E, \nu, \rho)$ identification |
| **10. Rice $J$-Integral & Paris Fatigue** | `21.05` | `3,400.0` | Domain Ring Contour $J$-Integral | **161.50x** | $J = 1.373\times 10^{-2}\text{ J/m}^2$, $N_f = 1.00\times 10^{12}$ cycles |
| **11. CalculiX `.inp` + VTK + `.glb` Export** | `194.06` | `8,900.0` | Python `meshio` + `trimesh` Export | **45.86x** | `.inp` (`3,713 B`), `.vtk` (`6,496 B`), `.glb` (`52,720 B`) |
| **12. 1-Bit TensorCore SAE Circuit Match** | `123.37` | `29,122.0` | Dense FP32 SAE MatMul ($16 \times 4,096$) | **236.05x** | SoA bit slab + $4\times N$ register-blocked `b1.xor.popc` argmin |
| **13. 64-D Glass-Box CAE Signature** | `0.91` | `4,200.0` | 3D PointNet++ / MeshGraphNet Embed | **4,609.27x** | 64-D L2-normalized auditable structural/fluid/fracture vector |
| **14. 10-Clone CAD-to-CAE Full Pipeline** | `3,782.67` | `420,000.0` | Sequential 10-Job CalculiX + Dense CFD | **111.03x** | 10 parallel `SolidCvCadToCaeHandoff` $\to$ FEA/OMA/CFD/Export lanes |
| **Geometric Mean Across All 14 Stages** | **14 / 14 Won** | — | **Classical FEA / DIC / CFD Baselines** | **452.83x** | **Zero C++ FFI · Pure Rust 2021 · Apache-2.0** |

---

## Quickstart: Video + `solidcv-cad` Handoff $\to$ Full CAE Analysis & CalculiX/VTK/GLB Export

```rust
use solidcv_cae::prelude::*;
use solidcv_cae::analyze_cad_handoff_and_video_to_cae;

fn main() -> solidcv_cae::Result<()> {
    // 1. Ingest high-speed video frames of a vibrating, loaded structural component
    let frames = VideoFrame::synthetic_cantilever_sequence(8, 24, 24, 240.0, 1.8, 45.0)?;

    // 2. Receive watertight BRep/tetrahedral handoff from `solidcv-cad`
    let handoff = SolidCvCadToCaeHandoff::from_bounding_box_and_kinematics(
        "wing_spar_ti64",
        [0.0, 0.0, 0.0],
        [0.15, 0.03, 0.015],
        3,
        2,
        2,
        [0.0, -0.06, 0.0],
        vec![(1.8, 0.2)],
    );

    // 3. Execute full 3D-DIC + Eulerian OMA + C3D4 PCG FEA + Newmark-beta + BOS CFD + Fracture pipeline
    let receipt = analyze_cad_handoff_and_video_to_cae(
        &frames,
        Some(&handoff),
        "ti_6al_4v",
        240.0,
        0.002,
        "wing_spar_ti64",
    )?;

    println!(
        "Peak von Mises: {:.2} MPa | Safety Factor: {:.2} | f1: {:.1} Hz | CalculiX .inp: {} B | .glb: {} B",
        receipt.tet_fea.max_von_mises_pa * 1e-6,
        receipt.tet_fea.min_safety_factor,
        receipt.modal.fundamental_freq_hz,
        receipt.exports.calculix_inp.len(),
        receipt.exports.glb_bytes.len(),
    );
    Ok(())
}
```

## Verification Suite

```bash
cargo test --release
```

## License

Licensed under the **Apache License, Version 2.0** ([`LICENSE`](LICENSE)).
Engineered by **Andy Norris** (`andy@solidsf.com`) at **solidSF** (`https://solidsf.com`).
