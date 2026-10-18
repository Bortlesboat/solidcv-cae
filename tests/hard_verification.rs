//! Hard mathematical, physical, numerical, and protocol verification suite for `solidcv-cae`.

use solidcv_cae::prelude::*;
use solidcv_cae::{
    analyze_cad_handoff_and_video_to_cae, analyze_video_to_cae, run_10_clone_solidcv_cae_suite,
};

#[test]
fn gate_01_so3_frame_indifference_and_cardano_eigenvalue_exactness() {
    reset_mind();
    // 1. Verify SO(3) rigid-body rotation invariance of nonlinear Green-Lagrange strain E = 1/2 (F^T F - I)
    for angle_deg in [-135.0_f64, -45.0, 0.0, 22.5, 60.0, 90.0, 179.0] {
        let theta = angle_deg.to_radians();
        let c = theta.cos();
        let s = theta.sin();
        let grad_u = [[c - 1.0, -s, 0.0], [s, c - 1.0, 0.0], [0.0, 0.0, 0.0]];
        let e = SymmetricTensor3D::green_lagrange_from_grad_u(&grad_u);
        assert!(
            e.equivalent_strain() < 1e-14,
            "Green-Lagrange strain must vanish identically under finite SO(3) rotation ({angle_deg} deg)"
        );
    }

    // 2. Verify Cardano analytical cubic eigenvalues match trace I_1 and J_2 invariants
    let sig = SymmetricTensor3D::new(240.0e6, -80.0e6, 50.0e6, 45.0e6, -20.0e6, 30.0e6);
    let p = sig.principal_values();
    assert!(p[0] >= p[1] && p[1] >= p[2]);
    let i1_from_p = p[0] + p[1] + p[2];
    assert!((i1_from_p - sig.trace()).abs() < 1e-3);
    let vm_from_p = (0.5 * ((p[0] - p[1]).powi(2) + (p[1] - p[2]).powi(2) + (p[2] - p[0]).powi(2)))
        .sqrt();
    assert!((vm_from_p - sig.von_mises()).abs() < 1e-3);
}

#[test]
fn gate_02_dic_3d_strain_stress_and_xor_dof_pruning() {
    reset_mind();
    let frames = VideoFrame::synthetic_cantilever_sequence(8, 32, 32, 240.0, 2.2, 42.0).unwrap();
    let prune = prune_cae_mesh_dofs(&frames[0], &frames[1], 8, 2, 1).unwrap();
    assert_eq!(prune.total_subdomains, 16);
    assert!(prune.dof_savings_ratio > 0.15);

    let ti = EngineeringMaterial::by_id("ti_6al_4v");
    let dic = extract_3d_dic_strain_stress_field(&frames, &ti, 5, 4, 0.002).unwrap();
    assert!(dic.xor_active_subsets > 0);
    assert!(dic.mean_zncc > 0.75);
    assert!(dic.max_von_mises_pa > 0.0);
    assert!(dic.min_safety_factor > 0.0);
}

#[test]
fn gate_03_eulerian_phase_modal_oma_and_mac_orthogonality() {
    reset_mind();
    let frames = VideoFrame::synthetic_cantilever_sequence(16, 32, 32, 240.0, 2.0, 45.0).unwrap();
    let oma = extract_eulerian_modal_oma(&frames, 240.0, 15.0, 0.002).unwrap();
    assert_eq!(oma.modes.len(), 4);
    assert!(oma.fundamental_freq_hz >= 15.0);
    assert!(oma.fundamental_damping_zeta > 0.0 && oma.fundamental_damping_zeta < 0.20);
    // Diagonal MAC entries must be 1.0, off-diagonal < 0.05
    for i in 0..4 {
        assert!((oma.mac_matrix[i][i] - 1.0).abs() < 1e-9);
    }
    assert!(oma.max_off_diagonal_mac < 0.05);
}

#[test]
fn gate_04_kuhn_6_tet_mesh_pcg_and_symplectic_newmark_beta() {
    reset_mind();
    let mesh =
        TetMesh3D::from_cantilever_box(0.16, 0.02, 0.03, 4, 2, 2, [0.0, -450.0, 0.0]).unwrap();
    assert_eq!(mesh.elements.len(), 4 * 2 * 2 * 6);
    assert!(mesh.all_positive_jacobians);
    assert!(mesh.min_jacobian_det > 0.0);
    assert!((mesh.total_volume_m3 - (0.16 * 0.02 * 0.03)).abs() < 1e-9);

    let mat = EngineeringMaterial::by_id("al_6061_t6");
    let sol = solve_tet_fea_static_pcg(&mesh, &mat, 100, 1e-8).unwrap();
    assert!(sol.pcg_iterations >= 1);
    assert!(sol.max_displacement_m > 0.0);
    assert!(sol.max_von_mises_pa > 0.0);
    assert!(sol.total_strain_energy_j > 0.0);

    let dyn_rec = solve_tet_fea_newmark_beta(&mesh, &mat, 0.0005, 64).unwrap();
    assert!(dyn_rec.unconditionally_stable);
    assert!(dyn_rec.max_energy_drift_ratio < 1e-12);
}

#[test]
fn gate_05_worldtube_cfd_hodge_poisson_and_richtmyer_shock() {
    reset_mind();
    let frames = VideoFrame::synthetic_cantilever_sequence(6, 32, 32, 240.0, 2.5, 50.0).unwrap();
    let cfd = extract_worldtube_cfd_and_schlieren(&frames, 240.0, 0.002, 1.225, 1.5e-5).unwrap();
    assert_eq!(cfd.grid_w, 32);
    assert_eq!(cfd.grid_h, 32);
    assert!(cfd.reynolds_number > 10.0);
    assert!(cfd.strouhal_number >= 0.12 && cfd.strouhal_number <= 0.24);
    assert!(cfd.max_vorticity_s_inv > 0.0);
}

#[test]
fn gate_06_inverse_vfm_thermal_fracture_and_counterfactual_ablation() {
    reset_mind();
    let frames = VideoFrame::synthetic_cantilever_sequence(8, 24, 24, 240.0, 1.6, 40.0).unwrap();
    let baseline = analyze_video_to_cae(&frames, "ti_6al_4v", 240.0, 0.002, "ti_beam").unwrap();
    assert!(baseline.inverse_material.identified_youngs_modulus_pa > 1.0e9);
    assert!(baseline.thermal_fracture.rice_j_integral_j_m2 > 0.0);
    assert!(baseline.thermal_fracture.paris_fatigue_cycles_to_failure > 100.0);

    // Run counterfactual ablation disabling nonlinear Green-Lagrange, thermoelastic coupling, and eigenmode 0
    let (ablated, summary) = with_counterfactual_ablation(
        AblationConfig {
            ablated_eigenmodes: vec![0],
            disable_nonlinear_green_lagrange: true,
            disable_richtmyer_shock_viscosity: true,
            disable_thermoelastic_coupling: true,
        },
        || analyze_video_to_cae(&frames, "ti_6al_4v", 240.0, 0.002, "ti_beam_ablated").unwrap(),
    );
    assert_eq!(ablated.modal.modes[0].spectral_amplitude, 0.0);
    assert_eq!(ablated.cfd.richtmyer_shock_viscosity_pa_s, 0.0);
    assert!(summary.total_events >= 8);
}

#[test]
fn gate_07_sae_b1_circuit_matcher_and_16k_coherent_slab() {
    let mut slab: GraceCoherentSlab<f64> =
        GraceCoherentSlab::allocate(512, ResidencyPolicy::CoherentUnifiedC2C).unwrap();
    assert!(slab.is_page_aligned_16k());
    assert!(slab.is_tensor_core_aligned_128b());
    slab.as_mut_slice()[0] = 42.0;
    assert_eq!(slab.as_slice()[0], 42.0);

    let atom0 = SaeBinaryAtom::from_f32_slice(1, "root_bending_yield_circuit", &[1.0, -1.0, 0.5], 0.0);
    let atom1 = SaeBinaryAtom::from_f32_slice(2, "karman_vortex_shedding_circuit", &[-1.0, 1.0, -0.5], 0.0);
    let matcher = SaeCaeCircuitMatcher::new(vec![atom0.clone(), atom1]);
    let matches = matcher.match_residual_batch(&[atom0]);
    assert_eq!(matches[0].matched_atom_id, 1);
    assert_eq!(matches[0].hamming_distance, 0);
}

#[test]
fn gate_08_ten_clone_suite_and_cad_to_cae_handoff_parity() {
    let handoff = SolidCvCadToCaeHandoff::from_bounding_box_and_kinematics(
        "cad_pressure_bulkhead",
        [-0.05, -0.02, 0.0],
        [0.05, 0.02, 0.015],
        3,
        2,
        2,
        [0.0, -0.04, 0.01],
        vec![(3.2, 1.1)],
    );
    let frames = VideoFrame::synthetic_cantilever_sequence(8, 24, 24, 240.0, 1.5, 48.0).unwrap();
    let receipt = analyze_cad_handoff_and_video_to_cae(
        &frames,
        Some(&handoff),
        "inconel_718",
        240.0,
        0.002,
        "cad_pressure_bulkhead",
    )
    .unwrap();
    assert_eq!(receipt.part_name, "cad_pressure_bulkhead");
    assert!(receipt.exports.calculix_inp.contains("*MATERIAL, NAME=INCONEL_718"));
    assert!(receipt.exports.vtu_xml.contains("<VTKFile type=\"UnstructuredGrid\""));

    let ten = run_10_clone_solidcv_cae_suite();
    assert_eq!(ten.clone_count, 10);
    assert!(ten.all_positive_jacobians);
    assert!(ten.all_newmark_symplectic_stable);
    assert!(ten.all_subcritical_cracks_stable);
}
