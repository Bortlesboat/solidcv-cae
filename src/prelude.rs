//! Prelude exports for `solidcv-cae` Video-to-CAE & Physics Analysis workflows.

pub use crate::cae_export::{
    export_cae_field_glb, export_calculix_abaqus_inp, export_vtk_unstructured_grid, export_vtu_xml,
    sample_ssf_data_ramp, CaeExportBundle, SSF_DATA_RAMP_RGB,
};
pub use crate::cfd_flow::{extract_worldtube_cfd_and_schlieren, WorldtubeCfdReceipt};
pub use crate::cloud_ws::{
    purchase_cents_to_microusd, retail_debit_microusd, unlock_vault_seal,
    SolidCvCaeLoopbackWsServer, SolidCvCaeWsClient, WsEngine, WsRpcRequest, WsRpcResponse,
    CREDITS_STORE_URL, DEFAULT_SOLIDCV_CAE_WS_URL,
};
pub use crate::core::{
    solidcv_cae_digest_hex, EngineeringMaterial, Error, Result, SolidCvCadToCaeHandoff,
    SymmetricTensor3D, VideoFrame,
};
pub use crate::dic_strain::{
    extract_3d_dic_strain_stress_field, DicFieldReceipt, DicSubsetSample,
};
pub use crate::hardware::{
    simd_dot_f64, simd_spmv_csr_f64, simd_xor_dof_carrier_u8, tensor_core_b1_argmin_256,
    GraceCoherentSlab, HardwareTopology, ResidencyPolicy,
};
pub use crate::inverse_material::{
    identify_constitutive_parameters_vfm, InverseMaterialFitReceipt,
};
pub use crate::mechinterp::{
    current_ablation, mind_summary, prune_cae_mesh_dofs, record_activation, record_circuit_node,
    reset_mind, snapshot_mind, with_counterfactual_ablation, AblationConfig, CaeDofPruningReceipt,
    CaeTheoremId, GlassBoxCaeSignature64D, MechInterpSummary, SaeBinaryAtom, SaeCaeCircuitMatcher,
    SaeCircuitMatch,
};
pub use crate::modal_oma::{
    extract_eulerian_modal_oma, modal_assurance_criterion, EigenMode3D, EulerianModalOmaReceipt,
};
pub use crate::tet_fea::{
    solve_tet_fea_newmark_beta, solve_tet_fea_static_pcg, NewmarkTransientReceipt,
    TetFeaSolutionReceipt, TetMesh3D,
};
pub use crate::thermal_fracture::{analyze_thermal_and_fracture, ThermalFractureReceipt};
