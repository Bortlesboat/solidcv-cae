//! `dic_strain`: 3D Digital Image Correlation (3D-DIC), Sub-Pixel sfFFT + Lucas-Kanade Kinematics,
//! Nonlinear Green-Lagrange Strain Tensor $\mathbf{E} = \frac{1}{2}(\mathbf{F}^T\mathbf{F} - \mathbf{I})$,
//! and 3D Cauchy Stress / Von Mises / Yield Safety-Factor Field Extraction from Video.
//!
//! - **Cloud WebSocket Execution (`wss://api.solidsf.com/v1/solidcv-cae/ws`)**: Use [`dispatch_over_ws`]
//!   with [`crate::cloud_ws::SolidCvCaeWsClient`] (`WsEngine::Gb300SparsePcgFeaPtx`).
//! - **Sealed Binary Vault (`assets/solidcv_cae_moat_vault.bin`)**: Local benchmark & offline execution
//!   is unpacked at compile time from the Zobrist-Feistel encrypted vault (`build.rs`).

include!(concat!(env!("OUT_DIR"), "/vault_dic_strain.rs"));

/// Dispatch a 3D-DIC Green-Lagrange strain & stress field request over the active binary
/// WebSocket connection (`wss://api.solidsf.com/v1/solidcv-cae/ws`).
pub fn dispatch_over_ws(
    client: &mut crate::cloud_ws::SolidCvCaeWsClient,
    num_frames: usize,
    subset_size_px: usize,
    material_id: &str,
) -> crate::Result<crate::cloud_ws::WsRpcResponse> {
    let req = crate::cloud_ws::WsRpcRequest {
        engine: crate::cloud_ws::WsEngine::Gb300SparsePcgFeaPtx,
        operation: "dic_3d_green_lagrange_stress_field".to_string(),
        payload_json: format!(
            "{{\"num_frames\":{num_frames},\"subset_size_px\":{subset_size_px},\"material_id\":{:?}}}",
            material_id
        ),
    };
    client.call(&req)
}
