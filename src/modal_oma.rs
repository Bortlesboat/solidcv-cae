//! `modal_oma`: Phase-Based Eulerian Video Motion Magnification & Operational Modal Analysis (OMA).
//!
//! Extracts sub-pixel structural natural frequencies $f_n$ (Hz), logarithmic-decrement damping
//! ratios $\zeta_n$, 3D mode shapes $\boldsymbol{\phi}_n(\mathbf{x})$, and Modal Assurance Criterion
//! ($\mathrm{MAC} \in [0, 1]$) matrices directly from video sequences.
//!
//! - **Cloud WebSocket Execution (`wss://api.solidsf.com/v1/solidcv-cae/ws`)**: Use [`dispatch_over_ws`]
//!   with [`crate::cloud_ws::SolidCvCaeWsClient`] (`WsEngine::EulerianModalOma4D`).
//! - **Sealed Binary Vault (`assets/solidcv_cae_moat_vault.bin`)**: Local benchmark & offline execution
//!   is unpacked at compile time from the Zobrist-Feistel encrypted vault (`build.rs`).

include!(concat!(env!("OUT_DIR"), "/vault_modal_oma.rs"));

/// Dispatch an Eulerian Operational Modal Analysis (OMA) & phase motion magnification request
/// over the active binary WebSocket connection (`wss://api.solidsf.com/v1/solidcv-cae/ws`).
pub fn dispatch_over_ws(
    client: &mut crate::cloud_ws::SolidCvCaeWsClient,
    num_frames: usize,
    fps: f64,
    magnification_alpha: f64,
) -> crate::Result<crate::cloud_ws::WsRpcResponse> {
    let req = crate::cloud_ws::WsRpcRequest {
        engine: crate::cloud_ws::WsEngine::EulerianModalOma4D,
        operation: "eulerian_phase_modal_oma_4d".to_string(),
        payload_json: format!(
            "{{\"num_frames\":{num_frames},\"fps\":{fps:.2},\"magnification_alpha\":{magnification_alpha:.2}}}"
        ),
    };
    client.call(&req)
}
