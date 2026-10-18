//! `cfd_flow`: Background-Oriented Schlieren (BOS), $(3+1)\text{D}$ Worldtube Navier-Stokes Velocity
//! & Vorticity Extraction, Q-Criterion Vortex Core Identification, Hodge-Helmholtz Pressure Poisson
//! Reconstruction, and Von Neumann-Richtmyer Shock Viscosity Analysis from Video.
//!
//! - **Cloud WebSocket Execution (`wss://api.solidsf.com/v1/solidcv-cae/ws`)**: Use [`dispatch_over_ws`]
//!   with [`crate::cloud_ws::SolidCvCaeWsClient`] (`WsEngine::WorldtubeNavierStokesCfd`).
//! - **Sealed Binary Vault (`assets/solidcv_cae_moat_vault.bin`)**: Local benchmark & offline execution
//!   is unpacked at compile time from the Zobrist-Feistel encrypted vault (`build.rs`).

include!(concat!(env!("OUT_DIR"), "/vault_cfd_flow.rs"));

/// Dispatch a $(3+1)\text{D}$ Worldtube Navier-Stokes & BOS Schlieren CFD analysis request over the
/// active binary WebSocket connection (`wss://api.solidsf.com/v1/solidcv-cae/ws`).
pub fn dispatch_over_ws(
    client: &mut crate::cloud_ws::SolidCvCaeWsClient,
    grid_w: usize,
    grid_h: usize,
    reynolds_number: f64,
) -> crate::Result<crate::cloud_ws::WsRpcResponse> {
    let req = crate::cloud_ws::WsRpcRequest {
        engine: crate::cloud_ws::WsEngine::WorldtubeNavierStokesCfd,
        operation: "worldtube_navier_stokes_bos_cfd".to_string(),
        payload_json: format!(
            "{{\"grid_w\":{grid_w},\"grid_h\":{grid_h},\"reynolds_number\":{reynolds_number:.1}}}"
        ),
    };
    client.call(&req)
}
