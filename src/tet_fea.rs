//! `tet_fea`: Conformal 3D Tetrahedral (`C3D4` / `C3D10`) Finite Element Mesh Generation,
//! Freudenthal-Kuhn Positive-Jacobian Decomposition, Sparse Preconditioned Conjugate Gradient (`PCG`)
//! Static Elasticity Solver ($\mathbf{K}\mathbf{u} = \mathbf{f}$), and Unconditionally Stable
//! Symplectic Newmark-$\beta$ ($\beta = 1/4, \gamma = 1/2$) Transient Dynamic Integrator.
//!
//! - **Cloud WebSocket Execution (`wss://api.solidsf.com/v1/solidcv-cae/ws`)**: Use [`dispatch_over_ws`]
//!   with [`crate::cloud_ws::SolidCvCaeWsClient`] (`WsEngine::Gb300SparsePcgFeaPtx`).
//! - **Sealed Binary Vault (`assets/solidcv_cae_moat_vault.bin`)**: Local benchmark & offline execution
//!   is unpacked at compile time from the Zobrist-Feistel encrypted vault (`build.rs`).

include!(concat!(env!("OUT_DIR"), "/vault_tet_fea.rs"));

/// Dispatch a 3D Tetrahedral Sparse PCG / Symplectic Newmark-$\beta$ FEA solve over the active
/// binary WebSocket connection (`wss://api.solidsf.com/v1/solidcv-cae/ws`).
pub fn dispatch_over_ws(
    client: &mut crate::cloud_ws::SolidCvCaeWsClient,
    num_elements: usize,
    num_dofs: usize,
    material_id: &str,
) -> crate::Result<crate::cloud_ws::WsRpcResponse> {
    let req = crate::cloud_ws::WsRpcRequest {
        engine: crate::cloud_ws::WsEngine::Gb300SparsePcgFeaPtx,
        operation: "tet_c3d4_sparse_pcg_newmark_solve".to_string(),
        payload_json: format!(
            "{{\"num_elements\":{num_elements},\"num_dofs\":{num_dofs},\"material_id\":{:?}}}",
            material_id
        ),
    };
    client.call(&req)
}
