//! `cloud_ws`: Binary RFC-6455 WebSocket Client for `solidSF` Proprietary Cloud CAE, FEA & CFD Engines.
//!
//! While `solidcv-cae` provides 100% local, zero-FFI execution for 3D-DIC Green-Lagrange strain
//! and Cauchy stress fields, Eulerian phase-based Operational Modal Analysis (OMA), conformal
//! tetrahedral (`C3D4`/`C3D10`) FEA, Virtual Fields Method (VFM) inverse material identification,
//! Abaqus/CalculiX `.inp` + VTK `.vtk`/`.vtu` + `.glb` export, and CAE mechanistic interpretability,
//! the following four proprietary `solidSF` cluster engines execute over a low-latency binary
//! WebSocket stream (`wss://api.solidsf.com/v1/solidcv-cae/ws` or `SOLIDCV_CAE_WS_URL`):
//!
//! 1. **`WsEngine::Gb300SparsePcgFeaPtx` (`0x01`)**: Remote NVIDIA Grace-Blackwell `GB300` (`sm_100a`) /
//!    `GB10` (`sm_120a`) NVLink-C2C multi-million-DOF sparse PCG & symplectic Newmark-$\beta$ FEA solver.
//! 2. **`WsEngine::EulerianModalOma4D` (`0x02`)**: Remote phase-based motion magnification, eigenfrequency
//!    extraction, and 3D Modal Assurance Criterion (`MAC`) identification.
//! 3. **`WsEngine::WorldtubeNavierStokesCfd` (`0x03`)**: Remote $(3+1)\text{D}$ Worldtube Background-Oriented
//!    Schlieren (BOS), Hodge-Helmholtz pressure Poisson, and Richtmyer-Meshkov shock CFD solver.
//! 4. **`WsEngine::InverseConstitutiveVfm` (`0x04`)**: Remote Virtual Fields Method (VFM) constitutive
//!    tensor identification & Rice $J$-integral / Paris-law fracture life horizon engine.

use crate::core::{Error, Result, STS_ERROR, STS_PAYMENT_REQUIRED};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::thread;
use std::time::Duration;

/// Default production WebSocket endpoint for `solidSF` proprietary CAE/FEA/CFD engines.
pub const DEFAULT_SOLIDCV_CAE_WS_URL: &str = "wss://api.solidsf.com/v1/solidcv-cae/ws";

/// Public self-serve `solidSF` Compute Credits storefront (`GET /api/fah/store`).
pub const CREDITS_STORE_URL: &str = "https://solidsf.com/compute-credits";

/// Stripe product contract identifier for `solidSF Credits` (`1 USD = 1,000,000 micro-USD`).
pub const CREDITS_PRODUCT_CONTRACT: &str = "solidsf.platform.billing.v2.product.credits";

/// Default platform markup percentage over raw provider/cluster cost (`30%`).
pub const DEFAULT_MARKUP_PERCENT: i64 = 30;

/// Welcome grant minted to new accounts (`10,000,000 micro-USD = $10.00`).
pub const WELCOME_CREDITS_MICROUSD: i64 = 10_000_000;

/// Referral award minted per verified referral (`5,000,000 micro-USD = $5.00`).
pub const REFERRER_CREDITS_MICROUSD: i64 = 5_000_000;

/// Minimum self-serve FAH store credit purchase (`$10.00`).
pub const FAH_STORE_MIN_PURCHASE_CENTS: i32 = 1_000;

/// Maximum self-serve FAH store credit purchase (`$1,000.00`; org packs scale to `$100,000.00`).
pub const FAH_STORE_MAX_PURCHASE_CENTS: i32 = 100_000;

/// Magic 6-byte binary frame preamble (`"SSFCA\x01"`).
pub const WS_FRAME_MAGIC: [u8; 6] = *b"SSFCA\x01";

/// Zobrist-Feistel encrypted vault seal (`#[used]` so the ciphertext remains in `.rodata`
/// across stripped binaries and static archives for reverse-engineering inspection).
#[used]
pub static SOLIDSF_ENCRYPTED_BOUNTY_BLOB: [u8; 737] = [
    0x1D, 0x0F, 0xDB, 0x6D, 0x15, 0x93, 0xA3, 0xD4, 0x3A, 0x0B, 0x23, 0xAE, 0xB9, 0x47, 0x4F, 0xB4,
    0x04, 0x0A, 0xD1, 0xAF, 0xA9, 0x55, 0x10, 0xED, 0x2E, 0x0F, 0x48, 0x43, 0x0E, 0x52, 0xF9, 0x42,
    0x8C, 0x28, 0x7B, 0xC3, 0xC4, 0xD9, 0x69, 0x03, 0xB9, 0x65, 0xDF, 0x78, 0x37, 0x20, 0x62, 0x55,
    0x14, 0xBD, 0xA9, 0x76, 0xD2, 0x6F, 0xAD, 0xF3, 0xF6, 0xD5, 0x38, 0xAE, 0x1D, 0xDA, 0x86, 0x9A,
    0x7B, 0xF6, 0x5E, 0xFA, 0xF2, 0xA7, 0x19, 0xC3, 0x26, 0xE0, 0x80, 0x63, 0x9D, 0x7A, 0x98, 0x33,
    0x72, 0x66, 0x54, 0x93, 0x95, 0xED, 0x58, 0xCF, 0x68, 0xE5, 0x2B, 0x10, 0x19, 0x8B, 0x65, 0x09,
    0xB1, 0x30, 0xAD, 0x25, 0x20, 0xBE, 0x4E, 0xFF, 0x35, 0x84, 0xE6, 0x99, 0x24, 0x25, 0x2F, 0x48,
    0xF1, 0x3A, 0xA6, 0x52, 0xDC, 0xA9, 0xE2, 0xB3, 0xD0, 0x5A, 0xD0, 0xAC, 0xBB, 0xD8, 0xF7, 0x8F,
    0x6D, 0x4D, 0xA9, 0x46, 0x34, 0x3E, 0x60, 0xC1, 0xFA, 0x53, 0xEE, 0x59, 0xC5, 0xC7, 0x50, 0x07,
    0xC1, 0xAD, 0x4C, 0x27, 0xA1, 0x8E, 0x13, 0x93, 0x57, 0x22, 0xBD, 0x75, 0xF8, 0x50, 0x75, 0x63,
    0x40, 0x3B, 0xD8, 0xA9, 0x65, 0x0B, 0x27, 0x8E, 0xD2, 0x62, 0x8B, 0x9C, 0x34, 0x8B, 0x15, 0x71,
    0x6A, 0x9C, 0x38, 0x41, 0x95, 0x1C, 0xB0, 0x94, 0xCA, 0x50, 0xED, 0xA5, 0x52, 0xC6, 0xFC, 0x47,
    0xD9, 0x50, 0xE6, 0x92, 0xF4, 0xEA, 0x35, 0xBA, 0x55, 0x56, 0xF1, 0x48, 0x5B, 0xEB, 0x9B, 0x8A,
    0x87, 0xA1, 0x2A, 0xEC, 0xC8, 0x80, 0x82, 0x34, 0x0D, 0xFB, 0xF4, 0x6A, 0x89, 0x81, 0x8E, 0x62,
    0x89, 0x14, 0x16, 0x6F, 0x70, 0xCB, 0x43, 0x94, 0x1F, 0x6F, 0x8C, 0xE9, 0x82, 0x2B, 0x5E, 0x86,
    0x01, 0x28, 0x30, 0x30, 0xE8, 0x35, 0xA1, 0x9B, 0xD8, 0xAB, 0xEA, 0xD1, 0x47, 0x0E, 0xB8, 0xBD,
    0x6E, 0x48, 0x33, 0x0A, 0x46, 0xC8, 0x5E, 0xC8, 0x2C, 0x3F, 0xAE, 0xD0, 0x2F, 0x5C, 0x70, 0x71,
    0xC6, 0xA2, 0xDB, 0x2C, 0x74, 0x4B, 0x66, 0xA2, 0x95, 0x36, 0xB0, 0xF9, 0x5A, 0xC6, 0x72, 0xF1,
    0xB4, 0xDD, 0x47, 0x64, 0x40, 0x53, 0x27, 0x23, 0x38, 0x0D, 0x0E, 0x8F, 0xCC, 0x47, 0x9A, 0xBB,
    0x9A, 0x10, 0x94, 0xFF, 0x0C, 0x88, 0x5C, 0xFE, 0x85, 0xB7, 0x21, 0xCF, 0x93, 0x3A, 0xE2, 0x9F,
    0x12, 0x5C, 0xFC, 0xA5, 0xD5, 0xA0, 0x0B, 0x16, 0x7C, 0x48, 0xD2, 0x48, 0x96, 0x57, 0x5D, 0xBF,
    0xDD, 0x80, 0xC7, 0x70, 0x7A, 0xCA, 0xB1, 0x67, 0xF7, 0x0F, 0xE1, 0x88, 0xD7, 0xB5, 0x93, 0x08,
    0xD9, 0xAF, 0x5A, 0x32, 0x6E, 0x8A, 0x31, 0xCB, 0xC4, 0xC5, 0x2D, 0xFB, 0xDA, 0xB0, 0x4F, 0xAA,
    0x0E, 0x96, 0x13, 0x82, 0x4D, 0xBA, 0xE6, 0xBC, 0x96, 0xA4, 0xF9, 0x3F, 0x5F, 0x78, 0xDF, 0x36,
    0x60, 0x73, 0x39, 0xAE, 0x99, 0xFD, 0xD3, 0x52, 0x30, 0x69, 0x09, 0x7B, 0x02, 0x1F, 0x6D, 0x74,
    0x30, 0x8A, 0x43, 0x79, 0xEC, 0x54, 0x7B, 0xB9, 0x8A, 0x39, 0xC4, 0x7A, 0x5C, 0x4F, 0xD8, 0xE6,
    0xA3, 0xBC, 0x95, 0x29, 0x02, 0x2A, 0x21, 0x2C, 0x0F, 0xA8, 0xA7, 0xDF, 0xB8, 0xA0, 0x1F, 0xBF,
    0x52, 0x95, 0x86, 0x9A, 0x4D, 0x52, 0x49, 0x1B, 0x81, 0xD3, 0x93, 0x7E, 0x71, 0x75, 0x4E, 0x7D,
    0x73, 0x95, 0x2A, 0xEB, 0xA3, 0x2A, 0x35, 0xAC, 0x1C, 0xC7, 0x19, 0x11, 0xBC, 0xCA, 0x41, 0x33,
    0xB7, 0x6E, 0x90, 0xF3, 0x1C, 0x45, 0xD2, 0x99, 0x4E, 0x56, 0xD5, 0x2F, 0x4C, 0x2B, 0x91, 0xF2,
    0x6E, 0x5D, 0xBF, 0xA1, 0xC5, 0xA5, 0x56, 0xBD, 0x59, 0x64, 0x4E, 0xF8, 0xBF, 0x0A, 0xE2, 0xF1,
    0xD3, 0xC2, 0x30, 0x1B, 0x42, 0xFC, 0xE8, 0xE4, 0x2A, 0xA1, 0x03, 0xA2, 0x96, 0x64, 0x89, 0xE3,
    0x67, 0xD2, 0xCC, 0x2E, 0x25, 0x99, 0xCE, 0x1C, 0x9E, 0x3C, 0x82, 0xBE, 0x85, 0xAE, 0x95, 0xE2,
    0x05, 0xB7, 0x22, 0x36, 0xA4, 0xA3, 0x3F, 0x19, 0x76, 0xEE, 0xB9, 0x3C, 0xB5, 0x30, 0x6D, 0x85,
    0x2A, 0x77, 0xEA, 0x7F, 0x3E, 0xA1, 0x91, 0x79, 0x2E, 0x68, 0x42, 0x08, 0x46, 0x47, 0xB6, 0x25,
    0xD2, 0xF0, 0xCC, 0x7D, 0x33, 0x05, 0x8C, 0xAC, 0xDC, 0xFF, 0xF2, 0x82, 0x3F, 0xA1, 0xA8, 0x09,
    0x5D, 0x5A, 0xE8, 0x4E, 0x56, 0x48, 0xAC, 0xEF, 0x4D, 0xD5, 0x3D, 0x6F, 0x28, 0x21, 0xA3, 0x6B,
    0x82, 0xFB, 0x9E, 0x28, 0x8D, 0x03, 0x9D, 0x13, 0x15, 0x8B, 0xA0, 0x67, 0x85, 0x18, 0xBC, 0x1A,
    0xA3, 0x54, 0xA5, 0x09, 0xFB, 0x3F, 0xCE, 0x20, 0xAE, 0xCA, 0xB3, 0x38, 0x54, 0x61, 0xBA, 0xC3,
    0xBD, 0x83, 0x6B, 0x05, 0xBA, 0x5F, 0x20, 0xF8, 0xBB, 0x7A, 0xFE, 0x01, 0xFE, 0xD6, 0x6B, 0xF2,
    0x6A, 0xD8, 0x3A, 0x86, 0x9B, 0xA7, 0x3F, 0x0A, 0xCC, 0x8C, 0xAC, 0xD3, 0x94, 0xF2, 0xBC, 0x03,
    0x62, 0x06, 0xE0, 0x83, 0x03, 0x7B, 0xA1, 0x00, 0x4F, 0x99, 0x95, 0xB2, 0x1C, 0x69, 0xAA, 0x74,
    0x74, 0x78, 0x25, 0x44, 0x2A, 0x28, 0x26, 0x06, 0xAB, 0x55, 0xB9, 0x99, 0x3B, 0x0C, 0x64, 0xEE,
    0x71, 0x31, 0x6D, 0x5F, 0x4E, 0xAD, 0x2C, 0xDD, 0xD4, 0x0E, 0xA7, 0x95, 0xFA, 0x46, 0x86, 0xDA,
    0x5F, 0x36, 0x64, 0x0F, 0x7E, 0xB1, 0xF8, 0x3F, 0x89, 0x60, 0xA3, 0x89, 0xFC, 0x67, 0x37, 0x71,
    0x3A, 0xE7, 0x18, 0x80, 0x8B, 0x26, 0xAD, 0x98, 0x80, 0x46, 0x0F, 0x72, 0xF5, 0x57, 0x71, 0x5F,
    0x9E,
];

/// Proprietary CAE engine opcode multiplexed over the binary WebSocket connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum WsEngine {
    /// NVIDIA Grace-Blackwell GB300/GB10 PTX 8.5+ Sparse PCG & Symplectic Newmark-$\beta$ FEA Solver.
    Gb300SparsePcgFeaPtx = 0x01,
    /// `EulerianModalOma4D` phase-based motion magnification, eigenfrequency & 3D MAC mode shape solver.
    EulerianModalOma4D = 0x02,
    /// `WorldtubeNavierStokesCfd` BOS Schlieren, Hodge-Helmholtz pressure Poisson & Richtmyer shock CFD.
    WorldtubeNavierStokesCfd = 0x03,
    /// `InverseConstitutiveVfm` Virtual Fields Method constitutive identification & Rice $J$-integral fracture.
    InverseConstitutiveVfm = 0x04,
}

impl WsEngine {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0x01 => Some(Self::Gb300SparsePcgFeaPtx),
            0x02 => Some(Self::EulerianModalOma4D),
            0x03 => Some(Self::WorldtubeNavierStokesCfd),
            0x04 => Some(Self::InverseConstitutiveVfm),
            _ => None,
        }
    }

    /// Retail base rate in `micro-USD` (`1 USD = 1,000,000 micro-USD`) per WebSocket RPC invocation,
    /// matching `solidSF` gateway `agentic_credit_ledger` (`credits_delta_microusd`).
    pub const fn base_rate_microusd(&self) -> i64 {
        match self {
            // 25 micro-USD / solve ($0.025 per 1,000 GB300 Sparse PCG FEA solves; 400k solves per $10 pack)
            Self::Gb300SparsePcgFeaPtx => 25,
            // 15 micro-USD / modal sweep ($0.015 per 1,000 Eulerian OMA sweeps; 666.6k sweeps per $10 pack)
            Self::EulerianModalOma4D => 15,
            // 10 micro-USD / CFD step ($0.010 per 1,000 Worldtube Navier-Stokes steps; 1.0M steps per $10 pack)
            Self::WorldtubeNavierStokesCfd => 10,
            // 5 micro-USD / inverse fit ($0.005 per 1,000 VFM constitutive fits; 2.0M fits per $10 pack)
            Self::InverseConstitutiveVfm => 5,
        }
    }
}

/// Retail debit in `micro-USD`: raw provider/cluster cost marked up by `markup_percent` (default 30%),
/// rounded up to the next integer `micro-USD`.
pub fn retail_debit_microusd(provider_cost_usd: f64, markup_percent: i64) -> Option<i64> {
    if !provider_cost_usd.is_finite() || provider_cost_usd <= 0.0 {
        return None;
    }
    let markup = markup_percent.clamp(0, 1000) as f64;
    let retail_usd = provider_cost_usd * (1.0 + markup / 100.0);
    let micro = (retail_usd * 1_000_000.0).ceil();
    if !micro.is_finite() || micro <= 0.0 {
        return None;
    }
    Some((micro as i64).max(1))
}

/// Convert a self-serve Stripe / FAH Store purchase in USD cents (`$10.00` = `1_000` cents)
/// to `micro-USD` credits (`1 cent = 10,000 micro-USD`).
pub fn purchase_cents_to_microusd(amount_cents: i32) -> Result<i64> {
    if !(FAH_STORE_MIN_PURCHASE_CENTS..=FAH_STORE_MAX_PURCHASE_CENTS).contains(&amount_cents) {
        return Err(Error::new(
            STS_ERROR,
            format!(
                "Invalid credit pack amount {} cents: must be between {} ($10) and {} ($1,000)",
                amount_cents, FAH_STORE_MIN_PURCHASE_CENTS, FAH_STORE_MAX_PURCHASE_CENTS
            ),
        ));
    }
    Ok((amount_cents as i64) * 10_000)
}

/// Decrypt the embedded Zobrist-Feistel vault seal (`SOLIDSF_ENCRYPTED_BOUNTY_BLOB`).
pub fn unlock_vault_seal() -> String {
    let tag: [u8; 8] = [0x53, 0x4F, 0x4C, 0x49, 0x44, 0x53, 0x46, 0x21];
    let master = u64::from_be_bytes(tag);
    let mut out = Vec::with_capacity(SOLIDSF_ENCRYPTED_BOUNTY_BLOB.len());
    for (idx, &enc_byte) in SOLIDSF_ENCRYPTED_BOUNTY_BLOB.iter().enumerate() {
        let mut z = master ^ ((idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        out.push(enc_byte ^ ((z & 0xFF) as u8));
    }
    String::from_utf8(out).unwrap_or_default()
}

/// Binary WebSocket RPC Request frame payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WsRpcRequest {
    pub engine: WsEngine,
    pub operation: String,
    pub payload_json: String,
}

/// Binary WebSocket RPC Response frame payload with `solidSF Credits` ledger receipt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WsRpcResponse {
    pub ok: bool,
    pub engine: WsEngine,
    pub server_exec_us: f64,
    pub credits_charged_microusd: i64,
    pub credits_remaining_microusd: i64,
    pub billing_status: String,
    pub result_json: String,
}

fn encode_rfc6455_binary_frame(payload: &[u8], mask_key: [u8; 4]) -> Vec<u8> {
    let mut frame = Vec::with_capacity(14 + payload.len());
    frame.push(0x82); // FIN=1, Opcode=0x2 (Binary)
    if payload.len() <= 125 {
        frame.push(0x80 | (payload.len() as u8));
    } else if payload.len() <= 65535 {
        frame.push(0x80 | 126);
        frame.extend_from_slice(&(payload.len() as u16).to_be_bytes());
    } else {
        frame.push(0x80 | 127);
        frame.extend_from_slice(&(payload.len() as u64).to_be_bytes());
    }
    frame.extend_from_slice(&mask_key);
    for (i, &b) in payload.iter().enumerate() {
        frame.push(b ^ mask_key[i & 3]);
    }
    frame
}

fn decode_rfc6455_binary_frame<R: Read>(stream: &mut R) -> Result<Vec<u8>> {
    let mut hdr = [0u8; 2];
    stream
        .read_exact(&mut hdr)
        .map_err(|e| Error::new(STS_ERROR, format!("WS frame header read failed: {e}")))?;
    let masked = (hdr[1] & 0x80) != 0;
    let mut len = (hdr[1] & 0x7F) as usize;
    if len == 126 {
        let mut ext = [0u8; 2];
        stream
            .read_exact(&mut ext)
            .map_err(|e| Error::new(STS_ERROR, format!("WS ext16 read failed: {e}")))?;
        len = u16::from_be_bytes(ext) as usize;
    } else if len == 127 {
        let mut ext = [0u8; 8];
        stream
            .read_exact(&mut ext)
            .map_err(|e| Error::new(STS_ERROR, format!("WS ext64 read failed: {e}")))?;
        len = u64::from_be_bytes(ext) as usize;
    }
    let mut mask = [0u8; 4];
    if masked {
        stream
            .read_exact(&mut mask)
            .map_err(|e| Error::new(STS_ERROR, format!("WS mask read failed: {e}")))?;
    }
    let mut payload = vec![0u8; len];
    stream
        .read_exact(&mut payload)
        .map_err(|e| Error::new(STS_ERROR, format!("WS payload read failed: {e}")))?;
    if masked {
        for (i, b) in payload.iter_mut().enumerate() {
            *b ^= mask[i & 3];
        }
    }
    Ok(payload)
}

/// Synchronous RFC-6455 Binary WebSocket client for `wss://api.solidsf.com/v1/solidcv-cae/ws`.
pub struct SolidCvCaeWsClient {
    endpoint: String,
    stream: TcpStream,
}

impl SolidCvCaeWsClient {
    /// Connect to `ws://host:port/v1/solidcv-cae/ws` and complete the RFC-6455 upgrade handshake.
    pub fn connect(ws_url: &str, api_key: Option<&str>) -> Result<Self> {
        let stripped = ws_url
            .strip_prefix("ws://")
            .or_else(|| ws_url.strip_prefix("wss://"))
            .unwrap_or(ws_url);
        let (host_port, path) = match stripped.find('/') {
            Some(idx) => (&stripped[..idx], &stripped[idx..]),
            None => (stripped, "/v1/solidcv-cae/ws"),
        };
        let stream = TcpStream::connect(host_port).map_err(|e| {
            Error::new(
                STS_ERROR,
                format!("Failed to connect to solidcv-cae WebSocket {host_port}: {e}"),
            )
        })?;
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .map_err(|e| Error::new(STS_ERROR, format!("set_read_timeout failed: {e}")))?;
        let mut client = Self {
            endpoint: ws_url.to_string(),
            stream,
        };
        client.handshake(host_port, path, api_key)?;
        Ok(client)
    }

    fn handshake(&mut self, host: &str, path: &str, api_key: Option<&str>) -> Result<()> {
        let auth_header = match api_key {
            Some(k) if !k.is_empty() => format!("Authorization: Bearer {k}\r\n"),
            _ => String::new(),
        };
        let req = format!(
            "GET {path} HTTP/1.1\r\n\
             Host: {host}\r\n\
             Upgrade: websocket\r\n\
             Connection: Upgrade\r\n\
             Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\
             Sec-WebSocket-Version: 13\r\n\
             X-SolidSF-Protocol: solidcv-cae-ws-v1\r\n\
             {auth_header}\r\n"
        );
        self.stream
            .write_all(req.as_bytes())
            .map_err(|e| Error::new(STS_ERROR, format!("WS handshake write failed: {e}")))?;

        let mut resp_buf = Vec::with_capacity(512);
        let mut byte = [0u8; 1];
        while !resp_buf.ends_with(b"\r\n\r\n") {
            let n = self
                .stream
                .read(&mut byte)
                .map_err(|e| Error::new(STS_ERROR, format!("WS handshake read failed: {e}")))?;
            if n == 0 {
                break;
            }
            resp_buf.push(byte[0]);
            if resp_buf.len() > 8192 {
                return Err(Error::new(STS_ERROR, "WS handshake header overflow"));
            }
        }
        let header_str = String::from_utf8_lossy(&resp_buf);
        if !header_str.starts_with("HTTP/1.1 101") {
            return Err(Error::new(
                STS_ERROR,
                format!("WebSocket upgrade rejected: {header_str}"),
            ));
        }
        Ok(())
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Execute a binary RPC call over the active WebSocket stream.
    pub fn call(&mut self, request: &WsRpcRequest) -> Result<WsRpcResponse> {
        let mut wire = Vec::with_capacity(64 + request.payload_json.len());
        wire.extend_from_slice(&WS_FRAME_MAGIC);
        wire.push(request.engine as u8);
        let json_bytes = serde_json::to_vec(request)
            .map_err(|e| Error::new(STS_ERROR, format!("WS serialize error: {e}")))?;
        wire.extend_from_slice(&json_bytes);

        let frame = encode_rfc6455_binary_frame(&wire, [0x53, 0x53, 0x46, 0x21]);
        self.stream
            .write_all(&frame)
            .map_err(|e| Error::new(STS_ERROR, format!("WS frame write error: {e}")))?;

        let resp_wire = decode_rfc6455_binary_frame(&mut self.stream)?;
        if resp_wire.len() < 7 || resp_wire[0..6] != WS_FRAME_MAGIC {
            return Err(Error::new(
                STS_ERROR,
                "Invalid SSFCA\\x01 binary preamble in WebSocket response",
            ));
        }
        let resp: WsRpcResponse = serde_json::from_slice::<WsRpcResponse>(&resp_wire[7..])
            .map_err(|e| Error::new(STS_ERROR, format!("WS response deserialize error: {e}")))?;
        if !resp.ok && resp.billing_status == "payment_required_402" {
            return Err(Error::new(
                STS_PAYMENT_REQUIRED,
                format!(
                    "402 Payment Required: solidSF Credits balance ({} micro-USD) insufficient for {:?} ({} micro-USD). Refill at {}",
                    resp.credits_remaining_microusd,
                    resp.engine,
                    resp.engine.base_rate_microusd(),
                    CREDITS_STORE_URL
                ),
            ));
        }
        Ok(resp)
    }
}

/// Ephemeral loopback RFC-6455 WebSocket server for local protocol verification and unit testing.
pub struct SolidCvCaeLoopbackWsServer {
    pub ws_url: String,
    handle: Option<thread::JoinHandle<()>>,
}

impl SolidCvCaeLoopbackWsServer {
    pub fn spawn_ephemeral(expected_calls: usize) -> Result<Self> {
        Self::spawn_with_credits(expected_calls, WELCOME_CREDITS_MICROUSD)
    }

    pub fn spawn_with_credits(expected_calls: usize, initial_credits_microusd: i64) -> Result<Self> {
        let listener = TcpListener::bind("127.0.0.1:0")
            .map_err(|e| Error::new(STS_ERROR, format!("bind loopback failed: {e}")))?;
        let addr = listener
            .local_addr()
            .map_err(|e| Error::new(STS_ERROR, format!("local_addr failed: {e}")))?;
        let ws_url = format!("ws://{addr}/v1/solidcv-cae/ws");

        let handle = thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut hdr = Vec::with_capacity(512);
                let mut b = [0u8; 1];
                while !hdr.ends_with(b"\r\n\r\n") {
                    if stream.read(&mut b).unwrap_or(0) == 0 {
                        return;
                    }
                    hdr.push(b[0]);
                }
                let hdr_str = String::from_utf8_lossy(&hdr);
                let is_owned_cluster = hdr_str.contains("Bearer ssf_internal_");
                let upgrade_resp = "HTTP/1.1 101 Switching Protocols\r\n\
                                    Upgrade: websocket\r\n\
                                    Connection: Upgrade\r\n\
                                    Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=\r\n\r\n";
                let _ = stream.write_all(upgrade_resp.as_bytes());

                let mut balance_microusd = initial_credits_microusd;

                for _ in 0..expected_calls {
                    let Ok(wire) = decode_rfc6455_binary_frame(&mut stream) else {
                        break;
                    };
                    if wire.len() < 7 || wire[0..6] != WS_FRAME_MAGIC {
                        break;
                    }
                    let engine = WsEngine::from_u8(wire[6]).unwrap_or(WsEngine::Gb300SparsePcgFeaPtx);
                    let req: WsRpcRequest = serde_json::from_slice(&wire[7..]).unwrap_or(WsRpcRequest {
                        engine,
                        operation: "unknown".to_string(),
                        payload_json: "{}".to_string(),
                    });

                    let rate_microusd = if is_owned_cluster {
                        0
                    } else {
                        engine.base_rate_microusd()
                    };

                    let resp = if is_owned_cluster || balance_microusd >= rate_microusd {
                        balance_microusd -= rate_microusd;
                        WsRpcResponse {
                            ok: true,
                            engine,
                            server_exec_us: match engine {
                                WsEngine::Gb300SparsePcgFeaPtx => 14.20,
                                WsEngine::EulerianModalOma4D => 11.80,
                                WsEngine::WorldtubeNavierStokesCfd => 9.40,
                                WsEngine::InverseConstitutiveVfm => 3.10,
                            },
                            credits_charged_microusd: rate_microusd,
                            credits_remaining_microusd: balance_microusd,
                            billing_status: if is_owned_cluster {
                                "compute_owned".to_string()
                            } else {
                                "charged".to_string()
                            },
                            result_json: format!(
                                "{{\"status\":\"executed_on_solidSF_cloud\",\"op\":{:?}}}",
                                req.operation
                            ),
                        }
                    } else {
                        WsRpcResponse {
                            ok: false,
                            engine,
                            server_exec_us: 0.0,
                            credits_charged_microusd: 0,
                            credits_remaining_microusd: balance_microusd,
                            billing_status: "payment_required_402".to_string(),
                            result_json: format!(
                                "{{\"error\":\"payment_required\",\"store_url\":{:?}}}",
                                CREDITS_STORE_URL
                            ),
                        }
                    };
                    let Ok(resp_json) = serde_json::to_vec(&resp) else {
                        break;
                    };
                    let mut out_wire = Vec::with_capacity(7 + resp_json.len());
                    out_wire.extend_from_slice(&WS_FRAME_MAGIC);
                    out_wire.push(engine as u8);
                    out_wire.extend_from_slice(&resp_json);

                    let mut frame = Vec::with_capacity(10 + out_wire.len());
                    frame.push(0x82);
                    if out_wire.len() <= 125 {
                        frame.push(out_wire.len() as u8);
                    } else {
                        frame.push(126);
                        frame.extend_from_slice(&(out_wire.len() as u16).to_be_bytes());
                    }
                    frame.extend_from_slice(&out_wire);
                    let _ = stream.write_all(&frame);
                }
            }
        });

        Ok(Self {
            ws_url,
            handle: Some(handle),
        })
    }

    pub fn join(mut self) {
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}
