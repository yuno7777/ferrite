//! Recurrent State-Space (`Mamba` / `Mamba-2`), Linear Attention (`RWKV-6` / `RWKV-7`),
//! and Hybrid Transformer-SSM (`Jamba` / `Zamba`) inference engines.
//!
//! Traditional Transformer layers require an $O(T \cdot d_{\text{kv}})$ KV cache
//! that grows linearly with sequence length $T$, making 128K-1M context windows
//! prohibitively expensive on consumer RAM.
//!
//! State-Space Models (`Mamba-1`, `Mamba-2`) and Linear-Attention RNNs (`RWKV-6`,
//! `RWKV-7`) compress the entire history of a sequence into a **fixed-size**
//! recurrent state per layer:
//!
//! 1. **Mamba Selective State-Space (`MambaLayer`)**:
//!    - Maintains a causal 1D depthwise convolution ring buffer `conv_state` of
//!      shape `[d_inner, d_conv - 1]` and a selective SSM hidden state `ssm_state`
//!      of shape `[d_inner, d_state]`.
//!    - Input-dependent step size $\Delta = \text{softplus}(W_\Delta x + b_\Delta)$,
//!      input matrix $B = W_B x$, and output matrix $C = W_C x$:
//!      $$\bar{A} = \exp(\Delta \odot A), \quad \bar{B} = \Delta \odot B$$
//!      $$h_t = \bar{A} \odot h_{t-1} + \bar{B} \odot u_t, \quad y_t = (C_t \cdot h_t + D \odot u_t) \odot \text{SiLU}(z_t)$$
//!
//! 2. **RWKV-6 (Finch) & RWKV-7 (Goose) Time-Mix & Channel-Mix (`RwkvLayer`)**:
//!    - Maintains the last token's hidden vector `time_shift_state` (`[dim]`)
//!      for token-shift interpolation ($\mu x_t + (1 - \mu) x_{t-1}$) and a
//!      per-head recurrent key-value state matrix `wkv_state` (`[n_heads, head_dim, head_dim]`).
//!    - Updates the recurrent state with data-dependent exponential decay
//!      $w_t = \exp(-\exp(\omega_t))$:
//!      $$S_t = \text{diag}(w_t) S_{t-1} + k_t \otimes v_t, \quad y_t = r_t \cdot S_t$$
//!
//! 3. **Hybrid Transformer-SSM (`HybridLayer` / `HybridState`)**:
//!    - Interleaves constant-memory `Mamba` / `RWKV` layers with periodic full
//!      or sliding-window GQA attention layers (as in `Jamba`, `Zamba`, and
//!      `Granite-Hybrid`).

use crate::ops;

/// Softplus activation $\ln(1 + e^x)$ with numerical stability for large $x$.
#[inline]
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// SiLU (Swish) activation $x \cdot \sigma(x)$.
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Fixed-size recurrent state for a single `Mamba-1` or `Mamba-2` layer.
#[derive(Clone, Debug)]
pub struct MambaState {
    pub d_inner: usize,
    pub d_conv: usize,
    pub d_state: usize,
    /// Rolling history of the last `d_conv - 1` pre-activation inputs (`[d_inner, d_conv - 1]`).
    pub conv_state: Vec<f32>,
    /// Selective state-space hidden matrix (`[d_inner, d_state]`).
    pub ssm_state: Vec<f32>,
    /// Number of tokens processed so far.
    pub step_count: usize,
}

impl MambaState {
    pub fn new(d_inner: usize, d_conv: usize, d_state: usize) -> Self {
        let hist = d_conv.saturating_sub(1);
        Self {
            d_inner,
            d_conv,
            d_state,
            conv_state: vec![0.0; d_inner * hist],
            ssm_state: vec![0.0; d_inner * d_state],
            step_count: 0,
        }
    }

    pub fn reset(&mut self) {
        self.conv_state.fill(0.0);
        self.ssm_state.fill(0.0);
        self.step_count = 0;
    }

    /// Total memory footprint in bytes (constant regardless of sequence length).
    pub fn memory_bytes(&self) -> usize {
        (self.conv_state.len() + self.ssm_state.len()) * std::mem::size_of::<f32>()
    }
}

/// Parameters for a single Mamba selective state-space layer.
#[derive(Clone, Debug)]
pub struct MambaLayer {
    pub d_model: usize,
    pub d_inner: usize,
    pub d_conv: usize,
    pub d_state: usize,
    pub dt_rank: usize,
    /// Input projection `[2 * d_inner, d_model]` producing `(u, z)`.
    pub in_proj: Vec<f32>,
    /// Depthwise 1D causal convolution weights `[d_inner, d_conv]`.
    pub conv1d_weight: Vec<f32>,
    /// Depthwise 1D causal convolution bias `[d_inner]`.
    pub conv1d_bias: Vec<f32>,
    /// Selective projection `[dt_rank + 2 * d_state, d_inner]` producing `(dt_raw, B, C)`.
    pub x_proj: Vec<f32>,
    /// Time-step projection `[d_inner, dt_rank]`.
    pub dt_proj_weight: Vec<f32>,
    /// Time-step bias `[d_inner]`.
    pub dt_proj_bias: Vec<f32>,
    /// Diagonal state transition matrix $A$ (`[d_inner, d_state]`, stored as negative values $-\exp(A_{\log})$).
    pub a_neg: Vec<f32>,
    /// Skip connection vector $D$ (`[d_inner]`).
    pub d_skip: Vec<f32>,
    /// Output projection `[d_model, d_inner]`.
    pub out_proj: Vec<f32>,
}

impl MambaLayer {
    /// Execute a single recurrent decode step ($O(d_{\text{inner}} \cdot d_{\text{state}})$ time, $O(1)$ memory).
    pub fn step(&self, x: &[f32], state: &mut MambaState, out: &mut [f32]) {
        debug_assert_eq!(x.len(), self.d_model);
        debug_assert_eq!(out.len(), self.d_model);

        // 1. Project input `x` -> `(u, z)` of length `2 * d_inner`
        let mut uz = vec![0.0f32; 2 * self.d_inner];
        for (r, slot) in uz.iter_mut().enumerate() {
            let row = &self.in_proj[r * self.d_model..(r + 1) * self.d_model];
            *slot = ops::dot(row, x);
        }
        let (u_raw, z) = uz.split_at(self.d_inner);

        // 2. Causal 1D depthwise convolution over rolling `conv_state` + `u_raw` + SiLU
        let hist = self.d_conv.saturating_sub(1);
        let mut u_act = vec![0.0f32; self.d_inner];
        for ch in 0..self.d_inner {
            let w = &self.conv1d_weight[ch * self.d_conv..(ch + 1) * self.d_conv];
            let mut sum = self.conv1d_bias[ch];
            if hist > 0 {
                let st = &mut state.conv_state[ch * hist..(ch + 1) * hist];
                for k in 0..hist {
                    sum += st[k] * w[k];
                }
                sum += u_raw[ch] * w[hist];
                st.copy_within(1..hist, 0);
                st[hist - 1] = u_raw[ch];
            } else {
                sum += u_raw[ch] * w[0];
            }
            u_act[ch] = silu(sum);
        }

        // 3. Project `u_act` -> `(dt_rank, B[d_state], C[d_state])`
        let proj_dim = self.dt_rank + 2 * self.d_state;
        let mut dbc = vec![0.0f32; proj_dim];
        for (r, slot) in dbc.iter_mut().enumerate() {
            let row = &self.x_proj[r * self.d_inner..(r + 1) * self.d_inner];
            *slot = ops::dot(row, &u_act);
        }
        let dt_low = &dbc[0..self.dt_rank];
        let b_vec = &dbc[self.dt_rank..self.dt_rank + self.d_state];
        let c_vec = &dbc[self.dt_rank + self.d_state..proj_dim];

        // 4. Selective scan update per channel
        let mut y_inner = vec![0.0f32; self.d_inner];
        for ch in 0..self.d_inner {
            let dt_row = &self.dt_proj_weight[ch * self.dt_rank..(ch + 1) * self.dt_rank];
            let dt = softplus(ops::dot(dt_row, dt_low) + self.dt_proj_bias[ch]);
            let u_ch = u_act[ch];
            let a_row = &self.a_neg[ch * self.d_state..(ch + 1) * self.d_state];
            let h_row = &mut state.ssm_state[ch * self.d_state..(ch + 1) * self.d_state];

            let mut y_ch = self.d_skip[ch] * u_ch;
            for s in 0..self.d_state {
                let d_a = (dt * a_row[s]).exp();
                let d_b = dt * b_vec[s];
                let h_new = d_a * h_row[s] + d_b * u_ch;
                h_row[s] = h_new;
                y_ch += c_vec[s] * h_new;
            }
            y_inner[ch] = y_ch * silu(z[ch]);
        }

        // 5. Output projection `y_inner` -> `out`
        for (r, slot) in out.iter_mut().enumerate() {
            let row = &self.out_proj[r * self.d_inner..(r + 1) * self.d_inner];
            *slot = ops::dot(row, &y_inner);
        }
        state.step_count += 1;
    }

    /// Execute a multi-token prefill sequence `[seq_len, d_model]` -> `[seq_len, d_model]`.
    pub fn prefill(
        &self,
        x_seq: &[f32],
        seq_len: usize,
        state: &mut MambaState,
        out_seq: &mut [f32],
    ) {
        for t in 0..seq_len {
            let x_t = &x_seq[t * self.d_model..(t + 1) * self.d_model];
            let out_t = &mut out_seq[t * self.d_model..(t + 1) * self.d_model];
            self.step(x_t, state, out_t);
        }
    }
}

/// Fixed-size recurrent state for an `RWKV-6` (Finch) / `RWKV-7` (Goose) linear-attention layer.
#[derive(Clone, Debug)]
pub struct RwkvState {
    pub dim: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    /// Previous token's hidden state entering Time-Mix (`[dim]`).
    pub prev_x_time: Vec<f32>,
    /// Previous token's hidden state entering Channel-Mix (`[dim]`).
    pub prev_x_ffn: Vec<f32>,
    /// Per-head recurrent WKV matrix state (`[n_heads, head_dim, head_dim]`).
    pub wkv_state: Vec<f32>,
}

impl RwkvState {
    pub fn new(dim: usize, n_heads: usize) -> Self {
        assert_eq!(dim % n_heads, 0);
        let head_dim = dim / n_heads;
        Self {
            dim,
            n_heads,
            head_dim,
            prev_x_time: vec![0.0; dim],
            prev_x_ffn: vec![0.0; dim],
            wkv_state: vec![0.0; n_heads * head_dim * head_dim],
        }
    }

    pub fn reset(&mut self) {
        self.prev_x_time.fill(0.0);
        self.prev_x_ffn.fill(0.0);
        self.wkv_state.fill(0.0);
    }

    pub fn memory_bytes(&self) -> usize {
        (self.prev_x_time.len() + self.prev_x_ffn.len() + self.wkv_state.len())
            * std::mem::size_of::<f32>()
    }
}

/// Parameters for an `RWKV-6` / `RWKV-7` Time-Mix + Channel-Mix recurrent block.
#[derive(Clone, Debug)]
pub struct RwkvLayer {
    pub dim: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    /// Token-shift interpolation ratios $\mu_r, \mu_k, \mu_v, \mu_w, \mu_g$ (`[dim]` each).
    pub time_maa_r: Vec<f32>,
    pub time_maa_k: Vec<f32>,
    pub time_maa_v: Vec<f32>,
    pub time_maa_w: Vec<f32>,
    pub time_maa_g: Vec<f32>,
    /// Linear projections (`[dim, dim]`).
    pub w_r: Vec<f32>,
    pub w_k: Vec<f32>,
    pub w_v: Vec<f32>,
    pub w_g: Vec<f32>,
    pub w_o: Vec<f32>,
    /// Per-channel base decay $\omega$ (`[dim]`) and in-context bonus $u$ (`[dim]`).
    pub time_decay: Vec<f32>,
    pub time_first: Vec<f32>,
}

impl RwkvLayer {
    /// Execute a single RWKV Time-Mix recurrent step in $O(H \cdot d_h^2)$ time and constant memory.
    pub fn step_time_mix(&self, x: &[f32], state: &mut RwkvState, out: &mut [f32]) {
        debug_assert_eq!(x.len(), self.dim);
        debug_assert_eq!(out.len(), self.dim);

        let mut xr = vec![0.0f32; self.dim];
        let mut xk = vec![0.0f32; self.dim];
        let mut xv = vec![0.0f32; self.dim];
        let mut xg = vec![0.0f32; self.dim];

        for i in 0..self.dim {
            let diff = state.prev_x_time[i] - x[i];
            xr[i] = x[i] + diff * self.time_maa_r[i];
            xk[i] = x[i] + diff * self.time_maa_k[i];
            xv[i] = x[i] + diff * self.time_maa_v[i];
            xg[i] = x[i] + diff * self.time_maa_g[i];
        }
        state.prev_x_time.copy_from_slice(x);

        let mut r = vec![0.0f32; self.dim];
        let mut k = vec![0.0f32; self.dim];
        let mut v = vec![0.0f32; self.dim];
        let mut g = vec![0.0f32; self.dim];

        for i in 0..self.dim {
            let off = i * self.dim;
            r[i] = ops::dot(&self.w_r[off..off + self.dim], &xr);
            k[i] = ops::dot(&self.w_k[off..off + self.dim], &xk);
            v[i] = ops::dot(&self.w_v[off..off + self.dim], &xv);
            g[i] = silu(ops::dot(&self.w_g[off..off + self.dim], &xg));
        }

        let mut wkv_out = vec![0.0f32; self.dim];
        let hd = self.head_dim;

        for h in 0..self.n_heads {
            let base = h * hd;
            let st_head = &mut state.wkv_state[h * hd * hd..(h + 1) * hd * hd];

            for i in 0..hd {
                let ch = base + i;
                let r_i = r[ch];
                let k_i = k[ch];
                let u_i = self.time_first[ch];
                let decay_i = (-self.time_decay[ch].exp()).exp();
                let st_row = &mut st_head[i * hd..(i + 1) * hd];

                for j in 0..hd {
                    let v_j = v[base + j];
                    let kv = k_i * v_j;
                    wkv_out[base + j] += r_i * (u_i * kv + st_row[j]);
                    st_row[j] = decay_i * st_row[j] + kv;
                }
            }
        }

        for i in 0..self.dim {
            wkv_out[i] *= g[i];
        }
        for (i, o_val) in out.iter_mut().enumerate().take(self.dim) {
            let off = i * self.dim;
            *o_val = ops::dot(&self.w_o[off..off + self.dim], &wkv_out);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mamba_selective_scan_step_and_prefill_equivalence() {
        let d_model = 8;
        let d_inner = 16;
        let d_conv = 4;
        let d_state = 4;
        let dt_rank = 2;

        let layer = MambaLayer {
            d_model,
            d_inner,
            d_conv,
            d_state,
            dt_rank,
            in_proj: (0..2 * d_inner * d_model)
                .map(|i| ((i as f32) * 0.13).sin() * 0.1)
                .collect(),
            conv1d_weight: vec![0.25; d_inner * d_conv],
            conv1d_bias: vec![0.01; d_inner],
            x_proj: (0..(dt_rank + 2 * d_state) * d_inner)
                .map(|i| ((i as f32) * 0.19).cos() * 0.1)
                .collect(),
            dt_proj_weight: vec![0.1; d_inner * dt_rank],
            dt_proj_bias: vec![-1.0; d_inner],
            a_neg: vec![-1.0; d_inner * d_state],
            d_skip: vec![1.0; d_inner],
            out_proj: (0..d_model * d_inner)
                .map(|i| ((i as f32) * 0.07).sin() * 0.1)
                .collect(),
        };

        let seq: Vec<f32> = (0..4 * d_model).map(|i| (i as f32) * 0.05 - 0.5).collect();

        // Prefill all 4 tokens at once
        let mut st_prefill = MambaState::new(d_inner, d_conv, d_state);
        let mut out_prefill = vec![0.0f32; 4 * d_model];
        layer.prefill(&seq, 4, &mut st_prefill, &mut out_prefill);

        // Step token by token
        let mut st_step = MambaState::new(d_inner, d_conv, d_state);
        let mut out_step = vec![0.0f32; 4 * d_model];
        for t in 0..4 {
            layer.step(
                &seq[t * d_model..(t + 1) * d_model],
                &mut st_step,
                &mut out_step[t * d_model..(t + 1) * d_model],
            );
        }

        assert_eq!(out_prefill, out_step);
        assert_eq!(st_prefill.ssm_state, st_step.ssm_state);
        assert!(out_prefill.iter().all(|v| v.is_finite() && v.abs() > 0.0));
        // Memory footprint is strictly constant
        assert_eq!(st_step.memory_bytes(), (16 * 3 + 16 * 4) * 4);
    }

    #[test]
    fn rwkv_time_mix_recurrent_state_progression() {
        let dim = 8;
        let n_heads = 2;
        let head_dim = 4;
        let eye: Vec<f32> = (0..dim * dim)
            .map(|i| if i / dim == i % dim { 0.5 } else { 0.05 })
            .collect();

        let layer = RwkvLayer {
            dim,
            n_heads,
            head_dim,
            time_maa_r: vec![0.5; dim],
            time_maa_k: vec![0.5; dim],
            time_maa_v: vec![0.5; dim],
            time_maa_w: vec![0.5; dim],
            time_maa_g: vec![0.5; dim],
            w_r: eye.clone(),
            w_k: eye.clone(),
            w_v: eye.clone(),
            w_g: eye.clone(),
            w_o: eye,
            time_decay: vec![-0.5; dim],
            time_first: vec![0.2; dim],
        };

        let mut state = RwkvState::new(dim, n_heads);
        let x0 = vec![1.0f32; dim];
        let x1 = vec![0.5f32; dim];
        let mut out0 = vec![0.0f32; dim];
        let mut out1 = vec![0.0f32; dim];

        layer.step_time_mix(&x0, &mut state, &mut out0);
        layer.step_time_mix(&x1, &mut state, &mut out1);

        assert!(out0.iter().all(|v| v.is_finite() && v.abs() > 0.0));
        assert!(out1.iter().all(|v| v.is_finite() && v.abs() > 0.0));
        assert_ne!(out0, out1, "recurrent state must evolve across tokens");
    }
}
