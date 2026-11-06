//! Zero-dependency multi-modal vision-language projection engine (CLIP / SigLIP / ViT).
//!
//! Supports pure-Rust image decoding (PPM P3/P6 and uncompressed BMP 24/32-bit),
//! bilinear image resampling, normalization, patch embeddings, bidirectional multi-head
//! vision transformer layers, and multi-modal MLP/linear projections to LLM activation space.

use std::fs;
use std::io::{Error, ErrorKind, Result};
use std::path::Path;

use crate::gguf::Gguf;
use crate::ops;
use crate::quant;

fn bad(msg: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidData, msg.into())
}

/// In-memory RGB image representation with float values.
#[derive(Clone, Debug, PartialEq)]
pub struct Image {
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    /// Data stored in planar RGB format: [C, H, W] where C=0: R, C=1: G, C=2: B.
    /// Length is `channels * height * width`.
    pub data: Vec<f32>,
}

impl Image {
    /// Creates a new empty or zeroed RGB image.
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            channels: 3,
            data: vec![0.0; 3 * width * height],
        }
    }

    /// Creates an image from interleaved RGB bytes [H, W, 3].
    pub fn from_interleaved_rgb(width: usize, height: usize, raw: &[u8]) -> Result<Self> {
        let expected = width
            .checked_mul(height)
            .and_then(|wh| wh.checked_mul(3))
            .ok_or_else(|| bad("Image dimension overflow"))?;
        if raw.len() != expected {
            return Err(bad(format!(
                "RGB buffer size mismatch: expected {expected} bytes, got {}",
                raw.len()
            )));
        }

        let mut data = vec![0.0f32; 3 * width * height];
        let plane_size = width * height;
        for i in 0..plane_size {
            let r = raw[i * 3] as f32 / 255.0;
            let g = raw[i * 3 + 1] as f32 / 255.0;
            let b = raw[i * 3 + 2] as f32 / 255.0;
            data[i] = r;
            data[plane_size + i] = g;
            data[2 * plane_size + i] = b;
        }

        Ok(Self {
            width,
            height,
            channels: 3,
            data,
        })
    }

    /// Parses Netpbm PPM image bytes (P6 binary or P3 ASCII format).
    pub fn from_ppm(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 3 {
            return Err(bad("PPM file too small"));
        }

        if bytes[0] != b'P' {
            return Err(bad("Not a PPM file (missing 'P' magic)"));
        }
        let format = bytes[1];
        let mut pos = 2;

        // Skip whitespace and comments
        let skip_ws = |bytes: &[u8], pos: &mut usize| {
            while *pos < bytes.len() {
                if bytes[*pos] == b'#' {
                    while *pos < bytes.len() && bytes[*pos] != b'\n' {
                        *pos += 1;
                    }
                } else if bytes[*pos].is_ascii_whitespace() {
                    *pos += 1;
                } else {
                    break;
                }
            }
        };

        if format != b'6' && format != b'3' {
            return Err(bad(format!("Unsupported PPM format P{}", format as char)));
        }

        // Read next token as integer
        let read_int = |bytes: &[u8], pos: &mut usize| -> Result<usize> {
            skip_ws(bytes, pos);
            let start = *pos;
            while *pos < bytes.len() && bytes[*pos].is_ascii_digit() {
                *pos += 1;
            }
            if start == *pos {
                return Err(bad("Expected integer token in PPM header"));
            }
            let s = std::str::from_utf8(&bytes[start..*pos])
                .map_err(|_| bad("Invalid ASCII in PPM"))?;
            s.parse::<usize>().map_err(|e| bad(e.to_string()))
        };

        let width = read_int(bytes, &mut pos)?;
        let height = read_int(bytes, &mut pos)?;
        let max_val = read_int(bytes, &mut pos)?;

        if max_val == 0 || max_val > 65535 {
            return Err(bad("PPM maxval must be between 1 and 65535"));
        }

        // Must consume exactly 1 whitespace character after maxval for P6
        if pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
            pos += 1;
        }

        let plane_size = width
            .checked_mul(height)
            .ok_or_else(|| bad("PPM dimensions overflow"))?;
        let mut data = vec![0.0f32; 3 * plane_size];

        if format == b'6' {
            // Binary format
            let scale = 1.0 / (max_val as f32);
            if max_val < 256 {
                let required = plane_size * 3;
                if bytes.len() - pos < required {
                    return Err(bad("Unexpected EOF in PPM P6 pixel data"));
                }
                for i in 0..plane_size {
                    let r = bytes[pos + i * 3] as f32 * scale;
                    let g = bytes[pos + i * 3 + 1] as f32 * scale;
                    let b = bytes[pos + i * 3 + 2] as f32 * scale;
                    data[i] = r;
                    data[plane_size + i] = g;
                    data[2 * plane_size + i] = b;
                }
            } else {
                let required = plane_size * 6;
                if bytes.len() - pos < required {
                    return Err(bad("Unexpected EOF in PPM P6 16-bit pixel data"));
                }
                for i in 0..plane_size {
                    let r_raw = u16::from_be_bytes([bytes[pos + i * 6], bytes[pos + i * 6 + 1]]);
                    let g_raw =
                        u16::from_be_bytes([bytes[pos + i * 6 + 2], bytes[pos + i * 6 + 3]]);
                    let b_raw =
                        u16::from_be_bytes([bytes[pos + i * 6 + 4], bytes[pos + i * 6 + 5]]);
                    data[i] = r_raw as f32 * scale;
                    data[plane_size + i] = g_raw as f32 * scale;
                    data[2 * plane_size + i] = b_raw as f32 * scale;
                }
            }
        } else {
            // P3 ASCII format
            let scale = 1.0 / (max_val as f32);
            for i in 0..plane_size {
                let r = read_int(bytes, &mut pos)? as f32 * scale;
                let g = read_int(bytes, &mut pos)? as f32 * scale;
                let b = read_int(bytes, &mut pos)? as f32 * scale;
                data[i] = r;
                data[plane_size + i] = g;
                data[2 * plane_size + i] = b;
            }
        }

        Ok(Self {
            width,
            height,
            channels: 3,
            data,
        })
    }

    /// Parses an uncompressed 24-bit or 32-bit Windows BMP image.
    pub fn from_bmp(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 54 {
            return Err(bad("BMP file too small (< 54 bytes)"));
        }
        if bytes[0] != b'B' || bytes[1] != b'M' {
            return Err(bad("Not a valid BMP file (missing 'BM' signature)"));
        }

        let pixel_offset =
            u32::from_le_bytes([bytes[10], bytes[11], bytes[12], bytes[13]]) as usize;
        let dib_size = u32::from_le_bytes([bytes[14], bytes[15], bytes[16], bytes[17]]) as usize;
        if dib_size < 40 {
            return Err(bad("Unsupported BMP DIB header size (< 40)"));
        }

        let width = i32::from_le_bytes([bytes[18], bytes[19], bytes[20], bytes[21]]);
        let height = i32::from_le_bytes([bytes[22], bytes[23], bytes[24], bytes[25]]);
        if width <= 0 {
            return Err(bad("BMP width must be positive"));
        }
        let w = width as usize;
        let top_down = height < 0;
        let h = height.unsigned_abs() as usize;

        let planes = u16::from_le_bytes([bytes[26], bytes[27]]);
        if planes != 1 {
            return Err(bad("BMP planes must be 1"));
        }
        let bpp = u16::from_le_bytes([bytes[28], bytes[29]]);
        if bpp != 24 && bpp != 32 {
            return Err(bad(format!(
                "Unsupported BMP bit depth: {bpp} bpp (expected 24 or 32)"
            )));
        }
        let compression = u32::from_le_bytes([bytes[30], bytes[31], bytes[32], bytes[33]]);
        if compression != 0 {
            return Err(bad("Compressed BMP files are not supported"));
        }

        if bytes.len() < pixel_offset {
            return Err(bad("BMP pixel offset exceeds file size"));
        }

        let bytes_per_pixel = (bpp / 8) as usize;
        let row_stride = (w * bytes_per_pixel + 3) & !3; // rows are 4-byte aligned
        let plane_size = w * h;
        let mut data = vec![0.0f32; 3 * plane_size];

        let pixel_data = &bytes[pixel_offset..];
        for row in 0..h {
            let y = if top_down { row } else { h - 1 - row };
            let row_offset = row * row_stride;
            if row_offset + w * bytes_per_pixel > pixel_data.len() {
                return Err(bad("Truncated BMP pixel array"));
            }
            let row_slice = &pixel_data[row_offset..row_offset + w * bytes_per_pixel];
            for x in 0..w {
                // BMP stores BGR or BGRA
                let b = row_slice[x * bytes_per_pixel] as f32 / 255.0;
                let g = row_slice[x * bytes_per_pixel + 1] as f32 / 255.0;
                let r = row_slice[x * bytes_per_pixel + 2] as f32 / 255.0;
                let idx = y * w + x;
                data[idx] = r;
                data[plane_size + idx] = g;
                data[2 * plane_size + idx] = b;
            }
        }

        Ok(Self {
            width: w,
            height: h,
            channels: 3,
            data,
        })
    }

    /// Loads an image from a path (detects BMP or PPM).
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let bytes = fs::read(path)?;
        if bytes.starts_with(b"BM") {
            Self::from_bmp(&bytes)
        } else if bytes.starts_with(b"P6") || bytes.starts_with(b"P3") {
            Self::from_ppm(&bytes)
        } else {
            Err(bad("Unrecognized image format (expected PPM or BMP)"))
        }
    }

    /// Serializes image to Netpbm PPM P6 binary format.
    pub fn to_ppm_p6(&self) -> Vec<u8> {
        let header = format!("P6\n{} {}\n255\n", self.width, self.height);
        let mut out = Vec::with_capacity(header.len() + self.width * self.height * 3);
        out.extend_from_slice(header.as_bytes());

        let plane_size = self.width * self.height;
        for i in 0..plane_size {
            let r = (self.data[i].clamp(0.0, 1.0) * 255.0).round() as u8;
            let g = (self.data[plane_size + i].clamp(0.0, 1.0) * 255.0).round() as u8;
            let b = (self.data[2 * plane_size + i].clamp(0.0, 1.0) * 255.0).round() as u8;
            out.push(r);
            out.push(g);
            out.push(b);
        }
        out
    }

    /// Resamples the image to `(target_w, target_h)` using bilinear interpolation.
    pub fn resize_bilinear(&self, target_w: usize, target_h: usize) -> Self {
        if self.width == target_w && self.height == target_h {
            return self.clone();
        }

        let mut out_data = vec![0.0f32; 3 * target_w * target_h];
        let x_ratio = if target_w > 1 {
            (self.width - 1) as f32 / (target_w - 1) as f32
        } else {
            0.0
        };
        let y_ratio = if target_h > 1 {
            (self.height - 1) as f32 / (target_h - 1) as f32
        } else {
            0.0
        };

        let in_plane = self.width * self.height;
        let out_plane = target_w * target_h;

        for y in 0..target_h {
            let y_src = y as f32 * y_ratio;
            let y_low = y_src.floor() as usize;
            let y_high = (y_low + 1).min(self.height - 1);
            let y_weight = y_src - y_low as f32;

            for x in 0..target_w {
                let x_src = x as f32 * x_ratio;
                let x_low = x_src.floor() as usize;
                let x_high = (x_low + 1).min(self.width - 1);
                let x_weight = x_src - x_low as f32;

                let out_idx = y * target_w + x;
                for c in 0..3 {
                    let c_in_offset = c * in_plane;
                    let p00 = self.data[c_in_offset + y_low * self.width + x_low];
                    let p10 = self.data[c_in_offset + y_low * self.width + x_high];
                    let p01 = self.data[c_in_offset + y_high * self.width + x_low];
                    let p11 = self.data[c_in_offset + y_high * self.width + x_high];

                    let top = p00 * (1.0 - x_weight) + p10 * x_weight;
                    let bottom = p01 * (1.0 - x_weight) + p11 * x_weight;
                    let val = top * (1.0 - y_weight) + bottom * y_weight;

                    out_data[c * out_plane + out_idx] = val;
                }
            }
        }

        Self {
            width: target_w,
            height: target_h,
            channels: 3,
            data: out_data,
        }
    }

    /// Normalizes the image in-place using channel means and standard deviations:
    /// `pixel = (pixel - mean) / std`.
    pub fn normalize(&mut self, mean: [f32; 3], std: [f32; 3]) {
        let plane_size = self.width * self.height;
        for c in 0..3 {
            let m = mean[c];
            let s = std[c].max(1e-7);
            let inv_s = 1.0 / s;
            let offset = c * plane_size;
            for val in &mut self.data[offset..offset + plane_size] {
                *val = (*val - m) * inv_s;
            }
        }
    }
}

/// Vision Transformer hyperparameters and architecture configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct VisionConfig {
    pub image_size: usize,
    pub patch_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub projection_dim: usize,
    pub has_class_token: bool,
    pub eps: f32,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            image_size: 224,
            patch_size: 14,
            hidden_size: 768,
            intermediate_size: 3072,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            projection_dim: 2048,
            has_class_token: true,
            eps: 1e-5,
            mean: [0.48145466, 0.4578275, 0.40821073], // CLIP default
            std: [0.26862954, 0.261_302_6, 0.275_777_1],
        }
    }
}

impl VisionConfig {
    pub fn num_patches_per_axis(&self) -> usize {
        self.image_size / self.patch_size
    }

    pub fn num_patches(&self) -> usize {
        let p = self.num_patches_per_axis();
        p * p
    }

    pub fn total_tokens(&self) -> usize {
        if self.has_class_token {
            self.num_patches() + 1
        } else {
            self.num_patches()
        }
    }
}

/// Vision Transformer self-attention block with bidirectional attention.
#[derive(Clone, Debug)]
pub struct VisionAttention {
    pub wq: Vec<f32>, // [hidden_size, hidden_size]
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub wo: Vec<f32>,
    pub bq: Option<Vec<f32>>,
    pub bk: Option<Vec<f32>>,
    pub bv: Option<Vec<f32>>,
    pub bo: Option<Vec<f32>>,
    pub heads: usize,
    pub hidden_size: usize,
}

impl VisionAttention {
    pub fn forward(&self, x: &[f32], num_tokens: usize, out: &mut [f32]) {
        let dim = self.hidden_size;
        let head_dim = dim / self.heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Temporary buffers
        let mut q = vec![0.0f32; num_tokens * dim];
        let mut k = vec![0.0f32; num_tokens * dim];
        let mut v = vec![0.0f32; num_tokens * dim];
        let mut attn_out = vec![0.0f32; num_tokens * dim];

        // Linear projections Q, K, V
        for t in 0..num_tokens {
            let in_tok = &x[t * dim..(t + 1) * dim];

            // Q
            let q_tok = &mut q[t * dim..(t + 1) * dim];
            ops::matvec(&self.wq, in_tok, q_tok);
            if let Some(b) = &self.bq {
                for (s, val) in q_tok.iter_mut().zip(b) {
                    *s += val;
                }
            }

            // K
            let k_tok = &mut k[t * dim..(t + 1) * dim];
            ops::matvec(&self.wk, in_tok, k_tok);
            if let Some(b) = &self.bk {
                for (s, val) in k_tok.iter_mut().zip(b) {
                    *s += val;
                }
            }

            // V
            let v_tok = &mut v[t * dim..(t + 1) * dim];
            ops::matvec(&self.wv, in_tok, v_tok);
            if let Some(b) = &self.bv {
                for (s, val) in v_tok.iter_mut().zip(b) {
                    *s += val;
                }
            }
        }

        // Bidirectional Multi-Head Attention
        let mut scores = vec![0.0f32; num_tokens];
        for h in 0..self.heads {
            let h_offset = h * head_dim;

            for i in 0..num_tokens {
                let qi = &q[i * dim + h_offset..i * dim + h_offset + head_dim];

                // Compute attention scores against all tokens j (bidirectional!)
                for j in 0..num_tokens {
                    let kj = &k[j * dim + h_offset..j * dim + h_offset + head_dim];
                    let score = ops::dot(qi, kj) * scale;
                    scores[j] = score;
                }

                // Softmax
                ops::softmax(&mut scores[..num_tokens]);

                // Accumulate weighted V
                let out_slot = &mut attn_out[i * dim + h_offset..i * dim + h_offset + head_dim];
                for s in out_slot.iter_mut() {
                    *s = 0.0;
                }
                for j in 0..num_tokens {
                    let weight = scores[j];
                    let vj = &v[j * dim + h_offset..j * dim + h_offset + head_dim];
                    for (slot, &val) in out_slot.iter_mut().zip(vj) {
                        *slot += weight * val;
                    }
                }
            }
        }

        // Output projection W_O
        for t in 0..num_tokens {
            let in_tok = &attn_out[t * dim..(t + 1) * dim];
            let out_tok = &mut out[t * dim..(t + 1) * dim];
            ops::matvec(&self.wo, in_tok, out_tok);
            if let Some(b) = &self.bo {
                for (s, val) in out_tok.iter_mut().zip(b) {
                    *s += val;
                }
            }
        }
    }
}

/// Vision Transformer Feed-Forward Network with GeLU activation.
#[derive(Clone, Debug)]
pub struct VisionMlp {
    pub up: Vec<f32>,   // [intermediate_size, hidden_size]
    pub down: Vec<f32>, // [hidden_size, intermediate_size]
    pub up_bias: Option<Vec<f32>>,
    pub down_bias: Option<Vec<f32>>,
    pub hidden_size: usize,
    pub intermediate_size: usize,
}

impl VisionMlp {
    pub fn forward(&self, x: &[f32], num_tokens: usize, out: &mut [f32]) {
        let mut inter = vec![0.0f32; self.intermediate_size];

        for t in 0..num_tokens {
            let in_tok = &x[t * self.hidden_size..(t + 1) * self.hidden_size];
            ops::matvec(&self.up, in_tok, &mut inter);
            if let Some(b) = &self.up_bias {
                for (s, val) in inter.iter_mut().zip(b) {
                    *s += val;
                }
            }

            // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            for v in inter.iter_mut() {
                let x_val = *v;
                let cube = x_val * x_val * x_val;
                let inner = 0.797_884_6 * (x_val + 0.044715 * cube);
                *v = 0.5 * x_val * (1.0 + inner.tanh());
            }

            let out_tok = &mut out[t * self.hidden_size..(t + 1) * self.hidden_size];
            ops::matvec(&self.down, &inter, out_tok);
            if let Some(b) = &self.down_bias {
                for (s, val) in out_tok.iter_mut().zip(b) {
                    *s += val;
                }
            }
        }
    }
}

/// Single Vision Transformer encoder layer.
#[derive(Clone, Debug)]
pub struct VisionLayer {
    pub ln1_weight: Vec<f32>,
    pub ln1_bias: Vec<f32>,
    pub ln2_weight: Vec<f32>,
    pub ln2_bias: Vec<f32>,
    pub attn: VisionAttention,
    pub mlp: VisionMlp,
    pub eps: f32,
}

impl VisionLayer {
    pub fn forward(&self, x: &mut [f32], num_tokens: usize) {
        let dim = self.attn.hidden_size;
        let mut norm_buf = vec![0.0f32; num_tokens * dim];
        let mut delta_buf = vec![0.0f32; num_tokens * dim];

        // LayerNorm 1 + Attention + Residual
        for t in 0..num_tokens {
            let in_tok = &x[t * dim..(t + 1) * dim];
            let out_tok = &mut norm_buf[t * dim..(t + 1) * dim];
            layer_norm(in_tok, &self.ln1_weight, &self.ln1_bias, self.eps, out_tok);
        }
        self.attn.forward(&norm_buf, num_tokens, &mut delta_buf);
        for (slot, delta) in x.iter_mut().zip(delta_buf.iter()) {
            *slot += delta;
        }

        // LayerNorm 2 + MLP + Residual
        for t in 0..num_tokens {
            let in_tok = &x[t * dim..(t + 1) * dim];
            let out_tok = &mut norm_buf[t * dim..(t + 1) * dim];
            layer_norm(in_tok, &self.ln2_weight, &self.ln2_bias, self.eps, out_tok);
        }
        self.mlp.forward(&norm_buf, num_tokens, &mut delta_buf);
        for (slot, delta) in x.iter_mut().zip(delta_buf.iter()) {
            *slot += delta;
        }
    }
}

/// Standard layer normalization: y = (x - mean) / sqrt(var + eps) * weight + bias.
pub fn layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32, out: &mut [f32]) {
    assert_eq!(x.len(), weight.len());
    assert_eq!(x.len(), bias.len());
    assert_eq!(x.len(), out.len());

    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv_std = 1.0 / (var + eps).sqrt();

    for i in 0..x.len() {
        out[i] = (x[i] - mean) * inv_std * weight[i] + bias[i];
    }
}

/// Multi-Modal Projector: maps Vision Transformer embeddings into LLM embedding space.
#[derive(Clone, Debug)]
pub enum MultiModalProjector {
    /// Single linear projection: [projection_dim, hidden_size]
    Linear {
        weight: Vec<f32>,
        bias: Option<Vec<f32>>,
        in_dim: usize,
        out_dim: usize,
    },
    /// 2-layer MLP projection: Linear -> GELU -> Linear
    Mlp {
        w1: Vec<f32>, // [out_dim, in_dim]
        b1: Option<Vec<f32>>,
        w2: Vec<f32>, // [out_dim, out_dim]
        b2: Option<Vec<f32>>,
        in_dim: usize,
        out_dim: usize,
    },
}

impl MultiModalProjector {
    pub fn forward(&self, in_tokens: &[f32], num_tokens: usize, out_tokens: &mut [f32]) {
        match self {
            Self::Linear {
                weight,
                bias,
                in_dim,
                out_dim,
            } => {
                for t in 0..num_tokens {
                    let in_tok = &in_tokens[t * in_dim..(t + 1) * in_dim];
                    let out_tok = &mut out_tokens[t * out_dim..(t + 1) * out_dim];
                    ops::matvec(weight, in_tok, out_tok);
                    if let Some(b) = bias {
                        for (s, val) in out_tok.iter_mut().zip(b) {
                            *s += val;
                        }
                    }
                }
            }
            Self::Mlp {
                w1,
                b1,
                w2,
                b2,
                in_dim,
                out_dim,
            } => {
                let mut mid = vec![0.0f32; *out_dim];
                for t in 0..num_tokens {
                    let in_tok = &in_tokens[t * in_dim..(t + 1) * in_dim];
                    ops::matvec(w1, in_tok, &mut mid);
                    if let Some(b) = b1 {
                        for (s, val) in mid.iter_mut().zip(b) {
                            *s += val;
                        }
                    }

                    // GELU
                    for v in mid.iter_mut() {
                        let x = *v;
                        let cube = x * x * x;
                        let inner = 0.797_884_6 * (x + 0.044715 * cube);
                        *v = 0.5 * x * (1.0 + inner.tanh());
                    }

                    let out_tok = &mut out_tokens[t * out_dim..(t + 1) * out_dim];
                    ops::matvec(w2, &mid, out_tok);
                    if let Some(b) = b2 {
                        for (s, val) in out_tok.iter_mut().zip(b) {
                            *s += val;
                        }
                    }
                }
            }
        }
    }
}

/// Complete Multi-Modal Vision Encoder (ViT + Positional Embedding + Projector).
#[derive(Clone, Debug)]
pub struct VisionEncoder {
    pub config: VisionConfig,
    /// Patch embedding weight: [hidden_size, 3 * patch_size * patch_size]
    pub patch_weight: Vec<f32>,
    pub patch_bias: Option<Vec<f32>>,
    /// Positional embedding: [total_tokens, hidden_size]
    pub pos_embed: Vec<f32>,
    /// Optional class token embedding: [hidden_size]
    pub class_token: Option<Vec<f32>>,
    /// Transformer encoder layers
    pub layers: Vec<VisionLayer>,
    /// Post-transformer LayerNorm
    pub post_ln_weight: Vec<f32>,
    pub post_ln_bias: Vec<f32>,
    /// Projection to LLM token embedding space
    pub projector: MultiModalProjector,
}

impl VisionEncoder {
    /// Extracts patches from `image` and projects to `[num_patches, hidden_size]`.
    pub fn embed_patches(&self, image: &Image, out: &mut [f32]) -> Result<()> {
        let p = self.config.patch_size;
        let p_axis = self.config.num_patches_per_axis();
        let hidden = self.config.hidden_size;
        let patch_pixels = 3 * p * p;

        if image.width != self.config.image_size || image.height != self.config.image_size {
            return Err(bad(format!(
                "Image size mismatch: expected {}x{}, got {}x{}",
                self.config.image_size, self.config.image_size, image.width, image.height
            )));
        }

        let mut patch_buf = vec![0.0f32; patch_pixels];
        let plane_size = image.width * image.height;

        for py in 0..p_axis {
            for px in 0..p_axis {
                let patch_idx = py * p_axis + px;

                // Extract patch pixel values into patch_buf [C, p, p]
                let mut buf_idx = 0;
                for c in 0..3 {
                    let c_offset = c * plane_size;
                    for dy in 0..p {
                        let y = py * p + dy;
                        let row_offset = c_offset + y * image.width;
                        for dx in 0..p {
                            let x = px * p + dx;
                            patch_buf[buf_idx] = image.data[row_offset + x];
                            buf_idx += 1;
                        }
                    }
                }

                // Linear projection to hidden_size
                let out_tok = &mut out[patch_idx * hidden..(patch_idx + 1) * hidden];
                ops::matvec(&self.patch_weight, &patch_buf, out_tok);
                if let Some(b) = &self.patch_bias {
                    for (slot, val) in out_tok.iter_mut().zip(b) {
                        *slot += val;
                    }
                }
            }
        }

        Ok(())
    }

    /// Encodes an image through the Vision Transformer and multi-modal projector,
    /// returning LLM-space token embeddings: `[num_tokens, projection_dim]`.
    pub fn encode_image(&self, image: &Image) -> Result<Vec<f32>> {
        // Preprocess: resize if needed and normalize
        let mut preprocessed =
            image.resize_bilinear(self.config.image_size, self.config.image_size);
        preprocessed.normalize(self.config.mean, self.config.std);

        let total_tokens = self.config.total_tokens();
        let hidden = self.config.hidden_size;
        let mut tokens = vec![0.0f32; total_tokens * hidden];

        let num_patches = self.config.num_patches();
        let patch_offset = if self.config.has_class_token {
            // Token 0 is class token
            if let Some(cls) = &self.class_token {
                tokens[..hidden].copy_from_slice(cls);
            }
            1
        } else {
            0
        };

        // Extract and project patches
        self.embed_patches(
            &preprocessed,
            &mut tokens[patch_offset * hidden..(patch_offset + num_patches) * hidden],
        )?;

        // Add positional embedding: [total_tokens, hidden]
        if self.pos_embed.len() >= total_tokens * hidden {
            for (s, p) in tokens
                .iter_mut()
                .zip(&self.pos_embed[..total_tokens * hidden])
            {
                *s += p;
            }
        }

        // Run through transformer layers
        for layer in &self.layers {
            layer.forward(&mut tokens, total_tokens);
        }

        // Post LayerNorm
        let mut normed_tokens = vec![0.0f32; total_tokens * hidden];
        for t in 0..total_tokens {
            let in_tok = &tokens[t * hidden..(t + 1) * hidden];
            let out_tok = &mut normed_tokens[t * hidden..(t + 1) * hidden];
            layer_norm(
                in_tok,
                &self.post_ln_weight,
                &self.post_ln_bias,
                self.config.eps,
                out_tok,
            );
        }

        // If class token was used, some architectures drop it before projecting patches to LLM
        let (proj_tokens, proj_count) = if self.config.has_class_token {
            // Usually, multi-modal LLMs (like LLaVA / CLIP) project the patch tokens [1..] to LLM space
            (&normed_tokens[hidden..], num_patches)
        } else {
            (&normed_tokens[..], num_patches)
        };

        let mut projected = vec![0.0f32; proj_count * self.config.projection_dim];
        self.projector
            .forward(proj_tokens, proj_count, &mut projected);

        Ok(projected)
    }

    /// Loads a multi-modal vision projection model from a GGUF file.
    pub fn from_gguf(gguf: &Gguf) -> Result<Self> {
        let load_f32_vec = |name: &str| -> Result<Vec<f32>> {
            let tensor = gguf
                .tensor(name)
                .ok_or_else(|| bad(format!("Missing required tensor: {name}")))?;
            let raw = gguf.tensor_bytes(tensor)?;
            let count = tensor.elements() as usize;
            let mut out = vec![0.0f32; count];
            quant::dequantize(tensor.ty, raw, &mut out)?;
            Ok(out)
        };

        let load_f32_opt = |name: &str| -> Result<Option<Vec<f32>>> {
            if let Some(tensor) = gguf.tensor(name) {
                let raw = gguf.tensor_bytes(tensor)?;
                let count = tensor.elements() as usize;
                let mut out = vec![0.0f32; count];
                quant::dequantize(tensor.ty, raw, &mut out)?;
                Ok(Some(out))
            } else {
                Ok(None)
            }
        };

        // Read configuration from GGUF metadata or infer from tensor dimensions
        let image_size = gguf
            .meta_u64("clip.vision.image_size")
            .map(|v| v as usize)
            .unwrap_or(224);
        let patch_size = gguf
            .meta_u64("clip.vision.patch_size")
            .map(|v| v as usize)
            .unwrap_or(14);
        let hidden_size = gguf
            .meta_u64("clip.vision.embedding_length")
            .map(|v| v as usize)
            .unwrap_or(768);
        let intermediate_size = gguf
            .meta_u64("clip.vision.feed_forward_length")
            .map(|v| v as usize)
            .unwrap_or(3072);
        let num_hidden_layers = gguf
            .meta_u64("clip.vision.block_count")
            .map(|v| v as usize)
            .unwrap_or(12);
        let num_attention_heads = gguf
            .meta_u64("clip.vision.attention.head_count")
            .map(|v| v as usize)
            .unwrap_or(12);
        let projection_dim = gguf
            .meta_u64("clip.vision.projection_dim")
            .map(|v| v as usize)
            .unwrap_or(2048);

        let config = VisionConfig {
            image_size,
            patch_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            projection_dim,
            has_class_token: true,
            eps: 1e-5,
            mean: [0.48145466, 0.4578275, 0.40821073],
            std: [0.26862954, 0.261_302_6, 0.275_777_1],
        };

        // Load patch embedding
        let patch_weight = load_f32_vec("v.patch_embd.weight")
            .or_else(|_| load_f32_vec("model.vision_tower.patch_embedding.weight"))?;
        let patch_bias = load_f32_opt("v.patch_embd.bias")
            .or_else(|_| load_f32_opt("model.vision_tower.patch_embedding.bias"))?;

        // Positional embedding and class token
        let pos_embed = load_f32_vec("v.position_embd.weight")
            .or_else(|_| load_f32_vec("model.vision_tower.position_embedding.weight"))?;
        let class_token = load_f32_opt("v.class_embd")
            .or_else(|_| load_f32_opt("model.vision_tower.class_token"))?;

        // Layers
        let mut layers = Vec::with_capacity(num_hidden_layers);
        for i in 0..num_hidden_layers {
            let ln1_weight = load_f32_vec(&format!("v.blk.{i}.ln1.weight"))
                .or_else(|_| load_f32_vec(&format!("model.vision_tower.layers.{i}.ln1.weight")))?;
            let ln1_bias = load_f32_vec(&format!("v.blk.{i}.ln1.bias"))
                .or_else(|_| load_f32_vec(&format!("model.vision_tower.layers.{i}.ln1.bias")))?;
            let ln2_weight = load_f32_vec(&format!("v.blk.{i}.ln2.weight"))
                .or_else(|_| load_f32_vec(&format!("model.vision_tower.layers.{i}.ln2.weight")))?;
            let ln2_bias = load_f32_vec(&format!("v.blk.{i}.ln2.bias"))
                .or_else(|_| load_f32_vec(&format!("model.vision_tower.layers.{i}.ln2.bias")))?;

            let wq = load_f32_vec(&format!("v.blk.{i}.attn_q.weight")).or_else(|_| {
                load_f32_vec(&format!("model.vision_tower.layers.{i}.attn_q.weight"))
            })?;
            let wk = load_f32_vec(&format!("v.blk.{i}.attn_k.weight")).or_else(|_| {
                load_f32_vec(&format!("model.vision_tower.layers.{i}.attn_k.weight"))
            })?;
            let wv = load_f32_vec(&format!("v.blk.{i}.attn_v.weight")).or_else(|_| {
                load_f32_vec(&format!("model.vision_tower.layers.{i}.attn_v.weight"))
            })?;
            let wo = load_f32_vec(&format!("v.blk.{i}.attn_out.weight")).or_else(|_| {
                load_f32_vec(&format!("model.vision_tower.layers.{i}.attn_out.weight"))
            })?;

            let bq = load_f32_opt(&format!("v.blk.{i}.attn_q.bias"))
                .or_else(|_| load_f32_opt(&format!("model.vision_tower.layers.{i}.attn_q.bias")))?;
            let bk = load_f32_opt(&format!("v.blk.{i}.attn_k.bias"))
                .or_else(|_| load_f32_opt(&format!("model.vision_tower.layers.{i}.attn_k.bias")))?;
            let bv = load_f32_opt(&format!("v.blk.{i}.attn_v.bias"))
                .or_else(|_| load_f32_opt(&format!("model.vision_tower.layers.{i}.attn_v.bias")))?;
            let bo = load_f32_opt(&format!("v.blk.{i}.attn_out.bias")).or_else(|_| {
                load_f32_opt(&format!("model.vision_tower.layers.{i}.attn_out.bias"))
            })?;

            let up = load_f32_vec(&format!("v.blk.{i}.ffn_up.weight")).or_else(|_| {
                load_f32_vec(&format!("model.vision_tower.layers.{i}.ffn_up.weight"))
            })?;
            let down = load_f32_vec(&format!("v.blk.{i}.ffn_down.weight")).or_else(|_| {
                load_f32_vec(&format!("model.vision_tower.layers.{i}.ffn_down.weight"))
            })?;
            let up_bias = load_f32_opt(&format!("v.blk.{i}.ffn_up.bias"))
                .or_else(|_| load_f32_opt(&format!("model.vision_tower.layers.{i}.ffn_up.bias")))?;
            let down_bias = load_f32_opt(&format!("v.blk.{i}.ffn_down.bias")).or_else(|_| {
                load_f32_opt(&format!("model.vision_tower.layers.{i}.ffn_down.bias"))
            })?;

            layers.push(VisionLayer {
                ln1_weight,
                ln1_bias,
                ln2_weight,
                ln2_bias,
                attn: VisionAttention {
                    wq,
                    wk,
                    wv,
                    wo,
                    bq,
                    bk,
                    bv,
                    bo,
                    heads: num_attention_heads,
                    hidden_size,
                },
                mlp: VisionMlp {
                    up,
                    down,
                    up_bias,
                    down_bias,
                    hidden_size,
                    intermediate_size,
                },
                eps: config.eps,
            });
        }

        let post_ln_weight = load_f32_vec("v.post_ln.weight")
            .or_else(|_| load_f32_vec("model.vision_tower.post_ln.weight"))?;
        let post_ln_bias = load_f32_vec("v.post_ln.bias")
            .or_else(|_| load_f32_vec("model.vision_tower.post_ln.bias"))?;

        // Multi-modal projector (Linear or 2-layer MLP)
        let projector = if let Some(w1) =
            load_f32_opt("mm.0.weight").or_else(|_| load_f32_opt("model.mm_projector.0.weight"))?
        {
            let b1 =
                load_f32_opt("mm.0.bias").or_else(|_| load_f32_opt("model.mm_projector.0.bias"))?;
            let w2 = load_f32_vec("mm.2.weight")
                .or_else(|_| load_f32_vec("model.mm_projector.2.weight"))?;
            let b2 =
                load_f32_opt("mm.2.bias").or_else(|_| load_f32_opt("model.mm_projector.2.bias"))?;
            MultiModalProjector::Mlp {
                w1,
                b1,
                w2,
                b2,
                in_dim: hidden_size,
                out_dim: projection_dim,
            }
        } else {
            let weight =
                load_f32_vec("mm.weight").or_else(|_| load_f32_vec("model.mm_projector.weight"))?;
            let bias =
                load_f32_opt("mm.bias").or_else(|_| load_f32_opt("model.mm_projector.bias"))?;
            MultiModalProjector::Linear {
                weight,
                bias,
                in_dim: hidden_size,
                out_dim: projection_dim,
            }
        };

        Ok(Self {
            config,
            patch_weight,
            patch_bias,
            pos_embed,
            class_token,
            layers,
            post_ln_weight,
            post_ln_bias,
            projector,
        })
    }
}
