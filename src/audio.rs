//! Multi-modal audio spectrogram projection engine (Whisper and Audio-LLM patch extraction).
//!
//! Provides zero-dependency audio waveform decoding (PCM WAV), Short-Time Fourier
//! Transform (STFT), triangular Log-Mel filterbank computation, audio frame patch
//! grouping, and projection into language model activation space.

use std::f32::consts::PI;
use std::io::{Error, ErrorKind, Result};

fn bad(msg: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidData, msg.into())
}

/// In-memory mono audio waveform normalized to `[-1.0, 1.0]`.
#[derive(Clone, Debug, PartialEq)]
pub struct AudioWaveform {
    pub sample_rate: usize,
    pub samples: Vec<f32>,
}

impl AudioWaveform {
    /// Create a new audio waveform from normalized float samples.
    pub fn new(samples: Vec<f32>, sample_rate: usize) -> Self {
        Self {
            sample_rate,
            samples,
        }
    }

    /// Create an audio waveform from 16-bit signed PCM samples.
    pub fn from_pcm16(raw_samples: &[i16], sample_rate: usize) -> Self {
        let samples = raw_samples
            .iter()
            .map(|&s| s as f32 / 32768.0)
            .collect::<Vec<f32>>();
        Self {
            sample_rate,
            samples,
        }
    }

    /// Duration of the audio in seconds.
    pub fn duration_seconds(&self) -> f32 {
        if self.sample_rate == 0 {
            0.0
        } else {
            self.samples.len() as f32 / self.sample_rate as f32
        }
    }

    /// Parse a RIFF/WAVE PCM audio file from raw bytes.
    pub fn from_wav_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 44 {
            return Err(bad("WAV buffer too small for standard header"));
        }

        if &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
            return Err(bad("Invalid RIFF/WAVE header"));
        }

        let mut offset = 12;
        let mut audio_format = 0u16;
        let mut num_channels = 0u16;
        let mut sample_rate = 0u32;
        let mut bits_per_sample = 0u16;
        let mut data_offset = 0usize;
        let mut data_len = 0usize;

        while offset + 8 <= bytes.len() {
            let chunk_id = &bytes[offset..offset + 4];
            let chunk_size = u32::from_le_bytes([
                bytes[offset + 4],
                bytes[offset + 5],
                bytes[offset + 6],
                bytes[offset + 7],
            ]) as usize;
            offset += 8;

            if chunk_id == b"fmt " {
                if chunk_size < 16 || offset + 16 > bytes.len() {
                    return Err(bad("Invalid fmt chunk size"));
                }
                audio_format = u16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
                num_channels = u16::from_le_bytes([bytes[offset + 2], bytes[offset + 3]]);
                sample_rate = u32::from_le_bytes([
                    bytes[offset + 4],
                    bytes[offset + 5],
                    bytes[offset + 6],
                    bytes[offset + 7],
                ]);
                bits_per_sample = u16::from_le_bytes([bytes[offset + 14], bytes[offset + 15]]);
                offset += chunk_size;
            } else if chunk_id == b"data" {
                data_offset = offset;
                data_len = chunk_size.min(bytes.len().saturating_sub(offset));
                break;
            } else {
                offset += chunk_size;
            }
        }

        if data_offset == 0 {
            return Err(bad("WAV missing data chunk"));
        }
        if audio_format != 1 {
            return Err(bad(format!(
                "Unsupported WAV audio format code: {audio_format} (only PCM=1 supported)"
            )));
        }
        if num_channels == 0 {
            return Err(bad("Invalid channel count 0"));
        }

        let data_bytes = &bytes[data_offset..data_offset + data_len];
        let channels = num_channels as usize;

        let samples = match bits_per_sample {
            16 => {
                let total_samples = data_bytes.len() / 2;
                let num_frames = total_samples / channels;
                let mut mono_samples = Vec::with_capacity(num_frames);

                for frame_idx in 0..num_frames {
                    let mut sum = 0.0f32;
                    for ch in 0..channels {
                        let sample_idx = (frame_idx * channels + ch) * 2;
                        let s = i16::from_le_bytes([
                            data_bytes[sample_idx],
                            data_bytes[sample_idx + 1],
                        ]);
                        sum += s as f32 / 32768.0;
                    }
                    mono_samples.push(sum / channels as f32);
                }
                mono_samples
            }
            8 => {
                let num_frames = data_bytes.len() / channels;
                let mut mono_samples = Vec::with_capacity(num_frames);
                for frame_idx in 0..num_frames {
                    let mut sum = 0.0f32;
                    for ch in 0..channels {
                        let b = data_bytes[frame_idx * channels + ch];
                        let s = (b as f32 - 128.0) / 128.0;
                        sum += s;
                    }
                    mono_samples.push(sum / channels as f32);
                }
                mono_samples
            }
            _ => {
                return Err(bad(format!(
                    "Unsupported bits per sample: {bits_per_sample}"
                )));
            }
        };

        Ok(Self {
            sample_rate: sample_rate as usize,
            samples,
        })
    }

    /// Resample waveform to a target sample rate using linear interpolation.
    pub fn resample_linear(&self, target_sample_rate: usize) -> Self {
        if self.sample_rate == target_sample_rate || self.samples.is_empty() {
            return self.clone();
        }

        let ratio = self.sample_rate as f64 / target_sample_rate as f64;
        let new_len = ((self.samples.len() as f64) / ratio).round() as usize;
        let mut out = Vec::with_capacity(new_len);

        for i in 0..new_len {
            let src_idx = i as f64 * ratio;
            let idx0 = src_idx.floor() as usize;
            let frac = (src_idx - idx0 as f64) as f32;

            if idx0 + 1 < self.samples.len() {
                let v0 = self.samples[idx0];
                let v1 = self.samples[idx0 + 1];
                out.push(v0 + frac * (v1 - v0));
            } else if idx0 < self.samples.len() {
                out.push(self.samples[idx0]);
            } else {
                out.push(0.0);
            }
        }

        Self {
            sample_rate: target_sample_rate,
            samples: out,
        }
    }
}

/// Configuration parameters for Mel spectrogram extraction.
#[derive(Clone, Debug, PartialEq)]
pub struct SpectrogramConfig {
    pub sample_rate: usize,
    pub n_fft: usize,
    pub hop_length: usize,
    pub n_mels: usize,
    pub f_min: f32,
    pub f_max: f32,
}

impl Default for SpectrogramConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            n_fft: 400,
            hop_length: 160,
            n_mels: 80,
            f_min: 0.0,
            f_max: 8000.0,
        }
    }
}

/// Computed 2D Log-Mel spectrogram.
/// Layout: `[n_frames, n_mels]` contiguous in row-major order.
#[derive(Clone, Debug, PartialEq)]
pub struct LogMelSpectrogram {
    pub n_frames: usize,
    pub n_mels: usize,
    pub data: Vec<f32>,
}

impl LogMelSpectrogram {
    /// Get the Mel energy slice for a specific time frame index.
    pub fn frame(&self, frame_idx: usize) -> &[f32] {
        let start = frame_idx * self.n_mels;
        &self.data[start..start + self.n_mels]
    }
}

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0f32.powf(mel / 2595.0) - 1.0)
}

/// Construct triangular Mel filterbank matrix of size `[n_mels, (n_fft / 2) + 1]`.
pub fn build_mel_filterbank(config: &SpectrogramConfig) -> Vec<Vec<f32>> {
    let num_bins = (config.n_fft / 2) + 1;
    let mel_min = hz_to_mel(config.f_min);
    let mel_max = hz_to_mel(config.f_max);

    let mut mel_points = Vec::with_capacity(config.n_mels + 2);
    for i in 0..=config.n_mels + 1 {
        let mel = mel_min + (mel_max - mel_min) * (i as f32 / (config.n_mels + 1) as f32);
        mel_points.push(mel_to_hz(mel));
    }

    let bin_freq_step = config.sample_rate as f32 / config.n_fft as f32;
    let mut filterbank = vec![vec![0.0f32; num_bins]; config.n_mels];

    for m in 0..config.n_mels {
        let f_left = mel_points[m];
        let f_center = mel_points[m + 1];
        let f_right = mel_points[m + 2];

        for (k, weight) in filterbank[m].iter_mut().enumerate() {
            let freq = k as f32 * bin_freq_step;
            if freq >= f_left && freq <= f_center {
                if f_center > f_left {
                    *weight = (freq - f_left) / (f_center - f_left);
                }
            } else if freq > f_center && freq <= f_right && f_right > f_center {
                *weight = (f_right - freq) / (f_right - f_center);
            }
        }
    }

    filterbank
}

/// Compute Hann window coefficients of size `n_fft`.
pub fn hann_window(size: usize) -> Vec<f32> {
    if size == 0 {
        return Vec::new();
    }
    (0..size)
        .map(|n| 0.5 * (1.0 - (2.0 * PI * n as f32 / size as f32).cos()))
        .collect()
}

/// Extract Log-Mel spectrogram from an audio waveform.
pub fn extract_log_mel_spectrogram(
    waveform: &AudioWaveform,
    config: &SpectrogramConfig,
) -> LogMelSpectrogram {
    let audio = if waveform.sample_rate != config.sample_rate {
        waveform.resample_linear(config.sample_rate)
    } else {
        waveform.clone()
    };

    let window = hann_window(config.n_fft);
    let filterbank = build_mel_filterbank(config);
    let num_freq_bins = (config.n_fft / 2) + 1;

    let num_samples = audio.samples.len();
    if num_samples < config.n_fft {
        let mut padded = audio.samples.clone();
        padded.resize(config.n_fft, 0.0);
        return extract_log_mel_spectrogram(
            &AudioWaveform::new(padded, config.sample_rate),
            config,
        );
    }

    let n_frames = (num_samples - config.n_fft) / config.hop_length + 1;
    let mut data = Vec::with_capacity(n_frames * config.n_mels);

    let mut stft_power = vec![0.0f32; num_freq_bins];

    for frame_idx in 0..n_frames {
        let frame_start = frame_idx * config.hop_length;
        let frame_samples = &audio.samples[frame_start..frame_start + config.n_fft];

        // Discrete Fourier Transform for real signal
        for (k, power) in stft_power.iter_mut().enumerate() {
            let mut real = 0.0f32;
            let mut imag = 0.0f32;
            let angle_step = 2.0 * PI * k as f32 / config.n_fft as f32;

            for (n, &s) in frame_samples.iter().enumerate() {
                let win_s = s * window[n];
                let angle = angle_step * n as f32;
                real += win_s * angle.cos();
                imag -= win_s * angle.sin();
            }

            *power = real * real + imag * imag;
        }

        // Apply Mel filterbank
        for filter in &filterbank[..config.n_mels] {
            let mut mel_energy = 0.0f32;
            for (k, &f_val) in filter.iter().take(num_freq_bins).enumerate() {
                mel_energy += f_val * stft_power[k];
            }
            // Clamp and convert to log-scale
            let log_mel = mel_energy.max(1e-5).log10();
            data.push(log_mel);
        }
    }

    LogMelSpectrogram {
        n_frames,
        n_mels: config.n_mels,
        data,
    }
}

/// Groups adjacent spectrogram frames into audio token patches.
#[derive(Clone, Debug)]
pub struct AudioPatchExtractor {
    pub patch_frames: usize,
    pub stride: usize,
}

impl AudioPatchExtractor {
    pub fn new(patch_frames: usize, stride: usize) -> Self {
        Self {
            patch_frames: patch_frames.max(1),
            stride: stride.max(1),
        }
    }

    /// Flatten groups of consecutive Mel frames into patch vectors.
    /// Each patch vector has dimension `patch_frames * n_mels`.
    pub fn extract_patches(&self, spec: &LogMelSpectrogram) -> Vec<Vec<f32>> {
        if spec.n_frames == 0 {
            return Vec::new();
        }

        let patch_dim = self.patch_frames * spec.n_mels;
        let mut patches = Vec::new();

        let mut start_frame = 0;
        while start_frame + self.patch_frames <= spec.n_frames {
            let mut patch = Vec::with_capacity(patch_dim);
            for f in 0..self.patch_frames {
                let frame_slice = spec.frame(start_frame + f);
                patch.extend_from_slice(frame_slice);
            }
            patches.push(patch);
            start_frame += self.stride;
        }

        patches
    }
}

/// Linear projection mapping audio patch features into LLM embedding space.
#[derive(Clone, Debug)]
pub struct AudioProjector {
    pub patch_dim: usize,
    pub llm_dim: usize,
    /// Projection matrix weights with layout `[llm_dim, patch_dim]`.
    pub weight: Vec<f32>,
    /// Optional projection bias vector of size `llm_dim`.
    pub bias: Option<Vec<f32>>,
}

impl AudioProjector {
    pub fn new(
        patch_dim: usize,
        llm_dim: usize,
        weight: Vec<f32>,
        bias: Option<Vec<f32>>,
    ) -> Result<Self> {
        if weight.len() != patch_dim * llm_dim {
            return Err(bad(format!(
                "Weight dimension mismatch: expected {} elements, got {}",
                patch_dim * llm_dim,
                weight.len()
            )));
        }
        if let Some(ref b) = bias {
            if b.len() != llm_dim {
                return Err(bad(format!(
                    "Bias dimension mismatch: expected {llm_dim} elements, got {}",
                    b.len()
                )));
            }
        }
        Ok(Self {
            patch_dim,
            llm_dim,
            weight,
            bias,
        })
    }

    /// Project audio patches into LLM token embedding vectors.
    pub fn project(&self, patches: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let mut projected = Vec::with_capacity(patches.len());

        for patch in patches {
            if patch.len() != self.patch_dim {
                return Err(bad(format!(
                    "Patch dimension mismatch: expected {}, got {}",
                    self.patch_dim,
                    patch.len()
                )));
            }

            let mut out = vec![0.0f32; self.llm_dim];
            for row in 0..self.llm_dim {
                let mut sum = 0.0f32;
                let w_row = &self.weight[row * self.patch_dim..(row + 1) * self.patch_dim];
                for (col, &w) in w_row.iter().enumerate() {
                    sum += w * patch[col];
                }
                if let Some(ref b) = self.bias {
                    sum += b[row];
                }
                out[row] = sum;
            }
            projected.push(out);
        }

        Ok(projected)
    }
}

/// Stitch text embeddings with projected audio tokens for multimodal prefill.
pub fn stitch_audio_and_text_embeddings(
    text_embeddings: &[f32],
    audio_tokens: &[Vec<f32>],
    insert_position: usize,
    llm_dim: usize,
) -> Result<Vec<f32>> {
    if text_embeddings.len() % llm_dim != 0 {
        return Err(bad("Text embedding size must be a multiple of llm_dim"));
    }

    let num_text_tokens = text_embeddings.len() / llm_dim;
    let clamped_pos = insert_position.min(num_text_tokens);

    let total_tokens = num_text_tokens + audio_tokens.len();
    let mut stitched = Vec::with_capacity(total_tokens * llm_dim);

    // Prefix text tokens
    let prefix_end = clamped_pos * llm_dim;
    stitched.extend_from_slice(&text_embeddings[..prefix_end]);

    // Audio tokens
    for token in audio_tokens {
        if token.len() != llm_dim {
            return Err(bad("Audio token dimension mismatch"));
        }
        stitched.extend_from_slice(token);
    }

    // Suffix text tokens
    stitched.extend_from_slice(&text_embeddings[prefix_end..]);

    Ok(stitched)
}
