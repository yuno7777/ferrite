use std::f32::consts::PI;

use ferrite::audio::{
    extract_log_mel_spectrogram, stitch_audio_and_text_embeddings, AudioPatchExtractor,
    AudioProjector, AudioWaveform, SpectrogramConfig,
};

fn create_wav_bytes(samples: &[i16], sample_rate: u32, num_channels: u16) -> Vec<u8> {
    let mut bytes = Vec::new();
    // RIFF header
    bytes.extend_from_slice(b"RIFF");
    let data_len = (samples.len() * 2) as u32;
    let file_size = 36 + data_len;
    bytes.extend_from_slice(&file_size.to_le_bytes());
    bytes.extend_from_slice(b"WAVE");

    // fmt subchunk
    bytes.extend_from_slice(b"fmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes()); // subchunk1 size (16 for PCM)
    bytes.extend_from_slice(&1u16.to_le_bytes()); // audio format 1 = PCM
    bytes.extend_from_slice(&num_channels.to_le_bytes());
    bytes.extend_from_slice(&sample_rate.to_le_bytes());
    let byte_rate = sample_rate * num_channels as u32 * 2;
    bytes.extend_from_slice(&byte_rate.to_le_bytes());
    let block_align = num_channels * 2;
    bytes.extend_from_slice(&block_align.to_le_bytes());
    bytes.extend_from_slice(&16u16.to_le_bytes()); // bits per sample

    // data subchunk
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data_len.to_le_bytes());
    for &sample in samples {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }

    bytes
}

#[test]
fn test_audio_pcm_and_wav_decoding() {
    let sample_rate = 16000;
    let num_samples = 1600; // 0.1 seconds

    // 440 Hz sine wave
    let mut raw_pcm = Vec::with_capacity(num_samples);
    for n in 0..num_samples {
        let t = n as f32 / sample_rate as f32;
        let val = (2.0 * PI * 440.0 * t).sin();
        let pcm = (val * 32000.0) as i16;
        raw_pcm.push(pcm);
    }

    let wav_bytes = create_wav_bytes(&raw_pcm, sample_rate as u32, 1);
    let waveform = AudioWaveform::from_wav_bytes(&wav_bytes).expect("parse wav");

    assert_eq!(waveform.sample_rate, sample_rate);
    assert_eq!(waveform.samples.len(), num_samples);
    assert!((waveform.duration_seconds() - 0.1).abs() < 1e-4);

    // Verify normalization
    for &s in &waveform.samples {
        assert!((-1.0..=1.0).contains(&s));
    }

    // Resample test (16000 Hz to 8000 Hz)
    let resampled = waveform.resample_linear(8000);
    assert_eq!(resampled.sample_rate, 8000);
    assert_eq!(resampled.samples.len(), num_samples / 2);
}

#[test]
fn test_log_mel_spectrogram_extraction() {
    let sample_rate = 16000;
    let num_samples = 3200; // 0.2 seconds

    let mut samples = Vec::with_capacity(num_samples);
    for n in 0..num_samples {
        let t = n as f32 / sample_rate as f32;
        let val = (2.0 * PI * 1000.0 * t).sin();
        samples.push(val);
    }

    let waveform = AudioWaveform::new(samples, sample_rate);
    let config = SpectrogramConfig {
        sample_rate: 16000,
        n_fft: 400,
        hop_length: 160,
        n_mels: 80,
        f_min: 0.0,
        f_max: 8000.0,
    };

    let spec = extract_log_mel_spectrogram(&waveform, &config);
    assert_eq!(spec.n_mels, 80);
    assert!(spec.n_frames > 0);
    assert_eq!(spec.data.len(), spec.n_frames * 80);

    // Check first frame values are valid log energies
    let frame0 = spec.frame(0);
    assert_eq!(frame0.len(), 80);
    for &energy in frame0 {
        assert!(!energy.is_nan());
        assert!(!energy.is_infinite());
    }
}

#[test]
fn test_audio_patch_extraction_and_linear_projection() {
    let n_frames = 10;
    let n_mels = 80;
    let dummy_data = vec![0.5f32; n_frames * n_mels];
    let spec = ferrite::audio::LogMelSpectrogram {
        n_frames,
        n_mels,
        data: dummy_data,
    };

    let extractor = AudioPatchExtractor::new(2, 2);
    let patches = extractor.extract_patches(&spec);
    assert_eq!(patches.len(), 5);
    assert_eq!(patches[0].len(), 2 * 80); // 160 features per patch

    // Projection into LLM dimension of 16
    let patch_dim = 160;
    let llm_dim = 16;
    let weight = vec![0.1f32; llm_dim * patch_dim];
    let bias = vec![0.05f32; llm_dim];

    let projector = AudioProjector::new(patch_dim, llm_dim, weight, Some(bias)).expect("projector");
    let tokens = projector.project(&patches).expect("project patches");

    assert_eq!(tokens.len(), 5);
    assert_eq!(tokens[0].len(), 16);

    // Expected value: 160 * (0.5 * 0.1) + 0.05 = 160 * 0.05 + 0.05 = 8.05
    let expected = 160.0 * (0.5 * 0.1) + 0.05;
    assert!((tokens[0][0] - expected).abs() < 1e-4);
}

#[test]
fn test_multimodal_audio_text_embedding_stitching() {
    let llm_dim = 4;
    // 3 text tokens: [1,1,1,1], [2,2,2,2], [3,3,3,3]
    let text = vec![
        1.0, 1.0, 1.0, 1.0, // Token 0
        2.0, 2.0, 2.0, 2.0, // Token 1
        3.0, 3.0, 3.0, 3.0, // Token 2
    ];

    // 2 audio tokens: [8,8,8,8], [9,9,9,9]
    let audio = vec![vec![8.0, 8.0, 8.0, 8.0], vec![9.0, 9.0, 9.0, 9.0]];

    // Insert audio at position 1 (between text Token 0 and Token 1)
    let stitched = stitch_audio_and_text_embeddings(&text, &audio, 1, llm_dim).expect("stitch");

    assert_eq!(stitched.len(), 5 * llm_dim);
    // Token 0: text[0]
    assert_eq!(&stitched[0..4], &[1.0, 1.0, 1.0, 1.0]);
    // Token 1: audio[0]
    assert_eq!(&stitched[4..8], &[8.0, 8.0, 8.0, 8.0]);
    // Token 2: audio[1]
    assert_eq!(&stitched[8..12], &[9.0, 9.0, 9.0, 9.0]);
    // Token 3: text[1]
    assert_eq!(&stitched[12..16], &[2.0, 2.0, 2.0, 2.0]);
    // Token 4: text[2]
    assert_eq!(&stitched[16..20], &[3.0, 3.0, 3.0, 3.0]);
}
