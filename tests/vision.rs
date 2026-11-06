use ferrite::gguf::{GgmlType, Gguf, Value};
use ferrite::model::{Model, State};
use ferrite::synth::Builder;
use ferrite::vision::{Image, VisionEncoder};

#[test]
fn test_ppm_p6_and_p3_round_trip_and_parsing() {
    // 1. Create a 2x2 RGB test image
    let mut img = Image::new(2, 2);
    // Pixel (0, 0): Red (1.0, 0.0, 0.0)
    // Pixel (1, 0): Green (0.0, 1.0, 0.0)
    // Pixel (0, 1): Blue (0.0, 0.0, 1.0)
    // Pixel (1, 1): White (1.0, 1.0, 1.0)
    let plane = 4;
    img.data[0] = 1.0;
    img.data[plane + 1] = 1.0;
    img.data[2 * plane + 2] = 1.0;
    img.data[3] = 1.0;
    img.data[plane + 3] = 1.0;
    img.data[2 * plane + 3] = 1.0;

    // Serialize to PPM P6
    let p6_bytes = img.to_ppm_p6();
    assert!(p6_bytes.starts_with(b"P6\n2 2\n255\n"));

    // Parse back
    let parsed_p6 = Image::from_ppm(&p6_bytes).expect("Valid P6 PPM parsing");
    assert_eq!(parsed_p6.width, 2);
    assert_eq!(parsed_p6.height, 2);
    assert_eq!(parsed_p6.channels, 3);
    for (orig, parsed) in img.data.iter().zip(&parsed_p6.data) {
        assert!((orig - parsed).abs() < 1e-2);
    }

    // 2. Test P3 ASCII PPM
    let p3_ascii = b"P3\n# A comment\n2 2\n255\n255 0 0  0 255 0\n0 0 255  255 255 255\n";
    let parsed_p3 = Image::from_ppm(p3_ascii).expect("Valid P3 PPM parsing");
    assert_eq!(parsed_p3.width, 2);
    assert_eq!(parsed_p3.height, 2);
    for (p6_val, p3_val) in parsed_p6.data.iter().zip(&parsed_p3.data) {
        assert!((p6_val - p3_val).abs() < 1e-4);
    }
}

#[test]
fn test_bmp_24bit_parsing() {
    // Construct uncompressed 24-bit Windows BMP: 2x2 image
    // Row width = 2 * 3 = 6 bytes -> padded to 8 bytes (4-byte aligned)
    let file_header_size = 14;
    let dib_header_size = 40;
    let pixel_offset = file_header_size + dib_header_size;
    let row_stride = 8;
    let image_data_size = row_stride * 2;
    let file_size = pixel_offset + image_data_size;

    let mut bmp = vec![0u8; file_size];
    // BM signature
    bmp[0] = b'B';
    bmp[1] = b'M';
    bmp[2..6].copy_from_slice(&(file_size as u32).to_le_bytes());
    bmp[10..14].copy_from_slice(&(pixel_offset as u32).to_le_bytes());

    // DIB header
    bmp[14..18].copy_from_slice(&(dib_header_size as u32).to_le_bytes());
    bmp[18..22].copy_from_slice(&2i32.to_le_bytes()); // width = 2
    bmp[22..26].copy_from_slice(&2i32.to_le_bytes()); // height = 2 (bottom-up)
    bmp[26..28].copy_from_slice(&1u16.to_le_bytes()); // planes = 1
    bmp[28..30].copy_from_slice(&24u16.to_le_bytes()); // 24 bpp
    bmp[30..34].copy_from_slice(&0u32.to_le_bytes()); // BI_RGB (uncompressed)

    // Row 0 (bottom row): (0, 1) Blue, (1, 1) White
    // Pixel 0: BGR = (255, 0, 0)
    // Pixel 1: BGR = (255, 255, 255)
    let r0 = pixel_offset;
    bmp[r0] = 255;
    bmp[r0 + 1] = 0;
    bmp[r0 + 2] = 0;
    bmp[r0 + 3] = 255;
    bmp[r0 + 4] = 255;
    bmp[r0 + 5] = 255;

    // Row 1 (top row): (0, 0) Red, (1, 0) Green
    // Pixel 0: BGR = (0, 0, 255)
    // Pixel 1: BGR = (0, 255, 0)
    let r1 = pixel_offset + row_stride;
    bmp[r1] = 0;
    bmp[r1 + 1] = 0;
    bmp[r1 + 2] = 255;
    bmp[r1 + 3] = 0;
    bmp[r1 + 4] = 255;
    bmp[r1 + 5] = 0;

    let img = Image::from_bmp(&bmp).expect("Valid 24-bit BMP");
    assert_eq!(img.width, 2);
    assert_eq!(img.height, 2);

    let plane = 4;
    // Check (0, 0) is Red
    assert!((img.data[0] - 1.0).abs() < 1e-2);
    assert!(img.data[plane] < 1e-2);
    assert!(img.data[2 * plane] < 1e-2);

    // Check (1, 0) is Green
    assert!(img.data[1] < 1e-2);
    assert!((img.data[plane + 1] - 1.0).abs() < 1e-2);
    assert!(img.data[2 * plane + 1] < 1e-2);

    // Check (0, 1) is Blue
    assert!(img.data[2] < 1e-2);
    assert!(img.data[plane + 2] < 1e-2);
    assert!((img.data[2 * plane + 2] - 1.0).abs() < 1e-2);

    // Check (1, 1) is White
    assert!((img.data[3] - 1.0).abs() < 1e-2);
    assert!((img.data[plane + 3] - 1.0).abs() < 1e-2);
    assert!((img.data[2 * plane + 3] - 1.0).abs() < 1e-2);
}

#[test]
fn test_bilinear_resampling_and_normalization() {
    let mut img = Image::new(2, 2);
    img.data = vec![
        0.0, 1.0, 0.0, 1.0, // R
        0.5, 0.5, 0.5, 0.5, // G
        1.0, 0.0, 1.0, 0.0, // B
    ];

    let resized = img.resize_bilinear(4, 4);
    assert_eq!(resized.width, 4);
    assert_eq!(resized.height, 4);
    assert_eq!(resized.data.len(), 3 * 16);

    // Corner values should be preserved
    let plane_4x4 = 16;
    assert!((resized.data[0] - 0.0).abs() < 1e-4);
    assert!((resized.data[3] - 1.0).abs() < 1e-4);
    assert!((resized.data[plane_4x4] - 0.5).abs() < 1e-4);

    // Center values should be linearly interpolated
    assert!((resized.data[1] - 0.3333).abs() < 0.05);

    // Normalization test
    let mut norm_img = resized.clone();
    let mean = [0.5, 0.5, 0.5];
    let std = [0.5, 0.5, 0.5];
    norm_img.normalize(mean, std);
    // (0.0 - 0.5) / 0.5 = -1.0
    assert!((norm_img.data[0] - (-1.0)).abs() < 1e-4);
    // (1.0 - 0.5) / 0.5 = 1.0
    assert!((norm_img.data[3] - 1.0).abs() < 1e-4);
}

#[test]
fn test_vision_encoder_from_gguf_fixture() {
    let dir = tempfile_dir("vision_test");
    std::fs::create_dir_all(&dir).ok();
    let file = dir.join("mmproj.gguf");

    let image_size = 28usize;
    let patch_size = 14usize;
    let hidden_size = 16usize;
    let intermediate_size = 32usize;
    let num_hidden_layers = 1usize;
    let num_attention_heads = 2usize;
    let projection_dim = 32usize;

    let patch_pixels = 3 * patch_size * patch_size; // 3 * 196 = 588
    let num_patches = (image_size / patch_size) * (image_size / patch_size); // 4
    let total_tokens = num_patches + 1; // 5

    let f32_to_bytes = |vals: &[f32]| -> Vec<u8> {
        let mut b = Vec::with_capacity(vals.len() * 4);
        for &v in vals {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b
    };

    let ones = |len: usize| vec![0.01f32; len];

    let mut b = Builder::new()
        .meta("general.architecture", Value::String("clip".into()))
        .meta("clip.vision.image_size", Value::U32(image_size as u32))
        .meta("clip.vision.patch_size", Value::U32(patch_size as u32))
        .meta(
            "clip.vision.embedding_length",
            Value::U32(hidden_size as u32),
        )
        .meta(
            "clip.vision.feed_forward_length",
            Value::U32(intermediate_size as u32),
        )
        .meta(
            "clip.vision.block_count",
            Value::U32(num_hidden_layers as u32),
        )
        .meta(
            "clip.vision.attention.head_count",
            Value::U32(num_attention_heads as u32),
        )
        .meta(
            "clip.vision.projection_dim",
            Value::U32(projection_dim as u32),
        );

    // Add tensors
    // Patch embedding
    b = b.tensor(
        "v.patch_embd.weight",
        &[patch_pixels as u64, hidden_size as u64],
        GgmlType::F32,
        f32_to_bytes(&ones(hidden_size * patch_pixels)),
    );
    b = b.tensor(
        "v.patch_embd.bias",
        &[hidden_size as u64],
        GgmlType::F32,
        f32_to_bytes(&ones(hidden_size)),
    );

    // Positional embedding and class token
    b = b.tensor(
        "v.position_embd.weight",
        &[hidden_size as u64, total_tokens as u64],
        GgmlType::F32,
        f32_to_bytes(&ones(total_tokens * hidden_size)),
    );
    b = b.tensor(
        "v.class_embd",
        &[hidden_size as u64],
        GgmlType::F32,
        f32_to_bytes(&ones(hidden_size)),
    );

    // Layer 0
    b = b.tensor(
        "v.blk.0.ln1.weight",
        &[hidden_size as u64],
        GgmlType::F32,
        f32_to_bytes(&vec![1.0f32; hidden_size]),
    );
    b = b.tensor(
        "v.blk.0.ln1.bias",
        &[hidden_size as u64],
        GgmlType::F32,
        f32_to_bytes(&vec![0.0f32; hidden_size]),
    );
    b = b.tensor(
        "v.blk.0.ln2.weight",
        &[hidden_size as u64],
        GgmlType::F32,
        f32_to_bytes(&vec![1.0f32; hidden_size]),
    );
    b = b.tensor(
        "v.blk.0.ln2.bias",
        &[hidden_size as u64],
        GgmlType::F32,
        f32_to_bytes(&vec![0.0f32; hidden_size]),
    );

    b = b.tensor(
        "v.blk.0.attn_q.weight",
        &[hidden_size as u64, hidden_size as u64],
        GgmlType::F32,
        f32_to_bytes(&ones(hidden_size * hidden_size)),
    );
    b = b.tensor(
        "v.blk.0.attn_k.weight",
        &[hidden_size as u64, hidden_size as u64],
        GgmlType::F32,
        f32_to_bytes(&ones(hidden_size * hidden_size)),
    );
    b = b.tensor(
        "v.blk.0.attn_v.weight",
        &[hidden_size as u64, hidden_size as u64],
        GgmlType::F32,
        f32_to_bytes(&ones(hidden_size * hidden_size)),
    );
    b = b.tensor(
        "v.blk.0.attn_out.weight",
        &[hidden_size as u64, hidden_size as u64],
        GgmlType::F32,
        f32_to_bytes(&ones(hidden_size * hidden_size)),
    );

    b = b.tensor(
        "v.blk.0.ffn_up.weight",
        &[hidden_size as u64, intermediate_size as u64],
        GgmlType::F32,
        f32_to_bytes(&ones(hidden_size * intermediate_size)),
    );
    b = b.tensor(
        "v.blk.0.ffn_down.weight",
        &[intermediate_size as u64, hidden_size as u64],
        GgmlType::F32,
        f32_to_bytes(&ones(intermediate_size * hidden_size)),
    );

    b = b.tensor(
        "v.post_ln.weight",
        &[hidden_size as u64],
        GgmlType::F32,
        f32_to_bytes(&vec![1.0f32; hidden_size]),
    );
    b = b.tensor(
        "v.post_ln.bias",
        &[hidden_size as u64],
        GgmlType::F32,
        f32_to_bytes(&vec![0.0f32; hidden_size]),
    );

    // Multi-modal projector: [projection_dim, hidden_size]
    b = b.tensor(
        "mm.weight",
        &[hidden_size as u64, projection_dim as u64],
        GgmlType::F32,
        f32_to_bytes(&ones(projection_dim * hidden_size)),
    );

    b.write(&file).expect("Write synthetic vision GGUF");

    let gguf = Gguf::open(&file).expect("Open synthetic vision GGUF");
    let encoder = VisionEncoder::from_gguf(&gguf).expect("Load VisionEncoder");

    assert_eq!(encoder.config.image_size, 28);
    assert_eq!(encoder.config.patch_size, 14);
    assert_eq!(encoder.config.projection_dim, 32);

    let test_img = Image::new(28, 28);
    let embeddings = encoder.encode_image(&test_img).expect("Encode image");

    // num_patches = 4, projection_dim = 32 -> 128 elements
    assert_eq!(embeddings.len(), 4 * 32);
    for val in embeddings {
        assert!(val.is_finite());
    }

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_multimodal_prefill_matches_sequential_forward() {
    let dir = tempfile_dir("mm_prefill");
    std::fs::create_dir_all(&dir).ok();
    let model_file = dir.join("model.gguf");

    let embedding = 16usize;
    let heads = 2usize;
    let kv_heads = 2usize;
    let layers = 1usize;
    let ffn = 32usize;
    let vocab = 32usize;

    let f32_to_bytes = |vals: &[f32]| -> Vec<u8> {
        let mut b = Vec::with_capacity(vals.len() * 4);
        for &v in vals {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b
    };

    let weights = |len: usize| {
        (0..len)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.05)
            .collect::<Vec<_>>()
    };

    let mut b = Builder::new()
        .meta("general.architecture", Value::String("llama".into()))
        .meta("llama.block_count", Value::U32(layers as u32))
        .meta("llama.embedding_length", Value::U32(embedding as u32))
        .meta("llama.attention.head_count", Value::U32(heads as u32))
        .meta("llama.attention.head_count_kv", Value::U32(kv_heads as u32))
        .meta("llama.feed_forward_length", Value::U32(ffn as u32))
        .meta("llama.context_length", Value::U32(128))
        .meta(
            "llama.rope.dimension_count",
            Value::U32((embedding / heads) as u32),
        )
        .meta("llama.rope.freq_base", Value::F32(10000.0));

    b = b.tensor(
        "token_embd.weight",
        &[embedding as u64, vocab as u64],
        GgmlType::F32,
        f32_to_bytes(&weights(vocab * embedding)),
    );
    b = b.tensor(
        "output_norm.weight",
        &[embedding as u64],
        GgmlType::F32,
        f32_to_bytes(&vec![1.0f32; embedding]),
    );
    b = b.tensor(
        "output.weight",
        &[embedding as u64, vocab as u64],
        GgmlType::F32,
        f32_to_bytes(&weights(vocab * embedding)),
    );

    b = b.tensor(
        "blk.0.attn_norm.weight",
        &[embedding as u64],
        GgmlType::F32,
        f32_to_bytes(&vec![1.0f32; embedding]),
    );
    b = b.tensor(
        "blk.0.attn_q.weight",
        &[embedding as u64, embedding as u64],
        GgmlType::F32,
        f32_to_bytes(&weights(embedding * embedding)),
    );
    b = b.tensor(
        "blk.0.attn_k.weight",
        &[embedding as u64, embedding as u64],
        GgmlType::F32,
        f32_to_bytes(&weights(embedding * embedding)),
    );
    b = b.tensor(
        "blk.0.attn_v.weight",
        &[embedding as u64, embedding as u64],
        GgmlType::F32,
        f32_to_bytes(&weights(embedding * embedding)),
    );
    b = b.tensor(
        "blk.0.attn_output.weight",
        &[embedding as u64, embedding as u64],
        GgmlType::F32,
        f32_to_bytes(&weights(embedding * embedding)),
    );

    b = b.tensor(
        "blk.0.ffn_norm.weight",
        &[embedding as u64],
        GgmlType::F32,
        f32_to_bytes(&vec![1.0f32; embedding]),
    );
    b = b.tensor(
        "blk.0.ffn_gate.weight",
        &[embedding as u64, ffn as u64],
        GgmlType::F32,
        f32_to_bytes(&weights(ffn * embedding)),
    );
    b = b.tensor(
        "blk.0.ffn_up.weight",
        &[embedding as u64, ffn as u64],
        GgmlType::F32,
        f32_to_bytes(&weights(ffn * embedding)),
    );
    b = b.tensor(
        "blk.0.ffn_down.weight",
        &[ffn as u64, embedding as u64],
        GgmlType::F32,
        f32_to_bytes(&weights(embedding * ffn)),
    );

    b.write(&model_file).expect("Write model file");

    let gguf = Gguf::open(&model_file).expect("Open model GGUF");
    let model = Model::load(&gguf).expect("Load model");

    let prefix = vec![1u32, 2u32];
    let suffix = vec![5u32];
    let num_img_tokens = 3usize;
    let image_tokens = vec![0.123f32; num_img_tokens * embedding];

    let mut state = State::with_context(&model.config, 64);

    // Multi-modal unified prefill
    model
        .prefill_multimodal(&mut state, &prefix, &image_tokens, &suffix, 0)
        .expect("Prefill multimodal");

    let total_tokens = prefix.len() + num_img_tokens + suffix.len(); // 2 + 3 + 1 = 6
    assert_eq!(total_tokens, 6);

    // Verify all logits are finite numbers
    assert_eq!(state.logits.len(), vocab);
    for &l in &state.logits {
        assert!(l.is_finite(), "Logit must be finite");
    }

    std::fs::remove_dir_all(&dir).ok();
}

fn tempfile_dir(sub: &str) -> std::path::PathBuf {
    let rand = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("ferrite_{sub}_{rand}"))
}
