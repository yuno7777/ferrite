//! Measure the matvec, which is where essentially all decode time goes.
//!
//! ```text
//! cargo run --release --example bench
//! ```
//!
//! Reports the fused kernel against the expand-then-dot reference at shapes
//! taken from real models, and reports effective bandwidth — that is the number
//! that matters, because single-token decode is memory bound, not compute
//! bound. Every weight is read once and used once.

use std::time::Instant;

use ferrite::gguf::{GgmlType, Gguf};
use ferrite::model::Weight;
use ferrite::synth::Builder;

/// Shapes lifted from Llama-3.2-1B and Llama-3.1-8B.
const SHAPES: [(&str, usize, usize); 4] = [
    ("1B  attn_q   2048x2048", 2048, 2048),
    ("1B  ffn_up   2048x8192", 2048, 8192),
    ("8B  attn_q   4096x4096", 4096, 4096),
    ("8B  ffn_down 14336x4096", 14336, 4096),
];

const TYPES: [GgmlType; 5] = [
    GgmlType::Q4_K,
    GgmlType::Q5_K,
    GgmlType::Q6_K,
    GgmlType::Q8_0,
    GgmlType::Q4_0,
];

fn bytes_for(ty: GgmlType, elements: usize) -> Vec<u8> {
    let (block, size) = ty.layout().expect("sized type");
    let len = elements as u64 / block * size;
    (0..len).map(|i| ((i * 37 + 11) % 251) as u8).collect()
}

/// Run `body` enough times to get past timer noise, return nanoseconds per call.
fn time(mut body: impl FnMut()) -> f64 {
    // Warm up: first touch of a fresh mapping is page faults, not arithmetic.
    for _ in 0..2 {
        body();
    }
    let mut runs = 0;
    let start = Instant::now();
    while start.elapsed().as_millis() < 250 {
        body();
        runs += 1;
    }
    start.elapsed().as_nanos() as f64 / runs as f64
}

fn main() -> std::io::Result<()> {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    println!("{threads} hardware threads\n");
    println!(
        "{:<26} {:>7} {:>11} {:>11} {:>8} {:>10}",
        "shape", "type", "reference", "fused", "speedup", "bandwidth"
    );

    for (label, cols, rows) in SHAPES {
        for ty in TYPES {
            let elements = cols * rows;
            let data = bytes_for(ty, elements);
            let stored = data.len();

            // Round-trip through a real GGUF so the benchmark measures the same
            // path inference takes, mmap included.
            let path = std::env::temp_dir().join("ferrite_bench.gguf");
            Builder::new()
                .tensor("w", &[cols as u64, rows as u64], ty, data)
                .write(&path)?;
            let gguf = Gguf::open(&path)?;
            let weight = Weight::bind(&gguf, "w")?;

            let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.37).sin()).collect();
            let mut out = vec![0.0; rows];
            let mut scratch = vec![0.0; cols];

            let reference = time(|| {
                weight
                    .matvec_reference(&x, &mut out, &mut scratch)
                    .expect("reference")
            });
            let fused = time(|| weight.matvec(&x, &mut out, &mut scratch).expect("fused"));

            // Bandwidth counts the stored weights, not the expanded f32: those
            // are the bytes that actually cross the memory bus.
            let seconds = fused / 1.0e9;
            let bandwidth = stored as f64 / seconds / (1024.0 * 1024.0 * 1024.0);

            println!(
                "{label:<26} {:>7} {:>9.2}ms {:>9.2}ms {:>7.2}x {:>7.1} GB/s",
                ty.name(),
                reference / 1.0e6,
                fused / 1.0e6,
                reference / fused,
                bandwidth
            );

            drop(gguf);
            std::fs::remove_file(&path).ok();
        }
        println!();
    }
    Ok(())
}
