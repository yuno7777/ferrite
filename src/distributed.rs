//! Zero-dependency distributed pipeline-parallel model inference.
//!
//! Enables partitioning large model layers across multiple nodes or local processes
//! over TCP sockets with zero external dependencies (no MPI, no PyTorch).
//!
//! Wire protocol:
//! 1. 4-byte magic (`FERR`)
//! 2. 1-byte message type (1 = Activation, 2 = Response, 3 = Shutdown)
//! 3. 4-byte sequence position (u32 LE)
//! 4. 4-byte batch size (u32 LE)
//! 5. 4-byte hidden dimension (u32 LE)
//! 6. 4-byte payload byte length (u32 LE = batch * dim * 4)
//! 7. Raw f32 activation payload

use std::io::{Error, ErrorKind, Read, Result, Write};
use std::net::{TcpListener, TcpStream};
use std::ops::Range;

use crate::model::{Layer, Model, State};
use crate::ops;

pub const DIST_MAGIC: [u8; 4] = *b"FERR";
pub const MSG_ACTIVATION: u8 = 1;
pub const MSG_RESPONSE: u8 = 2;
pub const MSG_SHUTDOWN: u8 = 3;
const MAX_PACKET_BYTES: usize = 256 * 1024 * 1024;

/// Packet carrying intermediate hidden state activations between distributed nodes.
#[derive(Clone, Debug, PartialEq)]
pub struct ActivationPacket {
    pub msg_type: u8,
    pub seq_pos: usize,
    pub batch_size: usize,
    pub dim: usize,
    pub data: Vec<f32>,
}

impl ActivationPacket {
    pub fn new(
        msg_type: u8,
        seq_pos: usize,
        batch_size: usize,
        dim: usize,
        data: Vec<f32>,
    ) -> Self {
        Self {
            msg_type,
            seq_pos,
            batch_size,
            dim,
            data,
        }
    }

    /// Serialize activation packet to writer.
    pub fn write_to<W: Write>(&self, w: &mut W) -> Result<()> {
        if !matches!(self.msg_type, MSG_ACTIVATION | MSG_RESPONSE | MSG_SHUTDOWN) {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "invalid distributed message type",
            ));
        }
        let expected = self
            .batch_size
            .checked_mul(self.dim)
            .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "activation dimensions overflow"))?;
        if self.msg_type != MSG_SHUTDOWN && self.data.len() != expected {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "activation payload length does not match dimensions",
            ));
        }
        if self.seq_pos > u32::MAX as usize
            || self.batch_size > u32::MAX as usize
            || self.dim > u32::MAX as usize
        {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "distributed packet field exceeds wire limits",
            ));
        }
        let byte_len = self
            .data
            .len()
            .checked_mul(4)
            .filter(|&n| n <= MAX_PACKET_BYTES && n <= u32::MAX as usize)
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::InvalidInput,
                    "distributed payload exceeds wire limit",
                )
            })?;
        w.write_all(&DIST_MAGIC)?;
        w.write_all(&[self.msg_type])?;
        w.write_all(&(self.seq_pos as u32).to_le_bytes())?;
        w.write_all(&(self.batch_size as u32).to_le_bytes())?;
        w.write_all(&(self.dim as u32).to_le_bytes())?;

        w.write_all(&(byte_len as u32).to_le_bytes())?;
        for value in &self.data {
            w.write_all(&value.to_le_bytes())?;
        }
        w.flush()?;
        Ok(())
    }

    /// Read activation packet from reader.
    pub fn read_from<R: Read>(r: &mut R) -> Result<Self> {
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if magic != DIST_MAGIC {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "invalid distributed magic",
            ));
        }

        let mut msg_type_buf = [0u8; 1];
        r.read_exact(&mut msg_type_buf)?;
        let msg_type = msg_type_buf[0];

        let mut u32_buf = [0u8; 4];

        r.read_exact(&mut u32_buf)?;
        let seq_pos = u32::from_le_bytes(u32_buf) as usize;

        r.read_exact(&mut u32_buf)?;
        let batch_size = u32::from_le_bytes(u32_buf) as usize;

        r.read_exact(&mut u32_buf)?;
        let dim = u32::from_le_bytes(u32_buf) as usize;

        r.read_exact(&mut u32_buf)?;
        let byte_len = u32::from_le_bytes(u32_buf) as usize;

        if byte_len % 4 != 0 {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "corrupt float payload length",
            ));
        }
        if byte_len > MAX_PACKET_BYTES {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "distributed payload exceeds wire limit",
            ));
        }
        if !matches!(msg_type, MSG_ACTIVATION | MSG_RESPONSE | MSG_SHUTDOWN) {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "invalid distributed message type",
            ));
        }
        let float_count = byte_len / 4;
        if msg_type == MSG_SHUTDOWN && (byte_len != 0 || batch_size != 0 || dim != 0) {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "invalid shutdown packet",
            ));
        }
        if msg_type != MSG_SHUTDOWN && batch_size.checked_mul(dim) != Some(float_count) {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "activation payload length does not match dimensions",
            ));
        }
        let mut data = vec![0.0f32; float_count];
        for value in &mut data {
            r.read_exact(&mut u32_buf)?;
            *value = f32::from_bits(u32::from_le_bytes(u32_buf));
        }

        Ok(Self {
            msg_type,
            seq_pos,
            batch_size,
            dim,
            data,
        })
    }
}

/// Pipeline partition configuration assigning layer ranges to nodes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PartitionConfig {
    pub node_rank: usize,
    pub total_nodes: usize,
    pub layer_range: Range<usize>,
}

/// Evaluates a partition of model layers over a single token or batch.
pub fn forward_partition(
    layers: &[Layer],
    config: &crate::model::Config,
    state: &mut State,
    layer_range: Range<usize>,
    x: &mut [f32],
    pos: usize,
) -> Result<()> {
    if layer_range.end > config.layers
        || layer_range.start > layer_range.end
        || layers.len() != layer_range.end - layer_range.start
        || x.len() != config.embedding
    {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "partition dimensions do not match model configuration",
        ));
    }
    let (head_dim, kv_dim, group) = (config.head_dim, config.kv_dim(), config.group());
    let att_dim = config.heads * head_dim;

    let is_sliding_layer = |idx: usize| (idx % 2 == 0) && config.sliding_window.is_some();
    let window = config.sliding_window.unwrap_or(usize::MAX);

    for (layer_idx, layer) in layers.iter().enumerate() {
        let global_idx = layer_range.start + layer_idx;
        ops::rms_norm(
            x,
            &layer.attn_norm,
            config.eps,
            &mut state.xb[..config.embedding],
        );

        layer.wq.matvec_threaded(
            &state.xb[..config.embedding],
            &mut state.q,
            state.threads,
            &mut state.workspace,
        )?;
        if let Some(bq) = &layer.bq {
            for (slot, b) in state.q.iter_mut().zip(bq) {
                *slot += b;
            }
        }
        if let Some(l) = &layer.lora_wq {
            l.apply(&state.xb[..config.embedding], &mut state.q)?;
        }

        layer.wk.matvec_threaded(
            &state.xb[..config.embedding],
            &mut state.k,
            state.threads,
            &mut state.workspace,
        )?;
        if let Some(bk) = &layer.bk {
            for (slot, b) in state.k.iter_mut().zip(bk) {
                *slot += b;
            }
        }
        if let Some(l) = &layer.lora_wk {
            l.apply(&state.xb[..config.embedding], &mut state.k)?;
        }

        layer.wv.matvec_threaded(
            &state.xb[..config.embedding],
            &mut state.v,
            state.threads,
            &mut state.workspace,
        )?;
        if let Some(bv) = &layer.bv {
            for (slot, b) in state.v.iter_mut().zip(bv) {
                *slot += b;
            }
        }
        if let Some(l) = &layer.lora_wv {
            l.apply(&state.xb[..config.embedding], &mut state.v)?;
        }

        for head in 0..config.heads {
            ops::rope_scaled(
                &mut state.q[head * head_dim..(head + 1) * head_dim],
                pos,
                config.rope_base,
                config.rope_dim,
                config.rope_scaling.as_ref(),
            );
        }
        for head in 0..config.kv_heads {
            ops::rope_scaled(
                &mut state.k[head * head_dim..(head + 1) * head_dim],
                pos,
                config.rope_base,
                config.rope_dim,
                config.rope_scaling.as_ref(),
            );
        }

        state.kv_cache.store(global_idx, pos, &state.k, &state.v);

        let scale = 1.0 / (head_dim as f32).sqrt();
        let is_sliding = is_sliding_layer(global_idx);
        for head in 0..config.heads {
            let kv_head = head / group;
            let query = &state.q[head * head_dim..(head + 1) * head_dim];
            let out = &mut state.xb[head * head_dim..(head + 1) * head_dim];
            state.kv_cache.attention(
                global_idx,
                query,
                out,
                ops::AttentionParams {
                    row_width: kv_dim,
                    offset: kv_head * head_dim,
                    len: pos + 1,
                    scale,
                    logit_cap: config.attn_logit_softcapping,
                    window: is_sliding.then_some(window),
                    position: pos,
                    tile_size: 64,
                    v_row_width: None,
                    v_offset: None,
                },
            );
        }

        layer.wo.matvec_threaded(
            &state.xb[..att_dim],
            &mut state.xb2[..config.embedding],
            state.threads,
            &mut state.workspace,
        )?;
        if let Some(l) = &layer.lora_wo {
            l.apply(&state.xb[..att_dim], &mut state.xb2[..config.embedding])?;
        }

        if let Some(post_attn_norm) = &layer.post_attn_norm {
            ops::rms_norm(
                &state.xb2[..config.embedding],
                post_attn_norm,
                config.eps,
                &mut state.xb[..config.embedding],
            );
            for (slot, delta) in x.iter_mut().zip(&state.xb[..config.embedding]) {
                *slot += delta;
            }
        } else {
            for (slot, delta) in x.iter_mut().zip(&state.xb2[..config.embedding]) {
                *slot += delta;
            }
        }

        ops::rms_norm(
            x,
            &layer.ffn_norm,
            config.eps,
            &mut state.xb[..config.embedding],
        );
        layer.eval_ffn(
            config,
            &state.xb[..config.embedding],
            1,
            &mut state.xb2[..config.embedding],
            &mut state.hb,
            &mut state.hb2,
            state.threads,
            &mut state.workspace,
        )?;

        if let Some(post_ffn_norm) = &layer.post_ffn_norm {
            ops::rms_norm(
                &state.xb2[..config.embedding],
                post_ffn_norm,
                config.eps,
                &mut state.xb[..config.embedding],
            );
            for (slot, delta) in x.iter_mut().zip(&state.xb[..config.embedding]) {
                *slot += delta;
            }
        } else {
            for (slot, delta) in x.iter_mut().zip(&state.xb2[..config.embedding]) {
                *slot += delta;
            }
        }
    }

    Ok(())
}

/// Run a distributed worker loop on a partitioned slice of model layers.
pub fn run_worker(model: &Model, layer_range: Range<usize>, listener: TcpListener) -> Result<()> {
    if layer_range.start >= layer_range.end || layer_range.end > model.layers.len() {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "worker layer range is outside the model",
        ));
    }
    let mut state = State::new(&model.config);
    let partitioned_layers = &model.layers[layer_range.clone()];

    for stream in listener.incoming() {
        let mut stream = stream?;
        loop {
            let packet = match ActivationPacket::read_from(&mut stream) {
                Ok(p) => p,
                Err(e) if e.kind() == ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e),
            };

            if packet.msg_type == MSG_SHUTDOWN {
                return Ok(());
            }
            if packet.msg_type != MSG_ACTIVATION
                || packet.batch_size != 1
                || packet.dim != model.config.embedding
                || packet.seq_pos >= state.context()
            {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    "activation packet does not match worker model",
                ));
            }

            let mut activations = packet.data;
            forward_partition(
                partitioned_layers,
                &model.config,
                &mut state,
                layer_range.clone(),
                &mut activations,
                packet.seq_pos,
            )?;

            let response = ActivationPacket::new(
                MSG_RESPONSE,
                packet.seq_pos,
                packet.batch_size,
                packet.dim,
                activations,
            );
            response.write_to(&mut stream)?;
        }
    }
    Ok(())
}

/// Client driver executing partitioned forward pass over remote worker.
pub struct DistributedDriver<'a> {
    pub model: &'a Model<'a>,
    pub local_layers: Range<usize>,
    pub remote_layers: Range<usize>,
    pub stream: TcpStream,
}

impl<'a> DistributedDriver<'a> {
    pub fn connect(
        model: &'a Model<'a>,
        local_layers: Range<usize>,
        remote_layers: Range<usize>,
        remote_addr: &str,
    ) -> Result<Self> {
        let stream = TcpStream::connect(remote_addr)?;
        Ok(Self {
            model,
            local_layers,
            remote_layers,
            stream,
        })
    }

    /// Perform a distributed single-token forward pass:
    /// 1. Embeds token and evaluates `local_layers` locally.
    /// 2. Sends intermediate activations to remote worker via TCP.
    /// 3. Receives remote activations.
    /// 4. Projects to vocabulary logits locally.
    pub fn forward(&mut self, state: &mut State, token: u32, position: usize) -> Result<()> {
        let c = &self.model.config;
        if position >= state.context() {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "position exceeds context",
            ));
        }

        // 1. Local token embedding
        self.model.token_embd.row(token as usize, &mut state.x)?;
        if let Some(scale) = c.embedding_scale {
            for v in state.x.iter_mut() {
                *v *= scale;
            }
        }

        // 2. Local partition
        if self.local_layers.end != self.remote_layers.start
            || self.local_layers.start != 0
            || self.remote_layers.end != self.model.layers.len()
            || self.local_layers.end > self.model.layers.len()
            || self.remote_layers.start >= self.remote_layers.end
        {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "distributed partitions must be contiguous and cover the model",
            ));
        }
        let local_slice = &self.model.layers[self.local_layers.clone()];
        let mut activations = std::mem::take(&mut state.x);
        forward_partition(
            local_slice,
            c,
            state,
            self.local_layers.clone(),
            &mut activations,
            position,
        )?;
        state.x = activations;

        // 3. Send activation to remote worker
        let packet =
            ActivationPacket::new(MSG_ACTIVATION, position, 1, c.embedding, state.x.clone());
        packet.write_to(&mut self.stream)?;

        // 4. Receive updated activation from remote worker
        let response = ActivationPacket::read_from(&mut self.stream)?;
        if response.msg_type != MSG_RESPONSE
            || response.seq_pos != position
            || response.batch_size != 1
            || response.dim != c.embedding
            || response.data.len() != c.embedding
        {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "invalid distributed worker response",
            ));
        }
        state.x.copy_from_slice(&response.data);

        // 5. Output norm and projection
        ops::rms_norm(
            &state.x,
            &self.model.output_norm,
            c.eps,
            &mut state.xb[..c.embedding],
        );
        self.model.output.matvec_threaded(
            &state.xb[..c.embedding],
            &mut state.logits,
            state.threads,
            &mut state.workspace,
        )?;
        if let Some(l) = &self.model.lora_output {
            l.apply(&state.xb[..c.embedding], &mut state.logits)?;
        }
        if let Some(cap) = c.final_logit_softcapping {
            for logit in state.logits.iter_mut() {
                *logit = cap * (*logit / cap).tanh();
            }
        }

        Ok(())
    }

    /// Gracefully shutdown remote worker.
    pub fn shutdown(&mut self) -> Result<()> {
        let shutdown_packet = ActivationPacket::new(MSG_SHUTDOWN, 0, 0, 0, Vec::new());
        shutdown_packet.write_to(&mut self.stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn activation_packet_round_trips_and_rejects_bad_shape() {
        let packet = ActivationPacket::new(MSG_ACTIVATION, 7, 1, 2, vec![1.25, -3.5]);
        let mut wire = Vec::new();
        packet.write_to(&mut wire).unwrap();
        assert_eq!(
            ActivationPacket::read_from(&mut wire.as_slice()).unwrap(),
            packet
        );

        let malformed = ActivationPacket::new(MSG_ACTIVATION, 0, 2, 2, vec![1.0]);
        assert_eq!(
            malformed.write_to(&mut Vec::new()).unwrap_err().kind(),
            ErrorKind::InvalidInput
        );
    }
}
