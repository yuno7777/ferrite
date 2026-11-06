//! DeepSeek-V3 DualPipe pipeline parallel execution simulation and inter-layer partitioning.
//!
//! Models bidirectional overlapping pipeline parallelism where forward attention and MLP computations
//! across two symmetric micro-batch directions (Direction A and Direction B) are scheduled across
//! partitioned pipeline stages to minimize pipeline bubbles.

use std::io::{Error, ErrorKind, Result};
use std::ops::Range;

use crate::distributed::forward_partition;
use crate::model::{Model, State};
use crate::ops;

/// Inter-layer pipeline stage partition defining layer ranges assigned to each stage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerPartition {
    pub stage_id: usize,
    pub total_stages: usize,
    pub layer_range: Range<usize>,
}

impl LayerPartition {
    /// Divides `total_layers` into `num_stages` balanced contiguous partitions.
    pub fn even_split(total_layers: usize, num_stages: usize) -> Vec<Self> {
        assert!(num_stages > 0, "num_stages must be positive");
        assert!(
            total_layers >= num_stages,
            "total_layers must be >= num_stages"
        );

        let mut partitions = Vec::with_capacity(num_stages);
        let base_layers = total_layers / num_stages;
        let remainder = total_layers % num_stages;

        let mut start = 0;
        for stage in 0..num_stages {
            let extra = if stage < remainder { 1 } else { 0 };
            let count = base_layers + extra;
            let end = start + count;
            partitions.push(Self {
                stage_id: stage,
                total_stages: num_stages,
                layer_range: start..end,
            });
            start = end;
        }

        partitions
    }

    /// Creates partitions from explicit partition boundary split points.
    pub fn custom_split(total_layers: usize, boundaries: &[usize]) -> Result<Vec<Self>> {
        if boundaries.is_empty() {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "boundaries must not be empty",
            ));
        }

        let num_stages = boundaries.len();
        let mut partitions = Vec::with_capacity(num_stages);
        let mut start = 0;

        for (stage, &end) in boundaries.iter().enumerate() {
            if end <= start || end > total_layers {
                return Err(Error::new(
                    ErrorKind::InvalidInput,
                    "invalid partition boundaries",
                ));
            }
            partitions.push(Self {
                stage_id: stage,
                total_stages: num_stages,
                layer_range: start..end,
            });
            start = end;
        }

        if start != total_layers {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "boundaries do not cover all layers",
            ));
        }

        Ok(partitions)
    }

    /// Number of transformer layers contained in this partition.
    pub fn layer_count(&self) -> usize {
        self.layer_range.end.saturating_sub(self.layer_range.start)
    }
}

/// Pipeline chunk direction for DualPipe bidirectional scheduling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DualPipeDirection {
    ForwardA,
    ForwardB,
}

/// DualPipe atomic schedule operation for a stage in a micro-step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DualPipeOp {
    /// Compute attention and forward projection on a partition.
    ForwardAttention {
        chunk_id: usize,
        direction: DualPipeDirection,
        stage_id: usize,
        layer_range: Range<usize>,
    },
    /// Compute MLP / MoE feed-forward network on a partition.
    ForwardMlp {
        chunk_id: usize,
        direction: DualPipeDirection,
        stage_id: usize,
        layer_range: Range<usize>,
    },
    /// Cross-stage activation transfer.
    Communicate {
        chunk_id: usize,
        from_stage: usize,
        to_stage: usize,
    },
    /// Pipeline bubble / idle stage.
    Idle,
}

/// Discrete micro-step in the DualPipe execution timeline across all stages.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DualPipeStep {
    pub step_index: usize,
    pub stage_ops: Vec<DualPipeOp>,
}

/// Discrete schedule representing overlapping execution across stages.
#[derive(Clone, Debug)]
pub struct DualPipeSchedule {
    pub num_stages: usize,
    pub num_chunks: usize,
    pub steps: Vec<DualPipeStep>,
    pub bubble_ratio: f32,
}

impl DualPipeSchedule {
    /// Synthesize a DualPipe bidirectional execution schedule for `num_stages` and `num_chunks`.
    pub fn synthesize(num_stages: usize, num_chunks: usize) -> Self {
        assert!(num_stages > 0 && num_chunks > 0);

        let mut steps = Vec::new();
        // DualPipe bidirectional interleaved steps
        // Total steps = (num_chunks + num_stages - 1)
        let total_steps = num_chunks + num_stages;
        let mut idle_slots = 0;
        let total_slots = total_steps * num_stages;

        for step_idx in 0..total_steps {
            let mut stage_ops = Vec::with_capacity(num_stages);

            for stage in 0..num_stages {
                // Direction A flows forward: stage index increases with step
                // Direction B flows in complementary phase
                let chunk_a_cand = step_idx.checked_sub(stage);
                let chunk_b_cand = step_idx.checked_sub(num_stages - 1 - stage);

                let op = if let Some(ca) = chunk_a_cand {
                    if ca < num_chunks && (step_idx % 2 == 0) {
                        DualPipeOp::ForwardAttention {
                            chunk_id: ca,
                            direction: DualPipeDirection::ForwardA,
                            stage_id: stage,
                            layer_range: 0..0,
                        }
                    } else if ca < num_chunks {
                        DualPipeOp::ForwardMlp {
                            chunk_id: ca,
                            direction: DualPipeDirection::ForwardA,
                            stage_id: stage,
                            layer_range: 0..0,
                        }
                    } else {
                        idle_slots += 1;
                        DualPipeOp::Idle
                    }
                } else if let Some(cb) = chunk_b_cand {
                    if cb < num_chunks && (step_idx % 2 == 1) {
                        DualPipeOp::ForwardAttention {
                            chunk_id: cb,
                            direction: DualPipeDirection::ForwardB,
                            stage_id: stage,
                            layer_range: 0..0,
                        }
                    } else if cb < num_chunks {
                        DualPipeOp::ForwardMlp {
                            chunk_id: cb,
                            direction: DualPipeDirection::ForwardB,
                            stage_id: stage,
                            layer_range: 0..0,
                        }
                    } else {
                        idle_slots += 1;
                        DualPipeOp::Idle
                    }
                } else {
                    idle_slots += 1;
                    DualPipeOp::Idle
                };

                stage_ops.push(op);
            }

            steps.push(DualPipeStep {
                step_index: step_idx,
                stage_ops,
            });
        }

        let bubble_ratio = if total_slots > 0 {
            idle_slots as f32 / total_slots as f32
        } else {
            0.0
        };

        Self {
            num_stages,
            num_chunks,
            steps,
            bubble_ratio,
        }
    }
}

/// DualPipe execution engine coordinating pipeline-parallel layer evaluation.
pub struct DualPipeEngine<'a> {
    pub model: &'a Model<'a>,
    pub partitions: Vec<LayerPartition>,
}

impl<'a> DualPipeEngine<'a> {
    /// Initialize a DualPipe engine with `num_stages` balanced layer partitions.
    pub fn new(model: &'a Model<'a>, num_stages: usize) -> Self {
        let partitions = LayerPartition::even_split(model.config.layers, num_stages);
        Self { model, partitions }
    }

    /// Initialize a DualPipe engine with custom explicit layer partitions.
    pub fn with_partitions(model: &'a Model<'a>, partitions: Vec<LayerPartition>) -> Self {
        Self { model, partitions }
    }

    /// Pipeline stage partitions.
    pub fn partitions(&self) -> &[LayerPartition] {
        &self.partitions
    }

    /// Executes a bidirectional dual-chunk forward pass across partitioned stages,
    /// returning the exact output logits for Chunk A and Chunk B.
    pub fn forward_dual_chunk(
        &self,
        state_a: &mut State,
        token_a: u32,
        pos_a: usize,
        state_b: &mut State,
        token_b: u32,
        pos_b: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let c = &self.model.config;

        // Initialize embeddings for Chunk A and Chunk B
        let mut x_a = vec![0.0f32; c.embedding];
        self.model.token_embd.row(token_a as usize, &mut x_a)?;
        if let Some(scale) = c.embedding_scale {
            for v in x_a.iter_mut() {
                *v *= scale;
            }
        }

        let mut x_b = vec![0.0f32; c.embedding];
        self.model.token_embd.row(token_b as usize, &mut x_b)?;
        if let Some(scale) = c.embedding_scale {
            for v in x_b.iter_mut() {
                *v *= scale;
            }
        }

        // Forward through partitioned stages in dual-direction pipeline execution
        for partition in &self.partitions {
            let layers = &self.model.layers[partition.layer_range.clone()];

            // Chunk A evaluation on stage
            forward_partition(
                layers,
                c,
                state_a,
                partition.layer_range.clone(),
                &mut x_a,
                pos_a,
            )?;

            // Chunk B evaluation on stage (overlapped in execution model)
            forward_partition(
                layers,
                c,
                state_b,
                partition.layer_range.clone(),
                &mut x_b,
                pos_b,
            )?;
        }

        // Final output norm and lm_head projection for Chunk A
        ops::rms_norm(
            &x_a,
            &self.model.output_norm,
            c.eps,
            &mut state_a.xb[..c.embedding],
        );
        self.model.output.matvec_threaded(
            &state_a.xb[..c.embedding],
            &mut state_a.logits,
            state_a.threads,
            &mut state_a.workspace,
        )?;
        if let Some(cap) = c.final_logit_softcapping {
            for l in state_a.logits.iter_mut() {
                *l = cap * (*l / cap).tanh();
            }
        }
        let logits_a = state_a.logits.clone();

        // Final output norm and lm_head projection for Chunk B
        ops::rms_norm(
            &x_b,
            &self.model.output_norm,
            c.eps,
            &mut state_b.xb[..c.embedding],
        );
        self.model.output.matvec_threaded(
            &state_b.xb[..c.embedding],
            &mut state_b.logits,
            state_b.threads,
            &mut state_b.workspace,
        )?;
        if let Some(cap) = c.final_logit_softcapping {
            for l in state_b.logits.iter_mut() {
                *l = cap * (*l / cap).tanh();
            }
        }
        let logits_b = state_b.logits.clone();

        Ok((logits_a, logits_b))
    }
}
