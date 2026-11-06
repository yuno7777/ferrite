//! Chunked prefill with asynchronous background compute and incremental cache streaming.
//!
//! Provides non-blocking execution of large prompt prefilling across background worker threads,
//! streaming incremental KV-cache chunk completion events and supporting cooperative cancellation
//! without blocking interactive decode loops.

use std::io::{Error, ErrorKind, Result};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::thread::{spawn, JoinHandle};

use crate::model::{Model, State};

fn bad(msg: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidInput, msg.into())
}

/// Incremental cache streaming event emitted during chunked prompt prefill.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefillChunkEvent {
    /// A chunk of prompt tokens was successfully prefilled into the KV-cache.
    ChunkProcessed {
        chunk_index: usize,
        chunk_tokens: usize,
        total_processed: usize,
        total_tokens: usize,
    },
    /// All prompt tokens have been successfully prefilled.
    Completed { total_tokens: usize },
    /// The prefill operation was cancelled cooperatively before completion.
    Cancelled { processed_tokens: usize },
}

#[derive(Clone, Copy)]
struct SendModelPtr(usize);

/// Asynchronous background worker for chunked prompt prefill with incremental cache streaming.
pub struct AsyncChunkedPrefill<'a> {
    cancel_flag: Arc<AtomicBool>,
    rx: Receiver<PrefillChunkEvent>,
    handle: Option<JoinHandle<Result<State>>>,
    total_tokens: usize,
    processed_tokens: usize,
    is_finished: bool,
    _marker: PhantomData<&'a ()>,
}

impl<'a> AsyncChunkedPrefill<'a> {
    /// Spawns a background thread to prefill `tokens` into `state` in chunks of `chunk_size`.
    ///
    /// Incremental chunk completion events are streamed across an internal channel
    /// and can be polled without blocking via [`poll_event`] or waited upon with [`wait_next_event`].
    pub fn spawn(
        model: &'a Model<'a>,
        mut state: State,
        tokens: Vec<u32>,
        start_pos: usize,
        chunk_size: usize,
    ) -> Result<Self> {
        let total_tokens = tokens.len();
        let eff_chunk = if chunk_size == 0 { 512 } else { chunk_size };

        let cancel_flag = Arc::new(AtomicBool::new(false));
        let cancel_clone = Arc::clone(&cancel_flag);
        let (tx, rx): (Sender<PrefillChunkEvent>, Receiver<PrefillChunkEvent>) = channel();

        let model_ptr = SendModelPtr(model as *const Model<'a> as usize);

        let handle = spawn(move || -> Result<State> {
            let model_ref = unsafe { &*(model_ptr.0 as *const Model<'a>) };
            let mut processed = 0;
            let mut chunk_index = 0;

            while processed < total_tokens {
                if cancel_clone.load(Ordering::Relaxed) {
                    tx.send(PrefillChunkEvent::Cancelled {
                        processed_tokens: processed,
                    })
                    .ok();
                    return Ok(state);
                }

                let end = (processed + eff_chunk).min(total_tokens);
                let chunk = &tokens[processed..end];
                let count = end - processed;

                model_ref.prefill(&mut state, chunk, start_pos + processed)?;

                processed = end;
                chunk_index += 1;

                let event = PrefillChunkEvent::ChunkProcessed {
                    chunk_index,
                    chunk_tokens: count,
                    total_processed: processed,
                    total_tokens,
                };

                if tx.send(event).is_err() {
                    // Receiver dropped, stop background prefill
                    break;
                }
            }

            if !cancel_clone.load(Ordering::Relaxed) {
                tx.send(PrefillChunkEvent::Completed {
                    total_tokens: processed,
                })
                .ok();
            }

            Ok(state)
        });

        Ok(Self {
            cancel_flag,
            rx,
            handle: Some(handle),
            total_tokens,
            processed_tokens: 0,
            is_finished: false,
            _marker: PhantomData,
        })
    }

    /// Poll for the next incremental cache streaming event without blocking.
    pub fn poll_event(&mut self) -> Option<PrefillChunkEvent> {
        if let Ok(event) = self.rx.try_recv() {
            match event {
                PrefillChunkEvent::ChunkProcessed {
                    total_processed, ..
                } => {
                    self.processed_tokens = total_processed;
                }
                PrefillChunkEvent::Completed { total_tokens } => {
                    self.processed_tokens = total_tokens;
                    self.is_finished = true;
                }
                PrefillChunkEvent::Cancelled { processed_tokens } => {
                    self.processed_tokens = processed_tokens;
                    self.is_finished = true;
                }
            }
            Some(event)
        } else {
            None
        }
    }

    /// Blocking wait for the next incremental cache streaming event.
    pub fn wait_next_event(&mut self) -> Option<PrefillChunkEvent> {
        if let Ok(event) = self.rx.recv() {
            match event {
                PrefillChunkEvent::ChunkProcessed {
                    total_processed, ..
                } => {
                    self.processed_tokens = total_processed;
                }
                PrefillChunkEvent::Completed { total_tokens } => {
                    self.processed_tokens = total_tokens;
                    self.is_finished = true;
                }
                PrefillChunkEvent::Cancelled { processed_tokens } => {
                    self.processed_tokens = processed_tokens;
                    self.is_finished = true;
                }
            }
            Some(event)
        } else {
            None
        }
    }

    /// Total number of tokens in the prompt.
    #[inline]
    pub fn total_tokens(&self) -> usize {
        self.total_tokens
    }

    /// Number of tokens currently committed to the KV-cache.
    #[inline]
    pub fn processed_tokens(&self) -> usize {
        self.processed_tokens
    }

    /// Returns true if all chunks have finished or if execution was cancelled.
    #[inline]
    pub fn is_finished(&self) -> bool {
        self.is_finished
    }

    /// Request cooperative cancellation of the background prefill task.
    pub fn cancel(&self) {
        self.cancel_flag.store(true, Ordering::Relaxed);
    }

    /// Join the background worker thread and retrieve the final populated model state.
    pub fn join(mut self) -> Result<State> {
        if let Some(handle) = self.handle.take() {
            handle
                .join()
                .map_err(|_| bad("background prefill thread panicked"))?
        } else {
            Err(bad("prefill thread already joined"))
        }
    }
}

impl<'a> Drop for AsyncChunkedPrefill<'a> {
    fn drop(&mut self) {
        self.cancel_flag.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}
