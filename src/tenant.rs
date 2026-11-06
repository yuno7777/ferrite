//! Multi-tenant inference engine with token bucket rate limiting and priority queuing.
//!
//! Provides tenant isolation, per-tenant rate limiting via token buckets,
//! concurrency caps, usage metrics accounting, and strict priority request queuing.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::time::Instant;

/// Priority level for tenant requests.
/// Lower integer value represents higher priority.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RequestPriority {
    Critical = 0,
    High = 1,
    Standard = 2,
    Background = 3,
}

impl PartialOrd for RequestPriority {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RequestPriority {
    fn cmp(&self, other: &Self) -> Ordering {
        // Lower numerical value means higher scheduling priority.
        // In max-heap BinaryHeap, we want lower value to be Greater.
        other.raw().cmp(&self.raw())
    }
}

impl RequestPriority {
    pub fn raw(&self) -> u8 {
        match self {
            Self::Critical => 0,
            Self::High => 1,
            Self::Standard => 2,
            Self::Background => 3,
        }
    }
}

/// Token bucket algorithm for rate limiting.
#[derive(Clone, Debug)]
pub struct TokenBucket {
    capacity: f64,
    tokens: f64,
    refill_rate_per_sec: f64,
    last_update: Instant,
}

impl TokenBucket {
    pub fn new(capacity: f64, refill_rate_per_sec: f64) -> Self {
        Self {
            capacity: capacity.max(1.0),
            tokens: capacity.max(1.0),
            refill_rate_per_sec: refill_rate_per_sec.max(0.001),
            last_update: Instant::now(),
        }
    }

    /// Refills available tokens based on elapsed duration.
    pub fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_update).as_secs_f64();
        if elapsed > 0.0 {
            self.tokens = (self.tokens + elapsed * self.refill_rate_per_sec).min(self.capacity);
            self.last_update = now;
        }
    }

    /// Attempts to consume the specified amount of tokens. Returns true if successful.
    pub fn try_consume(&mut self, amount: f64) -> bool {
        self.refill();
        if self.tokens >= amount {
            self.tokens -= amount;
            true
        } else {
            false
        }
    }

    /// Current available tokens.
    pub fn available(&mut self) -> f64 {
        self.refill();
        self.tokens
    }

    /// Maximum capacity of this bucket.
    pub fn capacity(&self) -> f64 {
        self.capacity
    }
}

/// Tenant configuration parameters.
#[derive(Clone, Debug)]
pub struct TenantConfig {
    pub id: String,
    pub api_key: String,
    pub priority: RequestPriority,
    pub rate_limit_rps: f64,
    pub burst_capacity: f64,
    pub max_concurrent_requests: usize,
}

impl TenantConfig {
    pub fn new(
        id: impl Into<String>,
        api_key: impl Into<String>,
        priority: RequestPriority,
        rate_limit_rps: f64,
        burst_capacity: f64,
        max_concurrent_requests: usize,
    ) -> Self {
        Self {
            id: id.into(),
            api_key: api_key.into(),
            priority,
            rate_limit_rps: rate_limit_rps.max(0.1),
            burst_capacity: burst_capacity.max(1.0),
            max_concurrent_requests: max_concurrent_requests.max(1),
        }
    }

    pub fn standard(id: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self::new(id, api_key, RequestPriority::Standard, 50.0, 100.0, 16)
    }

    pub fn enterprise(id: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self::new(id, api_key, RequestPriority::High, 200.0, 500.0, 64)
    }
}

/// Metrics and usage statistics tracked per tenant.
#[derive(Clone, Debug, Default)]
pub struct TenantMetrics {
    pub total_requests: u64,
    pub active_requests: usize,
    pub rate_limited_requests: u64,
    pub rejected_concurrency_requests: u64,
    pub total_prompt_tokens: u64,
    pub total_completion_tokens: u64,
}

#[derive(Debug)]
struct TenantState {
    config: TenantConfig,
    bucket: TokenBucket,
    active_requests: usize,
    metrics: TenantMetrics,
}

/// Rate limiting and admission errors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AdmissionError {
    Unauthorized,
    RateLimitExceeded { retry_after_secs: u64 },
    ConcurrencyLimitReached { current: usize, max: usize },
    TenantNotFound,
}

/// RAII Guard that decrements active concurrent request counter when dropped.
#[derive(Debug)]
pub struct TenantGuard {
    registry: Arc<TenantRegistry>,
    tenant_id: String,
}

impl TenantGuard {
    pub fn tenant_id(&self) -> &str {
        &self.tenant_id
    }

    /// Records token consumption for prompt and completion tokens.
    pub fn record_tokens(&self, prompt_tokens: usize, completion_tokens: usize) {
        if let Ok(mut map) = self.registry.tenants.write() {
            if let Some(state) = map.get_mut(&self.tenant_id) {
                state.metrics.total_prompt_tokens += prompt_tokens as u64;
                state.metrics.total_completion_tokens += completion_tokens as u64;
            }
        }
    }
}

impl Drop for TenantGuard {
    fn drop(&mut self) {
        if let Ok(mut map) = self.registry.tenants.write() {
            if let Some(state) = map.get_mut(&self.tenant_id) {
                state.active_requests = state.active_requests.saturating_sub(1);
                state.metrics.active_requests = state.active_requests;
            }
        }
    }
}

/// Multi-tenant registry managing credentials, rate limits, and metrics.
#[derive(Debug)]
pub struct TenantRegistry {
    tenants: RwLock<HashMap<String, TenantState>>,
    key_to_id: RwLock<HashMap<String, String>>,
}

impl Default for TenantRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl TenantRegistry {
    pub fn new() -> Self {
        Self {
            tenants: RwLock::new(HashMap::new()),
            key_to_id: RwLock::new(HashMap::new()),
        }
    }

    /// Registers a new tenant or updates existing configuration.
    pub fn register(&self, config: TenantConfig) {
        let bucket = TokenBucket::new(config.burst_capacity, config.rate_limit_rps);
        let id = config.id.clone();
        let key = config.api_key.clone();

        let state = TenantState {
            config,
            bucket,
            active_requests: 0,
            metrics: TenantMetrics::default(),
        };

        if let Ok(mut keys) = self.key_to_id.write() {
            keys.insert(key, id.clone());
        }
        if let Ok(mut map) = self.tenants.write() {
            map.insert(id, state);
        }
    }

    /// Resolves an API key or raw tenant ID to a tenant ID.
    pub fn resolve_tenant_id(&self, token_or_id: &str) -> Option<String> {
        let trimmed = token_or_id.trim();
        // Check API key first
        if let Ok(keys) = self.key_to_id.read() {
            if let Some(id) = keys.get(trimmed) {
                return Some(id.clone());
            }
        }
        // Check direct tenant ID
        if let Ok(map) = self.tenants.read() {
            if map.contains_key(trimmed) {
                return Some(trimmed.to_string());
            }
        }
        None
    }

    /// Gets configured priority for a tenant.
    pub fn get_priority(&self, tenant_id: &str) -> RequestPriority {
        if let Ok(map) = self.tenants.read() {
            if let Some(state) = map.get(tenant_id) {
                return state.config.priority;
            }
        }
        RequestPriority::Standard
    }

    /// Checks rate limits and concurrency limits for the tenant.
    /// On success returns an RAII `TenantGuard` that tracks the active request.
    pub fn acquire(self: &Arc<Self>, tenant_id: &str) -> Result<TenantGuard, AdmissionError> {
        let mut map = self
            .tenants
            .write()
            .map_err(|_| AdmissionError::TenantNotFound)?;
        let state = map
            .get_mut(tenant_id)
            .ok_or(AdmissionError::TenantNotFound)?;

        // Check concurrency cap
        if state.active_requests >= state.config.max_concurrent_requests {
            state.metrics.rejected_concurrency_requests += 1;
            return Err(AdmissionError::ConcurrencyLimitReached {
                current: state.active_requests,
                max: state.config.max_concurrent_requests,
            });
        }

        // Check token bucket rate limit
        if !state.bucket.try_consume(1.0) {
            state.metrics.rate_limited_requests += 1;
            let retry_after = (1.0 / state.config.rate_limit_rps).ceil() as u64;
            return Err(AdmissionError::RateLimitExceeded {
                retry_after_secs: retry_after.max(1),
            });
        }

        // Admission granted
        state.active_requests += 1;
        state.metrics.total_requests += 1;
        state.metrics.active_requests = state.active_requests;

        Ok(TenantGuard {
            registry: Arc::clone(self),
            tenant_id: tenant_id.to_string(),
        })
    }

    /// Retrieves a snapshot of the tenant metrics.
    pub fn get_metrics(&self, tenant_id: &str) -> Option<TenantMetrics> {
        let map = self.tenants.read().ok()?;
        map.get(tenant_id).map(|s| s.metrics.clone())
    }

    /// Returns all registered tenant IDs and their current metrics.
    pub fn all_metrics(&self) -> HashMap<String, TenantMetrics> {
        let mut result = HashMap::new();
        if let Ok(map) = self.tenants.read() {
            for (id, state) in map.iter() {
                result.insert(id.clone(), state.metrics.clone());
            }
        }
        result
    }
}

/// An entry in the priority request queue.
#[derive(Debug)]
struct QueueItem<T> {
    priority: RequestPriority,
    sequence: u64,
    payload: T,
}

impl<T> PartialEq for QueueItem<T> {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.sequence == other.sequence
    }
}

impl<T> Eq for QueueItem<T> {}

impl<T> PartialOrd for QueueItem<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for QueueItem<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        // First compare priority
        match self.priority.cmp(&other.priority) {
            Ordering::Equal => {
                // For identical priority, FIFO: lower sequence number comes first.
                // In max-heap BinaryHeap, lower sequence must be Greater.
                other.sequence.cmp(&self.sequence)
            }
            ord => ord,
        }
    }
}

/// Thread-safe priority request queue with FIFO ordering within each priority level.
#[derive(Debug)]
pub struct PriorityRequestQueue<T> {
    inner: Mutex<(BinaryHeap<QueueItem<T>>, u64)>,
    cond: Condvar,
}

impl<T> Default for PriorityRequestQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> PriorityRequestQueue<T> {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new((BinaryHeap::new(), 0)),
            cond: Condvar::new(),
        }
    }

    /// Enqueues an item with the given priority level.
    pub fn push(&self, priority: RequestPriority, payload: T) {
        if let Ok(mut guard) = self.inner.lock() {
            let seq = guard.1;
            guard.1 = seq.wrapping_add(1);
            guard.0.push(QueueItem {
                priority,
                sequence: seq,
                payload,
            });
            self.cond.notify_one();
        }
    }

    /// Attempts to pop the highest priority item non-blockingly.
    pub fn try_pop(&self) -> Option<T> {
        let mut guard = self.inner.lock().ok()?;
        guard.0.pop().map(|item| item.payload)
    }

    /// Current number of queued items.
    pub fn len(&self) -> usize {
        self.inner.lock().map(|g| g.0.len()).unwrap_or(0)
    }

    /// Whether the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
