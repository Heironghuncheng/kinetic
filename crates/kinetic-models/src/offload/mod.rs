//! The bookkeeping of routed experts kept on a device that cannot hold them
//! all: where they go, which ones it holds, and what the runtime did.

#[cfg(feature = "vulkan")]
pub(crate) mod assign;
pub(crate) mod cache;
pub(crate) mod plan;
pub(crate) mod stats;
