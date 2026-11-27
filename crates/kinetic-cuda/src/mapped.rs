use std::ops::Range;

use crate::error::{Error, Result};
use crate::kernels::{arg, Kernels};
use crate::memory::DeviceBuffer;
use crate::module::LaunchConfig;
use crate::pinned::PinnedBuffer;
use crate::stream::Stream;

const THREADS: u32 = 256;

impl Kernels {
    /// Queues a kernel that writes elements `range` of `x` into `out` from
    /// element `at`, straight into host memory.
    ///
    /// # Safety
    ///
    /// The destination elements must not be read or written, and neither
    /// buffer freed, until `stream` has passed the kernel.
    pub unsafe fn to_host(
        &self,
        out: &mut PinnedBuffer<f32>,
        at: usize,
        x: &DeviceBuffer<f32>,
        range: Range<usize>,
        stream: &Stream,
    ) -> Result<()> {
        x.check_holds(range.end, "source")?;
        let len = range.len();
        if at.checked_add(len).is_none_or(|end| end > out.len()) {
            return Err(Error::Shape(format!(
                "{len} elements at {at} of a pinned buffer of {}",
                out.len()
            )));
        }
        if len == 0 {
            return Ok(());
        }
        let (o, src, n) = (out.mapped_at(at), x.ptr_at(range.start), len as u32);
        let config = LaunchConfig::linear(n.div_ceil(THREADS), THREADS);
        // SAFETY: matches `to_host(float*, const float*, unsigned)`; both
        // ranges were checked, and the caller keeps the destination until
        // the stream passes the kernel.
        unsafe {
            self.to_host
                .launch(config, stream, &mut [arg(&o), arg(&src), arg(&n)])
        }
    }
}
