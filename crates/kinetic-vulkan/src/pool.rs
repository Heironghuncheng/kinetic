use std::rc::Rc;

use kinetic_quant::{Matrix, TensorType};

use crate::commands::Commands;
use crate::device::Device;
use crate::error::{Error, Result};
use crate::gemv::{block_len, DeviceMatrix};
use crate::layout::{device_layout, planes};
use crate::memory::{DeviceBuffer, HostBuffer};

/// Slots for experts of one shape: a matrix of `slots` experts one after
/// another, as the expert kernels read them, whose slots are filled from
/// host copies of single experts.
pub struct ExpertSlots {
    matrix: DeviceMatrix,
    slots: usize,
}

impl ExpertSlots {
    /// `slots` slots for experts of `rows` rows of `cols` weights of `ty`.
    pub fn new(
        device: &Rc<Device>,
        ty: TensorType,
        slots: usize,
        rows: usize,
        cols: usize,
    ) -> Result<ExpertSlots> {
        let Some(block) = block_len(ty) else {
            return Err(Error::Unsupported(ty));
        };
        let blocks = (slots * rows).checked_mul(cols / block);
        if slots == 0
            || !cols.is_multiple_of(32)
            || blocks.is_none_or(|b| u32::try_from(b).is_err())
        {
            return Err(Error::Shape(format!(
                "{slots} slots of {rows}x{cols} experts"
            )));
        }
        let bytes = slots * rows * cols / ty.block_len() * ty.block_bytes();
        Ok(ExpertSlots {
            matrix: DeviceMatrix {
                ty,
                rows: slots * rows,
                cols,
                bytes,
                // Kernels read whole 16-byte words, also of the last plane.
                data: DeviceBuffer::new(device, bytes.next_multiple_of(16))?,
            },
            slots,
        })
    }

    /// Every slot as one matrix.
    pub fn matrix(&self) -> &DeviceMatrix {
        &self.matrix
    }

    pub fn slots(&self) -> usize {
        self.slots
    }

    /// Bytes of one expert.
    pub fn expert_bytes(&self) -> usize {
        self.matrix.bytes / self.slots
    }

    /// Writes expert `m` to `staging` as [`ExpertSlots::record_copy`] takes
    /// it.
    pub fn stage(&self, m: &Matrix, staging: &mut [u8]) -> Result<()> {
        let fits = m.ty == self.matrix.ty
            && m.cols == self.matrix.cols
            && m.rows * self.slots == self.matrix.rows
            && staging.len() == self.expert_bytes();
        if !fits {
            return Err(Error::Shape(format!(
                "a {}x{} {} expert into slots of {} rows of {}",
                m.rows,
                m.cols,
                m.ty,
                self.matrix.rows / self.slots,
                self.matrix.cols
            )));
        }
        staging.copy_from_slice(&device_layout(m));
        Ok(())
    }

    /// Records the copy of the expert staged at byte `at` of `staging` into
    /// slot `slot`.
    pub fn record_copy(
        &self,
        staging: &HostBuffer<u8>,
        at: usize,
        slot: usize,
        commands: &mut Commands,
    ) -> Result<()> {
        let bytes = self.expert_bytes();
        if slot >= self.slots || at + bytes > staging.len() {
            return Err(Error::Shape(format!(
                "copy of {bytes} staged bytes at {at} into slot {slot} of {}",
                self.slots
            )));
        }
        let ty = self.matrix.ty;
        let blocks = bytes / ty.block_bytes();
        let fields: &[(usize, usize)] = match planes(ty) {
            Some(f) => f,
            None => &[(0, ty.block_bytes())],
        };
        let mut plane = 0;
        for &(_, len) in fields {
            let (src, dst) = (
                at + plane * blocks,
                plane * blocks * self.slots + slot * blocks * len,
            );
            // SAFETY: the staged expert lies inside `staging`, checked above,
            // and each of its planes goes to the part of the same plane of
            // the matrix that belongs to `slot`, which is below `slots`.
            unsafe {
                commands.copy(
                    staging.allocation(),
                    src,
                    self.matrix.data.allocation(),
                    dst,
                    blocks * len,
                )
            };
            plane += len;
        }
        Ok(())
    }
}
