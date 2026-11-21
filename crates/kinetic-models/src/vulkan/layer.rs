use std::rc::Rc;

use kinetic_vulkan::{Commands, ConvRing, DeltaState, Device, DeviceBuffer, KvCache, Residual};

use super::attention::VulkanAttention;
use super::delta_net::VulkanDeltaNet;
use super::ffn::VulkanFfn;
use super::moe::VulkanMoe;
use super::relative::{RelativeState, VulkanRelative};
use super::rope::RopeTables;
use super::short_conv::VulkanShortConv;
use super::work::{Needs, Work};
use super::Gpu;
use crate::blocks::Mixer;
use crate::model::Layer;
use crate::LoadError;

pub(super) type R = kinetic_vulkan::Result<()>;

pub(super) fn unsupported(what: &str) -> LoadError {
    LoadError(format!("{what} are not supported on Vulkan yet"))
}

/// A token mixer on the device.
enum VulkanMixer {
    Attention(VulkanAttention),
    ShortConv(VulkanShortConv),
    DeltaNet(VulkanDeltaNet),
    /// Attention by relative distance, run by its model's network.
    Relative(Box<VulkanRelative>),
}

/// What a sequence keeps of one layer.
pub(super) enum LayerState {
    Attention(KvCache),
    ShortConv(ConvRing),
    DeltaNet(DeltaState),
    Relative(Box<RelativeState>),
}

impl LayerState {
    pub fn size_bytes(&self) -> usize {
        match self {
            LayerState::Attention(c) => c.size_bytes(),
            LayerState::ShortConv(r) => r.size_bytes(),
            LayerState::DeltaNet(s) => s.size_bytes(),
            LayerState::Relative(s) => s.size_bytes(),
        }
    }
}

/// A decoder layer on the device; see [`Layer`].
pub(super) struct VulkanLayer {
    /// The input norm, which layers of a network leave to it.
    pub mixer_norm: Option<DeviceBuffer<f32>>,
    mixer: VulkanMixer,
    mixer_out_norm: Option<DeviceBuffer<f32>>,
    ffn_norm: DeviceBuffer<f32>,
    /// The feed-forward, or the shared expert beside routed ones.
    ffn: Option<VulkanFfn>,
    experts: Option<VulkanMoe>,
    ffn_out_norm: Option<DeviceBuffer<f32>>,
    residual_scale: f32,
}

impl VulkanLayer {
    pub fn upload(gpu: &Gpu, l: &Layer, ropes: &mut RopeTables) -> Result<VulkanLayer, LoadError> {
        let optional = |v: &Option<Vec<f32>>| v.as_deref().map(|v| gpu.vector(v)).transpose();
        let mixer = match &l.mixer {
            Mixer::Attention(a) => VulkanMixer::Attention(VulkanAttention::upload(gpu, a, ropes)?),
            Mixer::ShortConv(s) => VulkanMixer::ShortConv(VulkanShortConv::upload(gpu, s)?),
            Mixer::DeltaNet(d) => VulkanMixer::DeltaNet(VulkanDeltaNet::upload(gpu, d)?),
            Mixer::Relative(a) => VulkanMixer::Relative(Box::new(VulkanRelative::upload(gpu, a)?)),
            Mixer::Mamba2(_) | Mixer::Parallel(_) => {
                return Err(unsupported("Mamba-2 and parallel token mixers"))
            }
        };
        if l.output_scale.is_some() {
            return Err(unsupported("layer output scales"));
        }
        let relative = matches!(mixer, VulkanMixer::Relative(_));
        let (Some(ffn_norm), true) = (&l.ffn_norm, relative || l.mixer_norm.is_some()) else {
            return Err(unsupported("layers without input norms"));
        };
        let ffn = l.ffn.as_ref().map(|f| VulkanFfn::upload(gpu, f));
        let experts = l.experts.as_ref().map(|m| VulkanMoe::upload(gpu, m));
        Ok(VulkanLayer {
            mixer_norm: optional(&l.mixer_norm)?,
            mixer,
            mixer_out_norm: optional(&l.mixer_out_norm)?,
            ffn_norm: gpu.vector(ffn_norm)?,
            ffn: ffn.transpose()?,
            experts: experts.transpose()?,
            ffn_out_norm: optional(&l.ffn_out_norm)?,
            residual_scale: l.residual_scale.unwrap_or(1.0),
        })
    }

    pub fn size_bytes(&self) -> usize {
        let mixer = match &self.mixer {
            VulkanMixer::Attention(a) => a.size_bytes(),
            VulkanMixer::ShortConv(s) => s.size_bytes(),
            VulkanMixer::DeltaNet(d) => d.size_bytes(),
            VulkanMixer::Relative(a) => a.size_bytes(),
        };
        let ffn = self.ffn.as_ref().map_or(0, VulkanFfn::size_bytes);
        mixer + ffn + self.experts.as_ref().map_or(0, VulkanMoe::size_bytes)
    }

    /// What the mixer needs of the work buffers.
    pub fn needs(&self) -> Needs {
        match &self.mixer {
            VulkanMixer::Attention(a) => a.needs(),
            VulkanMixer::ShortConv(s) => s.needs(),
            VulkanMixer::DeltaNet(d) => d.needs(),
            VulkanMixer::Relative(a) => a.needs(),
        }
    }

    pub fn n_ff(&self) -> Option<usize> {
        self.ffn.as_ref().map(|f| f.gate.rows())
    }

    pub fn relative(&self) -> Option<&VulkanRelative> {
        match &self.mixer {
            VulkanMixer::Relative(a) => Some(a),
            _ => None,
        }
    }

    pub fn ffn_norm(&self) -> &DeviceBuffer<f32> {
        &self.ffn_norm
    }

    pub fn experts(&self) -> Option<&VulkanMoe> {
        self.experts.as_ref()
    }

    /// Past positions an attention mixer sees, when not all.
    pub fn window(&self) -> Option<usize> {
        match &self.mixer {
            VulkanMixer::Attention(a) => a.shape.window,
            VulkanMixer::Relative(a) => a.shape.heads.window,
            _ => None,
        }
    }

    /// Whether a rollback must replay the layer's kept tokens.
    pub fn replays(&self) -> bool {
        matches!(self.mixer, VulkanMixer::DeltaNet(_))
    }

    /// A fresh state for `positions` positions run in passes of up to
    /// `rows` tokens.
    pub fn new_state(
        &self,
        dev: &Rc<Device>,
        positions: usize,
        rows: usize,
    ) -> kinetic_vulkan::Result<LayerState> {
        Ok(match &self.mixer {
            VulkanMixer::Attention(a) => {
                LayerState::Attention(KvCache::new(dev, &a.shape, positions, rows)?)
            }
            VulkanMixer::ShortConv(s) => {
                LayerState::ShortConv(ConvRing::new(dev, s.channels(), s.taps, rows)?)
            }
            VulkanMixer::DeltaNet(d) => LayerState::DeltaNet(DeltaState::new(dev, &d.shape, rows)?),
            VulkanMixer::Relative(a) => {
                LayerState::Relative(Box::new(a.new_state(dev, positions, rows)?))
            }
        })
    }

    /// Empties `state` for a new sequence. Caches and rings need nothing:
    /// every slot a pass reads was written before it.
    pub fn clear(&self, state: &mut LayerState) -> R {
        match state {
            LayerState::DeltaNet(s) => s.clear(),
            LayerState::Attention(_) | LayerState::ShortConv(_) | LayerState::Relative(_) => Ok(()),
        }
    }

    /// Records the layer over the batch, whose input `w.h` holds the stream
    /// `w.x` after this layer's input norm. Leaves `w.h` holding `x` after
    /// the layer, normalized by `next`, the following layer's input norm or
    /// the output norm.
    pub fn record<'a>(
        &'a self,
        gpu: &Gpu,
        w: &'a Work,
        state: &LayerState,
        next: &DeviceBuffer<f32>,
        eps: f32,
        c: &mut Commands,
    ) -> R {
        match (&self.mixer, state) {
            (VulkanMixer::Attention(a), LayerState::Attention(cache)) => a.record(gpu, w, cache, c),
            (VulkanMixer::ShortConv(s), LayerState::ShortConv(ring)) => s.record(gpu, w, ring, c),
            (VulkanMixer::DeltaNet(d), LayerState::DeltaNet(st)) => d.record(gpu, w, st, c),
            (VulkanMixer::Relative(_), _) => unreachable!("relative attention runs in its network"),
            _ => unreachable!("state does not belong to this layer"),
        }?;
        let k = &gpu.kernels;
        let add = |post: Option<&'a DeviceBuffer<f32>>| Residual {
            y: &w.y,
            post,
            scale: self.residual_scale,
        };
        let mixed = add(self.mixer_out_norm.as_ref());
        k.norm(&w.batch, &w.x, Some(mixed), &self.ffn_norm, eps, &w.h, c)?;
        if let Some(ffn) = &self.ffn {
            ffn.record(gpu, w, c)?;
        }
        if let Some(moe) = &self.experts {
            moe.record(gpu, w, self.ffn.is_some(), c)?;
        }
        let fed = add(self.ffn_out_norm.as_ref());
        k.norm(&w.batch, &w.x, Some(fed), next, eps, &w.h, c)
    }

    /// Records what a rollback runs for this layer, if anything.
    pub fn record_replay(&self, gpu: &Gpu, w: &Work, state: &LayerState, c: &mut Commands) -> R {
        match (&self.mixer, state) {
            (VulkanMixer::DeltaNet(d), LayerState::DeltaNet(st)) => d.record_replay(gpu, w, st, c),
            _ => Ok(()),
        }
    }
}
