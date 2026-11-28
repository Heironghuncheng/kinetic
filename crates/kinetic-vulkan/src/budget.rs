#[cfg(not(metal_driver))]
use ash::vk;

use crate::device::Device;

#[cfg(not(metal_driver))]
impl Device {
    /// Device-local bytes free now, in the heap kernels' buffers come from,
    /// as the driver's memory budget reports them; none without the
    /// budget extension. Other programs' use counts against it.
    pub fn free_bytes(&self) -> Option<u64> {
        let physical = self.physical();
        let instance = &physical.instance.raw;
        // SAFETY: the physical device came from this instance.
        let extensions =
            unsafe { instance.enumerate_device_extension_properties(physical.raw) }.ok()?;
        let name = ash::ext::memory_budget::NAME;
        if !extensions
            .iter()
            .any(|e| e.extension_name_as_c_str() == Ok(name))
        {
            return None;
        }
        let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        let mut props = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
        // SAFETY: the device supports the budget extension, checked above,
        // and both structures outlive the call.
        unsafe { instance.get_physical_device_memory_properties2(physical.raw, &mut props) };
        let memory = props.memory_properties;
        let heaps = &memory.memory_heaps[..memory.memory_heap_count as usize];
        let largest = (0..heaps.len())
            .filter(|&h| heaps[h].flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
            .max_by_key(|&h| heaps[h].size)?;
        Some(budget.heap_budget[largest].saturating_sub(budget.heap_usage[largest]))
    }
}

#[cfg(metal_driver)]
impl Device {
    /// Device bytes free now; Metal does not say.
    pub fn free_bytes(&self) -> Option<u64> {
        None
    }
}
