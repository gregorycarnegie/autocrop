use std::sync::OnceLock;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::__cpuid;

/// Helper function to check if AVX2 is supported with result caching
#[inline]
fn is_avx2_supported() -> bool {
    static AVX2_SUPPORTED: OnceLock<bool> = OnceLock::new();

    *AVX2_SUPPORTED.get_or_init(|| unsafe {
        let info = __cpuid(7);
        ((info.ebx >> 5) & 1) != 0
    })
}

// Generic function to dispatch between SIMD-accelerated and fallback implementations
#[inline]
pub fn dispatch_simd<F, G, Args, Ret>(args: Args, avx2_fn: F, fallback_fn: G) -> Ret
where
    F: FnOnce(Args) -> Ret,
    G: FnOnce(Args) -> Ret,
{
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_avx2_supported() {
            return avx2_fn(args);
        }
    }

    fallback_fn(args)
}

#[inline]
pub fn compare_buffers(buffer: &[u8], signature: &[u8], offset: usize) -> bool {
    if buffer.len() < offset + signature.len() {
        return false;
    }
    &buffer[offset..offset + signature.len()] == signature
}
