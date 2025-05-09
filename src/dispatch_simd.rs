use std::sync::atomic::{AtomicBool, Ordering};

/// Helper function to check if AVX2 is supported with result caching
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
fn is_avx2_supported() -> bool {
    static AVX2_SUPPORTED: AtomicBool = AtomicBool::new(false);
    static CHECKED: AtomicBool = AtomicBool::new(false);
    
    if !CHECKED.load(Ordering::Relaxed) {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::__cpuid;
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::__cpuid;

        let supported = unsafe {
            let info = __cpuid(7);
            ((info.ebx >> 5) & 1) != 0
        };
        
        AVX2_SUPPORTED.store(supported, Ordering::Relaxed);
        CHECKED.store(true, Ordering::Relaxed);
    }
    
    AVX2_SUPPORTED.load(Ordering::Relaxed)
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
#[inline]
fn is_avx2_supported() -> bool {
    false
}

/// Generic function to dispatch between SIMD-accelerated and fallback implementations
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