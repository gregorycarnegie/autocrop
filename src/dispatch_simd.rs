use std::sync::OnceLock;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::{
    _mm256_loadu_si256, _mm256_movemask_epi8, __m256i,
    _mm256_cmpeq_epi8, _mm_loadu_si128, _mm_cmpeq_epi8,
    _mm_movemask_epi8, __m128i
};

/// Helper function to check if AVX2 is supported with result caching
#[inline]
fn is_avx2_supported() -> bool {
    static AVX2_SUPPORTED: OnceLock<bool> = OnceLock::new();
    
    *AVX2_SUPPORTED.get_or_init(|| {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::__cpuid;
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::__cpuid;

        unsafe {
            let info = __cpuid(7);
            ((info.ebx >> 5) & 1) != 0
        }
    })
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

// In dispatch_simd.rs

// First add the existing code you have there...

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
pub unsafe fn check_long_signature_avx2(buffer: &[u8], offset: usize, signature: &[u8]) -> bool {
    let sig_len = signature.len();
    let chunks = sig_len / 32;
    let remainder = sig_len % 32;
    
    // Compare 32 bytes at a time
    for i in 0..chunks {
        let buf_ptr = buffer.as_ptr().add(offset + i * 32) as *const __m256i;
        let sig_ptr = signature.as_ptr().add(i * 32) as *const __m256i;
        
        let buf_chunk = _mm256_loadu_si256(buf_ptr);
        let sig_chunk = _mm256_loadu_si256(sig_ptr);
        
        // Compare equality (0xFF where equal, 0x00 where different)
        let comparison = _mm256_cmpeq_epi8(buf_chunk, sig_chunk);
        
        // Convert to bitmask (1 bit per byte)
        let mask = _mm256_movemask_epi8(comparison);
        
        // If all 32 bytes match, mask will be 0xFFFFFFFF
        // Fix: Use u32 explicitly
        if mask as u32 != 0xFFFF_FFFFu32 {
            return false;
        }
    }
    
    // Check remaining bytes if any
    if remainder > 0 {
        let start = chunks * 32;
        return &buffer[offset + start..offset + sig_len] == &signature[start..sig_len];
    }
    
    true
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
pub unsafe fn check_medium_signature_sse2(buffer: &[u8], offset: usize, signature: &[u8]) -> bool {
    let sig_len = signature.len();
    let chunks = sig_len / 16;
    let remainder = sig_len % 16;
    
    // Compare 16 bytes at a time
    for i in 0..chunks {
        let buf_ptr = buffer.as_ptr().add(offset + i * 16) as *const __m128i;
        let sig_ptr = signature.as_ptr().add(i * 16) as *const __m128i;
        
        let buf_chunk = _mm_loadu_si128(buf_ptr);
        let sig_chunk = _mm_loadu_si128(sig_ptr);
        
        // Compare equality (0xFF where equal, 0x00 where different)
        let comparison = _mm_cmpeq_epi8(buf_chunk, sig_chunk);
        
        // Convert to bitmask (1 bit per byte)
        let mask = _mm_movemask_epi8(comparison);
        
        // If all 16 bytes match, mask will be 0xFFFF
        // Fix: Use appropriate size and type
        if mask as u16 != 0xFFFFu16 {
            return false;
        }
    }
    
    // Check remaining bytes if any
    if remainder > 0 {
        let start = chunks * 16;
        return &buffer[offset + start..offset + sig_len] == &signature[start..sig_len];
    }
    
    true
}

// Now add the compare_buffers helper
#[inline]
pub fn compare_buffers(buffer: &[u8], signature: &[u8], offset: usize) -> bool {
    if buffer.len() < offset + signature.len() {
        return false;
    }
    
    dispatch_simd(
        (buffer, signature, offset),
        |(buf, sig, off)| unsafe {
            // Choose SIMD strategy based on signature length
            if sig.len() >= 32 {
                check_long_signature_avx2(buf, off, sig)
            } else if sig.len() >= 16 {
                check_medium_signature_sse2(buf, off, sig)
            } else {
                // For short signatures, scalar comparison is often faster
                &buf[off..off + sig.len()] == sig
            }
        },
        |(buf, sig, off)| &buf[off..off + sig.len()] == sig
    )
}