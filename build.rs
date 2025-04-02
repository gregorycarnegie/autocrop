fn main() {
    // Enable AVX2 if on x86/x86_64 platforms
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        println!("cargo:rustc-cfg=has_avx2");
        println!("cargo:rustc-flag=-C target-feature=+avx2");
    }
}