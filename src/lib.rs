use ndarray::{Array1, Axis};
use numpy::{IntoPyArray, PyReadonlyArray2};
use pyo3::{prelude::*, types::PyTuple};
use std::sync::atomic::{AtomicBool, Ordering};

// For x86/x86_64 specific SIMD intrinsics
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::{
    _mm256_set1_pd, _mm256_storeu_pd, _mm256_loadu_pd, 
    _mm256_div_pd, _mm256_set_pd,
};

/// A lightweight struct representing a rectangle
#[derive(Debug, Clone, Copy)]
struct Rectangle {
    x: f64,
    y: f64,
    width: f64,
    height: f64,
}

/// Alias for the four integer coordinates of a bounding box.
type BoxCoordinates = (i32, i32, i32, i32);

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

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
/// Helper function for computing the prefix sum of 4 elements from a SIMD load.
/// The running sum (passed in as `init`) is updated and the resulting prefix values are returned.
unsafe fn simd_prefix_sum(values: &[f64; 4], init: &mut f64) -> [f64; 4] {
    let mut result = [0.0; 4];
    for i in 0..4 {
        *init += values[i];
        result[i] = *init;
    }
    result
}

/// Compute an optimized cumulative sum of a slice of f64 values.
fn cumsum(vec: &[f64]) -> Vec<f64> {
    if vec.is_empty() {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(vec.len());
    let mut running_sum = 0.0;

    // Check if we can use AVX2 SIMD
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let use_avx = is_avx2_supported();
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    let use_avx = false;

    // Use a chunk size that fits well in cache
    const CHUNK_SIZE: usize = 64;
    
    // Process chunks for better cache locality
    for chunk in vec.chunks(CHUNK_SIZE) {
        let mut local_sums = Vec::with_capacity(chunk.len());
        let mut local_sum = 0.0;
        
        if use_avx {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            unsafe {
                // Process 4 elements at a time using AVX
                const LANE_WIDTH: usize = 4;
                let chunk_len = chunk.len();
                let simd_iterations = chunk_len / LANE_WIDTH;
                
                for i in 0..simd_iterations {
                    // Load 4 consecutive values
                    let values = _mm256_loadu_pd(&chunk[i * LANE_WIDTH]);
                    let mut arr = [0.0; LANE_WIDTH];
                    _mm256_storeu_pd(arr.as_mut_ptr(), values);
                    
                    // Use the helper to compute the prefix sum of these 4 values
                    let prefix = simd_prefix_sum(&arr, &mut local_sum);
                    local_sums.extend_from_slice(&prefix);
                }
                
                // Handle remaining elements
                let remainder_start = simd_iterations * LANE_WIDTH;
                for i in remainder_start..chunk_len {
                    local_sum += chunk[i];
                    local_sums.push(running_sum + local_sum);
                }
            }
        } else {
            // Scalar fallback implementation
            for &val in chunk {
                local_sum += val;
                local_sums.push(running_sum + local_sum);
            }
        }
        
        running_sum += local_sum;
        result.extend(local_sums);
    }

    result
}

/// Calculate the alpha and beta values for histogram equalization.
#[pyfunction]
fn calc_alpha_beta(hist: Vec<f64>) -> PyResult<(f64, f64)> {
    if hist.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Histogram is empty - cannot calculate alpha/beta values"
        ));
    }

    for &x in &hist {
        if x < 0.0 || !x.is_finite() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Histogram contains negative or invalid values"
            ));
        }
    }

    let accumulator = cumsum(&hist);

    let total = match accumulator.last() {
        Some(&v) => v,
        None => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Failed to compute cumulative histogram"
        )),
    };

    if total <= 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Histogram sum is zero or negative - cannot perform clipping"
        ));
    }

    const CLIP_PERCENTAGE: f64 = 0.005;
    let clip_hist_percent = total * CLIP_PERCENTAGE;
    let max_limit = total - clip_hist_percent;

    let min_gray = match accumulator.binary_search_by(|&x| {
        if x <= clip_hist_percent {
            std::cmp::Ordering::Less
        } else {
            std::cmp::Ordering::Greater
        }
    }) {
        Ok(pos) => pos,
        Err(pos) => pos,
    };

    let mut max_gray = match accumulator.binary_search_by(|&x| {
        if x <= max_limit {
            std::cmp::Ordering::Less
        } else {
            std::cmp::Ordering::Greater
        }
    }) {
        Ok(pos) => pos,
        Err(pos) => pos,
    };

    if max_gray == 0 {
        max_gray = 255;
    }

    if max_gray <= min_gray {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Invalid histogram range: max_gray ({}) <= min_gray ({}). This suggests the image has insufficient contrast.",
                    max_gray, min_gray)
        ));
    }

    let alpha = 255.0 / (max_gray as f64 - min_gray as f64);
    let beta = -alpha * min_gray as f64;

    if !alpha.is_finite() || !beta.is_finite() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Calculated non-finite values: alpha={}, beta={}. Check input histogram.", alpha, beta)
        ));
    }

    Ok((alpha, beta))
}

/// Generates a gamma correction lookup table for intensity values from 0 to 255.
#[pyfunction]
fn gamma(gamma_value: f64, py: Python<'_>) -> PyObject {
    let mut lookup = Array1::<u8>::zeros(256);
    
    if gamma_value <= 1.0 {
        for i in 0..256 {
            lookup[i] = i as u8;
        }
    } else {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let use_avx = is_avx2_supported();
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        let use_avx = false;
        
        let inv_gamma = 1.0 / gamma_value;
        let scale = 255.0;
        
        if use_avx {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            unsafe {
                const LANE_WIDTH: usize = 4;
                let scale_vector = _mm256_set1_pd(scale);
                
                for i in (0..256).step_by(LANE_WIDTH) {
                    if i + LANE_WIDTH <= 256 {
                        let indices = _mm256_set_pd(
                            (i + 3) as f64,
                            (i + 2) as f64,
                            (i + 1) as f64,
                            i as f64,
                        );
                        
                        let normalized = _mm256_div_pd(indices, scale_vector);
                        
                        let mut result = [0.0; LANE_WIDTH];
                        _mm256_storeu_pd(result.as_mut_ptr(), normalized);
                        
                        for j in 0..LANE_WIDTH {
                            let val = (result[j].powf(inv_gamma) * scale) as u8;
                            lookup[i + j] = val;
                        }
                    } else {
                        for j in i..256 {
                            let normalized = (j as f64) / scale;
                            let val = (normalized.powf(inv_gamma) * scale) as u8;
                            lookup[j] = val;
                        }
                    }
                }
            }
        } else {
            for i in 0..256 {
                let normalized = (i as f64) / scale;
                let val = (normalized.powf(inv_gamma) * scale) as u8;
                lookup[i] = val;
            }
        }
    }
    
    lookup.into_pyarray(py).into()
}

/// Calculates output dimensions and scaling factor for resizing.
#[pyfunction]
fn calculate_dimensions(height: i32, width: i32, target_height: i32) -> (i32, f64) {
    let scaling_factor = target_height as f64 / height as f64;
    let new_width = (width as f64 * scaling_factor) as i32;
    (new_width, scaling_factor)
}

// Helper function calculate_mean_center (remains the same) ---
fn calculate_mean_center(arr: &PyReadonlyArray2<f64>) -> PyResult<Array1<f64>> {
    let arr_view = arr.as_array(); // Get ndarray view without copying
    if arr_view.shape()[0] == 0 {
        // Raise Python ValueError if input array is empty
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Input landmark array cannot be empty",
        ));
    }
    // Calculate mean along axis 0 (columns). Returns Option<Array1<f64>>
    arr_view.mean_axis(Axis(0))
        // Convert Option to Result<_, PyErr>
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err(
            "Failed to calculate mean axis, check array validity"
        ))
}


#[pyfunction]
// Change return signature back to PyObject for the first element
fn get_rotation_matrix(
    py: Python<'_>,
    left_eye_landmarks: PyReadonlyArray2<f64>,
    right_eye_landmarks: PyReadonlyArray2<f64>,
    scale_factor: f64,
) -> PyResult<(PyObject, f64)> { // Return PyObject and f64

    // --- Input validation ---
    let left_view = left_eye_landmarks.as_array();
    let right_view = right_eye_landmarks.as_array();
    if left_view.shape().get(1) != Some(&2) || right_view.shape().get(1) != Some(&2) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Landmark arrays must have shape (N, 2)",
        ));
    }
    // calculate_mean_center handles N=0 check below

    // --- Calculations (Corrected logic) ---
    let left_eye_center = calculate_mean_center(&left_eye_landmarks)?;
    let right_eye_center = calculate_mean_center(&right_eye_landmarks)?;

    let left_eye_x = left_eye_center[0];
    let left_eye_y = left_eye_center[1];
    let right_eye_x = right_eye_center[0];
    let right_eye_y = right_eye_center[1];

    // Angle calculation
    let dy = right_eye_y - left_eye_y;
    let dx = right_eye_x - left_eye_x;
    let angle = dy.atan2(dx).to_degrees();

    // Center calculation (midpoint of eye centers)
    let center_x_unscaled = (left_eye_x + right_eye_x) / 2.0;
    let center_y_unscaled = (left_eye_y + right_eye_y) / 2.0;

    // Apply scaling
    let center_x = if scale_factor > 1.0 { center_x_unscaled * scale_factor } else { center_x_unscaled };
    let center_y = if scale_factor > 1.0 { center_y_unscaled * scale_factor } else { center_y_unscaled };

    // Create ndarray for center point
    let center_point = PyTuple::new(py, [
        center_x.round() as i32,
        center_y.round() as i32,
    ])?;
    Ok((
        center_point.into(),
        angle
    ))
}

/// Computes the desired crop width/height based on detected face size and desired output.
#[inline]
fn compute_cropped_lengths(
    rect: &Rectangle,
    output_width: u32,
    output_height: u32,
    percent_face: u32
) -> (f64, f64) {
    let inv_percentage = 100.0 / percent_face as f64;
    let face_width = rect.width * inv_percentage;
    let face_height = rect.height * inv_percentage;

    if output_height >= output_width {
        let scaled_width = (output_width as f64 * face_height) / output_height as f64;
        (scaled_width, face_height)
    } else {
        let scaled_height = (output_height as f64 * face_width) / output_width as f64;
        (face_width, scaled_height)
    }
}

/// Calculates the final bounding box coordinates given a face rectangle and cropping parameters.
#[inline]
fn compute_edges(
    rect: &Rectangle, 
    cropped_width: f64, 
    cropped_height: f64, 
    top: u32, 
    bottom: u32, 
    left: u32, 
    right: u32
) -> BoxCoordinates {
    let left_offset = (left as f64) * 0.01 * cropped_width;
    let right_offset = (right as f64) * 0.01 * cropped_width;
    let top_offset = (top as f64) * 0.01 * cropped_height;
    let bottom_offset = (bottom as f64) * 0.01 * cropped_height;

    let half_width_diff = (rect.width - cropped_width) * 0.5;
    let half_width_sum = (rect.width + cropped_width) * 0.5;
    let half_height_diff = (rect.height - cropped_height) * 0.5;
    let half_height_sum = (rect.height + cropped_height) * 0.5;

    let left_edge = rect.x + half_width_diff - left_offset;
    let top_edge = rect.y + half_height_diff - top_offset;
    let right_edge = rect.x + half_width_sum + right_offset;
    let bottom_edge = rect.y + half_height_sum + bottom_offset;

    (
        left_edge.round() as i32,
        top_edge.round() as i32,
        right_edge.round() as i32,
        bottom_edge.round() as i32,
    )
}

/// Calculate the crop positions based on face detection.
#[pyfunction]
fn crop_positions(
    x_loc: f64,
    y_loc: f64,
    width_dim: f64,
    height_dim: f64,
    percent_face: u32,
    output_width: u32,
    output_height: u32,
    top: u32,
    bottom: u32,
    left: u32,
    right: u32,
) -> Option<BoxCoordinates> {
    if percent_face == 0 || percent_face > 100 || output_width == 0 || output_height == 0 {
        return None;
    }

    let rect = Rectangle {
        x: x_loc,
        y: y_loc,
        width: width_dim,
        height: height_dim,
    };

    let (cropped_width, cropped_height) = compute_cropped_lengths(&rect, output_width, output_height, percent_face);
    Some(compute_edges(&rect, cropped_width, cropped_height, top, bottom, left, right))
}

/// Module definition
#[pymodule]
fn autocrop_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register all functions at once
    m.add_function(wrap_pyfunction!(crop_positions, m)?)?;
    m.add_function(wrap_pyfunction!(calc_alpha_beta, m)?)?;
    m.add_function(wrap_pyfunction!(gamma, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_dimensions, m)?)?;
    m.add_function(wrap_pyfunction!(get_rotation_matrix, m)?)?;
    
    Ok(())
}
