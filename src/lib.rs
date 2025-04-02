use ndarray::{Array1, Array2};
use numpy::IntoPyArray;
use pyo3::prelude::*;

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

/// Helper function to check if AVX2 is supported
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
fn is_avx2_supported() -> bool {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::__cpuid;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::__cpuid;

    unsafe {
        let info = __cpuid(7);
        ((info.ebx >> 5) & 1) != 0
    }
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

/// Computes the mean of a 2D array along axis 0, returning a 1D array.
fn mean_axis0(array: &Array2<f64>) -> Array1<f64> {
    if array.is_empty() {
        return Array1::<f64>::zeros(0);
    }
    
    let rows = array.shape()[0];
    let cols = array.shape()[1];
    
    if rows == 0 {
        return Array1::<f64>::zeros(cols);
    }
    
    let mut result = Array1::<f64>::zeros(cols);
    
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let use_avx = is_avx2_supported();
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    let use_avx = false;
    
    if use_avx && rows >= 4 {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            const LANE_WIDTH: usize = 4;
            
            for col in 0..cols {
                let mut sum = 0.0;
                let mut i = 0;
                
                while i + LANE_WIDTH <= rows {
                    let values = _mm256_set_pd(
                        array[[i + 3, col]],
                        array[[i + 2, col]],
                        array[[i + 1, col]],
                        array[[i, col]],
                    );
                    
                    let mut partial_sum = [0.0; LANE_WIDTH];
                    _mm256_storeu_pd(partial_sum.as_mut_ptr(), values);
                    
                    for &val in &partial_sum {
                        sum += val;
                    }
                    
                    i += LANE_WIDTH;
                }
                
                for j in i..rows {
                    sum += array[[j, col]];
                }
                
                result[col] = sum / rows as f64;
            }
        }
    } else {
        for col in 0..cols {
            let mut sum = 0.0;
            for row in 0..rows {
                sum += array[[row, col]];
            }
            result[col] = sum / rows as f64;
        }
    }
    
    result
}

/// Concatenates two 2D arrays vertically.
fn concat_rows(array1: &Array2<f64>, array2: &Array2<f64>) -> Array2<f64> {
    let rows1 = array1.shape()[0];
    let rows2 = array2.shape()[0];
    let cols = array1.shape()[1];
    
    let mut result = Array2::<f64>::zeros((rows1 + rows2, cols));
    
    result.slice_mut(ndarray::s![0..rows1, ..]).assign(array1);
    result.slice_mut(ndarray::s![rows1.., ..]).assign(array2);
    
    result
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
/// Helper function to load landmark coordinates using SIMD into a preallocated Array2.
/// This reduces duplication in the get_rotation_matrix function.
unsafe fn load_landmarks_simd(landmarks_x: &[f64], landmarks_y: &[f64], arr: &mut Array2<f64>) {
    const LANE_WIDTH: usize = 4;
    let n = landmarks_x.len();
    let mut i = 0;
    while i + LANE_WIDTH <= n {
        let x_vals = _mm256_loadu_pd(&landmarks_x[i]);
        let y_vals = _mm256_loadu_pd(&landmarks_y[i]);
        let mut x_arr = [0.0; LANE_WIDTH];
        let mut y_arr = [0.0; LANE_WIDTH];
        _mm256_storeu_pd(x_arr.as_mut_ptr(), x_vals);
        _mm256_storeu_pd(y_arr.as_mut_ptr(), y_vals);
        for j in 0..LANE_WIDTH {
            arr[[i + j, 0]] = x_arr[j];
            arr[[i + j, 1]] = y_arr[j];
        }
        i += LANE_WIDTH;
    }
    for j in i..n {
        arr[[j, 0]] = landmarks_x[j];
        arr[[j, 1]] = landmarks_y[j];
    }
}

/// Computes the rotation matrix parameters for face alignment.
#[pyfunction]
fn get_rotation_matrix(
    py: Python<'_>,
    left_eye_landmarks_x: Vec<f64>,
    left_eye_landmarks_y: Vec<f64>,
    right_eye_landmarks_x: Vec<f64>,
    right_eye_landmarks_y: Vec<f64>,
    scale_factor: f64
) -> (PyObject, f64) {
    let num_left_landmarks = left_eye_landmarks_x.len();
    let num_right_landmarks = right_eye_landmarks_x.len();
    
    let mut left_arr = Array2::<f64>::zeros((num_left_landmarks, 2));
    let mut right_arr = Array2::<f64>::zeros((num_right_landmarks, 2));
    
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let use_avx = is_avx2_supported() && num_left_landmarks >= 4 && num_right_landmarks >= 4;
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    let use_avx = false;
    
    if use_avx {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            load_landmarks_simd(&left_eye_landmarks_x, &left_eye_landmarks_y, &mut left_arr);
            load_landmarks_simd(&right_eye_landmarks_x, &right_eye_landmarks_y, &mut right_arr);
        }
    } else {
        for i in 0..num_left_landmarks {
            left_arr[[i, 0]] = left_eye_landmarks_x[i];
            left_arr[[i, 1]] = left_eye_landmarks_y[i];
        }
        
        for i in 0..num_right_landmarks {
            right_arr[[i, 0]] = right_eye_landmarks_x[i];
            right_arr[[i, 1]] = right_eye_landmarks_y[i];
        }
    }
    
    let left_eye_center = mean_axis0(&left_arr);
    let right_eye_center = mean_axis0(&right_arr);
    
    let left_eye_x = left_eye_center[0];
    let left_eye_y = left_eye_center[1];
    let right_eye_x = right_eye_center[0];
    let right_eye_y = right_eye_center[1];
    
    let dy = left_eye_y - right_eye_y;
    let dx = left_eye_x - right_eye_x;
    let angle = dy.atan2(dx) * 180.0 / std::f64::consts::PI;
    
    let both_eyes = concat_rows(&left_arr, &right_arr);
    let face_center = mean_axis0(&both_eyes);
    
    let face_center_x = if scale_factor > 1.0 { face_center[0] * scale_factor } else { face_center[0] };
    let face_center_y = if scale_factor > 1.0 { face_center[1] * scale_factor } else { face_center[1] };
    
    let center = Array1::<i32>::from_vec(vec![
        face_center_x.round() as i32,
        face_center_y.round() as i32,
    ]);
    
    (center.into_pyarray(py).into(), angle)
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
