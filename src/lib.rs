use ndarray::{Array1, Array2, Array3, ArrayView2, ArrayView3, Axis, Ix3, s, ShapeError, Zip};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::{prelude::*, exceptions::PyValueError, types::PyBytes};
use std::sync::atomic::{AtomicBool, Ordering};

// For x86/x86_64 specific SIMD intrinsics
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::{
    // _mm256_set1_pd, _mm256_storeu_pd, _mm256_loadu_pd, 
    _mm256_set1_pd, _mm256_storeu_pd,
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

/// Compute the cumulative sum of a slice of f64 values. (Corrected, simple version)
fn cumsum(vec: &[f64]) -> Vec<f64> {
    if vec.is_empty() {
        return Vec::new();
    }
    let mut accumulator = 0.0;
    
    vec.iter().map(|&x| {
        accumulator += x; // Add current element to the running total
        accumulator     // Return the updated running total for this position
    }).collect::<Vec<f64>>() // Collect results into a new Vec
}

/// Reshapes a raw byte buffer (from Python) into a 3D NumPy array (H, W, 3).
///
/// Args:
///     input_bytes: A Python bytes object containing raw pixel data (e.g., RGBRGB...).
///     height: The desired height of the output array.
///     width: The desired width of the output array.
///
/// Returns:
///     A NumPy array with shape (height, width, 3) and dtype uint8.
///
/// Raises:
///     ValueError: If the input buffer length doesn't match height * width * 3.
#[pyfunction]
fn reshape_buffer_to_image<'py>(
    py: Python<'py>,
    input_bytes: Bound<'py, PyBytes>,
    height: usize,
    width: usize,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let bytes_slice: &[u8] = input_bytes.as_bytes();

    let channels = 3;
    let expected_len = height * width * channels;

    if bytes_slice.len() != expected_len {
        return Err(PyValueError::new_err(format!(
            "Input buffer length ({}) does not match expected length ({}) for shape ({}, {}, {})",
            bytes_slice.len(),
            expected_len,
            height,
            width,
            channels
        )));
    }

    let shape = Ix3(height, width, channels);

    let array_view: ArrayView3<'_, u8> =
        ArrayView3::from_shape(shape, bytes_slice)
        .map_err(|e: ShapeError| {
            PyValueError::new_err(format!("Failed to reshape buffer: {}", e))
        })?;

    let owned_array: Array3<u8> = array_view.to_owned();
    Ok(owned_array.into_pyarray(py))
}

/// Internal: Convert BGR image view to grayscale array.
fn bgr_to_gray(
    image_view: ArrayView3<u8>,
    use_rec709: bool, // True for Rec. 709, False for Rec. 601
) -> Array2<u8> {
    let shape = image_view.shape();
    let height = shape[0];
    let width = shape[1];
    let mut gray = Array2::<u8>::zeros((height, width));

    // Select coefficients
    let (r_coeff, g_coeff, b_coeff) = if use_rec709 {
        (0.2126, 0.7152, 0.0722) // Rec. 709
    } else {
        (0.299, 0.587, 0.114) // Rec. 601
    };

    // Manual iteration or slicing is appropriate. Using slicing from original:
    let b_channel = image_view.slice(s![.., .., 0]);
    let g_channel = image_view.slice(s![.., .., 1]);
    let r_channel = image_view.slice(s![.., .., 2]);

    for i in 0..height {
        for j in 0..width {
            let b = b_channel[[i, j]] as f32;
            let g = g_channel[[i, j]] as f32;
            let r = r_channel[[i, j]] as f32;
            let gray_val = (r_coeff * r + g_coeff * g + b_coeff * b).round().clamp(0.0, 255.0) as u8; // Added clamp
            gray[[i, j]] = gray_val;
        }
    }
    gray
}

/// Internal: Calculate 256-bin histogram from grayscale view.
fn calc_histogram(gray_view: ArrayView2<u8>) -> [f64; 256] {
    // Based on calc_histogram logic from lib.txt
    let mut hist: [f64; 256] = [0.0; 256];
    for &pixel_value in gray_view.iter() {
        hist[pixel_value as usize] += 1.0;
    }
    hist
}

/// Internal: Calculate alpha and beta values for histogram equalization.
/// Returns Option<(f64, f64)>: Some on success, None on failure (e.g., invalid histogram).
fn calc_alpha_beta(hist: &[f64]) -> Option<(f64, f64)> {
    if hist.iter().all(|&x| x == 0.0) || hist.is_empty() { // Check for empty or all-zero hist early
         return None; // Cannot proceed
    }
    if hist.iter().any(|&x| x < 0.0 || !x.is_finite()) { //
        return None; // Invalid values
    }

    let accumulator = cumsum(hist); // Assumes cumsum is available as a non-py function
    let total = match accumulator.last() {
        Some(&v) if v > 0.0 => v,
        _ => return None, // Failed to compute or sum is non-positive
    };

    const CLIP_PERCENTAGE: f64 = 0.005;
    let clip_hist_percent = total * CLIP_PERCENTAGE;
    let max_limit = total - clip_hist_percent;

    // Use position instead of binary_search for clarity if accumulator is monotonic
    let min_gray = accumulator.iter().position(|&x| x > clip_hist_percent).unwrap_or(0);
    let mut max_gray = accumulator.iter().position(|&x| x > max_limit).unwrap_or(hist.len() -1);

    if max_gray == 0 && hist.len() > 1 { // Handle edge case where max_limit is very high
        max_gray = hist.len() - 1; // Use the last bin index
    }
     if max_gray == 0 { max_gray = 1 } // Prevent division by zero if only one bin exists

    if max_gray <= min_gray {
        return None; // Invalid range
    }

    let alpha = 255.0 / (max_gray as f64 - min_gray as f64);
    let beta = -alpha * min_gray as f64;

    if !alpha.is_finite() || !beta.is_finite() {
        return None; // Invalid calculation result
    }

    Some((alpha, beta))
}

/// Internal: Apply scale and shift (convertScaleAbs logic) to an image view.
fn convert_scale_abs(
    image_view: ArrayView3<u8>,
    alpha: f64,
    beta: f64,
) -> Array3<u8> {
    // Based on convert_scale_abs logic from lib.txt
    let mut output_array: Array3<u8> = Array3::zeros(image_view.raw_dim());
    Zip::from(&mut output_array)
        .and(image_view)
        .for_each(|output_pixel: &mut u8, &input_pixel: &u8| {
            let scaled_shifted: f64 = (input_pixel as f64) * alpha + beta;
            let abs_result: f64 = scaled_shifted.abs();
            let rounded_result: f64 = abs_result.round();
            let clamped_result: f64 = rounded_result.clamp(0.0, 255.0);
            *output_pixel = clamped_result as u8;
        });
    output_array
}

/// Performs exposure correction entirely within Rust by chaining internal logic.
///
/// Args:
///     image: Input NumPy array (expects HxWx3 BGR u8).
///     exposure: Bool flag indicating whether to perform correction.
///     video: Bool flag indicating if the source is video (affects grayscale method).
///
/// Returns:
///     A new NumPy array, either the original or the corrected version.
#[pyfunction]
fn correct_exposure<'py>(
    py: Python<'py>,
    image: PyReadonlyArray3<'py, u8>,
    exposure: bool,
    video: bool,
) -> PyResult<Bound<'py, PyArray3<u8>>> {

    let input_view: ArrayView3<'_, u8> = image.as_array();

    if !exposure {
        let owned_copy: Array3<u8> = input_view.to_owned();
        return Ok(owned_copy.into_pyarray(py));
    }

    // Call internal functions
    let gray_array: Array2<u8> = bgr_to_gray(input_view, video);
    let hist: [f64; 256] = calc_histogram(gray_array.view()); // Pass view

    // Calculate alpha/beta using internal function, defaulting on failure
    let (alpha, beta) = calc_alpha_beta(&hist).unwrap_or((1.0, 0.0));

    // Apply scaling using internal function
    let final_result_array: Array3<u8> = convert_scale_abs(input_view, alpha, beta);

    // Convert final Rust ndarray back to Python NumPy array and return
    Ok(final_result_array.into_pyarray(py))
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

/// Creates a 2D rotation matrix based on eye landmark positions.
/// 
/// Args:
///   left_eye_landmarks: Array of left eye landmark coordinates with shape (N, 2)
///   right_eye_landmarks: Array of right eye landmark coordinates with shape (N, 2)
///   scale_factor: Scaling factor to apply to center coordinates
///
/// Returns:
///   A 2x3 affine transformation matrix for rotating around the midpoint of the eyes
#[pyfunction]
fn get_rotation_matrix<'py>(
    py: Python<'py>,
    left_eye_landmarks: PyReadonlyArray2<f64>,
    right_eye_landmarks: PyReadonlyArray2<f64>,
    scale_factor: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> { // Return PyObject and f64

    // --- Input validation ---
    let left_view = left_eye_landmarks.as_array();
    let right_view = right_eye_landmarks.as_array();
    if left_view.shape().get(1) != Some(&2) || right_view.shape().get(1) != Some(&2) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Landmark arrays must have shape (N, 2)",
        ));
    }
    // calculate_mean_center handles N=0 check below
    let left_eye_center = calculate_mean_center(&left_eye_landmarks)?;
    let right_eye_center = calculate_mean_center(&right_eye_landmarks)?;

    let left_eye_x = left_eye_center[0];
    let left_eye_y = left_eye_center[1];
    let right_eye_x = right_eye_center[0];
    let right_eye_y = right_eye_center[1];

    // Angle calculation
    let dy = right_eye_y - left_eye_y;
    let dx = right_eye_x - left_eye_x;
    let angle = dy.atan2(dx);

    // Center calculation (midpoint of eye centers)
    let center_x_unscaled = (left_eye_x + right_eye_x) / 2.0;
    let center_y_unscaled = (left_eye_y + right_eye_y) / 2.0;

    // Apply scaling
    let center_x = if scale_factor > 1.0 { center_x_unscaled * scale_factor } else { center_x_unscaled };
    let center_y = if scale_factor > 1.0 { center_y_unscaled * scale_factor } else { center_y_unscaled };

    //  scale of 1.0
    let alpha = angle.cos();
    let beta = angle.sin();
    
    // Calculate the matrix elements
    let m02 = (1.0 - alpha) * center_x - beta * center_y;
    let m12 = beta * center_x + (1.0 - alpha) * center_y;
    
    // Create a 2x3 matrix
    let matrix = Array2::from_shape_vec(
        (2, 3),
        vec![alpha, beta, m02, -beta, alpha, m12],
    ).map_err(|e| PyValueError::new_err(format!("Failed to create matrix: {}", e)))?;
    
    // Convert to numpy array and return
    Ok(matrix.into_pyarray(py))
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
    m.add_function(wrap_pyfunction!(crop_positions, m)?)?;
    m.add_function(wrap_pyfunction!(gamma, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_dimensions, m)?)?;
    m.add_function(wrap_pyfunction!(get_rotation_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(correct_exposure, m)?)?;
    m.add_function(wrap_pyfunction!(reshape_buffer_to_image, m)?)?;
    
    Ok(())
}
