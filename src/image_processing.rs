use ndarray::{Array1, Array2, Array3, ArrayView2, ArrayView3, Axis, Ix3, par_azip, s, Zip};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::{prelude::*, types::PyBytes};
use pyo3::exceptions::{PyValueError, PyRuntimeError};
use std::convert::Into;
use rayon::prelude::*;
use rayon::current_num_threads;
use parking_lot::Mutex;

use crate::ImportablePyModuleBuilder;
use crate::dispatch_simd::dispatch_simd;

// For x86/x86_64 specific SIMD intrinsics
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::{
    _mm256_set1_pd, _mm256_set_pd, _mm256_storeu_pd,
    _mm256_div_pd
};

/// Alias representing a rectangle
type Rectangle = (f64, f64, f64, f64);
/// Alias for the four integer padding values.
type Padding = (u32, u32, u32, u32);
/// Alias for the four integer coordinates of a bounding box.
type BoxCoordinates = (i32, i32, i32, i32);

// OPTIMIZATION: Reusable buffers for grayscale image processing
thread_local! {
    static GRAY_BUFFER: Mutex<Option<Array2<u8>>> = Mutex::new(None);
}

/// Compute the cumulative sum of a slice of f64 values. (Corrected, simple version)
fn cumsum(vec: &[f64]) -> Vec<f64> {
    if vec.is_empty() {
        return Vec::new();
    }
    
    // For small arrays, use the sequential approach
    if vec.len() < 1000 {
        let mut result = vec![0.0; vec.len()];
        let mut sum = 0.0;
        
        for (i, &x) in vec.iter().enumerate() {
            sum += x;
            result[i] = sum;
        }
        return result;
    }
    
    // For larger arrays, use rayon's parallel scan
    let mut result = vec.to_vec();
    
    // Use a slice of the Vec rather than the Vec itself
    result.as_mut_slice().par_chunks_mut(1000)
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            let mut sum = if chunk_idx == 0 { 0.0 } else {
                // Start with the sum from the end of the previous chunk
                vec[..chunk_idx * 1000].iter().sum()
            };
            
            for (i, val) in chunk.iter_mut().enumerate() {
                sum += vec[chunk_idx * 1000 + i];
                *val = sum;
            }
        });
    
    result
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
            bytes_slice.len(), expected_len, height, width, channels
        )));
    }

    // OPTIMIZATION: Use from_slice for zero-copy when possible
    if bytes_slice.as_ptr().align_offset(std::mem::align_of::<u8>()) == 0 {
        // Create shape tuple
        let shape = Ix3(height, width, channels);
        
        // Create Array3 properly - using view first, then to_owned()
        let array = unsafe {
            // Create a view first
            let view = ArrayView3::from_shape_ptr(shape, bytes_slice.as_ptr());
            // Then convert to owned array
            view.to_owned()
        };
        
        // Convert to Python array
        return Ok(array.into_pyarray(py));
    }

    // If zero-copy not possible, allocate a single buffer once
    let mut output = vec![0u8; expected_len];
    
    // Copy data in parallel chunks
    let row_size = width * channels;
    output.par_chunks_mut(row_size)
          .enumerate()
          .for_each(|(i, row)| {
              let src_offset = i * row_size;
              row.copy_from_slice(&bytes_slice[src_offset..src_offset + row_size]);
          });
    
    let shape = Ix3(height, width, channels);
    let owned_array = Array3::from_shape_vec(shape, output)
        .map_err(|e| PyValueError::new_err(format!("Failed to reshape array: {}", e)))?;
    
    Ok(owned_array.into_pyarray(py))
}

/// Generic function to round, clamp, and convert any float type to u8
#[inline]
fn round_clamp_to_u8<T>(value: T) -> u8 
where
    T: Into<f64>,
{
    let float_val: f64 = value.into();
    float_val.round().clamp(0.0, 255.0) as u8
}

/// Convert BGR image view to grayscale array (Parallelized with Rayon).
pub fn bgr_to_gray(
    image_view: ArrayView3<u8>,
    use_rec709: bool,
) -> Array2<u8> {
    let shape = image_view.shape();
    let height = shape[0];
    let width = shape[1];
    
    let mut gray = GRAY_BUFFER.with(|cell| {
        let mut guard = cell.lock();
        if let Some(ref mut buf) = *guard {
            if buf.shape()[0] == height && buf.shape()[1] == width {
                // Reuse existing buffer
                buf.fill(0);
                buf.clone()
            } else {
                // Wrong size, create new
                let new_buf = Array2::<u8>::zeros((height, width));
                *guard = Some(new_buf.clone());
                new_buf
            }
        } else {
            // No buffer yet, create new
            let new_buf = Array2::<u8>::zeros((height, width));
            *guard = Some(new_buf.clone());
            new_buf
        }
    });

    // Select coefficients
    let (r_coeff, g_coeff, b_coeff) = get_rgb_coefficients(use_rec709);

    // --- Create 2D views for each channel ---
    // Assuming BGR order: Axis 2, Index 0=B, 1=G, 2=R
    let b_channel = image_view.slice(s![.., .., 0]); // Shape (H x W)
    let g_channel = image_view.slice(s![.., .., 1]); // Shape (H x W)
    let r_channel = image_view.slice(s![.., .., 2]); // Shape (H x W)

    Zip::from(&mut gray)
        .and(&b_channel)
        .and(&g_channel)
        .and(&r_channel)
        .par_for_each(|g, &b, &g_val, &r| {
            // These are now individual u8 values from the corresponding HxW positions
            let b = b as f32;
            let g_val = g_val as f32;
            let r = r as f32;
            *g = round_clamp_to_u8(r_coeff * r + g_coeff * g_val + b_coeff * b);
        });

    gray
}

fn get_rgb_coefficients(use_rec709: bool) -> (f32, f32, f32) {
    if use_rec709 {
        (0.2126_f32, 0.7152_f32, 0.0722_f32)
    } else {
        (0.299_f32, 0.587_f32, 0.114_f32)
    }
}

/// Internal: Calculate 256-bin histogram from grayscale view.
fn calc_histogram(gray_view: ArrayView2<u8>) -> [f64; 256] {
    let chunk_len = gray_view.shape()[0] / current_num_threads().max(1);

    gray_view
        .axis_chunks_iter(Axis(0), chunk_len)
        .into_par_iter()
        // each thread builds its own local [f64;256]
        .map(|chunk| {
            let mut local = [0.0; 256];
            for &pix in chunk.iter() {
                local[pix as usize] += 1.0;
            }
            local
        })
        // and then Rayon reduces all those local arrays into one
        .reduce(
            || [0.0; 256],              // identity
            |mut acc, local| {         // reduction step
                for i in 0..256 {
                    acc[i] += local[i];
                }
                acc
            },
        )
}

/// Internal: Calculate alpha and beta values for histogram equalization.
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

/// Apply scale and shift (convertScaleAbs logic) to an image view (Parallelized with Rayon).
pub fn convert_scale_abs(
    image_view: ArrayView3<u8>,
    alpha: f64,
    beta: f64,
) -> Array3<u8> {
    let mut output_array: Array3<u8> = Array3::zeros(image_view.raw_dim());

    // Use par_azip for parallel element-wise operations
    par_azip!((output_pixel in &mut output_array, &input_pixel in &image_view) {
        // No need for abs() if input is u8 and alpha/beta produce results clamped to positive range
        *output_pixel = round_clamp_to_u8((input_pixel as f64) * alpha + beta);
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
        return Ok(input_view.to_owned().into_pyarray(py));
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

/// Python-facing gamma correction function
#[pyfunction]
fn gamma<'py>(gamma_value: f64, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u8>>> {
    // OPTIMIZATION: Cache gamma lookup tables
    thread_local! {
        static GAMMA_CACHE: Mutex<Vec<(f64, Vec<u8>)>> = Mutex::new(Vec::new());
    }

    // Try to find a cached table first
    let lookup = GAMMA_CACHE.with(|cache| {
        let mut cache_guard = cache.lock();
        
        // Look for an existing table with this gamma value
        for (cached_gamma, table) in cache_guard.iter() {
            if (cached_gamma - gamma_value).abs() < 1e-6 {
                return table.clone();
            }
        }
        
        // Not found, generate a new table
        // standard_impl faster for gamma < 1.0
        let new_table = {
            if gamma_value < 1.0 {
                gamma_standard_impl(gamma_value)
            } else {
                dispatch_simd(
                    gamma_value,
                    |g| unsafe {gamma_avx2_impl(g)},
                    |g| gamma_standard_impl(g)
                )         
            }            
        };
        
        // Cache the result (up to a reasonable limit)
        if cache_guard.len() < 10 {  // Limit cache size
            cache_guard.push((gamma_value, new_table.clone()));
        }
        
        new_table
    });
    
    let array = Array1::from_vec(lookup);
    Ok(array.into_pyarray(py))
}

/// AVX2-accelerated gamma correction lookup table generation
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub unsafe fn gamma_avx2_impl(gamma_value: f64) -> Vec<u8> {
    let inv_gamma = 1.0 / gamma_value;
    let scale = 255.0;
    let scale_vec = _mm256_set1_pd(scale);
    let mut lookup = vec![0u8; 256];
    
    for i in (0..256).step_by(4) {
        // Create vector of 4 consecutive indices
        let indices = if i + 3 < 256 {
            _mm256_set_pd((i + 3) as f64, (i + 2) as f64, (i + 1) as f64, i as f64)
        } else {
            // Handle edge case for last iteration
            let mut vals = [0.0; 4];
            for j in 0..4 {
                if i + j < 256 {
                    vals[j] = (i + j) as f64;
                }
            }
            _mm256_set_pd(vals[3], vals[2], vals[1], vals[0])
        };
        
        // Normalize by dividing by scale
        let normalized = _mm256_div_pd(indices, scale_vec);
        
        // Apply power function (no AVX2 intrinsic for pow, need to store and process)
        let mut temp_values = [0.0; 4];
        _mm256_storeu_pd(temp_values.as_mut_ptr(), normalized);
        
        for j in 0..4 {
            if i + j < 256 {
                let powered = temp_values[j].powf(inv_gamma);
                lookup[i + j] = round_clamp_to_u8(powered * scale);
            }
        }
    }
    
    lookup
}

/// Python-facing gamma correction function that uses AVX2 if available
pub fn gamma_standard_impl(gamma_value: f64) -> Vec<u8> {
    if gamma_value <= 1.0 {
        // For gamma ≤ 1.0, just return array with values 0-255
        (0..=255).collect()
    } else {
        // For gamma > 1.0, calculate values in parallel
        let inv_gamma = 1.0 / gamma_value;
        let scale = 255.0;
        
        (0..256).into_par_iter()
            .map(|i| {
                let normalized = (i as f64) / scale;
                round_clamp_to_u8(normalized.powf(inv_gamma) * scale)
            })
            .collect()
    }
}

/// Calculates output dimensions and scaling factor for resizing.
#[pyfunction]
fn calculate_dimensions(height: i32, width: i32, target_height: i32) -> (i32, f64) {
    let scaling_factor = target_height as f64 / height as f64;
    let new_width = (width as f64 * scaling_factor) as i32;
    (new_width, scaling_factor)
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
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    // --- Input validation ---
    let left_view = left_eye_landmarks.as_array();
    let right_view = right_eye_landmarks.as_array();
    if left_view.shape().get(1) != Some(&2) || right_view.shape().get(1) != Some(&2) {
        return Err(PyValueError::new_err(
            "Landmark arrays must have shape (N, 2)",
        ));
    }
    
    // Convert PyReadonlyArray to owned ndarray.Array first
    let left_eye_ndarray = left_view.to_owned();
    let right_eye_ndarray = right_view.to_owned();
    
    // Now we can use Rayon's parallel computations on the owned arrays
    let (left_mean, right_mean) = rayon::join(
        || {
            if left_eye_ndarray.shape()[0] == 0 {
                return Err(PyValueError::new_err(
                    "Left eye landmark array cannot be empty",
                ));
            }
            left_eye_ndarray.mean_axis(Axis(0))
                .ok_or_else(|| PyRuntimeError::new_err(
                    "Failed to calculate mean axis for left eye"
                ))
        },
        || {
            if right_eye_ndarray.shape()[0] == 0 {
                return Err(PyValueError::new_err(
                    "Right eye landmark array cannot be empty",
                ));
            }
            right_eye_ndarray.mean_axis(Axis(0))
                .ok_or_else(|| PyRuntimeError::new_err(
                    "Failed to calculate mean axis for right eye"
                ))
        }
    );
    
    let left_eye_center = left_mean?;
    let right_eye_center = right_mean?;

    // Rest of the function remains the same
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
    face: &Rectangle,         // (x, y, width, height)
    dimensions: (u32, u32),   // (width, height)
    percent_face: u32
) -> (f64, f64) {
    let inv_percentage = 100.0 / percent_face as f64;
    let face_width = face.2 * inv_percentage;
    let face_height = face.3 * inv_percentage;

    if dimensions.1 >= dimensions.0 {
        let scaled_width = (dimensions.0 as f64 * face_height) / dimensions.1 as f64;
        (scaled_width, face_height)
    } else {
        let scaled_height = (dimensions.1 as f64 * face_width) / dimensions.0 as f64;
        (face_width, scaled_height)
    }
}

/// Calculates the final bounding box coordinates given a face rectangle and cropping parameters.
#[inline]
fn compute_edges(
    face: &Rectangle,          // (x, y, width, height)
    cropped: (f64, f64),       // (crop_w, crop_h)
    pad: Padding,              // % padding
) -> BoxCoordinates {
    let (p1, p2) = (face.0 + face.2 * 0.5, face.1 + face.3 * 0.5);
    let (crop_w, crop_h) = cropped;
    let (pad_t, pad_b, pad_l, pad_r) = pad;

    // Helper function to calculate edge with padding
    let calc_edge = |p: f64, dim: f64, pad: u32| {
        (p + dim * (0.5 + (pad as f64) / 100.0)).round() as i32
    };

    (
        calc_edge(p1, -crop_w, pad_l),
        calc_edge(p2, -crop_h, pad_t),
        calc_edge(p1, crop_w, pad_r),
        calc_edge(p2, crop_h, pad_b),
    )
}

/// Calculate the crop positions based on face detection.
#[pyfunction]
fn crop_positions(
    face: Rectangle,         // (x, y, width, height)
    face_percent: u32,
    dimensions: (u32, u32),  // (width, height)
    padding: Padding         // (top, bottom, left, right)
) -> Option<BoxCoordinates> {
    if face_percent == 0 || face_percent > 100 || dimensions.0 == 0 || dimensions.1 == 0 {
        return None;
    }

    let cropped_dimensions = compute_cropped_lengths(
        &face, 
        dimensions, 
        face_percent
    );
    
    Some(compute_edges(
        &face, 
        cropped_dimensions, 
        padding
    ))
}

/// Module initialization
// #[pymodule]
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let builder = ImportablePyModuleBuilder::from(m.clone())?;
    
    // Add all functions in a single builder chain
    builder
        .add_function(wrap_pyfunction!(crop_positions, m)?)?
        .add_function(wrap_pyfunction!(gamma, m)?)?
        .add_function(wrap_pyfunction!(calculate_dimensions, m)?)?
        .add_function(wrap_pyfunction!(get_rotation_matrix, m)?)?
        .add_function(wrap_pyfunction!(correct_exposure, m)?)?
        .add_function(wrap_pyfunction!(reshape_buffer_to_image, m)?)?;
    
    Ok(())
}