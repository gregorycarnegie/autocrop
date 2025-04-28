use ndarray::{Array1, Array2, Array3, ArrayView2, ArrayView3, Axis, Dim, Ix3, par_azip, s};
use numpy::{IntoPyArray, PyArray, PyArray2, PyArray3, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::{prelude::*, types::PyBytes};
use pyo3::exceptions::{PyValueError, PyRuntimeError};
use std::sync::atomic::{AtomicBool, Ordering};
use rayon::prelude::*;
use rayon::current_num_threads;

// For x86/x86_64 specific SIMD intrinsics
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::{
    _mm256_set1_pd, _mm256_storeu_pd,
    _mm256_div_pd, _mm256_set_pd,
    _mm256_mul_pd, _mm256_add_pd
};

/// Alias representing a rectangle
type Rectangle = (f64, f64, f64, f64);
/// Alias for the four integer padding values.
type Padding = (u32, u32, u32, u32);
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
    
    // For small arrays, use the sequential approach
    if vec.len() < 1000 {
        let mut result = Vec::with_capacity(vec.len());
        let mut accumulator = 0.0;
        
        for &x in vec {
            accumulator += x;
            result.push(accumulator);
        }
        return result;
    }
    
    // For larger arrays, use a parallel approach
    let chunk_size = 1000;
    let num_chunks = (vec.len() + chunk_size - 1) / chunk_size; // Ceiling division
    
    // Step 1: Compute local sums for each chunk in parallel
    let local_results: Vec<(Vec<f64>, f64)> = vec.par_chunks(chunk_size)
        .map(|chunk| {
            let mut local_result = Vec::with_capacity(chunk.len());
            let mut sum = 0.0;
            
            for &val in chunk {
                sum += val;
                local_result.push(sum);
            }
            
            (local_result, sum)
        })
        .collect();
    
    // Step 2: Calculate prefix sums of chunk totals
    let mut chunk_prefixes = Vec::with_capacity(num_chunks);
    let mut prefix_sum = 0.0;
    
    for (_, chunk_sum) in &local_results {
        chunk_prefixes.push(prefix_sum);
        prefix_sum += chunk_sum;
    }
    
    // Step 3: Combine results
    let mut final_result = Vec::with_capacity(vec.len());
    
    for (chunk_idx, (local_chunk, _)) in local_results.iter().enumerate() {
        let chunk_prefix = chunk_prefixes[chunk_idx];
        
        for &local_val in local_chunk {
            final_result.push(local_val + chunk_prefix);
        }
    }
    
    final_result
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

    // Create rows in parallel
    let rows: Vec<Vec<u8>> = (0..height).into_par_iter()
        .map(|i| {
            let row_start = i * width * channels;
            let row_end = row_start + (width * channels);
            bytes_slice[row_start..row_end].to_vec()
        })
        .collect();
    
    // Flatten and convert to ndarray
    let flattened: Vec<u8> = rows.into_iter().flatten().collect();
    let shape = Ix3(height, width, channels);
    
    let owned_array = Array3::from_shape_vec(shape, flattened)
        .map_err(|e| PyValueError::new_err(format!("Failed to reshape array: {}", e)))?;
    
    Ok(owned_array.into_pyarray(py))
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn bgr_to_gray_avx2(
    image_view: ArrayView3<u8>,
    use_rec709: bool,
) -> Array2<u8> {
    let shape = image_view.shape();
    let height = shape[0];
    let width = shape[1];
    let mut gray = Array2::<u8>::zeros((height, width));

    // Select coefficients
    let (r_coeff, g_coeff, b_coeff) = if use_rec709 {
        (0.2126_f32, 0.7152_f32, 0.0722_f32)
    } else {
        (0.299_f32, 0.587_f32, 0.114_f32)
    };

    // Create AVX2 vectors for coefficients (4 doubles per vector)
    let r_vector = _mm256_set1_pd(r_coeff as f64);
    let g_vector = _mm256_set1_pd(g_coeff as f64);
    let b_vector = _mm256_set1_pd(b_coeff as f64);

    // Process rows in parallel
    par_azip!((mut gray_row in gray.axis_iter_mut(Axis(0)),
               image_row in image_view.axis_iter(Axis(0))) {
        
        let mut i = 0;
        // Process 4 pixels at a time with AVX2
        while i + 3 < width {
            // Load 4 pixels from each channel
            let mut r_vals = [0.0; 4];
            let mut g_vals = [0.0; 4];
            let mut b_vals = [0.0; 4];
            
            for j in 0..4 {
                b_vals[j] = image_row[[i + j, 0]] as f64;
                g_vals[j] = image_row[[i + j, 1]] as f64;
                r_vals[j] = image_row[[i + j, 2]] as f64;
            }
            
            // Create AVX vectors
            let r_pixels = _mm256_set_pd(r_vals[3], r_vals[2], r_vals[1], r_vals[0]);
            let g_pixels = _mm256_set_pd(g_vals[3], g_vals[2], g_vals[1], g_vals[0]);
            let b_pixels = _mm256_set_pd(b_vals[3], b_vals[2], b_vals[1], b_vals[0]);
            
            // Multiply by coefficients
            let r_contrib = _mm256_mul_pd(r_pixels, r_vector);
            let g_contrib = _mm256_mul_pd(g_pixels, g_vector);
            let b_contrib = _mm256_mul_pd(b_pixels, b_vector);
            
            // Sum contributions
            let sum1 = _mm256_add_pd(r_contrib, g_contrib);
            let gray_values = _mm256_add_pd(sum1, b_contrib);
            
            // Store results
            let mut result_array = [0.0; 4];
            _mm256_storeu_pd(result_array.as_mut_ptr(), gray_values);
            
            // Convert to u8 and store in output
            for j in 0..4 {
                gray_row[i + j] = result_array[j].round().clamp(0.0, 255.0) as u8;
            }
            
            i += 4;
        }
        
        // Handle remaining pixels with scalar code
        for j in i..width {
            let b = image_row[[j, 0]] as f32;
            let g = image_row[[j, 1]] as f32;
            let r = image_row[[j, 2]] as f32;
            let gray_val = (r_coeff * r + g_coeff * g + b_coeff * b).round().clamp(0.0, 255.0) as u8;
            gray_row[j] = gray_val;
        }
    });

    gray
}

/// Internal: Convert BGR image view to grayscale array (Parallelized with Rayon).
fn bgr_to_gray(
    image_view: ArrayView3<u8>,
    use_rec709: bool,
) -> Array2<u8> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_avx2_supported() {
            unsafe {
                return bgr_to_gray_avx2(image_view, use_rec709);
            }
        }
    }

    let shape = image_view.shape();
    let height = shape[0];
    let width = shape[1];
    let mut gray = Array2::<u8>::zeros((height, width)); // Output array (H x W)

    // Select coefficients
    let (r_coeff, g_coeff, b_coeff) = if use_rec709 {
        (0.2126_f32, 0.7152_f32, 0.0722_f32)
    } else {
        (0.299_f32, 0.587_f32, 0.114_f32)
    };

    // --- Create 2D views for each channel ---
    // Assuming BGR order: Axis 2, Index 0=B, 1=G, 2=R
    let b_channel = image_view.slice(s![.., .., 0]); // Shape (H x W)
    let g_channel = image_view.slice(s![.., .., 1]); // Shape (H x W)
    let r_channel = image_view.slice(s![.., .., 2]); // Shape (H x W)
    // ---------------------------------------

    // Now zip the 2D output array with the 2D channel views
    par_azip!((
        gray_pixel in &mut gray,
        &b in &b_channel,
        &g in &g_channel,
        &r in &r_channel
    ) {
        // These are now individual u8 values from the corresponding HxW positions
        let b = b as f32;
        let g = g as f32;
        let r = r as f32;
        let gray_val = (r_coeff * r + g_coeff * g + b_coeff * b).round().clamp(0.0, 255.0) as u8;
        *gray_pixel = gray_val;
    });

    gray
}


/// Internal: Calculate 256-bin histogram from grayscale view.
fn calc_histogram(gray_view: ArrayView2<u8>) -> [f64; 256] {
    let mut hist: [f64; 256] = [0.0; 256];
    
    // Create thread-local histograms
    let local_hists = gray_view.axis_chunks_iter(Axis(0), gray_view.shape()[0] / current_num_threads().max(1))
        .into_par_iter()
        .map(|chunk| {
            let mut local_hist = [0.0; 256];
            for &pixel_value in chunk.iter() {
                local_hist[pixel_value as usize] += 1.0;
            }
            local_hist
        })
        .collect::<Vec<[f64; 256]>>();
    
    // Combine thread-local histograms
    for local_hist in local_hists {
        for i in 0..256 {
            hist[i] += local_hist[i];
        }
    }
    
    hist
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

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn convert_scale_abs_avx2(
    image_view: ArrayView3<u8>,
    alpha: f64, 
    beta: f64,
) -> Array3<u8> {
    let shape = image_view.raw_dim();
    let mut output_array: Array3<u8> = Array3::zeros(shape.clone());
    
    let alpha_vector = _mm256_set1_pd(alpha);
    let beta_vector = _mm256_set1_pd(beta);
    
    // Process each channel of each row in parallel
    par_azip!((
        mut output_slice in output_array.axis_iter_mut(Axis(0)),
        input_slice in image_view.axis_iter(Axis(0))
    ) {
        // For each row, process each channel
        for c in 0..shape[2] {
            let mut j = 0;
            // Process 4 pixels at a time within this row and channel
            while j + 3 < shape[1] {
                // Load 4 input values
                let mut input_vals = [0.0; 4];
                for k in 0..4 {
                    input_vals[k] = input_slice[[j + k, c]] as f64;
                }
                
                // Create AVX vector
                let input_vector = _mm256_set_pd(
                    input_vals[3], input_vals[2], input_vals[1], input_vals[0]
                );
                
                // Apply scale and shift
                let scaled = _mm256_mul_pd(input_vector, alpha_vector);
                let result = _mm256_add_pd(scaled, beta_vector);
                
                // Store results
                let mut result_array = [0.0; 4];
                _mm256_storeu_pd(result_array.as_mut_ptr(), result);
                
                // Convert to u8 and store
                for k in 0..4 {
                    output_slice[[j + k, c]] = result_array[k].round().clamp(0.0, 255.0) as u8;
                }
                
                j += 4;
            }
            
            // Handle remaining values
            while j < shape[1] {
                let scaled_shifted = (input_slice[[j, c]] as f64) * alpha + beta;
                output_slice[[j, c]] = scaled_shifted.round().clamp(0.0, 255.0) as u8;
                j += 1;
            }
        }
    });
    
    output_array
}

/// Internal: Apply scale and shift (convertScaleAbs logic) to an image view (Parallelized with Rayon).
fn convert_scale_abs(
    image_view: ArrayView3<u8>,
    alpha: f64,
    beta: f64,
) -> Array3<u8> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_avx2_supported() {
            unsafe {
                return convert_scale_abs_avx2(image_view, alpha, beta);
            }
        }
    }
    let mut output_array: Array3<u8> = Array3::zeros(image_view.raw_dim());

    // Use par_azip for parallel element-wise operations
    par_azip!((output_pixel in &mut output_array, &input_pixel in &image_view) {
        let scaled_shifted: f64 = (input_pixel as f64) * alpha + beta;
        // No need for abs() if input is u8 and alpha/beta produce results clamped to positive range
        let rounded_result: f64 = scaled_shifted.round();
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

/// AVX2-accelerated gamma correction lookup table generation
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn gamma_avx2(gamma_value: f64) -> Vec<u8> {
    if gamma_value <= 1.0 {
        // For gamma ≤ 1.0, just return array with values 0-255
        return (0..=255).collect();
    }
    
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
                lookup[i + j] = (powered * scale).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    
    lookup
}

/// Python-facing gamma correction function that uses AVX2 if available
#[pyfunction]
fn gamma<'py>(gamma_value: f64, py: Python<'py>) -> PyResult<Bound<'py, PyArray<u8, Dim<[usize; 1]>>>> {
    let lookup: Vec<u8>;
    
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_avx2_supported() {
            unsafe {
                lookup = gamma_avx2(gamma_value);
                let array = Array1::from_vec(lookup);
                return Ok(array.into_pyarray(py));
            }
        }
    }
    
    // Standard implementation (fallback)
    if gamma_value <= 1.0 {
        // For gamma ≤ 1.0, just return array with values 0-255
        lookup = (0..=255).collect();
    } else {
        // For gamma > 1.0, calculate values in parallel
        let inv_gamma = 1.0 / gamma_value;
        let scale = 255.0;
        
        lookup = (0..256).into_par_iter()
            .map(|i| {
                let normalized = (i as f64) / scale;
                (normalized.powf(inv_gamma) * scale).round().clamp(0.0, 255.0) as u8
            })
            .collect();
    }
    
    let array = Array1::from_vec(lookup);
    Ok(array.into_pyarray(py))
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
        (p + dim * (0.5 + pad as f64)).round() as i32
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
