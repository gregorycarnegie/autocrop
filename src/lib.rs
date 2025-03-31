use ndarray::{Array1, Array2, Axis};
use numpy::IntoPyArray;
use pyo3::prelude::*;

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

/// Compute an optimized cumulative sum of a slice of f64 values.
fn cumsum(vec: &[f64]) -> Vec<f64> {
    if vec.is_empty() {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(vec.len());
    let mut running_sum = 0.0;

    // Use a chunk size that fits well in cache
    const CHUNK_SIZE: usize = 64;

    // Process chunks for better cache locality
    for chunk in vec.chunks(CHUNK_SIZE) {
        // Use a local accumulator for the chunk
        let mut local_sums = Vec::with_capacity(chunk.len());
        let mut local_sum = 0.0;
        
        // Calculate partial sums within the chunk
        for &val in chunk {
            local_sum += val;
            local_sums.push(running_sum + local_sum);
        }
        
        // Update the running sum and extend the result
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

    // Check for invalid histogram values - use early return pattern
    for &x in &hist {
        if x < 0.0 || !x.is_finite() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Histogram contains negative or invalid values"
            ));
        }
    }

    // Use the optimized cumsum function
    let accumulator = cumsum(&hist);

    let total = match accumulator.last() {
        Some(&v) => v,
        None => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Failed to compute cumulative histogram"
        )),
    };

    // Check for zero total (degenerate histogram)
    if total <= 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Histogram sum is zero or negative - cannot perform clipping"
        ));
    }

    // 0.5% clipping
    const CLIP_PERCENTAGE: f64 = 0.005;
    let clip_hist_percent = total * CLIP_PERCENTAGE;
    let max_limit = total - clip_hist_percent;

    // Find min_gray using binary search for better performance
    let min_gray = match accumulator.binary_search_by(|&x| {
        if x <= clip_hist_percent {
            std::cmp::Ordering::Less
        } else {
            std::cmp::Ordering::Greater
        }
    }) {
        Ok(pos) => pos,
        Err(pos) => pos, // binary_search_by returns insertion point if not found
    };

    // Find max_gray using binary search
    let mut max_gray = match accumulator.binary_search_by(|&x| {
        if x <= max_limit {
            std::cmp::Ordering::Less
        } else {
            std::cmp::Ordering::Greater
        }
    }) {
        Ok(pos) => pos,
        Err(pos) => pos, // binary_search_by returns insertion point if not found
    };

    // Handle edge case
    if max_gray == 0 {
        max_gray = 255;
    }

    if max_gray <= min_gray {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Invalid histogram range: max_gray ({}) <= min_gray ({}). This suggests the image has insufficient contrast.",
                    max_gray, min_gray)
        ));
    }

    // Calculate alpha and beta
    let alpha = 255.0 / (max_gray as f64 - min_gray as f64);
    let beta = -alpha * min_gray as f64;

    // Final validation
    if !alpha.is_finite() || !beta.is_finite() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Calculated non-finite values: alpha={}, beta={}. Check input histogram.", alpha, beta)
        ));
    }

    Ok((alpha, beta))
}

/// Generates a gamma correction lookup table for intensity values from 0 to 255.
/// Replaces the numba.njit gamma function from Python.
#[pyfunction]
fn gamma(gamma_value: f64, py: Python<'_>) -> PyObject {
    let mut lookup = Array1::<u8>::zeros(256);
    
    if gamma_value <= 1.0 {
        // If gamma <= 1.0, simply return a linear array from 0 to 255
        for i in 0..256 {
            lookup[i] = i as u8;
        }
    } else {
        // Precalculate 1.0/gamma_value and 255.0 to avoid repeated division
        let inv_gamma = 1.0 / gamma_value;
        let scale = 255.0;
        
        // Optimize the scalar version
        for i in 0..256 {
            // Avoid division by using precomputed value
            let normalized = (i as f64) / scale;
            let val = (normalized.powf(inv_gamma) * scale) as u8;
            lookup[i] = val;
        }
    }
    
    lookup.into_pyarray(py).into()
}

/// Calculates output dimensions and scaling factor for resizing.
/// Replaces the numba.njit calculate_dimensions function from Python.
#[pyfunction]
fn calculate_dimensions(height: i32, width: i32, target_height: i32) -> (i32, f64) {
    let scaling_factor = target_height as f64 / height as f64;
    let new_width = (width as f64 * scaling_factor) as i32;
    (new_width, scaling_factor)
}

/// Computes the mean of a 2D array along axis 0, returning a 1D array.
/// Replaces the numba.njit mean_axis0 function from Python.
fn mean_axis0(array: &Array2<f64>) -> Array1<f64> {
    // Use ndarray's built-in mean function which is already optimized
    array.mean_axis(Axis(0)).unwrap_or(Array1::<f64>::zeros(0))
}

/// Concatenates two 2D arrays vertically.
/// Replaces the numba.njit concat_rows function from Python.
fn concat_rows(array1: &Array2<f64>, array2: &Array2<f64>) -> Array2<f64> {
    // Check that the arrays have compatible shapes
    let rows1 = array1.shape()[0];
    let rows2 = array2.shape()[0];
    let cols = array1.shape()[1];
    
    // Create a new array with enough space for both inputs
    let mut result = Array2::<f64>::zeros((rows1 + rows2, cols));
    
    // Use slice assignment for better performance
    result.slice_mut(ndarray::s![0..rows1, ..]).assign(array1);
    result.slice_mut(ndarray::s![rows1.., ..]).assign(array2);
    
    result
}

/// Computes the rotation matrix parameters for face alignment.
/// Replaces the numba.njit get_rotation_matrix function from Python.
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
    
    // Create arrays with predetermined capacity
    let mut left_arr = Array2::<f64>::zeros((num_left_landmarks, 2));
    let mut right_arr = Array2::<f64>::zeros((num_right_landmarks, 2));
    
    // Fill arrays in a single pass
    for i in 0..num_left_landmarks {
        left_arr[[i, 0]] = left_eye_landmarks_x[i];
        left_arr[[i, 1]] = left_eye_landmarks_y[i];
    }
    
    for i in 0..num_right_landmarks {
        right_arr[[i, 0]] = right_eye_landmarks_x[i];
        right_arr[[i, 1]] = right_eye_landmarks_y[i];
    }
    
    // Calculate eye centers using ndarray's optimized mean function
    let left_eye_center = mean_axis0(&left_arr);
    let right_eye_center = mean_axis0(&right_arr);
    
    // Extract coordinates directly
    let left_eye_x = left_eye_center[0];
    let left_eye_y = left_eye_center[1];
    let right_eye_x = right_eye_center[0];
    let right_eye_y = right_eye_center[1];
    
    // Calculate angle
    let dy = left_eye_y - right_eye_y;
    let dx = left_eye_x - right_eye_x;
    let angle = dy.atan2(dx) * 180.0 / std::f64::consts::PI;
    
    // Concatenate arrays for both eyes using ndarray's stack
    let both_eyes = concat_rows(&left_arr, &right_arr);
    
    // Calculate face center
    let face_center = mean_axis0(&both_eyes);
    
    // Adjust for scaling - only multiply if needed
    let face_center_x = if scale_factor > 1.0 { face_center[0] * scale_factor } else { face_center[0] };
    let face_center_y = if scale_factor > 1.0 { face_center[1] * scale_factor } else { face_center[1] };
    
    // Create center as integers for rotation matrix
    let center = Array1::<i32>::from_vec(vec![
        face_center_x.round() as i32,
        face_center_y.round() as i32,
    ]);
    
    (center.into_pyarray(py).into(), angle)
}

/// Computes the desired crop width/height based on detected face size and desired output.
///
/// # Arguments
/// * `rect` - The bounding rectangle for the detected face.
/// * `output_width` - The target output width.
/// * `output_height` - The target output height.
/// * `percent_face` - The percentage of the face to include in the final crop.
///
/// # Returns
/// * `(cropped_width, cropped_height)` - The float dimensions of the cropped region.
#[inline]
fn compute_cropped_lengths(
    rect: &Rectangle,
    output_width: u32,
    output_height: u32,
    percent_face: u32
) -> (f64, f64) {
    // Precompute inverse percentage once
    let inv_percentage = 100.0 / percent_face as f64;
    let face_width = rect.width * inv_percentage;
    let face_height = rect.height * inv_percentage;

    // Avoid branch if possible by using math
    if output_height >= output_width {
        let scaled_width = (output_width as f64 * face_height) / output_height as f64;
        (scaled_width, face_height)
    } else {
        let scaled_height = (output_height as f64 * face_width) / output_width as f64;
        (face_width, scaled_height)
    }
}

/// Calculates the final bounding box coordinates given a face rectangle, cropped dimensions,
/// and top/bottom/left/right padding (as percentages).
///
/// # Arguments
/// * `rect` - The bounding rectangle of the detected face.
/// * `cropped_width` - The width of the intended cropped region.
/// * `cropped_height` - The height of the intended cropped region.
/// * `top` - The top padding percentage.
/// * `bottom` - The bottom padding percentage.
/// * `left` - The left padding percentage.
/// * `right` - The right padding percentage.
///
/// # Returns
/// * `(left_edge, top_edge, right_edge, bottom_edge)` - as integer coordinates.
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
    // Convert percentages to offsets in a single step
    let left_offset = (left as f64) * 0.01 * cropped_width;
    let right_offset = (right as f64) * 0.01 * cropped_width;
    let top_offset = (top as f64) * 0.01 * cropped_height;
    let bottom_offset = (bottom as f64) * 0.01 * cropped_height;

    // Precompute common expressions
    let half_width_diff = (rect.width - cropped_width) * 0.5;
    let half_width_sum = (rect.width + cropped_width) * 0.5;
    let half_height_diff = (rect.height - cropped_height) * 0.5;
    let half_height_sum = (rect.height + cropped_height) * 0.5;

    // Calculate edges
    let left_edge = rect.x + half_width_diff - left_offset;
    let top_edge = rect.y + half_height_diff - top_offset;
    let right_edge = rect.x + half_width_sum + right_offset;
    let bottom_edge = rect.y + half_height_sum + bottom_offset;

    // Round and convert to i32 in one step
    (
        left_edge.round() as i32,
        top_edge.round() as i32,
        right_edge.round() as i32,
        bottom_edge.round() as i32,
    )
}

/// Calculate the crop positions based on face detection.
///
/// # Arguments
/// * `x_loc` - The x-coordinate of the detected face.
/// * `y_loc` - The y-coordinate of the detected face.
/// * `width_dim` - The width of the detected face.
/// * `height_dim` - The height of the detected face.
/// * `percent_face` - The percentage of the face to include in the crop.
/// * `output_width` - The desired output width.
/// * `output_height` - The desired output height.
/// * `top` - The top padding percentage.
/// * `bottom` - The bottom padding percentage.
/// * `left` - The left padding percentage.
/// * `right` - The right padding percentage.
///
/// # Returns
/// * `Option<(i32, i32, i32, i32)>` - The coordinates of the crop box, or `None` if inputs are invalid.
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
    // Fast path check for invalid inputs
    if percent_face == 0 || percent_face > 100 || output_width == 0 || output_height == 0 {
        return None;
    }

    // Create rectangle struct
    let rect = Rectangle {
        x: x_loc,
        y: y_loc,
        width: width_dim,
        height: height_dim,
    };

    // Compute dimensions and edges
    let (cropped_width, cropped_height) = compute_cropped_lengths(&rect, output_width, output_height, percent_face);
    Some(compute_edges(&rect, cropped_width, cropped_height, top, bottom, left, right))
}

// Module definition
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