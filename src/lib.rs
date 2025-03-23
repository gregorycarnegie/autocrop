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

    for chunk in vec.chunks(CHUNK_SIZE) {
        // Process each chunk with a local accumulator
        let mut local_sum = 0.0;
        let mut local_results = Vec::with_capacity(chunk.len());

        for &val in chunk {
            local_sum += val;
            local_results.push(running_sum + local_sum);
        }

        // Update running sum and extend results
        running_sum += local_sum;
        result.extend(local_results);
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

    // Check for invalid histogram values
    if hist.iter().any(|&x| x < 0.0 || !x.is_finite()) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Histogram contains negative or invalid values"
        ));
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

    // Find min_gray using a more efficient approach
    let min_gray = match accumulator.iter().position(|&x| x > clip_hist_percent) {
        Some(pos) => pos,
        None => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Cannot find minimum gray value - histogram may be invalid"
        )),
    };

    // Find max_gray using a more efficient approach
    let mut max_gray = match accumulator.iter().rposition(|&x| x <= max_limit) {
        Some(pos) => pos + 1, // We want the first element greater than max_limit
        None => 0,
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
fn compute_cropped_lengths(
    rect: &Rectangle,
    output_width: u32,
    output_height: u32,
    percent_face: u32
) -> (f64, f64) {
    let inv_percentage = 100.0 / percent_face as f64;
    let face_crop  = (rect.width * inv_percentage, rect.height * inv_percentage);

    if output_height >= output_width {
        let scaled_width = output_width as f64 * face_crop .1 / output_height as f64;
        (scaled_width, face_crop .1)
    } else {
        let scaled_height = output_height as f64 * face_crop .0 / output_width as f64;
        (face_crop .0, scaled_height)
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
fn compute_edges(rect: &Rectangle, cropped_width: f64, cropped_height: f64, top: u32, bottom: u32, left: u32, right: u32) -> BoxCoordinates {
    // Convert top/bottom/left/right into fractional offsets.
    let left_offset = left as f64 * 0.01 * cropped_width;
    let right_offset = right as f64 * 0.01 * cropped_width;
    let top_offset = top as f64 * 0.01 * cropped_height;
    let bottom_offset = bottom as f64 * 0.01 * cropped_height;

    let left_edge = rect.x + (rect.width - cropped_width) * 0.5 - left_offset;
    let top_edge = rect.y + (rect.height - cropped_height) * 0.5 - top_offset;
    let right_edge = rect.x + (rect.width + cropped_width) * 0.5 + right_offset;
    let bottom_edge = rect.y + (rect.height + cropped_height) * 0.5 + bottom_offset;

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

/// A Python module implemented in Rust.
#[pymodule]
fn autocrop_rs<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(crop_positions, m)?)?;
    m.add_function(wrap_pyfunction!(calc_alpha_beta, m)?)?;
    Ok(())
}
