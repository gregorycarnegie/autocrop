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

/// Computes the cumulative sum (prefix sums) of a slice of f64 values.
fn cumsum(vec: &[f64]) -> Vec<f64> {
    vec.iter()
        .scan(0.0, |state, &x| {
            *state += x;
            Some(*state)
        })
        .collect()
}

/// Calculate the alpha and beta values for histogram equalization.
///
/// # Arguments
/// * `hist` - The histogram data as a vector of f64.
///
/// # Returns
/// * `(alpha, beta)` - The tuple of alpha and beta values.
///
/// # Errors
/// Returns a `PyValueError` if the histogram is empty, or if no valid clip points can be found.
#[pyfunction]
fn calc_alpha_beta(hist: Vec<f64>) -> PyResult<(f64, f64)> {
    if hist.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Histogram is empty"));
    }

    // Compute cumulative distribution.
    let accumulator = cumsum(&hist);
    let total = *accumulator.last().unwrap();

    // 0.5% clipping.
    let clip_hist_percent = total * 0.005;

    // Minimum gray.
    let min_gray = accumulator
    .iter()
    .position(|&x| x > clip_hist_percent)
    .ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Cannot find minimum gray value",
        )
    })?;

    // Maximum gray.
    let max_limit = total - clip_hist_percent;
    let mut max_gray  = accumulator
    .iter()
    .rposition(|&x| x >= max_limit)
    .ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Cannot find maximum gray value",
        )
    })?;

    // Fallback if we can't find anything above 0.
    if max_gray  == 0 {
        max_gray  = 255;
    }

    if max_gray <= min_gray {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Invalid histogram values: max_gray <= min_gray",
        ));
    }

    // Calculate alpha and beta.
    let alpha = 255.0 / (max_gray  as f64 - min_gray  as f64);
    let beta = -alpha * min_gray  as f64;

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
