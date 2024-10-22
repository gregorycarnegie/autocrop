use pyo3::prelude::*;

#[derive(Debug, Clone, Copy)]
struct Rectangle {
    x: f64,
    y: f64,
    width: f64,
    height: f64,
}

type BoxCoordinates = (i32, i32, i32, i32);

fn cumsum(vec: &[f64]) -> Vec<f64> {
    vec.iter()
        .scan(0.0, |state, &x| {
            *state += x;
            Some(*state)
        })
        .collect()
}

#[pyfunction]
/// Calculate the alpha and beta values for histogram equalization.
///
/// Args:
///     hist (List[float]): The histogram data.
///
/// Returns:
///     Tuple[float, float]: A tuple containing the alpha and beta values.
fn calc_alpha_beta(hist: Vec<f64>) -> PyResult<(f64, f64)> {
    if hist.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Histogram is empty"));
    }
    // Cumulative distribution from the histogram
    let accumulator = cumsum(&hist);
    let total = *accumulator.last().unwrap();

    // Locate points to clip
    let clip_hist_percent = total * 0.005;

    // Find minimum_gray
    let minimum_gray = match accumulator.iter().position(|&x| x > clip_hist_percent) {
        Some(pos) => pos,
        None => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Cannot find minimum gray value"));
        }
    };

    // Find maximum_gray
    let max_limit = total - clip_hist_percent;
    let mut maximum_gray = match accumulator.iter().rposition(|&x| x >= max_limit) {
        Some(pos) => pos,
        None => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Cannot find maximum gray value"));
        }
    };

    if maximum_gray == 0 {
        maximum_gray = 255;
    }

    if maximum_gray <= minimum_gray {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid histogram values"));
    }

    // Calculate alpha
    let alpha = 255.0 / (maximum_gray as f64 - minimum_gray as f64);

    // Calculate beta
    let beta = -alpha * minimum_gray as f64;

    Ok((alpha, beta))
}

fn cropped_lengths(rect: &Rectangle, output_width: u32, output_height: u32, percent_face: u32) -> (f64, f64) {
    let inv_percentage = 100.0 / percent_face as f64;
    let crop_size = (rect.width * inv_percentage, rect.height * inv_percentage);

    if output_height >= output_width {
        let cropped_width = output_width as f64 * crop_size.1 / output_height as f64;
        (cropped_width, crop_size.1)
    } else {
        let cropped_height = output_height as f64 * crop_size.0 / output_width as f64;
        (crop_size.0, cropped_height)
    }
}

fn calculate_edge(rect: &Rectangle, cropped_width: f64, cropped_height: f64, top: u32, bottom: u32, left: u32, right: u32) -> BoxCoordinates {
    let left_edge = rect.x + (rect.width - cropped_width) * 0.5 - left as f64 * 0.01 * cropped_width;
    let top_edge = rect.y + (rect.height - cropped_height) * 0.5 - top as f64 * 0.01 * cropped_height;
    let right_edge = rect.x + (rect.width + cropped_width) * 0.5 + right as f64 * 0.01 * cropped_width;
    let bottom_edge = rect.y + (rect.height + cropped_height) * 0.5 + bottom as f64 * 0.01 * cropped_height;

    (
        left_edge.round() as i32,
        top_edge.round() as i32,
        right_edge.round() as i32,
        bottom_edge.round() as i32,
    )
}

#[pyfunction]
/// Calculate the crop positions based on face detection.
///
/// Args:
///     x_loc (float): The x-coordinate of the detected face.
///     y_loc (float): The y-coordinate of the detected face.
///     width_dim (float): The width of the detected face.
///     height_dim (float): The height of the detected face.
///     percent_face (int): The percentage of the face to include in the crop.
///     output_width (int): The desired output width.
///     output_height (int): The desired output height.
///     top (int): The top padding percentage.
///     bottom (int): The bottom padding percentage.
///     left (int): The left padding percentage.
///     right (int): The right padding percentage.
///
/// Returns:
///     Optional[Tuple[int, int, int, int]]: The coordinates of the crop box, or None if inputs are invalid.
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

    let (cropped_width, cropped_height) = cropped_lengths(&rect, output_width, output_height, percent_face);
    Some(calculate_edge(&rect, cropped_width, cropped_height, top, bottom, left, right))
}

/// A Python module implemented in Rust.
#[pymodule]
fn autocrop_rs<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(crop_positions, m)?)?;
    m.add_function(wrap_pyfunction!(calc_alpha_beta, m)?)?;
    Ok(())
}
