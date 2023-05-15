use pyo3::prelude::*;

struct Rectangle {
    x: f64,
    y: f64,
    width: f64,
    height: f64,
}

type BoxCoordinates = (i32, i32, i32, i32);

fn cropped_lengths(rect: &Rectangle, output_width: u32, output_height: u32, percent_face: u32) -> (f64, f64) {
    let inveprcentage: f64 = 100.0 / percent_face as f64;
    let cropfactor: (f64, f64) = (rect.width as f64 * inveprcentage, rect.height as f64 * inveprcentage);
    if output_height >= output_width {
        (output_width as f64 * cropfactor.1 / output_height as f64, cropfactor.1)
    } else {
        (cropfactor.0, output_height as f64 * cropfactor.0 / output_width as f64)
    }
    
}

fn calculate_edge(rect: &Rectangle, cropped_width: f64, cropped_height: f64,  top: u32, bottom: u32, left: u32, right: u32) -> BoxCoordinates {
    ((rect.x + (rect.width - cropped_width) * 0.5 - left as f64 * 0.01 * cropped_width).round() as i32,
    (rect.y + (rect.height - cropped_height) * 0.5 - top as f64 * 0.01 * cropped_height).round() as i32,
    (rect.x + (rect.width + cropped_width) * 0.5 + right as f64 * 0.01 * cropped_width).round() as i32,
    (rect.y + (rect.height + cropped_height) * 0.5 + bottom as f64 * 0.01 * cropped_height).round() as i32)
}

#[pyfunction]
fn crop_positions(x_loc: f64, y_loc: f64, width_dim: f64, height_dim: f64, percent_face: u32, output_width: u32, output_height: u32, top: u32, bottom: u32, left: u32, right: u32) -> Option<BoxCoordinates> {
    let rect = Rectangle {
        x: x_loc,
        y: y_loc,
        width: width_dim,
        height: height_dim
    };

    if 0 < percent_face && percent_face <= 100 && output_width > 0 && output_height > 0 {
        let (cropped_width, cropped_height) = cropped_lengths(&rect, output_width, output_height, percent_face);
        Some(calculate_edge(&rect, cropped_width, cropped_height, top, bottom, left, right))
    } else {
        None
    }
}
/// A Python module implemented in Rust.
#[pymodule]
fn autocrop_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(crop_positions, m)?)?;
    Ok(())
}