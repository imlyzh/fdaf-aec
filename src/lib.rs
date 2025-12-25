use nalgebra::{DVector, DVectorView};
use num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

/// Implements an Acoustic Echo Canceller using the Frequency Domain Adaptive Filter (FDAF)
/// algorithm with the Overlap-Save method.
///
/// This struct holds the state for the AEC and processes audio in frames.
pub struct FdafAec<const FFT_SIZE: usize> {
    fft: Arc<dyn Fft<f32>>,
    ifft: Arc<dyn Fft<f32>>,
    weights: DVector<Complex<f32>>,
    far_end_buffer: DVector<f32>,
    x_t_buffer: [Complex<f32>; FFT_SIZE],
    e_t_buffer: [Complex<f32>; FFT_SIZE],
    y_t: DVector<f32>,
    psd: DVector<f32>,
    mu: f32,
    smoothing_factor: f32,
}

impl<const FFT_SIZE: usize> FdafAec<FFT_SIZE> {
    pub const FRAME_SIZE: usize = FFT_SIZE / 2;

    /// Creates a new `FdafAec` instance.
    ///
    /// # Arguments
    ///
    /// * `fft_size`: The size of the FFT. This determines the filter length and the trade-off
    ///   between computational complexity and filter performance. A larger `fft_size` provides
    ///   a longer filter, which can cancel more delayed echoes, but increases latency and
    ///   computational cost. Must be a power of two.
    /// * `step_size`: The learning rate (mu) for the adaptive filter. It controls how fast the
    ///   filter adapts. A larger value leads to faster convergence but can be less stable.
    ///   A typical value is between 0.1 and 1.0.
    pub fn new(step_size: f32) -> Self {
        assert!(
            Self::FRAME_SIZE > 0 && Self::FRAME_SIZE.is_power_of_two(),
            "FRAME_SIZE must be a power of two."
        );
        let mut fft_planner = FftPlanner::new();
        let fft = fft_planner.plan_fft_forward(FFT_SIZE);
        let ifft = fft_planner.plan_fft_inverse(FFT_SIZE);

        Self {
            fft,
            ifft,
            weights: DVector::from_element(FFT_SIZE, Complex::new(0.0, 0.0)),
            far_end_buffer: DVector::from_element(FFT_SIZE, 0.0),
            x_t_buffer: [Complex::new(0.0, 0.0); FFT_SIZE],
            e_t_buffer: [Complex::new(0.0, 0.0); FFT_SIZE],
            psd: DVector::from_element(FFT_SIZE, 1.0), // Initialize with 1 to avoid division by zero
            y_t: DVector::zeros(FFT_SIZE),
            mu: step_size,
            smoothing_factor: 0.98,
        }
    }

    /// Processes a frame of audio data to remove echo.
    ///
    /// # Arguments
    ///
    /// * `far_end_frame`: A slice representing the audio frame from the far-end (the reference signal, e.g., loudspeaker).
    ///   Its length must be `fft_size / 2`.
    /// * `mic_frame`: A slice representing the audio frame from the near-end microphone, containing both the
    ///   near-end speaker's voice and the echo from the far-end. Its length must be `fft_size / 2`.
    ///
    /// # Returns
    ///
    /// A `Vec<f32>` containing the echo-cancelled audio frame. The length of the vector is `fft_size / 2`.
    pub fn process<const FRAME_SIZE: usize>(
        &mut self,
        error_signal: &mut [f32; FRAME_SIZE],
        far_end_frame: &[f32; FRAME_SIZE],
        mic_frame: &[f32; FRAME_SIZE],
    ) {
        assert_eq!(FRAME_SIZE, FFT_SIZE / 2);
        // 1. Update far-end buffer (shift old data, add new data)
        // This creates a rolling window of the last `fft_size` samples.
        self.far_end_buffer
            .as_mut_slice()
            .copy_within(FRAME_SIZE.., 0);
        self.far_end_buffer
            .rows_mut(FRAME_SIZE, FRAME_SIZE)
            .copy_from_slice(far_end_frame);

        // 2. FFT of the far-end signal block
        for (idx, x) in self.far_end_buffer.iter().enumerate() {
            self.x_t_buffer[idx] = Complex::new(*x, 0.0);
        }
        self.fft.process(&mut self.x_t_buffer);
        let x_f = DVectorView::from_slice(&self.x_t_buffer, FFT_SIZE);

        // 3. Update Power Spectral Density (PSD) of the far-end signal
        for i in 0..FFT_SIZE {
            let power = x_f[i].norm_sqr();
            self.psd[i] =
                self.smoothing_factor * self.psd[i] + (1.0 - self.smoothing_factor) * power;
        }

        // 4. Estimate echo in frequency domain
        let mut y_f = self.weights.component_mul(&x_f);

        // 5. Inverse FFT of the estimated echo
        let mut y_t_complex = y_f.as_mut_slice();
        self.ifft.process(&mut y_t_complex);

        // IFFT normalization and extract real part
        let fft_size_f32 = FFT_SIZE as f32;
        for (idx, c) in y_t_complex.iter().enumerate() {
            self.y_t[idx] = c.re / fft_size_f32;
        }

        // 6. Extract the valid part of the convolution (Overlap-Save method)
        let estimated_echo = self.y_t.rows(FRAME_SIZE, FRAME_SIZE);

        // 7. Calculate the error signal (mic signal - estimated echo)
        for (idx, (mic, echo)) in mic_frame.iter().zip(estimated_echo.iter()).enumerate() {
            error_signal[idx] = mic - echo;
        }

        // 8. FFT of the error signal for weight update
        // The error signal is placed in the second half of the buffer (the first half
        // is zero-padded) to ensure correct time alignment for the gradient calculation.
        self.e_t_buffer = [Complex::new(0.0, 0.0); FFT_SIZE];
        for (i, &sample) in error_signal.iter().enumerate() {
            self.e_t_buffer[i + FRAME_SIZE] = Complex::new(sample, 0.0);
        }

        self.fft.process(&mut self.e_t_buffer);
        let e_f = DVectorView::from_slice(&self.e_t_buffer, FFT_SIZE);

        // 9. Update filter weights using Normalized LMS algorithm
        let mut gradient = x_f.map(|c| c.conj()).component_mul(&e_f);
        for i in 0..FFT_SIZE {
            // Normalize by the PSD of the far-end signal
            gradient[i] /= self.psd[i] + 1e-10; // Add a small epsilon for stability
        }
        self.weights += &gradient * Complex::new(self.mu, 0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_instance_and_process_frame() {
        const FFT_SIZE: usize = 512;
        const FRAME_SIZE: usize = FFT_SIZE / 2;
        const STEP_SIZE: f32 = 0.5;

        let mut aec = FdafAec::<FFT_SIZE>::new(STEP_SIZE);

        let far_end_frame = vec![0.0; FRAME_SIZE];
        let mic_frame = vec![0.1; FRAME_SIZE]; // Some non-zero value
        let mut error_signal = vec![0.0; FRAME_SIZE];

        aec.process(
            error_signal.first_chunk_mut::<256>().unwrap(),
            far_end_frame.first_chunk::<256>().unwrap(),
            mic_frame.first_chunk::<256>().unwrap(),
        );
        // Check for NaN or Infinity
        assert!(
            error_signal.iter().all(|&x| x.is_finite()),
            "Output contains NaN or Infinity"
        );
    }

    #[test]
    #[should_panic]
    fn test_new_with_non_power_of_two_fft_size() {
        FdafAec::<511>::new(0.5);
    }

    #[test]
    #[should_panic]
    fn test_process_with_wrong_frame_size() {
        let mut aec = FdafAec::<512>::new(0.5);
        let far_end_frame = vec![0.0; 128];
        let mic_frame = vec![0.0; 256];
        let mut error_signal = vec![0.0; 256];
        aec.process(
            error_signal.first_chunk_mut::<256>().unwrap(),
            far_end_frame.first_chunk::<256>().unwrap(),
            mic_frame.first_chunk::<256>().unwrap(),
        );
    }
}
