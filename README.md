# Multi-Loop Helicopter Control System Tuner

This Python implementation replicates MATLAB's `systune` functionality for designing a multi-loop control system for a helicopter. The system features an inner loop with Static Output Feedback (SOF) for stability augmentation and decoupling, and an outer loop with three PI controllers for setpoint tracking of pitch (theta), roll (phi), and yaw rate (r).

## Features

- **Automated Tuning**: Uses differential evolution and SLSQP optimization to tune controller parameters.
- **Multi-Loop Architecture**:
  - Inner Loop: SOF for stability and decoupling.
  - Outer Loop: PI controllers for theta, phi, and r tracking.
- **Performance Requirements**:
  - 1st-order tracking response with ~1s time constant.
  - <20% steady-state mismatch.
  - ≥5 dB gain margin, ≥40° phase margin.
  - Closed-loop poles |poles| < 25 rad/s.
- **Sequential Tuning**: Iterative per-channel tuning with joint refinement.
- **Diagnostics**: Step response plots, Bode plots, pole maps, and performance metrics.
- **Optional H∞ Synthesis**: For initial SOF matrix using `mixsyn` (requires `slycot`).

## Requirements

- Python 3.7+
- NumPy
- SciPy
- Control Systems Library (`python-control`)
- Matplotlib (for plotting)
- Slycot (optional, for H∞ synthesis)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/helicopter-control-tuner.git
   cd helicopter-control-tuner
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   If `slycot` installation fails (common on Windows), the script will attempt to install it automatically or skip H∞ features.

## Usage

Run the tuner directly:

```bash
python classic_multiloop_controller.py
```

This will:
- Initialize the helicopter system model.
- Perform sequential tuning (up to 10 cycles by default).
- Display tuned parameters.
- Generate plots (if Matplotlib is available).

### Customization

Modify the `main()` function in `classic_multiloop_controller.py` to adjust tuning options:

- `cycles`: Number of tuning cycles (0 for automatic up to max_cycles).
- `max_cycles`: Maximum cycles if not specified.
- `verbose`: Enable detailed output.
- `use_hinf_inner`: Use H∞ synthesis for initial SOF.
- `final_joint_refine`: Perform final joint optimization.

Example:
```python
results = tuner.sequential_tune(cycles=5, max_cycles=15, verbose=True, use_hinf_inner=True, final_joint_refine=True)
```

### Output

- Console: Tuned PI gains and SOF matrix.
- Plots: Step responses, Bode plots, pole map (saved as `full_diagnostics.png`).
- Data: JSON export of diagnostics (`diagnostics_data.json`).

## Helicopter Model

The system uses a linearized 8-state helicopter model with:
- States: Longitudinal/lateral velocities, angular rates, etc.
- Inputs: Collective, longitudinal cyclic, lateral cyclic.
- Outputs: Theta, phi, r, q, p.

## Algorithm Overview

1. **Sequential Tuning**:
   - Tune inner SOF rows per channel.
   - Tune outer PI gains per channel.
   - Iterate until requirements met or max cycles reached.

2. **Cost Function**:
   - Penalizes instability, poor tracking, overshoot, coupling, and control effort.
   - Weights: SS error (1000x), IAE (200x), overshoot (500x), rise time (100x), coupling (200x).

3. **Optimization**:
   - Global: Differential Evolution.
   - Local: SLSQP.

## Contributing

Contributions welcome! Please open issues for bugs or feature requests.

## License

[Specify your license, e.g., MIT]

## References

- Based on helicopter control design principles.
- Inspired by MATLAB's Control System Toolbox `systune`.
