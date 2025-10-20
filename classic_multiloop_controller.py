"""
Multi-Loop Helicopter Control System with Automated Tuning

This implementation replicates MATLAB's systune functionality for a helicopter
control system with inner and outer control loops.

Control Architecture:
- Inner Loop (SOF): Static Output Feedback for stability augmentation & decoupling
- Outer Loop (PI): Three PI controllers for setpoint tracking (theta, phi, r)

Requirements:
- Tracking: 1st-order response with ~1s time constant, <20% mismatch
- Margins: ≥5 dB gain margin, ≥40° phase margin
- Poles: |poles| < 25 rad/s
"""

import numpy as np
import scipy.linalg as la
import scipy.optimize as opt
from scipy.signal import cont2discrete
from typing import Tuple, Dict
import warnings
import time
warnings.filterwarnings('ignore')

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Matplotlib not available, plotting disabled.")
    plt = None  # ensure symbol exists when matplotlib is missing

try:
    import control as ct
except ImportError:
    print("Installing control library... (pip install control)")
    import subprocess
    subprocess.check_call(['pip', 'install', 'control'])
    import control as ct

try:
    import slycot
except ImportError:
    print("Installing slycot... (pip install slycot)")
    import subprocess
    subprocess.check_call(['pip', 'install', 'slycot'])
    import slycot

try:
    from control.robust import mixsyn
    HAS_MIXSYN = True
except ImportError:
    mixsyn = None
    HAS_MIXSYN = False
    print("mixsyn not available even after installing slycot.")


class HelicopterSystem:
    """Helicopter state-space model and control system"""
    
    def __init__(self):
        # System matrices
        self.A = np.array([
            [-0.0191, 0.017, 0.3839, -9.7924, -0.0008, -0.3371, 0, 0],
            [0.0136, -0.2994, 0.0237, -0.5859, -0.0017, -0.0257, 0.5374, 0],
            [0.0405, -0.0026, -1.8394, 0, 0.0024, 0.5281, 0, -0.0015],
            [0, 0, 0.9985, 0, 0, 0, 0, 0.0549],
            [0.001, -0.0017, -0.3381, 0.0322, -0.0349, -0.4032, 9.7777, 0.1168],
            [0.013, 0, -3.047, 0, -0.229, -10.6199, 0, -0.0333],
            [0, 0, -0.0033, 0, 0, 1, 0, 0.0598],
            [0.002, 0.006, -0.5412, 0, 0.0039, -1.8554, 0, -0.3487]
        ])
        
        self.B = np.array([
            [-10.3456, 1.0793, 0],
            [-0.7293, 0.0755, 0],
            [27.09, -4.7239, -0.1857],
            [0, 0, 0],
            [-1.082, -10.3713, 4.7239],
            [-27.2884, -156.4425, -1.069],
            [0, 0, 0],
            [-4.8969, -27.9728, -12.9304]
        ])
        
        # Measurements: theta, phi, r, q, p
        self.C = np.array([
            [0, 0, 0, 1, 0, 0, 0, 0],  # theta
            [0, 0, 0, 0, 0, 0, 1, 0],  # phi
            [0, 0, 0, 0, 0, 0, 0, 1],  # r
            [0, 0, 1, 0, 0, 0, 0, 0],  # q
            [0, 0, 0, 0, 0, 1, 0, 0]   # p
        ])
        
        self.D = np.zeros((5, 3))
        
        # Create state-space system
        self.plant = ct.StateSpace(self.A, self.B, self.C, self.D)
        
    def create_closed_loop(self, pi_params: np.ndarray, sof_matrix: np.ndarray = np.zeros((3,5))) -> ct.StateSpace:
        """
        Create closed-loop system with PI controllers and SOF

        Args:
            pi_params: [Kp1, Ki1, Kp2, Ki2, Kp3, Ki3] for three PI controllers
            sof_matrix: 3x5 static output feedback matrix

        Returns:
            Closed-loop state-space system
        """
        # Create PI controllers (Kp + Ki/s)
        s = ct.TransferFunction.s

        # PI for theta (output 0)
        C1 = ct.tf2ss(ct.TransferFunction([pi_params[0], pi_params[1]], [1, 0]))

        # PI for phi (output 1)
        C2 = ct.tf2ss(ct.TransferFunction([pi_params[2], pi_params[3]], [1, 0]))

        # PI for r (output 2)
        C3 = ct.tf2ss(ct.TransferFunction([pi_params[4], pi_params[5]], [1, 0]))

        # Lowpass filters (40 rad/s cutoff)
        wc = 40
        F = ct.tf2ss(ct.TransferFunction([wc], [1, wc]))

        # Build multi-loop system manually
        # This is simplified - full implementation would use interconnection
        try:
            # Extract outputs for outer loop: theta, phi, r (indices 0, 1, 2)
            C_outer = self.C[:3, :]

            # Create augmented system with integrators
            n_states = self.A.shape[0]
            n_int = 3  # Three integrators for PI controllers

            # Augmented A matrix
            A_aug = np.block([
                [self.A, np.zeros((n_states, n_int))],
                [-C_outer, np.zeros((n_int, n_int))]
            ])

            # Augmented B matrix
            B_aug = np.block([
                [self.B],
                [np.zeros((n_int, 3))]
            ])

            # Control law: u = -SOF*y - [Kp1, Kp2, Kp3]*e - [Ki1, Ki2, Ki3]*∫e
            # where e = r - y_outer
            Kp = np.diag([pi_params[0], pi_params[2], pi_params[4]])
            Ki = np.diag([pi_params[1], pi_params[3], pi_params[5]])

            # Feedback matrix
            K_total = np.hstack([sof_matrix @ self.C + Kp @ C_outer, Ki])


            # Closed-loop A matrix
            A_cl = A_aug - B_aug @ K_total


            # For step response: B_cl is reference input
            B_cl = np.block([[np.zeros((n_states, 3))], [np.eye(3)]])

            # Output is theta, phi, r
            C_cl = np.hstack([C_outer, np.zeros((3, n_int))])
            D_cl = Kp  # Reference feedforward

            sys_cl = ct.StateSpace(A_cl, B_cl, C_cl, D_cl)
            return sys_cl

        except Exception as e:
            print(f"Error creating closed-loop: {e}")
            # Return a dummy unstable system
            return ct.StateSpace([[-1]], [[1]], [[1]], [[0]])


class MultiLoopTuner:
    """Automated tuning for multi-loop helicopter control"""
    
    def __init__(self, helicopter: HelicopterSystem):
        self.heli = helicopter
        self.best_params = None
        self.best_sof = None
        self.best_cost = float('inf')
        self.eval_count = 0
        self.verbose = False
        self.unstable_count = 0
        self.plot_t_end = 10.0
        
    def cost_function(self, params: np.ndarray) -> float:
        """Cost function with improved weighting"""
        pi_params = params[:6]
        sof_flat = params[6:]
        sof_matrix = sof_flat.reshape(3, 5)
        
        try:
            sys_cl = self.heli.create_closed_loop(pi_params, sof_matrix)
            poles = np.linalg.eigvals(sys_cl.A)
            
            # Hard stability constraint
            if np.any(np.real(poles) > 0):
                self.unstable_count += 1
                if self.verbose and (self.unstable_count <= 3 or self.unstable_count % 100 == 0):
                    print(f"[cost] Unstable poles detected (#{self.unstable_count}); assigning large penalty.")
                return 1e6
                
            # Pole constraints (stronger): fast dynamics discouraged, push real parts left
            pole_cost = np.sum(np.maximum(0, np.abs(poles) - 25)**2) * 200
            pole_cost += np.sum(np.maximum(0, np.real(poles) + 0.5)**2) * 500
            
            # Step response analysis (compute SISO responses to avoid shape ambiguity)
            t = np.linspace(0, 10, 1000)
            
            cost = pole_cost
            
            # Performance metrics with improved weighting
            for i in range(3):
                # Main channel (input i -> output i)
                t_resp, y_main = ct.step_response(sys_cl, T=t, input=i, output=i)
                if y_main is None:
                    if self.verbose and self.eval_count % 100 == 0:
                        print(f"[cost] Invalid main response for channel {i}; penalizing.")
                    return 1e6
                y_main_array = np.asarray(y_main)
                if y_main_array.ndim == 0:
                    if self.verbose and self.eval_count % 100 == 0:
                        print(f"[cost] Invalid main response for channel {i}; penalizing.")
                    return 1e6
                y = np.squeeze(y_main_array)
                final_val = float(y[-1])

                # Rise time (target: ~1s)
                idx_10 = np.where(y >= 0.1)[0]
                idx_90 = np.where(y >= 0.9)[0]
                if len(idx_10) > 0 and len(idx_90) > 0:
                    rise_time = t_resp[idx_90[0]] - t_resp[idx_10[0]]
                    cost += (rise_time - 1.0)**2 * 100

                # Tracking error
                ss_error = abs(1.0 - final_val)
                cost += ss_error**2 * 1000

                # Overshoot penalty (target: < 10%)
                peak = float(np.max(y))
                if peak > 1:
                    cost += (peak - 1.1)**2 * 500

                # Coupling penalty (responses in other outputs for input i)
                for j in range(3):
                    if i != j:
                        _, y_cpl = ct.step_response(sys_cl, T=t, input=i, output=j)
                        if y_cpl is None:
                            if self.verbose and self.eval_count % 100 == 0:
                                print(f"[cost] Invalid coupling response i={i} -> j={j}; penalizing.")
                            return 1e6
                        y_cpl_array = np.asarray(y_cpl)
                        if y_cpl_array.ndim == 0:
                            coupling = abs(float(y_cpl_array))
                        else:
                            coupling = float(np.max(np.abs(np.squeeze(y_cpl_array))))
                        cost += coupling**2 * 200
            
            # Control effort penalty
            cost += np.sum(np.abs(pi_params)**2) * 0.1
            cost += np.sum(np.abs(sof_matrix)**2) * 0.1
            
            # Lightweight progress output every 50 evaluations
            self.eval_count += 1
            if self.verbose and self.eval_count % 50 == 0:
                print(f"[cost] Eval {self.eval_count}: cost={cost:.2f}")
            return float(cost)
            
        except Exception as e:
            if self.verbose:
                print(f"[cost] Exception during cost evaluation: {e}")
            return 1e6
    
    def tune(self, max_iter: int = 200, verbose: bool = False) -> Dict:
        """Robust tuning: global DE followed by local SLSQP, minimal prints."""
        self.verbose = verbose
        self.unstable_count = 0
        # Bounds
        bounds = [(-5, 5)] * 6 + [(-1, 1)] * 15
        
        # Seed guess (moderate)
        x0 = np.hstack([
            np.array([0.7, 1.0, -0.2, -0.6, 0.1, -0.9]),
            np.zeros(15)
        ])
        
        # Global search using differential evolution
        if verbose:
            print("[tune] Starting Differential Evolution (global search)...")
        de_result = opt.differential_evolution(
            self.cost_function,
            bounds=bounds,
            maxiter=150,
            popsize=15,
            tol=1e-3,
            polish=False,
            disp=verbose
        )
        if verbose:
            print(f"[tune] DE complete. Best cost: {de_result.fun:.3f}")
            if self.unstable_count > 0:
                print(f"[tune] DE explored {self.unstable_count} unstable candidates (penalized).")
        self.unstable_count = 0
        x_de = de_result.x
        
        # Local refinement
        if verbose:
            print("[tune] Starting SLSQP (local refinement)...")
        slsqp = opt.minimize(
            self.cost_function,
            x_de,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-7, 'disp': verbose}
        )
        best_x = slsqp.x if slsqp.success else x_de
        best_cost = float(self.cost_function(best_x))
        if verbose:
            print(f"[tune] SLSQP {'succeeded' if slsqp.success else 'did not converge'}. Best cost: {best_cost:.3f}")
            if self.unstable_count > 0:
                print(f"[tune] SLSQP encountered {self.unstable_count} unstable candidates (penalized).")
        
        self.best_params = best_x
        self.best_sof = best_x[6:].reshape(3, 5)
        
        return {
            'pi_params': best_x[:6],
            'sof_matrix': self.best_sof,
            'cost': best_cost,
            'success': True
        }

    def initial_sof_hinf(self):
        if not HAS_MIXSYN or mixsyn is None:
            print("mixsyn not available.")
            return None

        C_outer = self.heli.C[:3, :]
        G = ct.StateSpace(self.heli.A, self.heli.B, C_outer, np.zeros((3,3)))

        s = ct.TransferFunction.s
        wb = 2.0
        M = 2.0
        A = 0.01
        W1 = (s / M + wb) / (s + wb * A)
        W2 = None
        W3 = None

        try:
            k, gam, rpt = mixsyn(G, W1, W2, W3)
            print(f"Hinf optimal gamma: {gam}")
            k_static = ct.dcgain(k)
            return k_static
        except Exception as e:
            print(f"Hinf synthesis failed: {e}")
            return None

    def _channel_indices(self, channel: int) -> Tuple[int, int]:
        # Map channel -> (kp_idx, ki_idx)
        return (0, 1) if channel == 0 else (2, 3) if channel == 1 else (4, 5)

    def _build_params(self, base_pi: np.ndarray, base_sof: np.ndarray,
                      channel: int, kp: float, ki: float, sof_row: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pi_params = base_pi.copy()
        kp_idx, ki_idx = self._channel_indices(channel)
        pi_params[kp_idx] = kp
        pi_params[ki_idx] = ki
        sof_matrix = base_sof.copy()
        sof_matrix[channel, :] = sof_row
        return pi_params, sof_matrix

    def _cost_sof_row(self, channel: int, sof_row: np.ndarray,
                       base_pi: np.ndarray, base_sof: np.ndarray,
                       disable_pi_for_channel: bool = False) -> float:
        sof_matrix = base_sof.copy()
        sof_matrix[channel, :] = sof_row
        C_outer = self.heli.C
        G = ct.StateSpace(self.heli.A, self.heli.B, C_outer, np.zeros((5,3)))
        try:
            sys_cl = ct.ss(G.A - G.B @ sof_matrix @ G.C, G.B, G.C, G.D)
            poles = np.linalg.eigvals(sys_cl.A)
            if np.any(np.real(poles) > 0):
                return 1e6
            t = np.linspace(0, 5, 500)
            # Main response (input=channel -> output=channel)
            t_resp, y_main = ct.step_response(sys_cl, T=t, input=channel, output=channel)
            if y_main is None:
                return 1e6
            y_main_array = np.asarray(y_main)
            if y_main_array.ndim == 0:
                return 1e6
            y = np.squeeze(y_main_array)
            final_val = float(y[-1])
            # Penalize tracking error and overshoot
            ss_error = abs(1.0 - final_val)
            peak = float(np.max(y))
            rise_time = 0.0
            idx_10 = np.where(y >= 0.1)[0]
            idx_90 = np.where(y >= 0.9)[0]
            if len(idx_10) > 0 and len(idx_90) > 0:
                rise_time = t_resp[idx_90[0]] - t_resp[idx_10[0]]
            # Coupling penalties to other outputs
            coupling_cost = 0.0
            for j in range(5):
                if j == channel:
                    continue
                _, y_cpl = ct.step_response(sys_cl, T=t, input=channel, output=j)
                if y_cpl is None:
                    return 1e6
                coupling = float(np.max(np.abs(np.squeeze(np.asarray(y_cpl)))))
                coupling_cost += coupling**2 * 100
            # Integral absolute error over horizon
            iae = float(np.trapz(np.abs(1.0 - y), t_resp))
            # Aggregate cost (heavier SS error + IAE)
            cost = (
                ss_error**2 * 3000 +
                iae * 200 +
                max(0.0, peak - 1.05)**2 * 500 +
                (rise_time - 1.0)**2 * 100 +
                coupling_cost +
                np.sum(np.abs(sof_row)) * 0.1
            )
            return float(cost)
        except Exception:
            return 1e6

    def _cost_pi_channel(self, channel: int, kp: float, ki: float,
                         base_pi: np.ndarray, base_sof: np.ndarray) -> float:
        # For outer-loop (PI) tuning of one channel, zero other channels' PI gains
        pi_isolated = np.zeros_like(base_pi)
        kp_idx, ki_idx = self._channel_indices(channel)
        pi_isolated[kp_idx] = kp
        pi_isolated[ki_idx] = ki
        pi_params, sof_matrix = self._build_params(pi_isolated, base_sof, channel, kp, ki, base_sof[channel, :])
        try:
            sys_cl = self.heli.create_closed_loop(pi_params, sof_matrix)
            t = np.linspace(0, 5, 500)
            t_resp, y_main = ct.step_response(sys_cl, T=t, input=channel, output=channel)
            if y_main is None:
                return 1e6
            y_main_array = np.asarray(y_main)
            if y_main_array.ndim == 0:
                return 1e6
            y = np.squeeze(y_main_array)
            final_val = float(y[-1])
            ss_error = abs(1.0 - final_val)
            peak = float(np.max(y))
            rise_time = 0.0
            idx_10 = np.where(y >= 0.1)[0]
            idx_90 = np.where(y >= 0.9)[0]
            if len(idx_10) > 0 and len(idx_90) > 0:
                rise_time = t_resp[idx_90[0]] - t_resp[idx_10[0]]
            # Coupling penalties to other outputs
            coupling_cost = 0.0
            for j in range(3):
                if j == channel:
                    continue
                _, y_cpl = ct.step_response(sys_cl, T=t, input=channel, output=j)
                if y_cpl is None:
                    return 1e6
                coupling = float(np.max(np.abs(np.squeeze(np.asarray(y_cpl)))))
                coupling_cost += coupling**2 * 100
            iae = float(np.trapz(np.abs(1.0 - y), t_resp))
            cost = (
                ss_error**2 * 4000 +
                iae * 300 +
                max(0.0, peak - 1.05)**2 * 500 +
                (rise_time - 1.0)**2 * 100 +
                coupling_cost +
                (kp**2 + ki**2) * 0.1
            )
            poles = np.linalg.eigvals(sys_cl.A)
            if np.any(np.real(poles) > 0):
                return 1e6
            return float(cost)
        except Exception:
            return 1e6

    def _meets_requirements(self, sys_cl: ct.StateSpace) -> bool:
        # Check diagonal channels for tracking specs and coupling
        t = np.linspace(0, 5, 500)
        for i in range(3):
            t_resp, y = ct.step_response(sys_cl, T=t, input=i, output=i)
            if y is None:
                return False
            y_array = np.asarray(y)
            if y_array.ndim == 0:
                return False
            yv = np.squeeze(y_array)
            final_val = float(yv[-1])
            ss_error = abs(1.0 - final_val)
            peak = float(np.max(yv))
            idx_10 = np.where(yv >= 0.1)[0]
            idx_90 = np.where(yv >= 0.9)[0]
            if len(idx_10) > 0 and len(idx_90) > 0:
                rise_time = t_resp[idx_90[0]] - t_resp[idx_10[0]]
            else:
                rise_time = 10.0
            if ss_error > 0.05 or peak > 1.10 or rise_time > 2.5:
                return False
            # coupling
            for j in range(3):
                if i == j:
                    continue
                _, y_cpl = ct.step_response(sys_cl, T=t, input=i, output=j)
                if y_cpl is None:
                    return False
                coupling = float(np.max(np.abs(np.squeeze(np.asarray(y_cpl)))))
                if coupling > 0.2:
                    return False
        return True

    def sequential_tune(self, cycles: int = 0, max_cycles: int = 15, verbose: bool = True, use_hinf_inner: bool = False, final_joint_refine: bool = False) -> Dict:
        # Initialize with moderate seeds (as before working version)
        pi_params = np.array([0.7, 1.0, -0.2, -0.6, 0.1, -0.9], dtype=float)
        sof_matrix = np.zeros((3,5))
        if use_hinf_inner:
            sof_init = self.initial_sof_hinf()
            if sof_init is not None:
                sof_matrix = sof_init
                if verbose:
                    print("[seq] Used Hinf synthesis for initial SOF matrix.")

        cycle_idx = 0
        target_cycles = cycles if cycles and cycles > 0 else max_cycles

        # Add convergence tracking
        prev_params = None
        prev_sof = None

        # Warm-up: Tune inner loops for 5 iterations each
        if verbose:
            print("[seq] Warm-up: Tuning inner loops for 5 iterations...")
        for ch in range(3):
            if verbose:
                print(f"[seq] Warm-up inner loop {ch+1}/3...")
            def cost_sof(x):
                return self._cost_sof_row(
                    ch, x, pi_params, sof_matrix, disable_pi_for_channel=True
                )
            bounds_row = [(-1, 1)] * 5
            de_row = opt.differential_evolution(cost_sof, bounds_row, maxiter=5, popsize=15, tol=1e-3, polish=False, disp=False)
            sof_matrix[ch, :] = de_row.x
            if verbose:
                print(f"[seq] Warm-up SOF row {ch} -> {de_row.x}")

        while cycle_idx < target_cycles:
            # Adaptive iteration counts (reduce effort in later cycles)
            de_iters = max(50, 150 - cycle_idx * 10)  # Start 150, reduce to 50
            slsqp_iters = max(30, 80 - cycle_idx * 5)  # Start 80, reduce to 30

            if verbose:
                label = f"{cycle_idx+1}/{target_cycles}"
                print(f"[seq] Tuning cycle {label}")
            # Store previous values
            if cycle_idx > 0:
                prev_params = pi_params.copy()
                prev_sof = sof_matrix.copy()
            if verbose:
                print("[seq] Tuning all inner loops (SOF rows)...")
            for ch in range(3):
                if verbose:
                    print(f"[seq] Inner loop {ch+1}/3: SOF row tuning...")
                # Inner: optimize sof row with PI disabled for this channel
                def cost_sof(x):
                    return self._cost_sof_row(
                        ch, x, pi_params, sof_matrix, disable_pi_for_channel=(cycle_idx == 0)
                    )
                bounds_row = [(-1, 1)] * 5
                de_row = opt.differential_evolution(cost_sof, bounds_row, maxiter=de_iters, popsize=15, tol=1e-3, polish=False, disp=False)
                sof_row_opt = de_row.x
                # Local refinement (SLSQP) warm-started from DE result
                try:
                    slsqp_row = opt.minimize(
                        cost_sof,
                        sof_row_opt,
                        method='SLSQP',
                        bounds=bounds_row,
                        options={'maxiter': 80, 'ftol': 1e-7, 'disp': False}
                    )
                    if slsqp_row.success:
                        sof_row_opt = slsqp_row.x
                except Exception:
                    pass
                # Update SOF row
                sof_matrix[ch, :] = sof_row_opt
                if verbose:
                    print(f"[seq] SOF row {ch} -> {sof_row_opt}")

            if verbose:
                print("[seq] Tuning all outer loops (PI gains)...")
            for ch in range(3):
                if verbose:
                    print(f"[seq] Outer loop {ch+1}/3: PI tuning...")
                def cost_pi(x):
                    return self._cost_pi_channel(ch, x[0], x[1], pi_params, sof_matrix)
                bounds_pi = [(-8, 8), (-8, 8)]
                de_pi = opt.differential_evolution(cost_pi, bounds_pi, maxiter=60, popsize=15, tol=1e-3, polish=False, disp=False)
                kp_opt, ki_opt = de_pi.x
                # Local refinement for PI
                try:
                    slsqp_pi = opt.minimize(
                        cost_pi,
                        np.array([kp_opt, ki_opt]),
                        method='SLSQP',
                        bounds=bounds_pi,
                        options={'maxiter': 100, 'ftol': 1e-7, 'disp': False}
                    )
                    if slsqp_pi.success:
                        kp_opt, ki_opt = float(slsqp_pi.x[0]), float(slsqp_pi.x[1])
                except Exception:
                    pass
                kp_idx, ki_idx = self._channel_indices(ch)
                pi_params[kp_idx] = kp_opt
                pi_params[ki_idx] = ki_opt
                if verbose:
                    print(f"[seq] PI gains ch{ch}: Kp={kp_opt:.3f}, Ki={ki_opt:.3f}")
                # Quick performance check for this channel
                sys_cl_tmp = self.heli.create_closed_loop(pi_params, sof_matrix)
                if not self._meets_requirements(sys_cl_tmp):
                    if verbose:
                        print(f"[seq] Channel {ch} interim performance below specs; will continue refinement.")
            # After all channels, check overall
            # Check convergence after cycle completes
            if cycle_idx > 0 and prev_params is not None:
                param_change = np.linalg.norm(pi_params - prev_params)
                sof_change = np.linalg.norm(sof_matrix - prev_sof)
                
                if verbose:
                    print(f"[seq] Param change: {param_change:.6f}, SOF change: {sof_change:.6f}")
                
                # Early stopping if changes are tiny
                if param_change < 1e-4 and sof_change < 1e-4:
                    if verbose:
                        print("[seq] Converged (minimal parameter changes). Stopping early.")
                    break
            sys_cl = self.heli.create_closed_loop(pi_params, sof_matrix)
            if verbose:
                print("[seq] Completed all channels; evaluating overall performance...")
            if self._meets_requirements(sys_cl):
                if verbose:
                    print("[seq] Overall requirements met.")
                break
            else:
                if verbose:
                    print("[seq] Overall requirements NOT met; starting another cycle if available.")
            cycle_idx += 1
            if cycle_idx >= target_cycles:
                break

        if final_joint_refine:
            if verbose:
                print("[seq] Performing final joint refinement...")
            self.verbose = verbose
            bounds = [(-5, 5)] * 6 + [(-1, 1)] * 15
            x_init = np.hstack([pi_params, sof_matrix.flatten()])
            de_result = opt.differential_evolution(
                self.cost_function,
                bounds,
                init='latinhypercube',
                maxiter=100,
                popsize=15,
                tol=1e-4,
                workers=1,
                disp=verbose
            )
            x_de = de_result.x
            self.best_cost = de_result.fun
            slsqp_result = opt.minimize(
                self.cost_function,
                x_de,
                bounds=bounds,
                method='SLSQP',
                options={'maxiter': 150, 'ftol': 1e-6, 'disp': verbose}
            )
            if slsqp_result.fun < self.best_cost:
                best_x = slsqp_result.x
                self.best_cost = slsqp_result.fun
            else:
                best_x = x_de
            pi_params = best_x[:6]
            sof_matrix = best_x[6:].reshape(3, 5)

        # Save results
        self.best_params = pi_params
        self.best_sof = sof_matrix
        self.best_cost = float(self.cost_function(np.hstack([pi_params, sof_matrix.flatten()])))
        success = self._meets_requirements(self.heli.create_closed_loop(self.best_params, self.best_sof))
        return {
            'pi_params': self.best_params,
            'sof_matrix': self.best_sof,
            'cost': self.best_cost,
            'success': bool(success),
            'cycles_run': cycle_idx + 1
        }

    def plot_full_diagnostics(self, filename: str = "full_diagnostics.png"):
        if self.best_params is None or self.best_sof is None:
            print("No tuning results. Run tuning first.")
            return
        sys_cl = self.heli.create_closed_loop(self.best_params, self.best_sof)
        import matplotlib.pyplot as mpl
        import matplotlib.gridspec as gridspec
        fig = mpl.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 2, figure=fig)
        # Top-left: 3x3 step responses (Actual vs Desired)
        gs_steps = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs[0, 0], wspace=0.35, hspace=0.35)
        t = np.linspace(0, float(self.plot_t_end), max(200, int(self.plot_t_end * 100)))
        inputs = ['theta-ref', 'phi-ref', 'r-ref']
        outputs = ['theta', 'phi', 'r']
        # Collect traces for export
        traces = { 'matrix': [] }
        for i in range(3):
            row_traces = []
            for j in range(3):
                ax = fig.add_subplot(gs_steps[i, j])
                t_resp, y_siso = ct.step_response(sys_cl, T=t, input=j, output=i)
                yv = np.squeeze(y_siso) if y_siso is not None else np.zeros_like(t_resp)
                ax.plot(t_resp, yv, 'b-', linewidth=2, label='Actual')
                if i == j:
                    desired = np.ones_like(t_resp)
                else:
                    desired = np.zeros_like(t_resp)
                # Ensure t_resp and desired are valid arrays
                t_plot = t_resp if t_resp is not None else np.zeros_like(desired)
                desired_plot = desired if desired is not None else np.zeros_like(t_plot)
                ax.plot(t_plot, desired_plot, 'r--', linewidth=1.2, label='Desired')
                ax.grid(True, alpha=0.3)
                if i == 2:
                    ax.set_xlabel('Time (s)')
                if j == 0:
                    ax.set_ylabel('Amplitude')
                ax.set_title(f'From: {inputs[j]}\nTo: {outputs[i]}', fontsize=9)
                if i == 0 and j == 0:
                    ax.legend(fontsize=8)
                row_traces.append({
                    't': t_resp.tolist() if t_resp is not None else [],
                    'actual': yv.tolist(),
                    'desired': desired.tolist(),
                    'from': inputs[j],
                    'to': outputs[i]
                })
            traces['matrix'].append(row_traces)
        fig.suptitle('Closed-Loop Diagnostics', fontsize=14, fontweight='bold')
        # Top-right: Bode magnitude and phase (diagonal channels open-loop approx.)
        ax_mag = fig.add_subplot(gs[0, 1])
        ax_phase = ax_mag.twinx()
        colors = ['tab:blue', 'tab:orange', 'tab:green']
        w = np.logspace(-3, 2, 400)
        for ch, c in enumerate(colors):
            # Approximate open-loop: response from input ch to output ch with unity feedback
            # Using closed-loop to get indicative frequency behavior
            sys_tf = ct.ss2tf(sys_cl)
            gij = ct.TransferFunction(sys_tf.num[ch][ch], sys_tf.den[ch][ch])
            mag, phase, ww = ct.bode(gij, w, Plot=False)
            if mag is None:
                continue
            ax_mag.semilogx(ww, 20*np.log10(np.maximum(mag, 1e-6)), color=c, label=f'{outputs[ch]}')
        ax_mag.set_ylabel('Magnitude (dB)')
        ax_mag.set_xlabel('Frequency (rad/s)')
        ax_mag.grid(True, which='both', alpha=0.3)
        ax_mag.legend()
        # Bottom-left: Another margin-like view (reuse mag for consistency)
        ax_mag2 = fig.add_subplot(gs[1, 0])
        for ch, c in enumerate(colors):
            sys_tf = ct.ss2tf(sys_cl)
            gij = ct.TransferFunction(sys_tf.num[ch][ch], sys_tf.den[ch][ch])
            mag, phase, ww = ct.bode(gij, w, Plot=False)
            if mag is None:
                continue
            ax_mag2.semilogx(ww, 20*np.log10(np.maximum(mag, 1e-6)), color=c, label=f'{outputs[ch]}')
        ax_mag2.set_ylabel('Magnitude (dB)')
        ax_mag2.set_xlabel('Frequency (rad/s)')
        ax_mag2.grid(True, which='both', alpha=0.3)
        ax_mag2.legend()
        # Bottom-right: Closed-loop pole map
        ax_poles = fig.add_subplot(gs[1, 1])
        poles = np.linalg.eigvals(sys_cl.A)
        ax_poles.scatter(np.real(poles), np.imag(poles), c='tab:blue', marker='x')
        ax_poles.axvline(0, color='k', linewidth=1)
        ax_poles.axhline(0, color='k', linewidth=1)
        ax_poles.set_xlabel('Real Axis (1/s)')
        ax_poles.set_ylabel('Imag Axis (1/s)')
        ax_poles.set_title('Closed-loop pole location')
        ax_poles.grid(True, alpha=0.3)
        mpl.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
        mpl.savefig(filename)
        mpl.show()
        # Save raw data alongside plot
        try:
            import json
            export = {
                'pi_params': self.best_params.tolist(),
                'sof_matrix': self.best_sof.tolist(),
                'poles': {'real': np.real(poles).tolist(), 'imag': np.imag(poles).tolist()},
                'step_traces': traces
            }
            with open('diagnostics_data.json', 'w') as f:
                json.dump(export, f, indent=2)
        except Exception as e:
            print(f"Warning: could not write diagnostics_data.json: {e}")
    
    def display_results(self, concise: bool = True):
        """Display tuned controller parameters; concise by default."""
        if self.best_params is None:
            print("No tuning results available. Run tune() first.")
            return
        if self.best_sof is None:
            print("Tuning produced no SOF matrix. Run tune() first.")
            return
        
        if concise:
            kp1, ki1, kp2, ki2, kp3, ki3 = self.best_params[:6]
            print("PI gains:")
            print(f"  theta: Kp={kp1:.4f}, Ki={ki1:.4f}")
            print(f"  phi  : Kp={kp2:.4f}, Ki={ki2:.4f}")
            print(f"  r    : Kp={kp3:.4f}, Ki={ki3:.4f}")
            print("SOF D (3x5) [rows: ds, dc, dT | cols: theta, phi, r, q, p]:")
            for i in range(3):
                print("  " + " ".join(f"{self.best_sof[i,j]: .4f}" for j in range(5)))
        else:
            # Verbose MATLAB-like description if needed
            controllers = [
                ('PI1 (theta)', 0, 1),
                ('PI2 (phi)', 2, 3),
                ('PI3 (r)', 4, 5)
            ]
            print("\n" + "="*60)
            print("TUNED CONTROLLER PARAMETERS")
            print("="*60)
            print("\n--- PI Controllers ---")
            for name, kp_idx, ki_idx in controllers:
                print(f"\nBlock: {name}")
                print("         1")
                print("  Kp + Ki * ---")
                print("         s")
                print(f"  with Kp = {self.best_params[kp_idx]:.3f}, Ki = {self.best_params[ki_idx]:.3f}")
                print("  Continuous-time PI controller in parallel form.")
            print("\n--- Static Output Feedback (SOF) ---")

            print("\nD matrix (3x5):")
            print("           theta      phi        r         q         p")
            for i, output in enumerate(['ds', 'dc', 'dT']):
                print(f"{output:3s}  ", end="")
                for j in range(5):
                    print(f"{self.best_sof[i, j]:10.4f}", end=" ")
                print()
    
    def plot_responses(self):
        """Plot step responses of tuned system"""
        if self.best_params is None:
            print("No tuning results. Run tune() first.")
            return

        if not HAS_MATPLOTLIB:
            print("Matplotlib not available, skipping plot.")
            return
        if self.best_sof is None:
            print("No SOF matrix available, skipping plot.")
            return

        # Create closed-loop system
        sys_cl = self.heli.create_closed_loop(self.best_params, self.best_sof)

        # Step responses computed per SISO pair
        t = np.linspace(0, float(self.plot_t_end), max(200, int(self.plot_t_end * 100)))

        # Plot (import locally to satisfy static analyzers)
        import matplotlib.pyplot as mpl
        fig, axes = mpl.subplots(3, 3, figsize=(14, 10))
        fig.suptitle('Closed-Loop Step Responses', fontsize=14, fontweight='bold')

        inputs = ['theta-ref', 'phi-ref', 'r-ref']
        outputs = ['theta', 'phi', 'r']

        for i in range(3):
            for j in range(3):
                ax = axes[i, j]
                t_resp, y_siso = ct.step_response(sys_cl, T=t, input=j, output=i)
                if y_siso is not None:
                    y_plot = np.squeeze(np.asarray(y_siso))
                    t_plot = t_resp if t_resp is not None else t
                    ax.plot(t_plot, y_plot, 'b-', linewidth=2)
                    # Highlight diagonal (main tracking)
                    if i == j:
                        ax.plot(t_plot, y_plot, 'b-', linewidth=2.5)
                        ax.axhline(y=1, color='r', linestyle='--', alpha=0.5)
                        ax.set_facecolor('#f0f8ff')
                ax.grid(True, alpha=0.3)
                ax.set_xlabel('Time (s)')

                ax.set_ylabel('Amplitude')
                ax.set_title(f'{inputs[j]} → {outputs[i]}')

        mpl.tight_layout()
        mpl.show()
        
        # Minimal console performance summary suppressed unless explicitly requested elsewhere

def main():
    """Main execution"""
    # Create helicopter system
    print("[main] Initializing helicopter system...")
    heli = HelicopterSystem()
    
    # Create tuner
    print("[main] Creating tuner...")
    tuner = MultiLoopTuner(heli)
    
    # Perform sequential per-channel tuning cycles until requirements met or limit reached
    print("[main] Starting sequential tuning process...")
    results = tuner.sequential_tune(cycles=0, max_cycles=10, verbose=True, use_hinf_inner=False , final_joint_refine=True)
    print(f"[main] Sequential tuning finished after {results.get('cycles_run', '?')} cycles. Success={results.get('success')}.")
    if not results.get('success'):
        print("[main] Warning: Requirements not met after maximum cycles.")
    
    # Display results
    print("[main] Displaying results...")
    tuner.display_results(concise=True)
    
    # Plot responses (optional): comment out by default to reduce output
    if HAS_MATPLOTLIB:
        tuner.plot_t_end = 10.0
        print("[main] Plotting responses...")
        tuner.plot_responses()
        print("[main] Plotting full diagnostics...")
        tuner.plot_full_diagnostics(filename="full_diagnostics.png")
    
    return tuner


if __name__ == "__main__":
    start_time = time.time()
    tuner = main()
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")