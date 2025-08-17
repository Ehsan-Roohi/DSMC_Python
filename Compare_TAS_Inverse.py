#!/usr/bin/env python
# -*- coding: utf-8 -*-
# --- DSMC CODE - V29.2 - Comparative Parallel Execution (NameError Fixed) ---

import numpy as np
import matplotlib.pyplot as plt
import numba
import time
from scipy.signal import savgol_filter
import pandas as pd
from joblib import Parallel, delayed

# ===================================================================
# 1. Simulation Core
# ===================================================================
MASS_AR = 39.948e-3 / 6.022e23
KB = 1.380649e-23

@numba.jit(nopython=True)
def calculate_vhs_cross_section_numba(vr_mag):
    d_ref = 4.17e-10; t_ref = 273.0; omega_vhs = 0.81
    if vr_mag < 1e-9: return 1e-30
    exponent = omega_vhs - 0.5
    c_ref_sq = 2 * KB * t_ref / MASS_AR
    gamma_val = 1.04533
    d_sq = (d_ref**2) * ((c_ref_sq / vr_mag**2)**exponent) * (1 / gamma_val)
    return np.pi * d_sq

@numba.jit(nopython=True)
def perform_collisions_sbt_subcell(particles, indices_in_cell, cell_start_x, cell_width, num_sub_cells, cell_vol, dt, fnum, dist_sum_array, count_array, main_cell_idx):
    num_particles_in_main_cell = len(indices_in_cell)
    if num_particles_in_main_cell < 2: return
    num_sub_cells = max(1, num_sub_cells)
    sub_cell_width = cell_width / num_sub_cells
    sub_cell_groups = [[np.int64(x) for x in range(0)] for _ in range(num_sub_cells)]
    for p_idx in indices_in_cell:
        relative_pos = particles[p_idx, 0] - cell_start_x
        sub_cell_idx = int(relative_pos / sub_cell_width)
        if 0 <= sub_cell_idx < num_sub_cells:
            sub_cell_groups[sub_cell_idx].append(p_idx)
    sub_cell_vol = cell_vol / num_sub_cells
    if sub_cell_vol < 1e-30: return
    for sc_idx in range(num_sub_cells):
        indices_in_sub_cell = sub_cell_groups[sc_idx]
        num_particles_in_sub_cell = len(indices_in_sub_cell)
        if num_particles_in_sub_cell < 2: continue
        for i in range(num_particles_in_sub_cell - 1):
            p1_idx = indices_in_sub_cell[i]
            remaining_particles_count = num_particles_in_sub_cell - (i + 1)
            j_offset = np.random.randint(0, remaining_particles_count)
            p2_idx = indices_in_sub_cell[i + 1 + j_offset]
            vr = particles[p1_idx, 1:4] - particles[p2_idx, 1:4]
            vr_mag = np.sqrt(np.sum(vr**2))
            if vr_mag < 1e-9: continue
            sigma_t = calculate_vhs_cross_section_numba(vr_mag)
            collision_prob = (float(remaining_particles_count) * fnum * dt * sigma_t * vr_mag) / sub_cell_vol
            if np.random.rand() < collision_prob:
                dist = np.abs(particles[p1_idx, 0] - particles[p2_idx, 0])
                dist_sum_array[main_cell_idx] += dist
                count_array[main_cell_idx] += 1
                vcm = 0.5 * (particles[p1_idx, 1:4] + particles[p2_idx, 1:4])
                cos_chi = 2 * np.random.rand() - 1.0
                sin_chi = np.sqrt(1.0 - cos_chi**2)
                phi_chi = 2.0 * np.pi * np.random.rand()
                vr_prime = np.array([vr_mag * sin_chi * np.cos(phi_chi), vr_mag * sin_chi * np.sin(phi_chi), vr_mag * cos_chi])
                particles[p1_idx, 1:4] = vcm + 0.5 * vr_prime
                particles[p2_idx, 1:4] = vcm - 0.5 * vr_prime

def run_dsmc_simulation(sim_params, run_index):
    np.random.seed(int(time.time()) + run_index)
    method = sim_params['METHOD']
    LX = sim_params['LX']; RHO_INIT = sim_params['RHO_INIT']; T_INIT = sim_params['T_INIT']
    NUM_CELLS_X = sim_params['NUM_CELLS_X']; PARTICLES_PER_CELL_INIT = sim_params['PARTICLES_PER_CELL_INIT']
    TOTAL_TIME = sim_params['TOTAL_TIME']; DT = sim_params['DT']
    MIN_PARTICLES_FOR_STATS = 2

    particles = initialize_shock_tube(sim_params)
    N_DENSITY_REAL = RHO_INIT / MASS_AR
    CELL_VOLUME_CONCEPTUAL = (LX / NUM_CELLS_X) * ((LX/10) * (LX/10))
    FNUM = (N_DENSITY_REAL * CELL_VOLUME_CONCEPTUAL) / PARTICLES_PER_CELL_INIT
    NUM_STEPS = int(TOTAL_TIME / DT)
    
    sample_points_in_time = np.linspace(0, TOTAL_TIME, 5)
    sample_steps = {int(round(t / DT)) for t in sample_points_in_time}
    sample_steps.add(NUM_STEPS)

    results_history = {}
    cell_width = LX / NUM_CELLS_X

    initial_props = sample_properties(particles, NUM_CELLS_X, cell_width, FNUM, CELL_VOLUME_CONCEPTUAL, MIN_PARTICLES_FOR_STATS)
    initial_props['sof'] = np.full(NUM_CELLS_X, np.nan)
    initial_props['subcell_mfp_ratio'] = np.full(NUM_CELLS_X, np.nan)
    results_history[0.0] = initial_props
    
    subcell_config = np.ones(NUM_CELLS_X, dtype=np.int32)
    accumulated_dist_sum = np.zeros(NUM_CELLS_X, dtype=np.float64)
    accumulated_collision_count = np.zeros(NUM_CELLS_X, dtype=np.int64)

    current_step = 0
    while current_step < NUM_STEPS:
        if method == 'tai_tas':
            sum_cell_counts = np.zeros(NUM_CELLS_X, dtype=np.int64)
            steps_in_block = min(sim_params['ADAPTATION_INTERVAL'], NUM_STEPS - current_step)
        else:
            steps_in_block = 1

        for _ in range(steps_in_block):
            if current_step >= NUM_STEPS: break
            current_step += 1
            particles[:, 0] += particles[:, 1] * DT
            hit_left = particles[:, 0] < 0; particles[hit_left, 1] *= -1; particles[hit_left, 0] *= -1
            hit_right = particles[:, 0] > LX; particles[hit_right, 1] *= -1; particles[hit_right, 0] = 2 * LX - particles[hit_right, 0]
            cell_indices = np.clip((particles[:, 0] / cell_width).astype(np.int64), 0, NUM_CELLS_X - 1)
            sorted_particle_indices = np.argsort(cell_indices)
            cell_counts = np.bincount(cell_indices, minlength=NUM_CELLS_X)
            if method == 'tai_tas':
                sum_cell_counts += cell_counts
            else:
                subcell_config = np.maximum(1, np.floor(cell_counts / sim_params['TARGET_PPC_SUBCELL'] + 0.5).astype(np.int32))
            
            # <--- CORRECTION IS HERE ---
            # The variable 'counts' was incorrectly used instead of 'cell_counts'.
            cell_start_indices = np.concatenate(([0], np.cumsum(cell_counts[:-1])))

            for i in range(NUM_CELLS_X):
                if cell_counts[i] > 1:
                    start, end = cell_start_indices[i], cell_start_indices[i] + cell_counts[i]
                    indices_in_cell_i = sorted_particle_indices[start:end]
                    cell_start_x = i * cell_width
                    perform_collisions_sbt_subcell(particles, indices_in_cell_i, cell_start_x, cell_width, subcell_config[i], CELL_VOLUME_CONCEPTUAL, DT, FNUM, accumulated_dist_sum, accumulated_collision_count, i)
            if current_step in sample_steps:
                time_key = current_step * DT
                if abs(time_key) > 1e-15:
                    props = sample_properties(particles, NUM_CELLS_X, cell_width, FNUM, CELL_VOLUME_CONCEPTUAL, MIN_PARTICLES_FOR_STATS)
                    avg_dist = np.full(NUM_CELLS_X, np.nan); valid_counts = accumulated_collision_count > 0
                    avg_dist[valid_counts] = accumulated_dist_sum[valid_counts] / accumulated_collision_count[valid_counts]
                    temp = props['temperature']; density = props['density']
                    mfp = np.full_like(temp, np.nan)
                    for idx in range(NUM_CELLS_X):
                        if not np.isnan(temp[idx]) and not np.isnan(density[idx]) and temp[idx]>0 and density[idx]>0:
                            avg_vr = np.sqrt(16*KB*temp[idx]/(np.pi*MASS_AR)); avg_sigma = calculate_vhs_cross_section_numba(avg_vr)
                            mfp[idx] = 1.0/(np.sqrt(2)*density[idx]*avg_sigma)
                    with np.errstate(divide='ignore',invalid='ignore'): props['sof'] = avg_dist/mfp
                    with np.errstate(divide='ignore',invalid='ignore'): props['subcell_mfp_ratio'] = (cell_width/subcell_config)/mfp
                    results_history[time_key] = props
                    accumulated_dist_sum.fill(0); accumulated_collision_count.fill(0)
        if method == 'tai_tas':
            avg_cell_counts = sum_cell_counts / steps_in_block
            max_avg_count = np.max(avg_cell_counts)
            if max_avg_count > 1e-9:
                normalized_density = avg_cell_counts / max_avg_count
                adaptive_rule = 1.0 - normalized_density
                subcell_config = np.clip(1+((sim_params['MAX_SUB_CELLS']-1)*adaptive_rule).astype(np.int32), 1, sim_params['MAX_SUB_CELLS'])
    return results_history

def initialize_shock_tube(params):
    num_cells = params['NUM_CELLS_X']; lx = params['LX']; ppc_left = params['PARTICLES_PER_CELL_INIT']
    t_left = params['T_INIT']; rho_ratio = params['RHO_RATIO']
    ppc_right = max(1, int(ppc_left / rho_ratio))
    num_cells_left = num_cells // 2; num_cells_right = num_cells - num_cells_left
    total_particles = (ppc_left * num_cells_left) + (ppc_right * num_cells_right)
    particles = np.zeros((total_particles, 4))
    cell_width = lx / num_cells; v_thermal_std = np.sqrt(KB * t_left / MASS_AR)
    current_pos = 0
    for i in range(num_cells_left):
        start_idx, end_idx = current_pos, current_pos + ppc_left
        particles[start_idx:end_idx, 0] = i * cell_width + np.random.rand(ppc_left) * cell_width
        particles[start_idx:end_idx, 1:4] = np.random.normal(0, v_thermal_std, (ppc_left, 3))
        current_pos = end_idx
    for i in range(num_cells_left, num_cells):
        start_idx, end_idx = current_pos, current_pos + ppc_right
        particles[start_idx:end_idx, 0] = i * cell_width + np.random.rand(ppc_right) * cell_width
        particles[start_idx:end_idx, 1:4] = np.random.normal(0, v_thermal_std, (ppc_right, 3))
        current_pos = end_idx
    print(f"Initialized {total_particles} particles.")
    return particles

def sample_properties(particles_state, num_cells, cell_width, fnum, cell_vol, min_parts):
    density_profile, velocity_profile, temp_profile = (np.full(num_cells, np.nan) for _ in range(3))
    cell_indices = np.clip((particles_state[:, 0] / cell_width).astype(np.int64), 0, num_cells - 1)
    sorted_indices = np.argsort(cell_indices)
    counts = np.bincount(cell_indices, minlength=num_cells)
    starts = np.concatenate(([0], np.cumsum(counts[:-1])))
    for i in range(num_cells):
        if counts[i] > 0: density_profile[i] = counts[i] * fnum / cell_vol
        if counts[i] >= min_parts:
            indices_in_cell = sorted_indices[starts[i]:starts[i]+counts[i]]
            cell_velocities = particles_state[indices_in_cell, 1:4]
            mean_vel_cell = np.mean(cell_velocities, axis=0)
            velocity_profile[i] = mean_vel_cell[0]
            if counts[i] > 1:
                thermal_vel_sq = np.sum((cell_velocities - mean_vel_cell)**2)
                temp_profile[i] = (MASS_AR * thermal_vel_sq) / (3 * KB * (counts[i] - 1))
            else: temp_profile[i] = 0.0
    return {'density': density_profile, 'velocity': velocity_profile, 'temperature': temp_profile}

def average_ensemble_runs(all_run_data, keys_to_average):
    if not all_run_data: return {}
    sample_times = sorted(all_run_data[0].keys())
    averaged_results = {t: {} for t in sample_times}
    for t in sample_times:
        for key in keys_to_average:
            all_data_for_t_and_key = [run[t][key] for run in all_run_data if t in run and key in run[t]]
            if all_data_for_t_and_key:
                stacked_data = np.vstack([arr for arr in all_data_for_t_and_key if arr is not None])
                averaged_results[t][key] = np.nanmean(stacked_data, axis=0)
    return averaged_results

# ===================================================================
# 2. Main Execution Block
# ===================================================================
if __name__ == "__main__":
    BASE_PARAMS = {
        'LX': 1.0e-6, 'RHO_INIT': 1.78, 'T_INIT': 273.0,
        'NUM_CELLS_X': 4400, 'DT': 6.744e-13,
        'PARTICLES_PER_CELL_INIT': 100, 'TOTAL_TIME': 0.4e-9, 'RHO_RATIO': 10.0,
    }
    NUM_ENSEMBLE_RUNS = 2500
    
    print("--- Starting STANDARD TAS Simulation ---")
    params_standard = BASE_PARAMS.copy()
    params_standard['METHOD'] = 'standard_tas'
    params_standard['TARGET_PPC_SUBCELL'] = 2.0
    results_standard_all = Parallel(n_jobs=50, verbose=10)(
        delayed(run_dsmc_simulation)(params_standard, i) for i in range(NUM_ENSEMBLE_RUNS)
    )
    averaged_results_standard = average_ensemble_runs(results_standard_all, ['density', 'velocity', 'temperature', 'sof', 'subcell_mfp_ratio'])

    print("\n--- Starting TAI-TAS (Inverse) Simulation ---")
    params_inverse = BASE_PARAMS.copy()
    params_inverse['METHOD'] = 'tai_tas'
    params_inverse['MAX_SUB_CELLS'] = 800
    params_inverse['ADAPTATION_INTERVAL'] = 50
    results_inverse_all = Parallel(n_jobs=50, verbose=10)(
        delayed(run_dsmc_simulation)(params_inverse, i) for i in range(NUM_ENSEMBLE_RUNS)
    )
    averaged_results_inverse = average_ensemble_runs(results_inverse_all, ['density', 'velocity', 'temperature', 'sof', 'subcell_mfp_ratio'])
    
    # --- Plotting: Direct Comparison of Final State ---
    cell_width = BASE_PARAMS['LX'] / BASE_PARAMS['NUM_CELLS_X']
    cell_centers = (np.arange(BASE_PARAMS['NUM_CELLS_X']) + 0.5) * cell_width
    final_time = max(averaged_results_standard.keys())

    plt.rcParams.update({'font.size': 16, 'axes.labelsize': 18, 'axes.titlesize': 20,
                         'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 14})
    
    fig, axes = plt.subplots(3, 2, figsize=(18, 22), sharex=True)
    axes_flat = axes.flatten()
    fig.suptitle(f"Direct Comparison of TAS Methods at t = {final_time*1e9:.2f} ns (N_runs={NUM_ENSEMBLE_RUNS})", fontsize=24)
    
    plot_info = {
        'Density': {'data_key': 'density', 'ax': axes_flat[0], 'ylabel': 'Number Density ($m^{-3}$)'},
        'Velocity': {'data_key': 'velocity', 'ax': axes_flat[1], 'ylabel': 'Bulk Velocity (m/s)'},
        'Temperature': {'data_key': 'temperature', 'ax': axes_flat[2], 'ylabel': 'Temperature (K)'},
        'SOF': {'data_key': 'sof', 'ax': axes_flat[3], 'ylabel': 'SOF Ratio'},
        'Subcell/MFP': {'data_key': 'subcell_mfp_ratio', 'ax': axes_flat[4], 'ylabel': r'Subcell $\Delta$x / MFP'}
    }

    for name, p_info in plot_info.items():
        ax = p_info['ax']
        data_key = p_info['data_key']

        if final_time in averaged_results_standard and data_key in averaged_results_standard[final_time]:
            data_std = pd.Series(averaged_results_standard[final_time][data_key])
            valid_std = ~data_std.isna()
            if np.sum(valid_std) > 51:
                smoothed_std = savgol_filter(data_std[valid_std], 51, 2)
                ax.plot(cell_centers[valid_std], smoothed_std, label='Standard TAS', linestyle='--', lw=2.5)
            else:
                ax.plot(cell_centers[valid_std], data_std[valid_std], label='Standard TAS', linestyle='--', lw=2.5)

        if final_time in averaged_results_inverse and data_key in averaged_results_inverse[final_time]:
            data_inv = pd.Series(averaged_results_inverse[final_time][data_key])
            valid_inv = ~data_inv.isna()
            if np.sum(valid_inv) > 51:
                smoothed_inv = savgol_filter(data_inv[valid_inv], 51, 2)
                ax.plot(cell_centers[valid_inv], smoothed_inv, label='TAI-TAS (Inverse)', linestyle='-', lw=2.0)
            else:
                ax.plot(cell_centers[valid_inv], data_inv[valid_inv], label='TAI-TAS (Inverse)', linestyle='-', lw=2.0)

        ax.set_xlabel('Position x (m)')
        ax.set_ylabel(p_info['ylabel'])
        ax.set_title(f"{name} Profile")
        ax.grid(True, linestyle=':')
        ax.legend()
        
    fig.delaxes(axes_flat[5])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("dsmc_comparison_results.eps", format='eps')
    plt.show()