# -*- coding: utf-8 -*-
"""
DSMC Relaxation Simulation - COMPLETE VERSION with Fixed Cell Count
====================================================================

نسخه کامل با تعداد سلول منطقی:
✅ 9 روش برخوردی کامل: SBT, DCP, NTC, GBT, SSBT, SGBT, MFS, NN, DCP-VR
✅ تعداد سلول‌های منطقی و قابل اجرا
✅ آنالیز آماری کامل
✅ انیمیشن evolution correlation
✅ تمام نمودارها و پارامترها
✅ کنترل probability exceed
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numba
from numba import njit
import time
import gc
from scipy.special import gammaln
import math
import os
import sys

# Environment detection
try:
    import google.colab
    IN_COLAB = True
    print("🔍 Detected Google Colab environment")
except ImportError:
    try:
        if 'GOOGLE_CLOUD_PROJECT' in os.environ or 'GCP_PROJECT' in os.environ:
            IN_COLAB = False
            IN_GOOGLE_CLOUD = True
            print("🔍 Detected Google Cloud Jupyter environment")
        else:
            IN_COLAB = False
            IN_GOOGLE_CLOUD = False
            print("🔍 Running in local environment")
    except:
        IN_COLAB = False
        IN_GOOGLE_CLOUD = False

# Set matplotlib backend
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['savefig.transparent'] = False
plt.rcParams['savefig.facecolor'] = 'white'

try:
    from IPython import get_ipython
    if get_ipython() is not None:
        get_ipython().run_line_magic('matplotlib', 'inline')
        print("📊 Matplotlib inline mode enabled")
except:
    pass

# Method mapping for collision algorithms - UPDATED WITH DCP-VR
METHOD_MAP = {1: 'SBT', 2: 'DCP', 3: 'NTC', 4: 'GBT', 5: 'SSBT', 6: 'SGBT', 7: 'MFS', 8: 'NN', 9: 'DCP-VR'}

# Physical constants
MASS_AR = 39.948e-3 / 6.022e23
KB = 1.380649e-23
D_REF_AR = 4.17e-10
T_REF_AR = 273.0
OMEGA_VHS = 0.50
PI = 3.141592654

# Simulation parameters (will be adjusted dynamically)
LX = 1.0e-6
RHO_INIT = 1.78
T_INIT = 273.0
NUM_CELLS_X = 100
PARTICLES_PER_CELL_INIT = 5
TOTAL_PARTICLES_SIM = 500
N_DENSITY_REAL = RHO_INIT / MASS_AR

CELL_VOLUME_CONCEPTUAL = LX / NUM_CELLS_X
FNUM = (N_DENSITY_REAL * CELL_VOLUME_CONCEPTUAL) / PARTICLES_PER_CELL_INIT
DT = 1.0e-12
TOTAL_TIME = 4.0e-6  # کاهش از 4.0e-7 به 2.0e-7 برای سرعت بیشتر
SAMPLING_INTERVAL = 25

# Statistical analysis parameters
MAXD = 20
MTC = 20
MXC = 10
CENTRAL_CELL_IDX = 50

# Fixed Poisson Distribution Functions
def calculate_poisson_probability_safe(k, lambda_val):
    """Safe Poisson probability calculation using log-space"""
    if lambda_val <= 0:
        return 0.0 if k != 0 else 1.0
    if k < 0:
        return 0.0
    
    try:
        if k == 0:
            return math.exp(-lambda_val)
        log_prob = k * math.log(lambda_val) - lambda_val - gammaln(k + 1)
        return math.exp(log_prob)
    except (OverflowError, ValueError):
        return 0.0

def calculate_poisson_pmf(k_values, lambda_val):
    """Calculate Poisson PMF for an array of k values"""
    probabilities = np.zeros_like(k_values, dtype=float)
    for i, k in enumerate(k_values):
        probabilities[i] = calculate_poisson_probability_safe(int(k), lambda_val)
    return probabilities

# COMPLETE Animation Storage for Auto-Correlation
class CorrelationAnimationData:
    def __init__(self, max_time_steps=200, cells_to_analyze=5):
        self.max_time_steps = max_time_steps
        self.cells_to_analyze = cells_to_analyze
        self.time_snapshots = []
        self.correlation_snapshots = []
        self.time_lags = np.arange(-MTC, MTC + 1) * DT * SAMPLING_INTERVAL
        self.property_names = ['N', 'u', 'v', 'w', 'T']
        self.colors = ['blue', 'green', 'red', 'cyan', 'magenta']
        self.current_step = 0
        
    def add_snapshot(self, time_step, correlation_data, cells_list):
        """Add a correlation snapshot for animation"""
        if self.current_step % max(1, 200 // self.max_time_steps) == 0:
            self.time_snapshots.append(time_step * DT * 1e9)
            
            normalized_corr = np.zeros_like(correlation_data)
            for cell_idx in range(correlation_data.shape[0]):
                for prop_idx in range(correlation_data.shape[1]):
                    zero_lag_var = correlation_data[cell_idx, prop_idx, MTC]
                    if abs(zero_lag_var) > 1e-12:
                        normalized_corr[cell_idx, prop_idx, :] = correlation_data[cell_idx, prop_idx, :] / zero_lag_var
                    else:
                        normalized_corr[cell_idx, prop_idx, :] = np.zeros(correlation_data.shape[2])
            
            self.correlation_snapshots.append(normalized_corr.copy())
            
            if len(self.time_snapshots) > self.max_time_steps:
                self.time_snapshots.pop(0)
                self.correlation_snapshots.pop(0)
        
        self.current_step += 1
    
    def create_animation(self, output_filename='dsmc_correlation_evolution.mp4'):
        """Create animation of correlation function evolution"""
        if len(self.correlation_snapshots) < 10:
            print("Insufficient data for animation. Need at least 10 snapshots.")
            return
        
        print(f"Creating animation with {len(self.correlation_snapshots)} frames...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), facecolor='white')
        fig.patch.set_facecolor('white')
        fig.suptitle('Evolution of Temporal Auto-Correlation Functions', fontsize=16, fontweight='bold', y=0.98)
        
        plt.subplots_adjust(left=0.08, bottom=0.08, right=0.95, top=0.92, wspace=0.25, hspace=0.4)
        axes_flat = axes.flatten()
        
        lines = []
        for prop_idx in range(5):
            ax = axes_flat[prop_idx]
            ax.set_facecolor('white')
            ax.set_title(f'Property: {self.property_names[prop_idx]}', fontsize=12, fontweight='bold', pad=15)
            ax.set_xlabel('Time Lag τ (ns)', fontsize=10)
            ax.set_ylabel('Normalized Correlation R(τ)', fontsize=10)
            ax.tick_params(axis='x', labelsize=16)
            ax.tick_params(axis='y', labelsize=16)
            ax.grid(True, alpha=0.3, color='gray')
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
            ax.set_ylim(-1.2, 1.2)
            ax.set_xlim(self.time_lags[0] * 1e9, self.time_lags[-1] * 1e9)
            
            cell_lines = []
            for cell_idx in range(min(3, self.cells_to_analyze)):
                line, = ax.plot([], [], 
                               color=self.colors[cell_idx], 
                               linewidth=2.5, 
                               alpha=0.9,
                               label=f'Cell {cell_idx+1}')
                cell_lines.append(line)
            ax.legend(loc='upper right', fontsize=8)
            lines.append(cell_lines)
        
        info_ax = axes_flat[5]
        info_ax.axis('off')
        info_ax.set_facecolor('white')
        info_text = info_ax.text(0.05, 0.95, '', fontsize=10, fontweight='bold',
                                verticalalignment='top', fontfamily='monospace',
                                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightblue", alpha=1.0),
                                transform=info_ax.transAxes)
        
        def animate(frame):
            if frame >= len(self.correlation_snapshots):
                return []
            
            current_time = self.time_snapshots[frame]
            current_corr = self.correlation_snapshots[frame]
            time_progress = frame / len(self.correlation_snapshots)
            
            all_artists = []
            for prop_idx in range(5):
                for cell_idx, line in enumerate(lines[prop_idx]):
                    if cell_idx < current_corr.shape[0]:
                        y_data = current_corr[cell_idx, prop_idx, :]
                        if len(y_data) >= 3:
                            y_data_smooth = np.convolve(y_data, np.ones(3)/3, mode='same')
                        else:
                            y_data_smooth = y_data
                        line.set_data(self.time_lags * 1e9, y_data_smooth)
                        all_artists.append(line)
            
            info_text.set_text(f'Simulation Time: {current_time:.2f} ns\n'
                              f'Frame: {frame + 1}/{len(self.correlation_snapshots)}\n'
                              f'Progress: {time_progress*100:.1f}%\n'
                              f'\nCorrelation Analysis:\n'
                              f'• Temporal auto-correlation\n'
                              f'• Multi-cell comparison\n'
                              f'• Real-time evolution\n'
                              f'\nProperties:\n'
                              f'• N: Particle number\n'
                              f'• u,v,w: Velocity components\n'
                              f'• T: Temperature')
            all_artists.append(info_text)
            
            return all_artists
        
        print("🎬 Creating animation...")
        anim = animation.FuncAnimation(fig, animate, frames=len(self.correlation_snapshots),
                                     interval=300, blit=True, repeat=True)
        
        # Try to save animation
        success = False
        
        try:
            print("📹 Attempting MP4 creation...")
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=3, metadata=dict(artist='DSMC Simulation'), bitrate=1200)
            anim.save(output_filename, writer=writer, dpi=80)
            print(f"✅ MP4 animation saved: {output_filename}")
            success = True
        except Exception as e:
            print(f"❌ MP4 creation failed: {e}")
        
        if not success:
            try:
                gif_filename = output_filename.replace('.mp4', '.gif')
                print(f"📸 Attempting GIF creation: {gif_filename}")
                anim.save(gif_filename, writer='pillow', fps=2)
                print(f"✅ GIF animation saved: {gif_filename}")
                success = True
            except Exception as e:
                print(f"❌ GIF creation failed: {e}")
        
        if not success:
            try:
                print("🌐 Creating HTML animation...")
                html_filename = output_filename.replace('.mp4', '.html')
                html_anim = anim.to_jshtml()
                with open(html_filename, 'w') as f:
                    f.write(html_anim)
                
                if IN_COLAB or 'jupyter' in sys.modules:
                    from IPython.display import HTML, display
                    display(HTML(html_anim))
                
                print(f"✅ HTML animation created: {html_filename}")
                success = True
            except Exception as e:
                print(f"❌ HTML animation failed: {e}")
        
        if not success:
            print("❌ All animation creation methods failed!")
        
        return anim

# Initialize animation storage
try:
    correlation_animation = CorrelationAnimationData()
    print("✅ Animation system initialized")
except Exception as e:
    print(f"⚠ Animation initialization failed: {e}")
    correlation_animation = None

# Numba-Jitted Core Functions
@njit(nopython=True)
def gamma_function_approx(x):
    """Gamma function approximation"""
    a = 1.0
    y = x
    if y < 1.0:
        a = a / y
        y = y + 1.0
    while y >= 2.0:
        y = y - 1.0
        a = a * y
    y = y - 1.0
    gamma_poly = 1.0 - 0.5748646*y + 0.9512363*y**2 - 0.6998588*y**3 + 0.4245549*y**4 - 0.1010678*y**5
    return a * gamma_poly

@njit(nopython=True)
def calculate_vhs_sigma_g(vr_mag):
    """Calculate VHS collision cross-section times relative velocity"""
    if vr_mag < 1e-9: return 0.0
    exponent = OMEGA_VHS - 0.5
    c_ref_sq = 2 * KB * T_REF_AR / MASS_AR
    gamma_val = gamma_function_approx(2.5 - OMEGA_VHS)
    sigma_g = (PI * D_REF_AR**2) * vr_mag * ((c_ref_sq / vr_mag**2)**exponent) / gamma_val
    return sigma_g

@njit(nopython=True)
def perform_post_collision(p1_idx, p2_idx, particles, rng_state):
    """Perform post-collision velocity update"""
    vr = particles[p1_idx, 1:4] - particles[p2_idx, 1:4]
    vr_mag = np.sqrt(np.sum(vr**2))
    if vr_mag < 1e-9: return False
    vcm = 0.5 * (particles[p1_idx, 1:4] + particles[p2_idx, 1:4])
    cos_chi = 2 * rng_state.random() - 1.0
    sin_chi = np.sqrt(1.0 - cos_chi**2)
    phi_chi = 2.0 * PI * rng_state.random()
    rand_vec = np.array([sin_chi * np.cos(phi_chi), sin_chi * np.sin(phi_chi), cos_chi])
    vr_prime = vr_mag * rand_vec
    particles[p1_idx, 1:4] = vcm + 0.5 * vr_prime
    particles[p2_idx, 1:4] = vcm - 0.5 * vr_prime
    return True

@njit(nopython=True)
def corrected_sbt_scheme(particles, lx, indices_in_cell, cell_vol, dt, fnum, rng_state):
    """Corrected SBT implementation according to the paper formula"""
    n_accepted = 0
    n_prob_exceed = 0
    num_particles_in_cell = len(indices_in_cell)
    
    if num_particles_in_cell < 2: 
        return 0, float(max(0, num_particles_in_cell - 1)), 0.0, 0

    n_selected = float(num_particles_in_cell - 1)
    sum_sep = 0.0
    
    for i in range(num_particles_in_cell - 1):
        k = num_particles_in_cell - i
        if k <= 0: break
            
        j_offset = int(rng_state.random() * k)
        j_local = (i + 1) + j_offset
        
        if j_local >= num_particles_in_cell: continue
            
        p1_idx, p2_idx = indices_in_cell[i], indices_in_cell[j_local]
        
        vr = particles[p1_idx, 1:4] - particles[p2_idx, 1:4]
        vr_mag = np.sqrt(np.sum(vr**2))
        if vr_mag < 1e-9: continue
            
        sigma_g = calculate_vhs_sigma_g(vr_mag)
        weight_factor = num_particles_in_cell - i
        collision_prob = fnum * weight_factor * sigma_g * dt / cell_vol
        
        if collision_prob > 1.0: n_prob_exceed += 1
        
        if rng_state.random() < collision_prob:
            if perform_post_collision(p1_idx, p2_idx, particles, rng_state):
                n_accepted += 1
                delta_x = np.abs(particles[p1_idx, 0] - particles[p2_idx, 0])
                sum_sep += min(delta_x, lx - delta_x)
    
    return n_accepted, n_selected, sum_sep, n_prob_exceed

@njit(nopython=True)
def standard_sbt_scheme(particles, lx, indices_in_cell, cell_vol, dt, fnum, rng_state):
    """Standard SBT wrapper"""
    n_acc, n_sel, sum_sep, n_prob_exceed = corrected_sbt_scheme(particles, lx, indices_in_cell, cell_vol, dt, fnum, rng_state)
    return n_acc, n_sel, sum_sep

# Fix for the perform_collisions function - replace the problematic sections

@njit(nopython=True)
def perform_collisions(method, particles, lx, indices_in_cell, cell_vol, dt, fnum, rng_state, sigma_g_max_cell, n_sel_fraction, duplicate_check_array):
    """Master collision function implementing all 9 methods - FIXED VERSION with DCP-VR"""
    n_accepted = 0
    n_selected = 0.0
    sum_separation = 0.0
    n_prob_exceed = 0
    num_particles_in_cell = len(indices_in_cell)
    
    if num_particles_in_cell == 0:
        return 0, sigma_g_max_cell, 0.0, 0.0, 0
    elif num_particles_in_cell == 1:
        return 0, sigma_g_max_cell, 0.0, 0.0, 0

    # Calculate effective particles per cell for optimization
    # Use the actual number of particles in this cell as a proxy
    effective_particles_per_cell = float(num_particles_in_cell)

    if method == 1: # SBT - Corrected implementation
        n_acc, n_sel, sum_sep, n_prob_ex = corrected_sbt_scheme(
            particles, lx, indices_in_cell, cell_vol, dt, fnum, rng_state)
        return n_acc, sigma_g_max_cell, n_sel, sum_sep, n_prob_ex

    elif method == 2: # DCP - Distance-based Collision Pairing
        num_pairs = num_particles_in_cell * (num_particles_in_cell - 1) // 2
        if num_pairs == 0: return 0, sigma_g_max_cell, 0.0, 0.0, 0
        pairs = np.empty((num_pairs, 2), dtype=np.int32)
        weights = np.empty(num_pairs, dtype=np.float64)
        pair_count = 0
        for i in range(num_particles_in_cell):
            for j in range(i + 1, num_particles_in_cell):
                p1_idx, p2_idx = indices_in_cell[i], indices_in_cell[j]
                delta_x = np.abs(particles[p1_idx, 0] - particles[p2_idx, 0])
                distance = min(delta_x, lx - delta_x)
                weights[pair_count] = 1.0 / (distance + 1e-10)
                pairs[pair_count, 0] = p1_idx
                pairs[pair_count, 1] = p2_idx
                pair_count += 1
        sum_of_sigma_g = 0.0
        for i in range(num_pairs):
            p1_idx, p2_idx = pairs[i,0], pairs[i,1]
            vr = particles[p1_idx, 1:4] - particles[p2_idx, 1:4]
            sum_of_sigma_g += calculate_vhs_sigma_g(np.sqrt(np.sum(vr**2)))
        expected_collisions = (sum_of_sigma_g * fnum * dt) / cell_vol
        num_collisions_to_perform = int(np.floor(expected_collisions + rng_state.random()))
        n_selected = float(num_collisions_to_perform)
        for _ in range(num_collisions_to_perform):
            if np.sum(weights) < 1e-12: break
            selected_idx = np.argmax(weights)
            p1_idx, p2_idx = pairs[selected_idx, 0], pairs[selected_idx, 1]
            if perform_post_collision(p1_idx, p2_idx, particles, rng_state):
                n_accepted += 1
                delta_x = np.abs(particles[p1_idx, 0] - particles[p2_idx, 0])
                sum_separation += min(delta_x, lx - delta_x)
            weights[selected_idx] = 0.0
        return n_accepted, sigma_g_max_cell, n_selected, sum_separation, n_prob_exceed

    elif method == 9: # DCP-VR - Distance-based Collision Pairing with Velocity-weighted Rates
        num_pairs = num_particles_in_cell * (num_particles_in_cell - 1) // 2
        if num_pairs == 0: return 0, sigma_g_max_cell, 0.0, 0.0, 0
        pairs = np.empty((num_pairs, 2), dtype=np.int32)
        weights = np.empty(num_pairs, dtype=np.float64)
        pair_count = 0
        for i in range(num_particles_in_cell):
            for j in range(i + 1, num_particles_in_cell):
                p1_idx, p2_idx = indices_in_cell[i], indices_in_cell[j]
                delta_x = np.abs(particles[p1_idx, 0] - particles[p2_idx, 0])
                distance = min(delta_x, lx - delta_x)
                # NEW: Calculate sigma_g for this pair and use it in the weight
                vr = particles[p1_idx, 1:4] - particles[p2_idx, 1:4]
                vr_mag = np.sqrt(np.sum(vr**2))
                sigma_g = calculate_vhs_sigma_g(vr_mag)
                weights[pair_count] = sigma_g / (distance + 1e-10)  # This is the key change from DCP
                pairs[pair_count, 0] = p1_idx
                pairs[pair_count, 1] = p2_idx
                pair_count += 1
        sum_of_sigma_g = 0.0
        for i in range(num_pairs):
            p1_idx, p2_idx = pairs[i,0], pairs[i,1]
            vr = particles[p1_idx, 1:4] - particles[p2_idx, 1:4]
            sum_of_sigma_g += calculate_vhs_sigma_g(np.sqrt(np.sum(vr**2)))
        expected_collisions = (sum_of_sigma_g * fnum * dt) / cell_vol
        num_collisions_to_perform = int(np.floor(expected_collisions + rng_state.random()))
        n_selected = float(num_collisions_to_perform)
        for _ in range(num_collisions_to_perform):
            if np.sum(weights) < 1e-12: break
            selected_idx = np.argmax(weights)
            p1_idx, p2_idx = pairs[selected_idx, 0], pairs[selected_idx, 1]
            if perform_post_collision(p1_idx, p2_idx, particles, rng_state):
                n_accepted += 1
                delta_x = np.abs(particles[p1_idx, 0] - particles[p2_idx, 0])
                sum_separation += min(delta_x, lx - delta_x)
            weights[selected_idx] = 0.0
        return n_accepted, sigma_g_max_cell, n_selected, sum_separation, n_prob_exceed

    elif method == 3: # NTC - No Time Counter (FIXED)
        # Adaptive prescan samples based on effective particles per cell
        if effective_particles_per_cell < 1.0:
            if effective_particles_per_cell <= 0.1:
                num_prescan_samples = min(num_particles_in_cell, 50)  
            elif effective_particles_per_cell <= 0.5:
                num_prescan_samples = min(num_particles_in_cell, 75)
            else:
                num_prescan_samples = min(num_particles_in_cell, 90)
        else:
            num_prescan_samples = min(num_particles_in_cell, 100)  
            
        for _ in range(num_prescan_samples):
            p1_idx_local = rng_state.integers(0, num_particles_in_cell)
            p2_idx_local = rng_state.integers(0, num_particles_in_cell)
            while p1_idx_local == p2_idx_local:
                p2_idx_local = rng_state.integers(0, num_particles_in_cell)
           
            p1_idx, p2_idx = indices_in_cell[p1_idx_local], indices_in_cell[p2_idx_local]
            vr = particles[p1_idx, 1:4] - particles[p2_idx, 1:4]
            sigma_g_current = calculate_vhs_sigma_g(np.sqrt(np.sum(vr**2)))
           
            if sigma_g_current > sigma_g_max_cell:
                sigma_g_max_cell = sigma_g_current
        
        num_pairs_float = 0.5 * num_particles_in_cell * (num_particles_in_cell - 1)
        if sigma_g_max_cell <= 0: 
            sigma_g_max_cell = 1e-18
        num_pairs_to_select = int(num_pairs_float * fnum * sigma_g_max_cell * dt / cell_vol + rng_state.random())
        n_selected = float(num_pairs_to_select)
        
        for _ in range(num_pairs_to_select):
            p1_idx_local = rng_state.integers(0, num_particles_in_cell)
            p2_idx_local = rng_state.integers(0, num_particles_in_cell)
            while p1_idx_local == p2_idx_local:
                p2_idx_local = rng_state.integers(0, num_particles_in_cell)
           
            p1_idx, p2_idx = indices_in_cell[p1_idx_local], indices_in_cell[p2_idx_local]
            vr = particles[p1_idx, 1:4] - particles[p2_idx, 1:4]
            sigma_g_current = calculate_vhs_sigma_g(np.sqrt(np.sum(vr**2)))
           
            if sigma_g_current > sigma_g_max_cell:
                sigma_g_max_cell = sigma_g_current
                if perform_post_collision(p1_idx, p2_idx, particles, rng_state):
                    n_accepted += 1
                    delta_x = np.abs(particles[p1_idx, 0] - particles[p2_idx, 0])
                    sum_separation += min(delta_x, lx - delta_x)
            elif rng_state.random() < sigma_g_current / sigma_g_max_cell:
                if perform_post_collision(p1_idx, p2_idx, particles, rng_state):
                    n_accepted += 1
                    delta_x = np.abs(particles[p1_idx, 0] - particles[p2_idx, 0])
                    sum_separation += min(delta_x, lx - delta_x)
        return n_accepted, sigma_g_max_cell, n_selected, sum_separation, n_prob_exceed

    elif method == 4: # GBT - Generalized Bernoulli Trials
        N = num_particles_in_cell
        N_sel = int(n_sel_fraction)
        
        if N_sel >= N - 1 or N_sel < 1 or N < 2:
            n_acc, n_sel_fallback, sum_sep = standard_sbt_scheme(particles, lx, indices_in_cell, cell_vol, dt, fnum, rng_state)
            return n_acc, sigma_g_max_cell, n_sel_fallback, sum_sep, 0
        
        n_selected = float(N_sel)
        
        all_local_indices = np.arange(N)
        for i in range(N_sel):
            k = rng_state.integers(i, N)
            all_local_indices[i], all_local_indices[k] = all_local_indices[k], all_local_indices[i]
        reordered_global_indices = indices_in_cell[all_local_indices]
        
        k_k_const = (N * (N - 1.0)) / (N_sel * (2.0 * N - N_sel - 1.0))
        prob_const = fnum * dt / cell_vol
        
        for i in range(N_sel):
            if i + 1 >= N: continue
            j = rng_state.integers(i + 1, N)
            p1_idx = reordered_global_indices[i]
            p2_idx = reordered_global_indices[j]
            vr = particles[p1_idx, 1:4] - particles[p2_idx, 1:4]
            vr_mag = np.sqrt(np.sum(vr**2))
            if vr_mag < 1e-9: continue
            sigma_g = calculate_vhs_sigma_g(vr_mag)
            k_k = k_k_const * (N - (i + 1.0))
            collision_prob = k_k * prob_const * sigma_g
            if collision_prob > 1.0:
                n_prob_exceed += 1
            if rng_state.random() < collision_prob:
                if perform_post_collision(p1_idx, p2_idx, particles, rng_state):
                    n_accepted += 1
                    delta_x = np.abs(particles[p1_idx, 0] - particles[p2_idx, 0])
                    sum_separation += min(delta_x, lx - delta_x)
        return n_accepted, sigma_g_max_cell, n_selected, sum_separation, n_prob_exceed

    elif method == 5: # SSBT - Simplified Stochastic Bernoulli Trials
        prob_const = fnum * dt / cell_vol
        multiplier = (num_particles_in_cell - 1.0) / 2.0
        n_selected = float(num_particles_in_cell)
        for i_local in range(num_particles_in_cell):
            p1_idx = indices_in_cell[i_local]
            j_local = rng_state.integers(0, num_particles_in_cell)
            while j_local == i_local:
                j_local = rng_state.integers(0, num_particles_in_cell)
            p2_idx = indices_in_cell[j_local]
            vr = particles[p1_idx, 1:4] - particles[p2_idx, 1:4]
            vr_mag = np.sqrt(np.sum(vr**2))
            if vr_mag < 1e-9: continue
            sigma_g = calculate_vhs_sigma_g(vr_mag)
            collision_prob = multiplier * prob_const * sigma_g
            if collision_prob > 1.0:
                n_prob_exceed += 1
            if rng_state.random() < collision_prob:
                if perform_post_collision(p1_idx, p2_idx, particles, rng_state):
                    n_accepted += 1
                    delta_x = np.abs(particles[p1_idx, 0] - particles[p2_idx, 0])
                    sum_separation += min(delta_x, lx - delta_x)
        return n_accepted, sigma_g_max_cell, n_selected, sum_separation, n_prob_exceed

    elif method == 6: # SGBT - Simplified Generalized Bernoulli Trials
        N = num_particles_in_cell
        N_sel = int(n_sel_fraction)
        
        if N_sel >= N - 1 or N_sel < 1 or N < 2:
            n_acc, n_sel_fallback, sum_sep = standard_sbt_scheme(particles, lx, indices_in_cell, cell_vol, dt, fnum, rng_state)
            return n_acc, sigma_g_max_cell, n_sel_fallback, sum_sep, 0
        
        n_selected = float(N_sel)
        
        all_local_indices = np.arange(N)
        for i in range(N_sel):
            k = rng_state.integers(i, N)
            all_local_indices[i], all_local_indices[k] = all_local_indices[k], all_local_indices[i]
        reordered_global_indices = indices_in_cell[all_local_indices]
        
        prob_const = fnum * dt / cell_vol
        multiplier = (N * (N - 1.0)) / (N_sel * 2.0)
        
        for i in range(N_sel):
            p1_idx = reordered_global_indices[i]
            j = rng_state.integers(0, N)
            while j == i:
                j = rng_state.integers(0, N)
            p2_idx = reordered_global_indices[j]
            
            if duplicate_check_array[p1_idx] == p2_idx and duplicate_check_array[p2_idx] == p1_idx:
                continue
            
            vr = particles[p1_idx, 1:4] - particles[p2_idx, 1:4]
            vr_mag = np.sqrt(np.sum(vr**2))
            if vr_mag < 1e-9: continue
            sigma_g = calculate_vhs_sigma_g(vr_mag)
            collision_prob = multiplier * prob_const * sigma_g
            
            if collision_prob > 1.0:
                n_prob_exceed += 1
            
            if rng_state.random() < collision_prob:
                if perform_post_collision(p1_idx, p2_idx, particles, rng_state):
                    n_accepted += 1
                    duplicate_check_array[p1_idx] = p2_idx
                    duplicate_check_array[p2_idx] = p1_idx
                    delta_x = np.abs(particles[p1_idx, 0] - particles[p2_idx, 0])
                    sum_separation += min(delta_x, lx - delta_x)
        return n_accepted, sigma_g_max_cell, n_selected, sum_separation, n_prob_exceed

    elif method == 7: # MFS - Majorant Frequency Scheme (FIXED)
        # Adaptive prescan samples based on effective particles per cell
        if effective_particles_per_cell < 1.0:
            if effective_particles_per_cell <= 0.1:
                num_prescan_samples = min(num_particles_in_cell, 50)
            elif effective_particles_per_cell <= 0.5:
                num_prescan_samples = min(num_particles_in_cell, 75)
            else:
                num_prescan_samples = min(num_particles_in_cell, 90)
        else:
            num_prescan_samples = min(num_particles_in_cell, 100)
            
        for _ in range(num_prescan_samples):
            p1_idx_local = rng_state.integers(0, num_particles_in_cell)
            p2_idx_local = rng_state.integers(0, num_particles_in_cell)
            while p1_idx_local == p2_idx_local:
                p2_idx_local = rng_state.integers(0, num_particles_in_cell)
           
            p1_idx, p2_idx = indices_in_cell[p1_idx_local], indices_in_cell[p2_idx_local]
            vr = particles[p1_idx, 1:4] - particles[p2_idx, 1:4]
            sigma_g_current = calculate_vhs_sigma_g(np.sqrt(np.sum(vr**2)))
           
            if sigma_g_current > sigma_g_max_cell:
                sigma_g_max_cell = sigma_g_current
        
        cell_time = 0.0
        n_sel_counter = 0
        while cell_time < dt:
            num_pairs_float = 0.5 * num_particles_in_cell * (num_particles_in_cell - 1)
            if num_pairs_float < 1 or sigma_g_max_cell <= 0:
                break 
            nu_max = num_pairs_float * fnum * sigma_g_max_cell / cell_vol
            delta_t_c = -np.log(1.0 - rng_state.random()) / nu_max
            cell_time += delta_t_c
            n_sel_counter += 1
            if cell_time < dt:
                p1_idx_local = rng_state.integers(0, num_particles_in_cell)
                p2_idx_local = rng_state.integers(0, num_particles_in_cell)
                while p1_idx_local == p2_idx_local:
                    p2_idx_local = rng_state.integers(0, num_particles_in_cell)
                p1_idx, p2_idx = indices_in_cell[p1_idx_local], indices_in_cell[p2_idx_local]
                vr = particles[p1_idx, 1:4] - particles[p2_idx, 1:4]
                sigma_g_current = calculate_vhs_sigma_g(np.sqrt(np.sum(vr**2)))
                
                if sigma_g_current > sigma_g_max_cell:
                    sigma_g_max_cell = sigma_g_current
                    if perform_post_collision(p1_idx, p2_idx, particles, rng_state):
                        n_accepted += 1
                        delta_x = np.abs(particles[p1_idx, 0] - particles[p2_idx, 0])
                        sum_separation += min(delta_x, lx - delta_x)
                elif rng_state.random() < sigma_g_current / sigma_g_max_cell:
                    if perform_post_collision(p1_idx, p2_idx, particles, rng_state):
                        n_accepted += 1
                        delta_x = np.abs(particles[p1_idx, 0] - particles[p2_idx, 0])
                        sum_separation += min(delta_x, lx - delta_x)
        return n_accepted, sigma_g_max_cell, float(n_sel_counter), sum_separation, n_prob_exceed

    elif method == 8: # NN - Nearest Neighbor (FIXED)
        # Adaptive prescan samples based on effective particles per cell
        if effective_particles_per_cell < 1.0:
            if effective_particles_per_cell <= 0.1:
                num_prescan_samples = min(num_particles_in_cell, 50)
            elif effective_particles_per_cell <= 0.5:
                num_prescan_samples = min(num_particles_in_cell, 75)
            else:
                num_prescan_samples = min(num_particles_in_cell, 90)
        else:
            num_prescan_samples = min(num_particles_in_cell, 100)
            
        for _ in range(num_prescan_samples):
            p1_idx_local = rng_state.integers(0, num_particles_in_cell)
            p2_idx_local = rng_state.integers(0, num_particles_in_cell)
            while p1_idx_local == p2_idx_local:
                p2_idx_local = rng_state.integers(0, num_particles_in_cell)
           
            p1_idx, p2_idx = indices_in_cell[p1_idx_local], indices_in_cell[p2_idx_local]
            vr = particles[p1_idx, 1:4] - particles[p2_idx, 1:4]
            sigma_g_current = calculate_vhs_sigma_g(np.sqrt(np.sum(vr**2)))
           
            if sigma_g_current > sigma_g_max_cell:
                sigma_g_max_cell = sigma_g_current
        
        num_pairs_float = 0.5 * num_particles_in_cell * (num_particles_in_cell - 1)
        if sigma_g_max_cell <= 0: 
            sigma_g_max_cell = 1e-18
        num_pairs_to_select = int(num_pairs_float * fnum * sigma_g_max_cell * dt / cell_vol + rng_state.random())
        n_selected = float(num_pairs_to_select)
        if n_selected == 0:
            return 0, sigma_g_max_cell, 0.0, 0.0, 0
            
        N = num_particles_in_cell
        cell_positions = particles[indices_in_cell, 0]
        dist_matrix = np.full((N, N), np.finfo(np.float64).max, dtype=np.float64)
        for i in range(N):
            for j in range(i + 1, N):
                delta_x = np.abs(cell_positions[i] - cell_positions[j])
                dist = min(delta_x, lx - delta_x)
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        nearest_neighbor_map = np.argmin(dist_matrix, axis=1)
        
        for _ in range(num_pairs_to_select):
            p1_local_idx = rng_state.integers(0, N)
            p2_local_idx = nearest_neighbor_map[p1_local_idx]
            p1_idx, p2_idx = indices_in_cell[p1_local_idx], indices_in_cell[p2_local_idx]
            vr = particles[p1_idx, 1:4] - particles[p2_idx, 1:4]
            sigma_g_current = calculate_vhs_sigma_g(np.sqrt(np.sum(vr**2)))
            
            if sigma_g_current > sigma_g_max_cell:
                sigma_g_max_cell = sigma_g_current
                if perform_post_collision(p1_idx, p2_idx, particles, rng_state):
                    n_accepted += 1
                    sum_separation += dist_matrix[p1_local_idx, p2_local_idx]
            elif rng_state.random() < sigma_g_current / sigma_g_max_cell:
                if perform_post_collision(p1_idx, p2_idx, particles, rng_state):
                    n_accepted += 1
                    sum_separation += dist_matrix[p1_local_idx, p2_local_idx]
        return n_accepted, sigma_g_max_cell, n_selected, sum_separation, n_prob_exceed

    return 0, sigma_g_max_cell, 0.0, 0.0, 0

@njit(nopython=True)
def calculate_theoretical_collision_frequency(avg_temp, n_density):
    """Calculate theoretical collision frequency using VHS model"""
    d_ref_sq = D_REF_AR**2
    temp_ratio_term = (avg_temp / T_REF_AR)**(1.0 - OMEGA_VHS)
    sqrt_term = np.sqrt(PI * KB * T_REF_AR / MASS_AR)
    cf_th = 4.0 * n_density * d_ref_sq * sqrt_term * temp_ratio_term
    return cf_th

def calculate_numerical_collision_frequency(num_sim_collisions, num_particles, time_interval):
    """Calculate numerical collision frequency"""
    if num_particles == 0 or time_interval == 0:
        return 0.0
    cf_num = num_sim_collisions / (0.5 * num_particles * time_interval)
    return cf_num

def calculate_max_collision_probability(particles_per_cell, sigma_g_max_estimate, fnum, dt, cell_vol, method='SBT'):
    """Calculate maximum collision probability for different methods"""
    if particles_per_cell < 2:
        return 0.0
    
    if method in ['SBT', 'DCP', 'DCP-VR', 'SSBT']:
        # SBT-based methods
        max_multiplier = particles_per_cell
        max_prob = fnum * max_multiplier * sigma_g_max_estimate * dt / cell_vol
    elif method in ['NTC', 'MFS', 'NN']:
        # NTC-based methods که از sigma_g_max استفاده می‌کنند
        num_pairs = 0.5 * particles_per_cell * (particles_per_cell - 1)
        max_prob = fnum * num_pairs * sigma_g_max_estimate * dt / cell_vol
    elif method in ['GBT', 'SGBT']:
        # GBT-based methods (approximate)
        max_multiplier = particles_per_cell * 0.5  # تقریبی
        max_prob = fnum * max_multiplier * sigma_g_max_estimate * dt / cell_vol
    else:
        # Default to SBT formula
        max_multiplier = particles_per_cell
        max_prob = fnum * max_multiplier * sigma_g_max_estimate * dt / cell_vol
    
    return max_prob

def calculate_sbt_max_probability(particles_per_cell, sigma_g_max_estimate, fnum, dt, cell_vol):
    """Calculate maximum collision probability for SBT method (backward compatibility)"""
    return calculate_max_collision_probability(particles_per_cell, sigma_g_max_estimate, fnum, dt, cell_vol, 'SBT')

def calculate_simulation_parameters(particles_per_cell, n_sel_for_gbt_sgbt=None):
    """Calculate simulation parameters - FIXED VERSION (Logical Cell Count) with Exceed Control"""
    global PARTICLES_PER_CELL_INIT, TOTAL_PARTICLES_SIM, FNUM, DT, NUM_CELLS_X, CELL_VOLUME_CONCEPTUAL, CENTRAL_CELL_IDX
    
    PARTICLES_PER_CELL_INIT = particles_per_cell
    
    # Physical constants
    sigma_ref = PI * D_REF_AR**2
    mfp = 1 / (np.sqrt(2) * sigma_ref * N_DENSITY_REAL)
    cm = np.sqrt(2 * KB * T_INIT / MASS_AR)
    tc = mfp / cm
    
    print(f"\n=== Physical Parameters ===")
    print(f"Mean free path (λ): {mfp*1e6:.3f} μm")
    print(f"Most probable velocity (cm): {cm:.1f} m/s") 
    print(f"Mean collision time (tc): {tc*1e12:.3f} ps")
    print(f"Reference cross-section: {sigma_ref:.2e} m²")
    
    # Maximum sigma_g estimate
    v_thermal = np.sqrt(2 * KB * T_INIT / MASS_AR)
    v_max_estimate = 4.0 * v_thermal
    sigma_g_max_estimate = PI * D_REF_AR**2 * v_max_estimate
    
    print(f"Thermal velocity: {v_thermal:.1f} m/s")
    print(f"Maximum velocity estimate: {v_max_estimate:.1f} m/s")
    print(f"Maximum sigma_g estimate: {sigma_g_max_estimate:.2e} m²⋅m/s")
    
    # FIXED: رویکرد منطقی برای محاسبه تعداد سلول‌ها
    # شروع از محدودیت فیزیکی اصلی: dx < λ/3
    min_cells_physics = LX / (mfp / 3.0)  # اصلی‌ترین محدودیت
    
    # FIXED: Grid Refinement approach for low exceed with fractional particles
    if particles_per_cell < 1.0:
        # برای ذرات کسری، شبکه ریزتر برای کاهش exceed
        if particles_per_cell <= 0.1:
            print(f"🔬 ULTRA FINE GRID MODE (≤ 0.1):")
            # برای کاهش exceed از 20% به 5%، نیاز به شبکه خیلی ریز
            base_cells = max(min_cells_physics, 550)  # افزایش از 50 به 300
            #safety_multiplier = 4.50  # افزایش قابل توجه برای ریزسازی
            safety_multiplier = 1.5 + (1.0 - particles_per_cell) * 5.0  # حداکثر 3.5
            NUM_CELLS_X = int(base_cells * safety_multiplier)
            print(f"  Fine mesh for exceed reduction")
            print(f"  Base cells: {base_cells:.0f}")
            print(f"  Grid refinement multiplier: {safety_multiplier:.1f}")
            print(f"  Target: Reduce exceed from ~20% to ~5%")
        elif particles_per_cell <= 0.2:
            print(f"🔬 FINE GRID MODE (0.1-0.2):")
            base_cells = max(min_cells_physics, 250)
            safety_multiplier = 2.5
            NUM_CELLS_X = int(base_cells * safety_multiplier)
            print(f"  Fine mesh optimization")
            print(f"  Base cells: {base_cells:.0f}")
            print(f"  Refinement multiplier: {safety_multiplier:.1f}")
        elif particles_per_cell <= 0.5:
            print(f"🔬 MEDIUM GRID MODE (0.2-0.5):")
            base_cells = max(min_cells_physics, 200)
            safety_multiplier = 2.0
            NUM_CELLS_X = int(base_cells * safety_multiplier)
            print(f"  Medium mesh optimization")
            print(f"  Base cells: {base_cells:.0f}")
            print(f"  Multiplier: {safety_multiplier:.1f}")
        else:
            print(f"🔧 FRACTIONAL MODE (0.5-1.0):")
            base_cells = max(min_cells_physics, 150)
            safety_multiplier = 1.4
            NUM_CELLS_X = int(base_cells * safety_multiplier)
            print(f"  Fractional optimization")
            print(f"  Base cells: {base_cells:.0f}")
            print(f"  Multiplier: {safety_multiplier:.1f}")
    else:
        # برای ذرات عادی، استفاده از محدودیت فیزیکی با حاشیه ایمنی کم
        NUM_CELLS_X = int(min_cells_physics * 1.2)  # فقط 20% حاشیه ایمنی
        print(f"Normal particles approach:")
        print(f"  Physics requirement: {min_cells_physics:.0f} cells")
        print(f"  Safety margin: 20%")
    
    print(f"\n🎯 LOGICAL: Starting with {NUM_CELLS_X} cells")
    print(f"⚡ Fine mesh applies to ALL collision methods (SBT, NTC, DCP, DCP-VR, etc.)")
    if particles_per_cell < 1.0:
        print(f"🔬 Mesh refinement active for fractional particles")
        print(f"📊 All methods (SBT, NTC, MFS, NN, DCP, DCP-VR, etc.) will use fine mesh")
        print(f"🎯 Target: Reduce exceed from ~20% to ~5% for all methods")
    
    # Show multiplier calculations for GBT/SGBT if relevant
    if n_sel_for_gbt_sgbt is not None:
        N = particles_per_cell
        N_sel = n_sel_for_gbt_sgbt
        if N_sel < N - 1 and N_sel >= 1:
            gbt_mult_const = (N * (N - 1.0)) / (N_sel * (2.0 * N - N_sel - 1.0))
            gbt_mult_max = gbt_mult_const * (N - 1.0)
            sgbt_mult = (N * (N - 1.0)) / (N_sel * 2.0)
            print(f"For N={N}, N_sel={N_sel}:")
            print(f"  GBT multiplier: constant={gbt_mult_const:.2f}, maximum={gbt_mult_max:.2f}")
            print(f"  SGBT multiplier: {sgbt_mult:.2f}")
            
            # Apply additional scaling for high multipliers
            max_multiplier = max(gbt_mult_max, sgbt_mult)
            if max_multiplier > 2.0:
                multiplier_safety = np.sqrt(max_multiplier / 2.0)
                NUM_CELLS_X = int(NUM_CELLS_X * multiplier_safety)
                print(f"  Additional scaling for high multipliers: {multiplier_safety:.2f}")
                print(f"  Final cells: {NUM_CELLS_X}")
    
    # FIXED: Very aggressive target probability for fine mesh exceed control
    if particles_per_cell <= 0.1:
        target_prob = 0.15  # خیلی سخت‌گیرانه برای کاهش exceed به 5%
        max_iterations = 25  # تکرارهای بیشتر برای تنظیم دقیق
        print(f"🎯 ULTRA FINE MESH: Target probability < {target_prob}")
        print(f"🔬 Grid refinement for exceed reduction: 20% → 5%")
    elif particles_per_cell <= 0.2:
        target_prob = 0.25  # سخت‌گیرانه برای شبکه ریز
        max_iterations = 22
        print(f"🎯 FINE MESH: Target probability < {target_prob}")
    elif particles_per_cell <= 0.5:
        target_prob = 0.35  # متوسط برای تراکم کم
        max_iterations = 20
        print(f"🎯 MEDIUM MESH: Target probability < {target_prob}")
    elif particles_per_cell < 1.0:
        target_prob = 0.55  # متوسط برای کسری
        max_iterations = 15
        print(f"🎯 FRACTIONAL: Target probability < {target_prob}")
    else:
        target_prob = 0.60  # عادی
        max_iterations = 15
        print(f"🎯 NORMAL: Target probability < {target_prob}")
    
    print(f"\n=== FINE MESH Parameter Tuning (Exceed Control) ===")
    print(f"Target: dx/dt = 1.0 AND collision probability < {target_prob}")
    print(f"🎯 Goal: Achieve exceed ratio < 5% through mesh refinement")
    
    for iteration in range(max_iterations):
        cell_width = LX / NUM_CELLS_X
        CELL_VOLUME_CONCEPTUAL = cell_width
        TOTAL_PARTICLES_SIM = int(NUM_CELLS_X * PARTICLES_PER_CELL_INIT)
        FNUM = (N_DENSITY_REAL * CELL_VOLUME_CONCEPTUAL) / PARTICLES_PER_CELL_INIT
        
        # محاسبه گام زمانی برای حفظ dx/dt = 1
        dx_over_mfp = cell_width / mfp
        dt_over_tc = dx_over_mfp  # اجبار dx/dt = 1
        DT = dt_over_tc * tc
        
        # بررسی محدودیت‌ها
        physics_ok = dx_over_mfp < (1.0/3.0)  # dx < λ/3
        stability_ok = dt_over_tc < 1.0        # dt < tc
        
        # محاسبه احتمال برخورد
        sbt_prob = calculate_sbt_max_probability(PARTICLES_PER_CELL_INIT, sigma_g_max_estimate, FNUM, DT, CELL_VOLUME_CONCEPTUAL)
        prob_ok = sbt_prob < target_prob
        
        # پیش‌بینی probability exceed
        expected_exceed = max(0, (sbt_prob - 1.0) / sbt_prob * 100) if sbt_prob > 1.0 else 0
        
        print(f"Iter {iteration}: Cells={NUM_CELLS_X}, P={sbt_prob:.3f}, Expected_exceed={expected_exceed:.1f}%")
        
        # اگر همه شرایط OK است و exceed خیلی کم است، خروج
        if physics_ok and stability_ok and prob_ok and expected_exceed < 3.0:
            print(f"✓ Converged with very low exceed after {iteration+1} iterations!")
            break
        elif physics_ok and stability_ok and prob_ok and expected_exceed < 6.0:
            print(f"✓ Converged with acceptable exceed after {iteration+1} iterations!")
            break
        
        # تنظیم adaptive برای کاهش exceed
        if expected_exceed > 25.0:  # exceed خیلی بالا
            # افزایش بسیار قوی
            increase = max(int(NUM_CELLS_X * 0.3), 100)
            NUM_CELLS_X += increase
            print(f"  → EMERGENCY: Massive increase by {increase} (critical exceed)")
        elif expected_exceed > 15.0:
            # افزایش قوی
            increase = max(int(NUM_CELLS_X * 0.25), 75)
            NUM_CELLS_X += increase
            print(f"  → CRITICAL: Large increase by {increase} (very high exceed)")
        elif expected_exceed > 10.0 or not prob_ok:
            # افزایش متوسط
            increase = max(int(NUM_CELLS_X * 0.2), 50)
            NUM_CELLS_X += increase
            print(f"  → AGGRESSIVE: Increase by {increase} (high exceed)")
        elif not physics_ok:
            increase = max(int(NUM_CELLS_X * 0.15), 30)
            NUM_CELLS_X += increase
            print(f"  → Increase by {increase} (dx too large)")
        elif expected_exceed > 5.0:
            # حتی برای exceed متوسط هم افزایش بدهیم
            increase = max(int(NUM_CELLS_X * 0.1), 25)
            NUM_CELLS_X += increase
            print(f"  → Moderate increase by {increase} (moderate exceed)")
        
        # محدودیت بالایی adaptive برای mesh refinement
        if particles_per_cell <= 0.1:
            max_cells_limit = 3000  # حد بالا برای شبکه خیلی ریز
        elif particles_per_cell <= 0.2:
            max_cells_limit = 2500
        elif particles_per_cell <= 0.5:
            max_cells_limit = 2000
        else:
            max_cells_limit = 1500
            
        if NUM_CELLS_X > max_cells_limit:
            print(f"🔬 MESH REFINEMENT LIMIT: Cells capped at {max_cells_limit} (was {NUM_CELLS_X})")
            NUM_CELLS_X = max_cells_limit
            break
        
        if iteration == max_iterations - 1:
            print(f"⚠ Reached iteration limit - May have some exceed")
    
    # چک نهایی
    final_sbt_prob = calculate_sbt_max_probability(PARTICLES_PER_CELL_INIT, sigma_g_max_estimate, FNUM, DT, CELL_VOLUME_CONCEPTUAL)
    final_expected_exceed = max(0, (final_sbt_prob - 1.0) / final_sbt_prob * 100) if final_sbt_prob > 1.0 else 0
    
    if final_expected_exceed > 15.0:
        print(f"🚨 HIGH EXCEED WARNING: Expected {final_expected_exceed:.1f}%")
        print(f"🔧 Consider: More cells or smaller time steps")
    elif final_expected_exceed > 8.0:
        print(f"⚠ MODERATE EXCEED: Expected {final_expected_exceed:.1f}%")
        print(f"💡 Results still useful but not ideal")
    else:
        print(f"✅ LOW EXCEED: Expected {final_expected_exceed:.1f}%")
        print(f"🎯 Good probability control!")
    
    # Update central cell index
    CENTRAL_CELL_IDX = NUM_CELLS_X // 2
    
    # Final calculations
    cell_width = LX / NUM_CELLS_X
    CELL_VOLUME_CONCEPTUAL = cell_width
    TOTAL_PARTICLES_SIM = int(NUM_CELLS_X * PARTICLES_PER_CELL_INIT)
    FNUM = (N_DENSITY_REAL * CELL_VOLUME_CONCEPTUAL) / PARTICLES_PER_CELL_INIT
    
    dx_over_mfp = cell_width / mfp
    dt_over_tc = DT / tc
    dx_dt_ratio = dx_over_mfp / dt_over_tc
    
    final_sbt_prob = calculate_sbt_max_probability(PARTICLES_PER_CELL_INIT, sigma_g_max_estimate, FNUM, DT, CELL_VOLUME_CONCEPTUAL)
    final_expected_exceed = max(0, (final_sbt_prob - 1.0) / final_sbt_prob * 100) if final_sbt_prob > 1.0 else 0
    
    print(f"\n=== FINAL Parameters (Fine Mesh Exceed Control) ===")
    print(f"🎯 Particles per cell: {PARTICLES_PER_CELL_INIT:.2f}")
    print(f"✅ Total particles: {TOTAL_PARTICLES_SIM:,}")
    print(f"🔬 Number of cells: {NUM_CELLS_X:,} (FINE MESH FOR EXCEED CONTROL)")
    print(f"📏 Cell width: {cell_width*1e9:.1f} nm (REFINED)")
    print(f"⏱️ Time step (dt): {DT*1e12:.1f} ps (AUTO-ADJUSTED)")
    print(f"🔢 FNUM: {FNUM:.3e}")
    
    print(f"\n✅ Physical Checks:")
    print(f"Cell size / λ ratio: {dx_over_mfp:.4f} (< 0.333: {dx_over_mfp < 1.0/3.0})")
    print(f"dt / tc ratio: {dt_over_tc:.4f} (< 1.0: {dt_over_tc < 1.0})")
    print(f"dx/dt ratio: {dx_dt_ratio:.6f} (≈ 1.0)")
    
    final_sbt_prob = calculate_sbt_max_probability(PARTICLES_PER_CELL_INIT, sigma_g_max_estimate, FNUM, DT, CELL_VOLUME_CONCEPTUAL)
    final_expected_exceed = max(0, (final_sbt_prob - 1.0) / final_sbt_prob * 100) if final_sbt_prob > 1.0 else 0
    
    print(f"\n🎯 Mesh Refinement Results:")
    print(f"Max collision probability: {final_sbt_prob:.4f}")
    print(f"Expected exceed ratio: {final_expected_exceed:.1f}%")
    
    if final_sbt_prob < 1.0:
        print(f"✅ Perfect: No probability exceed!")
        performance = "PERFECT"
    elif final_expected_exceed < 3.0:
        print(f"✅ Excellent: Very low exceed (TARGET ACHIEVED!)")
        performance = "EXCELLENT"
    elif final_expected_exceed < 6.0:
        print(f"✅ Very Good: Low exceed (Close to target)")
        performance = "VERY_GOOD"
    elif final_expected_exceed < 10.0:
        print(f"⚡ Good: Moderate exceed")
        performance = "GOOD"
    else:
        print(f"⚠ Warning: Still high exceed - May need even finer mesh")
        performance = "NEEDS_MORE_REFINEMENT"
    
    # Mesh refinement analysis
    if particles_per_cell <= 0.1:
        coarse_cells = int(min_cells_physics * 1.2)  # تخمین سلول‌های درشت
        refinement_factor = NUM_CELLS_X / coarse_cells
        print(f"\n🔬 Mesh Refinement Analysis:")
        print(f"Coarse mesh estimate: {coarse_cells} cells")
        print(f"Fine mesh actual: {NUM_CELLS_X} cells")
        print(f"Refinement factor: {refinement_factor:.1f}x")
        print(f"Expected computational cost: {refinement_factor:.1f}x higher")
        
        if final_expected_exceed <= 5.0:
            print(f"🎯 SUCCESS: Exceed reduced to ≤ 5% through mesh refinement!")
        else:
            print(f"💡 For exceed < 5%, try refinement factor: {refinement_factor * 1.5:.1f}x")
    
    print(f"\n🏆 Performance Level: {performance}")
    
    # منطق بررسی نهایی
    if particles_per_cell < 1.0:
        expected_cells_with_particles = int(NUM_CELLS_X * particles_per_cell)
        empty_cell_ratio = (NUM_CELLS_X - expected_cells_with_particles) / NUM_CELLS_X * 100
        print(f"\n📊 Fine Mesh Particle Analysis:")
        print(f"Expected cells with particles: {expected_cells_with_particles:,}")
        print(f"Expected empty cells: {NUM_CELLS_X - expected_cells_with_particles:,}")
        print(f"Empty cell ratio: {empty_cell_ratio:.1f}%")
        print(f"Particle density efficiency: {particles_per_cell*100:.0f}%")
    
    print(f"====================================\n")
    
    return DT

def initialize_particles(rng_state):
    """Initialize particles with Maxwell-Boltzmann velocity distribution"""
    particles = np.zeros((TOTAL_PARTICLES_SIM, 4))
    print("Initializing particles with Maxwell-Boltzmann velocity distribution...")
    particles[:, 0] = rng_state.random(TOTAL_PARTICLES_SIM) * LX
    v_thermal_std = np.sqrt(KB * T_INIT / MASS_AR)
    particles[:, 1:4] = rng_state.normal(0, v_thermal_std, (TOTAL_PARTICLES_SIM, 3))
    particles[:, 1:4] -= np.mean(particles[:, 1:4], axis=0)
    print(f"Initialized {TOTAL_PARTICLES_SIM} particles.")
    return particles

def plot_results(sampled_speeds, final_temp, method_name, freq_ratio_history, time_history, sof_history, acceptance_ratio, mfp, prob_exceed_history):
    """Plot comprehensive results"""
    plt.rcParams.update({'font.size': 26, 'axes.titlesize': 28, 'axes.labelsize': 26,
                         'xtick.labelsize': 20, 'ytick.labelsize': 20, 'legend.fontsize': 22})
    fig, axes = plt.subplots(2, 2, figsize=(22, 18))
    fig.suptitle(f'DSMC Relaxation Analysis - COMPLETE (Method: {method_name})', fontsize=22)
    ax1, ax2, ax3, ax4 = axes.flatten()

    # Plot 1: Final Speed Distribution
    if len(sampled_speeds) > 0:
        ax1.hist(sampled_speeds, bins=100, density=True, label='DSMC Results', alpha=0.7, color='dodgerblue')
        v_max_range = np.max(sampled_speeds) * 1.15
        v_theory = np.linspace(0, v_max_range, 500)
        pv_theory = (4 * np.pi * (MASS_AR / (2 * np.pi * KB * final_temp))**1.5 * v_theory**2 * np.exp(-MASS_AR * v_theory**2 / (2 * KB * final_temp)))
        ax1.plot(v_theory, pv_theory, 'r-', linewidth=2.5, label=f'Maxwell-Boltzmann (T={final_temp:.1f}K)')
    ax1.set_xlabel('Speed (m/s)'); ax1.set_ylabel('Probability Density'); ax1.set_title('Final Speed Distribution')
    ax1.legend(); ax1.grid(True, linestyle=':')

    # Plot 2: Collision Frequency Ratio vs. Time
    if time_history and freq_ratio_history and len(freq_ratio_history) > 0:
        plot_len = min(len(time_history), len(freq_ratio_history))
        ax2.plot(np.array(time_history[:plot_len]) * 1e9, freq_ratio_history[:plot_len], 'g-')
        ax2.axhline(1.0, color='r', linestyle='--', label='Ideal Ratio = 1.0')
    else:
        ax2.text(0.5, 0.5, 'No collision frequency data', 
                transform=ax2.transAxes, ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
    ax2.set_xlabel('Time (ns)'); ax2.set_ylabel('Collision Frequency Ratio'); ax2.set_title('Evolution of Collision Frequency Ratio')
    ax2.legend(); ax2.grid(True, linestyle=':'); ax2.set_ylim(0, 1.2)
    
    # Plot 3: Mean Separation Distance vs. Time
    if time_history and sof_history:
        plot_len = min(len(time_history), len(sof_history))
        ax3.plot(np.array(time_history[:plot_len]) * 1e9, sof_history[:plot_len], 'm-')
    ax3.set_xlabel('Time (ns)'); ax3.set_ylabel('Mean Separation / MFP'); ax3.set_title('Evolution of Collision Separation (SoF)')
    ax3.grid(True, linestyle=':'); ax3.set_ylim(bottom=0)

    # Plot 4: Probability Exceed Ratio vs. Time
    if time_history and prob_exceed_history:
        plot_len = min(len(time_history), len(prob_exceed_history))
        ax4.plot(np.array(time_history[:plot_len]) * 1e9, [ratio * 100 for ratio in prob_exceed_history[:plot_len]], 'orange', linewidth=2)
    ax4.set_xlabel('Time (ns)'); ax4.set_ylabel('Probability Exceed Ratio (%)'); ax4.set_title('Percentage of Collisions with P > 1')
    ax4.grid(True, linestyle=':'); ax4.set_ylim(bottom=0)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_filename = f'DSMC_Complete_Results_{method_name}.eps'
    plt.savefig(output_filename, format='eps')
    print(f"\nComplete results plot saved as {output_filename}")
    plt.show()

def plot_advanced_stats(density_fluctuation_hist, temporal_corr_sums_multi, spatial_corr_sums, nsmpt_multi, nsmpx, navg_theory, method_name, dt, cell_width, cells_to_analyze):
    """Plot advanced statistical analysis - COMPLETE"""
    plt.rcParams.update({'font.size': 24, 'axes.titlesize': 26, 'axes.labelsize': 24, 'xtick.labelsize': 20, 'ytick.labelsize': 20, 'legend.fontsize': 20})
    fig, axes = plt.subplots(3, 1, figsize=(15, 24))
    fig.suptitle(f'Advanced Statistical Analysis - COMPLETE (Method: {method_name})', fontsize=26)
    
    # Plot 1: Density Fluctuation Distribution
    ax1 = axes[0]
    dev_range = np.arange(-MAXD, MAXD + 1)
    total_samples = np.sum(density_fluctuation_hist)
    if total_samples > 0:
        prob_density_fluc = density_fluctuation_hist / total_samples
        ax1.bar(dev_range, prob_density_fluc, width=1.0, label='DSMC Simulation', alpha=0.7, color='skyblue')
        
        k_values = dev_range + navg_theory
        valid_indices = k_values >= 0
        valid_k = k_values[valid_indices]
        
        if navg_theory > 0:
            poisson_prob_full = np.zeros_like(k_values, dtype=float)
            poisson_prob_valid = calculate_poisson_pmf(valid_k, navg_theory)
            poisson_prob_full[valid_indices] = poisson_prob_valid
            ax1.plot(dev_range, poisson_prob_full, 'r--', linewidth=3, 
                    label=f'Poisson Theory (λ={navg_theory:.2f})', markersize=4)
            
            observed = prob_density_fluc[valid_indices]
            expected = poisson_prob_valid
            if len(observed) > 0 and len(expected) > 0:
                chi_square = np.sum((observed - expected)**2 / (expected + 1e-10))
                ax1.text(0.02, 0.98, f'χ² = {chi_square:.3f}', transform=ax1.transAxes, 
                        verticalalignment='top', fontsize=12, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax1.set_title('Number Density Fluctuation Distribution'); 
    ax1.set_xlabel('Deviation from Mean (N - <N>)'); 
    ax1.set_ylabel('Probability')
    ax1.legend(); ax1.grid(True, linestyle=':'); ax1.set_xlim(left=-MAXD, right=MAXD)

    # Plot 2: Multi-Cell Temporal Auto-Correlation
    ax2 = axes[1]
    time_lags = np.arange(-MTC, MTC + 1) * dt * SAMPLING_INTERVAL
    prop_labels = ['N', 'u', 'v', 'w', 'T']
    colors = ['b', 'g', 'r', 'c', 'm']
    
    if nsmpt_multi > MTC:
        print(f"Temporal correlation samples used: {nsmpt_multi - MTC}")
        
        def smooth_data(data, window_size=3):
            if len(data) < window_size:
                return data
            smoothed = np.zeros_like(data)
            for i in range(len(data)):
                start_idx = max(0, i - window_size//2)
                end_idx = min(len(data), i + window_size//2 + 1)
                smoothed[i] = np.mean(data[start_idx:end_idx])
            return smoothed
        
        for prop_idx in range(5):
            for cell_idx, cell_num in enumerate(cells_to_analyze):
                zero_lag_variance = temporal_corr_sums_multi[cell_idx, prop_idx, MTC] / (nsmpt_multi - MTC)
                if abs(zero_lag_variance) > 1e-9:
                    normalized_corr = temporal_corr_sums_multi[cell_idx, prop_idx, :] / (nsmpt_multi - MTC) / zero_lag_variance
                    smoothed_corr = smooth_data(normalized_corr, window_size=5)
                    
                    line_style = '-' if cell_idx == 0 else '--' if cell_idx == 1 else '-.' if cell_idx == 2 else ':' if cell_idx == 3 else '-'
                    alpha = 0.9 if cell_idx == 0 else 0.7
                    linewidth = 2.0 if cell_idx == 0 else 1.5
                    label = f'{prop_labels[prop_idx]}' if cell_idx == 0 else ""
                    
                    ax2.plot(time_lags * 1e9, smoothed_corr, 
                            color=colors[prop_idx], linestyle=line_style, alpha=alpha,
                            label=label, linewidth=linewidth, 
                            marker='o' if cell_idx == 0 and prop_idx < 2 else None, markersize=3)
    else:
        ax2.text(0.5, 0.5, f'Insufficient data: {nsmpt_multi} samples (need > {MTC})', 
                transform=ax2.transAxes, ha='center', va='center', fontsize=14, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
    
    ax2.set_title(f'Multi-Cell Temporal Auto-Correlation (Smoothed, N_samples={nsmpt_multi-MTC if nsmpt_multi > MTC else 0})'); 
    ax2.set_xlabel('Time Lag, τ (ns)'); ax2.set_ylabel('Normalized Correlation, R(τ)')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); ax2.grid(True, linestyle=':')
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax2.set_ylim(-1.1, 1.1)
    
    info_text = f"Cells: {cells_to_analyze}\nSamples: {nsmpt_multi-MTC if nsmpt_multi > MTC else 0}\nSmoothing: 5-point moving avg"
    ax2.text(0.02, 0.02, info_text, transform=ax2.transAxes, 
             verticalalignment='bottom', fontsize=9, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # Plot 3: Spatial Correlation
    ax3 = axes[2]
    space_lags = np.arange(-MXC, MXC + 1) * cell_width
    if nsmpx > 0:
        for i in range(5):
            zero_lag_variance = spatial_corr_sums[i, MXC] / nsmpx
            if abs(zero_lag_variance) > 1e-9:
                normalized_corr = spatial_corr_sums[i, :] / nsmpx / zero_lag_variance
                ax3.plot(space_lags * 1e6, normalized_corr, 'o-', color=colors[i], label=prop_labels[i], markersize=3)
    ax3.set_title(f'Spatial Correlation Relative to Cell {CENTRAL_CELL_IDX}'); ax3.set_xlabel('Spatial Lag, Δx (μm)'); ax3.set_ylabel('Normalized Correlation, G(Δx)')
    ax3.legend(); ax3.grid(True, linestyle=':')
    ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax3.axvline(x=0, color='k', linestyle='-', alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    output_filename = f'DSMC_Advanced_Stats_Complete_{method_name}.eps'
    plt.savefig(output_filename, format='eps', bbox_inches='tight')
    print(f"Advanced stats plot saved as {output_filename}")
    plt.show()

def run_dsmc_simulation(method_code, n_sel_fraction):
    """Main DSMC simulation function with comprehensive analysis - COMPLETE"""
    global correlation_animation
    
    collision_method_name = METHOD_MAP.get(method_code, 'Unknown')
    print(f"--- DSMC Relaxation using method: {collision_method_name} (COMPLETE VERSION) ---")
    
    rng_state = np.random.default_rng(seed=42)
    particles = initialize_particles(rng_state)
    sigma_g_max = np.full(NUM_CELLS_X, 1e-18, dtype=np.float64)

    total_accepted_collisions, total_selected_pairs, total_collision_separation = 0.0, 0.0, 0.0
    total_prob_exceed_collisions = 0.0
    last_sampled_total_collisions = 0.0
    last_sampled_prob_exceed = 0.0
    
    density_fluctuation_hist = np.zeros(2 * MAXD + 1, dtype=np.int64)
    navg_theory = PARTICLES_PER_CELL_INIT
    
    NUM_CELLS_TO_ANALYZE = 5
    CENTRAL_CELL_IDX = NUM_CELLS_X // 2
    
    if NUM_CELLS_X >= 5:
        cells_to_analyze = [
            NUM_CELLS_X // 4,           
            CENTRAL_CELL_IDX,           
            3 * NUM_CELLS_X // 4,       
            NUM_CELLS_X // 8,           
            7 * NUM_CELLS_X // 8        
        ]
    else:
        cells_to_analyze = list(range(min(NUM_CELLS_TO_ANALYZE, NUM_CELLS_X)))
        CENTRAL_CELL_IDX = cells_to_analyze[len(cells_to_analyze)//2] if cells_to_analyze else 0
    
    print(f"Analyzing temporal correlation for cells: {cells_to_analyze}")
    print(f"Central cell for spatial correlation: {CENTRAL_CELL_IDX}")
    
    temporal_samples_storage_multi = np.zeros((len(cells_to_analyze), 5, 2 * MTC + 1))
    temporal_corr_sums_multi = np.zeros((len(cells_to_analyze), 5, 2 * MTC + 1))
    nsmpt_multi = 0
    
    spatial_corr_sums = np.zeros((5, 2 * MXC + 1))
    nsmpx = 0
    
    MAX_HISTORY_SAMPLES = 2000
    MAX_SPEED_SAMPLES = 50000
    
    time_history, freq_ratio_history, sof_history, prob_exceed_history = [], [], [], []
    sampled_speeds_accumulator = []
    history_sample_count = 0
    
    start_time = time.time()
    num_steps = int(TOTAL_TIME / DT)
    speed_sampling_start_step = int(TOTAL_TIME * 0.9 / DT)
    cell_width = LX / NUM_CELLS_X
    sigma_ref = PI * D_REF_AR**2
    mfp = 1 / (np.sqrt(2) * sigma_ref * N_DENSITY_REAL)

    cells_processed = 0
    cells_with_collisions = 0
    empty_cells = 0
    single_particle_cells = 0

    print(f"Starting {collision_method_name} simulation with {num_steps} steps...")
    print(f"Total simulation time: {TOTAL_TIME*1e9:.1f} ns")
    print(f"Time step: {DT*1e12:.3f} ps")

    for step in range(1, num_steps + 1):
        particles[:, 0] += particles[:, 1] * DT
        particles[:, 0] %= LX
        
        step_accepted_collisions, step_selected_pairs, step_sum_separation = 0.0, 0.0, 0.0
        step_prob_exceed = 0.0
        duplicate_check_array = np.full(TOTAL_PARTICLES_SIM, -1, dtype=np.int64)
        cell_indices = (particles[:, 0] / cell_width).astype(np.int64)

        step_cells_processed = 0
        step_cells_with_collisions = 0
        step_empty_cells = 0
        step_single_particle_cells = 0

        for i in range(NUM_CELLS_X):
            indices_in_cell_i = np.where(cell_indices == i)[0]
            step_cells_processed += 1
            
            if len(indices_in_cell_i) == 0:
                step_empty_cells += 1
            elif len(indices_in_cell_i) == 1:
                step_single_particle_cells += 1
            
            n_acc, new_sigma_g_max, n_sel, sum_sep, n_prob_exceed = perform_collisions(
                method_code, particles, LX, indices_in_cell_i, 
                CELL_VOLUME_CONCEPTUAL, DT, FNUM, rng_state, 
                sigma_g_max[i], n_sel_fraction, duplicate_check_array
            )
            
            if n_acc > 0:
                step_cells_with_collisions += 1
                
            sigma_g_max[i] = new_sigma_g_max
            step_accepted_collisions += n_acc
            step_selected_pairs += n_sel
            step_sum_separation += sum_sep
            step_prob_exceed += n_prob_exceed

        cells_processed += step_cells_processed
        cells_with_collisions += step_cells_with_collisions
        empty_cells += step_empty_cells
        single_particle_cells += step_single_particle_cells

        total_accepted_collisions += step_accepted_collisions
        total_selected_pairs += step_selected_pairs
        total_collision_separation += step_sum_separation
        total_prob_exceed_collisions += step_prob_exceed

        if step % SAMPLING_INTERVAL == 0 and step > MTC:
            current_time = step * DT
            
            if history_sample_count < MAX_HISTORY_SAMPLES:
                time_history.append(current_time)
                history_sample_count += 1
            else:
                time_history.pop(0)
                time_history.append(current_time)
            
            collisions_this_interval = total_accepted_collisions - last_sampled_total_collisions
            prob_exceed_this_interval = total_prob_exceed_collisions - last_sampled_prob_exceed
            
            prob_exceed_ratio = (prob_exceed_this_interval / collisions_this_interval) if collisions_this_interval > 0 else 0.0
            
            if len(freq_ratio_history) >= MAX_HISTORY_SAMPLES:
                freq_ratio_history.pop(0)
                sof_history.pop(0)
                prob_exceed_history.pop(0)
            
            time_this_interval = SAMPLING_INTERVAL * DT
            current_temp = (0.5 * MASS_AR / (1.5 * KB)) * np.mean(np.sum(particles[:,1:4]**2, axis=1))

            cf_numerical = calculate_numerical_collision_frequency(collisions_this_interval, TOTAL_PARTICLES_SIM, time_this_interval)
            cf_theoretical = calculate_theoretical_collision_frequency(current_temp, N_DENSITY_REAL)
            ratio = cf_numerical / cf_theoretical if cf_theoretical > 0 else 0
            
            freq_ratio_history.append(ratio)
            prob_exceed_history.append(prob_exceed_ratio)
            last_sampled_total_collisions = total_accepted_collisions
            last_sampled_prob_exceed = total_prob_exceed_collisions
            
            sof_interval = (step_sum_separation / step_accepted_collisions) / mfp if step_accepted_collisions > 0 and mfp > 0 else 0
            sof_history.append(sof_interval)

            cell_counts = np.bincount(cell_indices, minlength=NUM_CELLS_X)
            deviations = cell_counts - int(navg_theory)
            for dev in deviations:
                if -MAXD <= dev <= MAXD:
                    density_fluctuation_hist[dev + MAXD] += 1

            current_cell_props = np.zeros((5, NUM_CELLS_X))
            for i in range(NUM_CELLS_X):
                indices = np.where(cell_indices == i)[0]
                N_i = len(indices)
                current_cell_props[0, i] = N_i
                if N_i > 1:
                    v_avg = np.mean(particles[indices, 1:4], axis=0)
                    current_cell_props[1:4, i] = v_avg
                    v_particles = particles[indices, 1:4]
                    v_mean_sq = np.mean(np.sum(v_particles**2, axis=1))
                    temp_i = (MASS_AR * v_mean_sq) / (3 * KB)
                    current_cell_props[4, i] = temp_i
                elif N_i == 1:
                    current_cell_props[1:4, i] = particles[indices[0], 1:4]
                    current_cell_props[4, i] = 0.0

            for cell_idx, cell_num in enumerate(cells_to_analyze):
                temporal_samples_storage_multi[cell_idx, :, :-1] = temporal_samples_storage_multi[cell_idx, :, 1:]
                temporal_samples_storage_multi[cell_idx, :, -1] = current_cell_props[:, cell_num]
            
            nsmpt_multi += 1
            if nsmpt_multi > MTC:
                for cell_idx, cell_num in enumerate(cells_to_analyze):
                    center_values_t = temporal_samples_storage_multi[cell_idx, :, MTC]
                    time_means = np.mean(temporal_samples_storage_multi[cell_idx, :, :], axis=1)
                    time_stds = np.std(temporal_samples_storage_multi[cell_idx, :, :], axis=1)
                    
                    avg_particles = time_means[0]
                    if avg_particles > 20:
                        scaling_factor = np.sqrt(avg_particles)
                    else:
                        scaling_factor = 1.0
                    
                    for lag in range(2 * MTC + 1):
                        past_values = temporal_samples_storage_multi[cell_idx, :, lag]
                        
                        for prop_idx in range(5):
                            if time_stds[prop_idx] > 1e-10:
                                if prop_idx == 0:
                                    if avg_particles > 1.0:
                                        poisson_std = np.sqrt(avg_particles)
                                        center_norm = (center_values_t[prop_idx] - time_means[prop_idx]) / poisson_std
                                        past_norm = (past_values[prop_idx] - time_means[prop_idx]) / poisson_std
                                    else:
                                        center_norm = 0.0
                                        past_norm = 0.0
                                elif prop_idx == 4:
                                    temp_scale = time_stds[prop_idx] * scaling_factor
                                    center_norm = (center_values_t[prop_idx] - time_means[prop_idx]) / temp_scale
                                    past_norm = (past_values[prop_idx] - time_means[prop_idx]) / temp_scale
                                else:
                                    vel_scale = time_stds[prop_idx] * np.sqrt(scaling_factor)
                                    center_norm = (center_values_t[prop_idx] - time_means[prop_idx]) / vel_scale
                                    past_norm = (past_values[prop_idx] - time_means[prop_idx]) / vel_scale
                                
                                correlation_products = center_norm * past_norm
                                temporal_corr_sums_multi[cell_idx, prop_idx, lag] += correlation_products
                            else:
                                temporal_corr_sums_multi[cell_idx, prop_idx, lag] += 0.0

            if nsmpt_multi > MTC and step % (SAMPLING_INTERVAL * 5) == 0 and correlation_animation is not None:
                try:
                    correlation_animation.add_snapshot(step, temporal_corr_sums_multi / (nsmpt_multi - MTC), cells_to_analyze)
                except Exception as e:
                    print(f"⚠ Snapshot collection failed: {e}")
                    correlation_animation = None

            nsmpx += 1
            center_props_s = current_cell_props[:, CENTRAL_CELL_IDX]
            spatial_means = np.mean(current_cell_props, axis=1)
            
            for lag in range(-MXC, MXC + 1):
                cell_idx_to_compare = (CENTRAL_CELL_IDX + lag + NUM_CELLS_X) % NUM_CELLS_X
                compare_props = current_cell_props[:, cell_idx_to_compare]
                correlation_products = (center_props_s - spatial_means) * (compare_props - spatial_means)
                spatial_corr_sums[:, lag + MXC] += correlation_products

        if step >= speed_sampling_start_step and step % (SAMPLING_INTERVAL * 2) == 0:
            if len(sampled_speeds_accumulator) < MAX_SPEED_SAMPLES:
                sample_size = min(1000, TOTAL_PARTICLES_SIM)
                sample_indices = np.random.choice(TOTAL_PARTICLES_SIM, size=sample_size, replace=False)
                particle_speeds = np.sqrt(np.sum(particles[sample_indices, 1:4]**2, axis=1))
                sampled_speeds_accumulator.extend(particle_speeds)

        if step % (num_steps // 10) == 0:
            progress = step / num_steps * 100
            current_time_ns = step * DT * 1e9
            
            history_size = len(time_history) * 4
            speed_size = len(sampled_speeds_accumulator)
            total_samples = history_size + speed_size
            
            print(f"Step: {step}/{num_steps} ({progress:.1f}%) - Time: {current_time_ns:.1f} ns")
            print(f"  Data samples: {total_samples:,} (History: {history_size:,}, Speeds: {speed_size:,})")
            
            avg_empty_cells = empty_cells / step if step > 0 else 0
            avg_single_cells = single_particle_cells / step if step > 0 else 0
            avg_collision_cells = cells_with_collisions / step if step > 0 else 0
            print(f"  Cell stats (avg/step): Empty: {avg_empty_cells:.1f}, Single: {avg_single_cells:.1f}, Collisions: {avg_collision_cells:.1f}")
            
            if len(freq_ratio_history) > 0:
                print(f"  Current frequency ratio: {freq_ratio_history[-1]:.4f}")
            
            if len(prob_exceed_history) > 0:
                print(f"  Current exceed ratio: {prob_exceed_history[-1]*100:.1f}%")
            
            gc.collect()

    end_time = time.time()
    print(f"\nSimulation finished in {end_time - start_time:.2f} seconds.")
    
    final_temp = (0.5 * MASS_AR / (1.5 * KB)) * np.mean(np.sum(particles[:,1:4]**2, axis=1))
    print(f"\nFinal Equilibrium Temperature: {final_temp:.2f} K")
    
    acceptance_ratio = total_accepted_collisions / total_selected_pairs if total_selected_pairs > 0 else 0
    mean_separation = total_collision_separation / total_accepted_collisions if total_accepted_collisions > 0 else 0
    sof = mean_separation / mfp if mfp > 0 else 0
    overall_prob_exceed_ratio = total_prob_exceed_collisions / total_accepted_collisions if total_accepted_collisions > 0 else 0
    
    cm = np.sqrt(2 * KB * T_INIT / MASS_AR)
    tc = mfp / cm
    dx_dimensionless = cell_width / mfp
    dt_dimensionless = DT / tc
    dx_dt_ratio = dx_dimensionless / dt_dimensionless
    
    print(f"Total Collisions Accepted: {int(total_accepted_collisions)}")
    print(f"Total Pairs Selected: {int(total_selected_pairs)}")
    print(f"Collision Acceptance Ratio: {acceptance_ratio:.4f}")
    print(f"Overall Probability Exceed Ratio: {overall_prob_exceed_ratio:.4f}")
    print(f"Mean Free Path (MFP): {mfp:.4e} m")
    print(f"Mean Collision Separation / MFP (SoF): {sof:.4f}")
    print(f"dx/dt (normalized): {dx_dt_ratio:.10f} (should be ≈ 1.0)")
    
    if method_code in [4, 6]:
        requested_n_sel = int(n_sel_fraction)
        if requested_n_sel >= PARTICLES_PER_CELL_INIT - 1:
            print(f"N_sel requested: {requested_n_sel} (>= {PARTICLES_PER_CELL_INIT - 1}), used Standard SBT instead")
        else:
            actual_n_sel = requested_n_sel
            print(f"N_sel used: {actual_n_sel} (per cell with {PARTICLES_PER_CELL_INIT} particles)")

    if freq_ratio_history:
        avg_ratio = np.mean(freq_ratio_history)
        print(f"\n📊 Collision Frequency Analysis:")
        print(f"Average frequency ratio: {avg_ratio:.4f}")
        print(f"Standard deviation: {np.std(freq_ratio_history):.4f}")
        print(f"Min/Max: {min(freq_ratio_history):.4f} / {max(freq_ratio_history):.4f}")
        print(f"Expected for correct implementation: ≈ 1.0")
        if 0.95 <= avg_ratio <= 1.05:
            print(f"✓ EXCELLENT: Frequency ratio within 5% of ideal")
        elif 0.90 <= avg_ratio <= 1.10:
            print(f"✓ GOOD: Frequency ratio within 10% of ideal")
        else:
            print(f"⚠ NEEDS ADJUSTMENT: Frequency ratio deviates significantly from ideal")

    if prob_exceed_history:
        avg_exceed = np.mean(prob_exceed_history) * 100
        print(f"\n📊 Probability Exceed Analysis:")
        print(f"Average exceed ratio: {avg_exceed:.2f}%")
        print(f"Standard deviation: {np.std(prob_exceed_history)*100:.2f}%")
        print(f"Min/Max: {min(prob_exceed_history)*100:.2f}% / {max(prob_exceed_history)*100:.2f}%")
        
        # دستورالعمل‌های عملی
        if avg_exceed < 3.0:
            print(f"✅ EXCELLENT: Very low exceed ratio!")
            print(f"🎯 Perfect setup for accurate DSMC simulation")
        elif avg_exceed < 8.0:
            print(f"✅ GOOD: Acceptable exceed ratio")
            print(f"🎯 Results are reliable and physically meaningful")
        elif avg_exceed < 15.0:
            print(f"⚡ MODERATE: Some exceed present")
            print(f"💡 Consider: {PARTICLES_PER_CELL_INIT * 2:.1f} particles/cell for better accuracy")
        elif avg_exceed < 25.0:
            print(f"⚠️ HIGH: Significant exceed detected")
            print(f"💡 RECOMMENDATION: Use {PARTICLES_PER_CELL_INIT * 3:.1f} particles/cell")
            print(f"💡 Alternative: Increase cells to {NUM_CELLS_X * 2}")
        else:
            print(f"🚨 CRITICAL: Very high exceed (>{avg_exceed:.0f}%)")
            print(f"🔧 URGENT FIXES:")
            print(f"   • Use ≥ {max(2.0, PARTICLES_PER_CELL_INIT * 5):.1f} particles/cell")
            print(f"   • Or increase cells to {NUM_CELLS_X * 3}")
            print(f"   • Or reduce time step by 50%")
            print(f"💡 Current setup not recommended for accurate physics")

    total_steps = num_steps
    print(f"\n=== Cell Statistics Summary ===")
    print(f"Average empty cells per step: {empty_cells / total_steps:.2f}")
    print(f"Average single-particle cells per step: {single_particle_cells / total_steps:.2f}")
    print(f"Average cells with collisions per step: {cells_with_collisions / total_steps:.2f}")
    print(f"Total cells processed: {cells_processed:,}")
    
    gc.collect()

    if sampled_speeds_accumulator:
        plot_results(np.array(sampled_speeds_accumulator), final_temp, collision_method_name, 
                    freq_ratio_history, time_history, sof_history, acceptance_ratio, mfp, prob_exceed_history)
    else:
        print("⚠ No speed data collected for final distribution plot")
    
    plot_advanced_stats(density_fluctuation_hist, temporal_corr_sums_multi, spatial_corr_sums, 
                       nsmpt_multi, nsmpx, navg_theory, collision_method_name, DT, cell_width, cells_to_analyze)
    
    print("\n" + "="*60)
    print("CREATING AUTO-CORRELATION EVOLUTION ANIMATION")
    print("="*60)
    
    try:
        if correlation_animation and len(correlation_animation.correlation_snapshots) > 10:
            print(f"Creating animation with {len(correlation_animation.correlation_snapshots)} snapshots...")
            anim = correlation_animation.create_animation(f'dsmc_{collision_method_name.lower()}_correlation_evolution.mp4')
            print("✓ Auto-correlation evolution animation created successfully!")
        else:
            print(f"⚠ Insufficient snapshots for animation")
    except Exception as e:
        print(f"⚠ Error creating animation: {e}")
    
    print("="*60)
    print(f"✅ DSMC {collision_method_name} simulation completed successfully!")
    print("📊 All plots and analyses have been generated.")

if __name__ == "__main__":
    print("🚀 === COMPLETE DSMC Simulation (Fixed Cell Count + All Features) ===")
    print("✅ تمام 9 روش برخوردی کامل: SBT, DCP, NTC, GBT, SSBT, SGBT, MFS, NN, DCP-VR")
    print("✅ تعداد سلول‌های منطقی و قابل اجرا")
    print("✅ آنالیز آماری کامل")
    print("✅ انیمیشن evolution correlation")
    print("✅ کنترل probability exceed")
    print("✅ تمام نمودارها و پارامترها")
    print("✅ NEW: DCP-VR method with velocity-weighted collision rates")
    
    print("\n🚨 IMPORTANT: Probability Exceed Guidelines:")
    print("   🎯 Target: < 5% exceed (high accuracy)")
    print("   ✅ Good: 5-10% exceed (acceptable)")
    print("   ⚠️  Moderate: 10-20% exceed (usable)")
    print("   🚨 High: > 20% exceed (needs mesh refinement)")
    print("")
    print("🔬 Mesh Refinement Strategy:")
    print("   • Fine mesh (more cells) → Lower exceed ratio")
    print("   • Automatic dt adjustment to maintain dx/dt = 1")
    print("   • Higher computational cost but better accuracy")
    print("   • Essential for fractional particle simulations")
    
    # Environment-specific hints
    if IN_COLAB:
        print("🔧 Google Colab detected:")
        print("💡 For quick testing: particles=2.0, method=1 (SBT)")
        print("📂 Generated files will be available in the file browser")
    elif 'IN_GOOGLE_CLOUD' in locals() and IN_GOOGLE_CLOUD:
        print("🔧 Google Cloud Jupyter detected:")
        print("💡 For quick testing: particles=2.0, method=1 (SBT)")
        print("⚠️ If animation fails, try: !sudo apt-get install ffmpeg")
    else:
        print("🔧 Local environment detected")
    
    # Enhanced guidance for fractional particles
    print("\n📋 Particle Count Guidelines:")
    print("   🟢 RECOMMENDED: ≥ 2.0 particles/cell")
    print("      → Low exceed, good statistics, reasonable speed")
    print("   🟡 ACCEPTABLE: 1.0-2.0 particles/cell") 
    print("      → Moderate exceed, acceptable performance")
    print("   🟠 CHALLENGING: 0.5-1.0 particles/cell")
    print("      → Higher exceed, slower, for special studies")
    print("   🔴 DIFFICULT: ≤ 0.5 particles/cell")
    print("      → High exceed, very slow, expert use only")
    print("      → Consider: 2.0+ particles for better results")
    
    # Get particles per cell from user
    while True:
        try:
            particles_input = float(input("\n🔢 Enter average particles per cell (e.g., 0.5, 1.0, 2.0, 5.0): ") or "2.0")
            if particles_input > 0:
                if particles_input <= 0.1:
                    print(f"🔴 ULTRA LOW: {particles_input:.2f} particles/cell")
                    print("   📊 Expected exceed: 15-25% (will use FINE MESH)")
                    print("   🔬 Fine mesh refinement will reduce exceed to ~5%")
                    print("   ⏱️  Computational cost: 3-5x higher")
                    print("   💡 Alternative: Try 1.0+ particles for faster simulation")
                    confirm = input("   Continue with fine mesh refinement? (y/n): ").lower()
                    if confirm not in ['y', 'yes']:
                        print("   👍 Consider 1.0+ particles for faster results!")
                        continue
                elif particles_input <= 0.2:
                    print(f"🟠 LOW DENSITY: {particles_input:.2f} particles/cell")
                    print("   📊 Expected exceed: 10-20% (will use MEDIUM MESH)")
                    print("   🔬 Mesh refinement will reduce exceed to ~5%")
                    print("   ⏱️  Computational cost: 2-3x higher")
                elif particles_input <= 0.5:
                    print(f"🟡 FRACTIONAL: {particles_input:.2f} particles/cell")
                    print("   📊 Expected exceed: 5-15% (optimized mesh)")
                    print("   This will trigger enhanced scaling for numerical stability")
                    print("   Expect moderate computational cost")
                elif particles_input < 1.0:
                    print(f"🟡 FRACTIONAL: {particles_input:.2f} particles/cell")
                    print("   This will trigger enhanced scaling for numerical stability")
                    print("   Expect moderate exceed (5-15%) and longer computation time")
                else:
                    print(f"🟢 GOOD CHOICE: {particles_input:.2f} particles/cell")
                    print("   Expected exceed < 5% with good performance")
                break
            else:
                print("❌ Please enter a value > 0.")
        except ValueError:
            print("❌ Please enter a valid number.")
    
    # Display method options
    print("\n🎯 Available collision methods (All use fine mesh for fractional particles):")
    for code, name in METHOD_MAP.items():
        if particles_input < 1.0:
            print(f"   [{code}] {name} (with automatic mesh refinement)")
        else:
            print(f"   [{code}] {name}")
    
    # Special note for new DCP-VR method
    print("\n🆕 NEW METHOD:")
    print("   [9] DCP-VR: Enhanced Distance-based Collision Pairing")
    print("       → Uses velocity-weighted collision rates (σ×g) instead of uniform weights")
    print("       → Better physical accuracy compared to standard DCP")
    print("       → Recommended for studies comparing spatial biasing effects")
    
    if particles_input < 1.0:
        print(f"\n🔬 NOTE: Fine mesh refinement will be applied to whichever method you choose")
        print(f"📊 Expected: ~{1600*particles_input:.0f} total particles in fine mesh")
        print(f"🎯 Target exceed reduction: 20% → 5% for any method")
    
    # Get collision method from user
    while True:
        choice = input("\nChoose method (1-9, default=1 for SBT): ") or "1"
        
        if choice in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
            method_code = int(choice)
            n_sel_fraction = 0.5
            n_sel_for_params = None
            
            # Get N_sel for GBT and SGBT methods
            if choice in ['4', '6']:
                method_name = {'4': 'GBT', '6': 'SGBT'}[choice]
                while True:
                    try:
                        n_sel_input = int(input(f"⚙️ Enter N_sel for {method_name} (recommended: 2-{max(2, int(particles_input)-1)}): ") or "2")
                        if n_sel_input >= 1:
                            n_sel_fraction = n_sel_input
                            n_sel_for_params = n_sel_input
                            break
                        else:
                            print("❌ N_sel must be >= 1.")
                    except ValueError:
                        print("❌ Please enter an integer.")
            
            # Special info for DCP-VR
            if choice == '9':
                print("\n🆕 DCP-VR Selected:")
                print("   • Weight = (σ×g) / distance instead of 1 / distance")
                print("   • Physically motivated collision rate weighting")
                print("   • Better represents actual collision probabilities")
                print("   • Expect similar performance to DCP but with improved physics")
            
            # Calculate simulation parameters
            print("\n📊 Calculating optimized simulation parameters...")
            if particles_input < 1.0:
                print("🔧 Applying enhanced scaling for fractional particles...")
            
            calculate_simulation_parameters(particles_input, n_sel_for_params)
            
            # Start simulation
            print(f"\n🏁 Starting COMPLETE {METHOD_MAP.get(method_code, 'Unknown')} simulation...")
            if particles_input < 1.0:
                print("⏱️ Fractional particles require more time and resources...")
            else:
                print("⏱️ This may take several minutes for comprehensive analysis...")
            
            if IN_COLAB or ('IN_GOOGLE_CLOUD' in locals() and IN_GOOGLE_CLOUD):
                print("💡 Progress will be shown below, all plots will be displayed automatically")
                print("📁 Files will be saved and can be downloaded from the file browser")
            
            run_dsmc_simulation(method_code=method_code, n_sel_fraction=n_sel_fraction)
            break
        else:
            print("❌ Please choose a number between 1-9.")