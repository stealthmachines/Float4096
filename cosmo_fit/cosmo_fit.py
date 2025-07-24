# root_folder/cosmo_fit/cosmo_fit.py
import os
import re
import logging
import time
from tqdm.auto import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sympy as sp
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
# tests/test_float4096.py
import pytest
import numpy as np
import sympy as sp
import math
import pickle
from numbers import Number
from float4096 import (
    Float4096,
    Float4096Array,
    fib_real,
    native_prime_product,
    sqrt,
    log10,
    logspace,
    linspace,
    max,
    abs,
    pow_f4096,
    prepare_prime_interpolation,
    pi_val,
    exp,
    sin,
    cos,
)

# Configuration
CONFIG = {
    'max_n': 500,
    'beta_steps': 10,
    'r_values': [0.5, 1.0, 2.0],
    'k_values': [0.5, 1.0, 2.0],
    'Omega': 1.0,
    'base': 2.0,
    'scale': 1.0,
    'tolerance': 0.05,
    'batch_size': 100,
    'codata_file': 'allascii.txt',
    'supernova_file': 'hlsp_ps1cosmo_panstarrs_gpc1_all_model_v1_lcparam-full.txt',
    'n_jobs': 4,
    'joblib_timeout': 300,
    'use_tqdm': True,
    'optimization_bounds': [
        (1e-5, 10), (1e-5, 10), (1e-5, 10), (1.5, 10), (1e-5, 100),  # r, k, Omega, base, scale
        (1e-5, 10), (1e-5, 10), (1e-5, 10),  # alpha, beta, gamma
        (1e-5, 100), (1e5, 1e9)  # H0_emergent, c0_emergent
    ],
    'optimization_maxiter': 100,
}

# Symbolic constants
phi = (Float4096(1) + sqrt(Float4096(5))) / Float4096(2)
sqrt5 = sqrt(Float4096(5))
Omega_sym = sp.Symbol("Omega")
s_sym = sp.Symbol("s")

# Set up logging
logging.basicConfig(
    filename='symbolic_cosmo_fit.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)

def P_nb(n_beta, prime_interp):
    return Float4096(prime_interp(float(n_beta)))

def solve_n_beta_for_prime(p_target, prime_interp, bracket=(0.1, 20)):
    def objective(n_beta): return P_nb(n_beta, prime_interp) - Float4096(p_target)
    result = root_scalar(objective, bracket=bracket, method='brentq')
    if result.converged:
        return Float4096(result.root)
    else:
        raise ValueError(f"Could not solve for n_beta corresponding to prime {p_target}")

def F_bin(x_val):
    x_val = Float4096(x_val)
    return (pow_f4096(phi, x_val) - pow_f4096(Float4096(-1)/phi, x_val)) / sqrt5

def Pi_x(x_val, s):
    s = sp.sympify(s)
    return exp(sp.I * pi_val() * Float4096(x_val)) * sp.zeta(s, sp.Rational(1, 2))

def D_x(x_val, s, prime_interp):
    s = sp.sympify(s)
    x_val = Float4096(x_val)
    P = P_nb(x_val, prime_interp)
    F = F_bin(x_val)
    zeta_val = Float4096(sp.zeta(s))
    product = phi * F * pow_f4096(Float4096(2), x_val) * P * zeta_val * Float4096(Omega_sym)
    return sqrt(product) * pow_f4096(Float4096(s), Float4096(-1))

def F_x(x_val, s, prime_interp):
    return D_x(x_val, s, prime_interp) * Pi_x(x_val, s)

class GoldenClassField:
    def __init__(self, s_list, x_list, prime_interp):
        self.s_list = [sp.sympify(s) for s in s_list]
        self.x_list = [Float4096(x) for x in x_list]
        self.prime_interp = prime_interp
        self.field_generators = []
        self.field_names = []
        self.construct_class_field()

    def construct_class_field(self):
        for s in self.s_list:
            for x in self.x_list:
                f = F_x(x, s, self.prime_interp)
                self.field_generators.append(f)
                self.field_names.append(f"F_{float(x):.4f}_s_{s}")

    def as_dict(self):
        return dict(zip(self.field_names, self.field_generators))

def D(n, beta, r=1.0, k=1.0, Omega=1.0, base=2, scale=1.0, prime_interp=None):
    try:
        n = Float4096(n)
        beta = Float4096(beta)
        r = Float4096(r)
        k = Float4096(k)
        Omega = Float4096(Omega)
        base = Float4096(base)
        scale = Float4096(scale)
        Fn_beta = fib_real(n + beta)
        Pn_beta = native_prime_product(int(float(n + beta)), prime_interp=prime_interp)
        dyadic = pow_f4096(base, n + beta)
        val = scale * phi * Fn_beta * dyadic * Pn_beta * Omega
        val = max(val, Float4096("1e-30"))
        return sqrt(val) * pow_f4096(r, k)
    except Exception as e:
        logging.debug(f"D failed for n={n}, beta={beta}: {e}")
        return Float4096("1e-30")

def invert_D(value, r=1.0, k=1.0, Omega=1.0, base=2, scale=1.0, max_n=500, steps=500, prime_interp=None):
    try:
        value = Float4096(value)
        r = Float4096(r)
        k = Float4096(k)
        Omega = Float4096(Omega)
        base = Float4096(base)
        scale = Float4096(scale)
        candidates = []
        log_val = log10(max(abs(value), Float4096("1e-30")))
        scale_factors = logspace(log_val - Float4096(2), log_val + Float4096(2), 10)
        max_n = min(max_n, max(50, int(float(10 * log_val))))
        n_values = linspace(Float4096(0), Float4096(max_n), steps)
        beta_values = linspace(Float4096(0), Float4096(1), 10)
        for n in n_values:
            for beta in beta_values:
                for dynamic_scale in scale_factors:
                    val = D(n, beta, r, k, Omega, base, scale * dynamic_scale, prime_interp)
                    if val.isfinite():
                        diff = abs(val - value)
                        candidates.append((diff, n, beta, dynamic_scale))
        if not candidates:
            return None, None, None
        best = min(candidates, key=lambda x: x[0])
        return best[1], best[2], best[3]
    except Exception as e:
        logging.warning(f"invert_D failed: {e}")
        return None, None, None

def parse_codata_ascii(filename):
    constants = []
    pattern = re.compile(r"^\s*(.*?)\s{2,}([0-9Ee\+\-\.]+)\s+([0-9Ee\+\-\.]+|exact)\s+(\S+)")
    try:
        with open(filename, "r") as f:
            for line in f:
                if line.startswith("Quantity") or line.strip() == "" or line.startswith("-"):
                    continue
                m = pattern.match(line)
                if m:
                    name, value_str, uncert_str, unit = m.groups()
                    try:
                        value = float(value_str.replace("e", "E"))
                        uncertainty = None if uncert_str == "exact" else float(uncert_str.replace("e", "E"))
                        constants.append({
                            "name": name.strip(),
                            "value": value,
                            "uncertainty": uncertainty,
                            "unit": unit.strip()
                        })
                    except:
                        continue
        df = pd.DataFrame(constants)
        logging.info(f"Parsed {len(df)} constants from {filename}")
        return df
    except FileNotFoundError:
        logging.error(f"Input file {filename} not found")
        raise
    except Exception as e:
        logging.error(f"Error parsing {filename}: {e}")
        raise

def parse_supernova_data(filename):
    try:
        df = np.genfromtxt(
            filename,
            delimiter=' ',
            names=True,
            comments='#',
            dtype=None,
            encoding=None
        )
        df = pd.DataFrame({
            'z': df['zcmb'],
            'mu': df['mb'] - Float4096(-19.3),
            'mu_err': df['dmb']
        })
        df = df[df['mu'].notnull() & df['mu_err'].notnull() & (df['mu_err'] > 0)]
        logging.info(f"Parsed {len(df)} valid supernova data points from {filename}")
        return df
    except FileNotFoundError:
        logging.error(f"Supernova file {filename} not found")
        raise
    except Exception as e:
        logging.error(f"Error parsing {filename}: {e}")
        raise

def generate_emergent_constants(n_max, beta_steps, r_values, k_values, Omega, base, scale, prime_interp):
    candidates = []
    n_values = linspace(Float4096(0), Float4096(n_max), 500)
    beta_values = linspace(Float4096(0), Float4096(1), beta_steps)
    r_values = Float4096Array(r_values)
    k_values = Float4096Array(k_values)
    Omega = Float4096(Omega)
    base = Float4096(base)
    scale = Float4096(scale)
    
    s_list = [sp.Rational(i, beta_steps) for i in range(1, beta_steps + 1)]
    x_list = [float(n) for n in n_values[:10]]
    field = GoldenClassField(s_list, x_list, prime_interp)
    field_dict = field.as_dict()
    
    iterator = n_values if not CONFIG['use_tqdm'] else tqdm(n_values, desc="Generating emergent constants")
    for n in iterator:
        for beta in beta_values:
            for r in r_values:
                for k in k_values:
                    val = D(n, beta, r, k, Omega, base, scale, prime_interp)
                    if val.isfinite():
                        candidates.append({
                            'n': float(n), 'beta': float(beta), 'value': float(val),
                            'r': float(r), 'k': float(k), 'scale': float(scale)
                        })
                    n_est, beta_est, scale_est = invert_D(val, r, k, Omega, base, scale, prime_interp=prime_interp)
                    if n_est is not None:
                        val_inv = D(n_est, beta_est, r, k, Omega, base, scale * scale_est, prime_interp)
                        if val_inv.isfinite():
                            candidates.append({
                                'n': float(n_est), 'beta': float(beta_est), 'value': float(val_inv),
                                'r': float(r), 'k': float(k), 'scale': float(scale)
                            })
    for (name, val) in field_dict.items():
        val_abs = Float4096(sp.Abs(val))
        if val_abs.isfinite():
            candidates.append({
                'n': float(x_list[0]), 'beta': float(s_list[0]), 'value': float(val_abs),
                'r': float(r_values[0]), 'k': float(k_values[0]), 'scale': float(scale),
                'source': name
            })
    
    return pd.DataFrame(candidates)

def fit_single_constant(row, r, k, Omega, base, scale, max_n, steps, prime_interp):
    try:
        val = Float4096(row['value'])
        if val <= Float4096(0) or val > Float4096("1e50"):
            return None
        n, beta, dynamic_scale = invert_D(val, r, k, Omega, base, scale, max_n, steps, prime_interp)
        if n is None:
            return None
        approx = D(n, beta, r, k, Omega, base, scale * dynamic_scale, prime_interp)
        error = abs(val - approx)
        rel_error = error / max(abs(val), Float4096("1e-30"))
        return {
            "name": row['name'],
            "value": float(val),
            "unit": row['unit'],
            "n": float(n),
            "beta": float(beta),
            "approx": float(approx),
            "error": float(error),
            "rel_error": float(rel_error),
            "uncertainty": row['uncertainty'],
            "scale": float(dynamic_scale)
        }
    except Exception as e:
        logging.warning(f"Failed inversion for {row['name']}: {e}")
        return None

def a_of_z(z):
    return Float4096(1) / (Float4096(1) + z)

def Omega(z, Omega0, alpha):
    a_z = a_of_z(z)
    return Omega0 / pow_f4096(a_z, Float4096(alpha))

def s(z, s0, beta):
    return s0 * pow_f4096(Float4096(1) + z, -Float4096(beta))

def G(z, k, r0, Omega0, s0, alpha, beta):
    return Omega(z, Omega0, alpha) * pow_f4096(Float4096(k), Float4096(2)) * r0 / s(z, s0, beta)

def H(z, k, r0, Omega0, s0, alpha, beta, H0_emergent, Om_m, Om_de):
    Gz = G(z, k, r0, Omega0, s0, alpha, beta)
    Hz_sq = pow_f4096(H0_emergent, Float4096(2)) * (Om_m * Gz * pow_f4096(Float4096(1) + z, Float4096(3)) + Om_de)
    return sqrt(Hz_sq)

def emergent_c(z, Omega0, alpha, gamma, c0_emergent):
    return c0_emergent * pow_f4096(Omega(z, Omega0, alpha) / Omega0, Float4096(gamma))

def compute_luminosity_distance_grid(z_max, params, n=500):
    k, r0, Omega0, s0, alpha, beta, gamma, H0_emergent, c0_emergent, Om_m, Om_de = params
    z_grid = linspace(Float4096(0), Float4096(z_max), n)
    c_z = emergent_c(z_grid, Omega0, alpha, gamma, c0_emergent)
    H_z = H(z_grid, k, r0, Omega0, s0, alpha, beta, H0_emergent, Om_m, Om_de)
    integrand_values = c_z / H_z
    integral_grid = Float4096Array([Float4096(0)] + [
        sum((integrand_values[:-1] + integrand_values[1:]) / Float4096(2) * np.diff([float(z) for z in z_grid]))
    ])
    d_c = interp1d([float(z) for z in z_grid], [float(i) for i in integral_grid], kind='cubic', fill_value="extrapolate")
    return lambda z: Float4096(1 + z) * Float4096(d_c(float(z)))

def model_mu(z_arr, params):
    d_L_func = compute_luminosity_distance_grid(max(z_arr), params)
    d_L_vals = Float4096Array([d_L_func(float(z)) for z in z_arr])
    return Float4096(5) * log10(d_L_vals) + Float4096(25)

def combined_error(params, df_codata, df_supernova, prime_interp):
    r, k, Omega, base, scale, alpha, beta, gamma, H0_emergent, c0_emergent = params
    Om_m = Float4096(0.3)  # Initial guess, to be optimized
    Om_de = Float4096(0.7)  # Initial guess, to be optimized
    params_cosmo = [r, k, Omega, Float4096(1.0), alpha, beta, gamma, H0_emergent, c0_emergent, Om_m, Om_de]
    
    # CODATA error
    df_fit = symbolic_fit_all_constants(
        df_codata, r=r, k=k, Omega=Omega, base=base, scale=scale,
        max_n=CONFIG['max_n'], steps=CONFIG['beta_steps'], prime_interp=prime_interp
    )
    if df_fit.empty:
        codata_error = float('inf')
    else:
        errors = df_fit['rel_error'].dropna().values
        errors = np.array([float(e) for e in errors])
        threshold = np.percentile(errors, 95)
        filtered_errors = errors[errors <= threshold]
        codata_error = np.sum(filtered_errors ** 2) if len(filtered_errors) > 0 else float('inf')
    
    # Supernova error
    z_arr = Float4096Array(df_supernova['z'])
    mu_observed = Float4096Array(df_supernova['mu'])
    mu_err = Float4096Array(df_supernova['mu_err'])
    try:
        mu_model = model_mu(z_arr, params_cosmo)
        chi2 = sum(((mu_model - mu_observed) / mu_err) ** Float4096(2))
        supernova_error = float(chi2)
    except Exception as e:
        logging.error(f"Supernova error calculation failed: {e}")
        supernova_error = float('inf')
    
    # Combine errors (weighted sum)
    return codata_error + supernova_error

def symbolic_fit_all_constants(df, r=1.0, k=1.0, Omega=1.0, base=2, scale=1.0, max_n=500, steps=500, prime_interp=None):
    r, k, Omega, base, scale = [Float4096(x) for x in (r, k, Omega, base, scale)]
    results = []
    output_file = "symbolic_fit_results.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        pd.DataFrame(columns=[
            'name', 'value', 'unit', 'n', 'beta', 'approx', 'error', 'rel_error', 'uncertainty', 'scale'
        ]).to_csv(f, sep="\t", index=False)
    
    iterator = range(0, len(df), CONFIG['batch_size']) if not CONFIG['use_tqdm'] else tqdm(
        range(0, len(df), CONFIG['batch_size']), desc="Fitting CODATA batches"
    )
    for start in iterator:
        batch = df.iloc[start:start + CONFIG['batch_size']]
        try:
            batch_results = Parallel(
                n_jobs=CONFIG['n_jobs'], timeout=CONFIG['joblib_timeout'], backend='loky'
            )(delayed(fit_single_constant)(row, r, k, Omega, base, scale, max_n, steps, prime_interp)
              for _, row in batch.iterrows())
            batch_results = [r for r in batch_results if r is not None]
            results.extend(batch_results)
            with open(output_file, 'a', encoding='utf-8') as f:
                pd.DataFrame(batch_results).to_csv(f, sep="\t", index=False, header=False, lineterminator='\n')
                f.flush()
        except Exception as e:
            logging.error(f"Parallel processing failed for batch {start//CONFIG['batch_size'] + 1}: {e}")
            batch_results = [fit_single_constant(row, r, k, Omega, base, scale, max_n, steps, prime_interp)
                             for _, row in batch.iterrows()]
            batch_results = [r for r in batch_results if r is not None]
            results.extend(batch_results)
            with open(output_file, 'a', encoding='utf-8') as f:
                pd.DataFrame(batch_results).to_csv(f, sep="\t", index=False, header=False, lineterminator='\n')
                f.flush()
    return pd.DataFrame(results)

def main():
    start_time = time.time()
    prime_interp = prepare_prime_interpolation()
    
    # Parse data
    if not os.path.exists(CONFIG['codata_file']):
        raise FileNotFoundError(f"{CONFIG['codata_file']} not found")
    df_codata = parse_codata_ascii(CONFIG['codata_file'])
    logging.info(f"Parsed {len(df_codata)} constants")
    
    df_supernova = None
    if os.path.exists(CONFIG['supernova_file']):
        df_supernova = parse_supernova_data(CONFIG['supernova_file'])
        logging.info(f"Parsed {len(df_supernova)} supernova data points")
    
    # Generate emergent constants
    emergent_df = generate_emergent_constants(
        CONFIG['max_n'], CONFIG['beta_steps'], CONFIG['r_values'], CONFIG['k_values'],
        CONFIG['Omega'], CONFIG['base'], CONFIG['scale'], prime_interp
    )
    logging.info("Generated emergent constants")
    
    # Optimize combined parameters
    init_params = [1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 0.5, 0.5, 70.0, 299792458.0]  # Initial guesses for H0, c0
    try:
        res = minimize(
            combined_error, init_params, args=(df_codata, df_supernova, prime_interp),
            bounds=CONFIG['optimization_bounds'], method='L-BFGS-B',
            options={'maxiter': CONFIG['optimization_maxiter']}
        )
        if res.success:
            r_opt, k_opt, Omega_opt, base_opt, scale_opt, alpha_opt, beta_opt, gamma_opt, H0_opt, c0_opt = res.x
            print(f"Optimization complete. Found parameters:\nr = {r_opt:.6f}, k = {k_opt:.6f}, "
                  f"Omega = {Omega_opt:.6f}, base = {base_opt:.6f}, scale = {scale_opt:.6f}, "
                  f"alpha = {alpha_opt:.6f}, beta = {beta_opt:.6f}, gamma = {gamma_opt:.6f}, "
                  f"H0_emergent = {H0_opt:.6f}, c0_emergent = {c0_opt:.6f}")
        else:
            logging.warning(f"Optimization failed: {res.message}")
            r_opt, k_opt, Omega_opt, base_opt, scale_opt, alpha_opt, beta_opt, gamma_opt, H0_opt, c0_opt = init_params
    except Exception as e:
        logging.error(f"Optimization failed: {e}")
        r_opt, k_opt, Omega_opt, base_opt, scale_opt, alpha_opt, beta_opt, gamma_opt, H0_opt, c0_opt = init_params
    
    # Fit CODATA constants
    fitted_df = symbolic_fit_all_constants(
        df_codata, r=r_opt, k=k_opt, Omega=Omega_opt, base=base_opt, scale=scale_opt,
        max_n=CONFIG['max_n'], steps=CONFIG['beta_steps'], prime_interp=prime_interp
    )
    fitted_df_sorted = fitted_df.sort_values("error")
    
    # Output CODATA results
    print("\nTop 20 best CODATA fits:")
    print(fitted_df_sorted.head(20).to_string(index=False))
    print("\nTop 20 worst CODATA fits:")
    print(fitted_df_sorted.tail(20).to_string(index=False))
    
    # Fit supernova data
    if df_supernova is not None:
        try:
            params_cosmo = [
                r_opt, k_opt, Omega_opt, Float4096(1.0), alpha_opt, beta_opt, gamma_opt,
                H0_opt, c0_opt, Float4096(0.3), Float4096(0.7)
            ]
            mu_fit = model_mu(Float4096Array(df_supernova['z']), params_cosmo)
            df_supernova['mu_model'] = [float(mu) for mu in mu_fit]
            df_supernova['residuals'] = df_supernova['mu'] - df_supernova['mu_model']
            residuals_rms = float(sqrt(mean(Float4096Array(df_supernova['residuals']) ** Float4096(2))))
            print(f"\nSupernova fit results:\nMinimum residual: {min(df_supernova['residuals']):.4f}")
            print(f"Maximum residual: {max(df_supernova['residuals']):.4f}")
            print(f"Residual RMS: {residuals_rms:.4f}")
            with open("cosmo_fit_results.txt", 'w', encoding='utf-8') as f:
                df_supernova.to_csv(f, sep="\t", index=False)
                f.flush()
            logging.info("Saved supernova results to cosmo_fit_results.txt")
        except Exception as e:
            logging.error(f"Supernova fitting failed: {e}")
    
    # Generate plots
    try:
        plt.figure(figsize=(10, 5))
        plt.hist(fitted_df_sorted['error'], bins=50, color='skyblue', edgecolor='black')
        plt.title('Histogram of Absolute Errors in CODATA Fit')
        plt.xlabel('Absolute Error')
        plt.ylabel('Count')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('codata_error_histogram.png')
        plt.close()
        
        plt.figure(figsize=(10, 5))
        plt.scatter(fitted_df_sorted['n'], fitted_df_sorted['error'], alpha=0.5, s=15, c='orange', edgecolors='black')
        plt.title('CODATA Absolute Error vs Symbolic Dimension n')
        plt.xlabel('n')
        plt.ylabel('Absolute Error')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('codata_error_vs_n.png')
        plt.close()
        
        if df_supernova is not None:
            plt.figure(figsize=(10, 6))
            plt.errorbar(df_supernova['z'], df_supernova['mu'], yerr=df_supernova['mu_err'], fmt='.', alpha=0.5, label='Pan-STARRS1 SNe')
            plt.plot(df_supernova['z'], df_supernova['mu_model'], 'r-', label='Symbolic Emergent Gravity Model')
            plt.xlabel('Redshift (z)')
            plt.ylabel('Distance Modulus (μ)')
            plt.title('Supernova Distance Modulus using Symbolic Parameters')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('supernova_fit.png')
            plt.close()
            
            plt.figure(figsize=(10, 4))
            plt.errorbar(df_supernova['z'], df_supernova['residuals'], yerr=df_supernova['mu_err'], fmt='.', alpha=0.5)
            plt.axhline(0, color='red', linestyle='--')
            plt.xlabel('Redshift (z)')
            plt.ylabel('Residuals (μ_data - μ_model)')
            plt.title('Residuals of Symbolic Model')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('supernova_residuals.png')
            plt.close()
        logging.info("Generated plots: codata_error_histogram.png, codata_error_vs_n.png, supernova_fit.png, supernova_residuals.png")
    except Exception as e:
        logging.error(f"Plot generation failed: {e}")
    
    logging.info(f"Total execution time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
