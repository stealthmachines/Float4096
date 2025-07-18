# cosmo_fit/cosmo_fit.py
import os
import signal
import sys
import logging
import time
from tqdm import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sympy as sp
from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import interp1d
from float4096 import (
    Float4096,
    Float4096Array,
    ComplexFloat4096,
    GRAElement,
    GoldenClassField,
    sqrt,
    exp,
    log,
    log10,
    sin,
    cos,
    pi_val,
    linspace,
    logspace,
    mean,
    stddev,
    abs,
    max,
    D,
    D_x,
    F_x,
    invert_D,
    ds_squared,
    g9,
    R9,
    grad9_r_n,
    Gamma_n,
    D_n,
    T,
    Xi_n,
    psi_9,
    E_n,
    edge_weight,
    Coil,
    Spin,
    Splice,
    Reflect,
    Coil_n,
    Spin_n,
    Splice_n,
    Reflect_n,
    recursive_time,
    frequency,
    charge,
    field_yield,
    action,
    energy,
    force,
    voltage,
    labeled_output,
    field_automorphisms,
    field_tension,
    prepare_prime_interpolation,
    native_zeta,
    native_prime_product,
    compute_spline_coefficients,
    native_cubic_spline,
    P_nb,
    solve_n_beta_for_prime,
)

# Configuration
CONFIG = {
    'n_max': 5000,
    'beta_steps': 10,
    'r_values': [0.5, 1.0, 2.0],
    'k_values': [0.5, 1.0, 2.0],
    'Omega': 1.0,
    'base': 2.0,
    'scale': 1.0,
    'tolerance': 0.05,
    'batch_size': 100,
    'codata_file': 'categorized_allascii.txt',
    'supernova_file': 'hlsp_ps1cosmo_panstarrs_gpc1_all_model_v1_lcparam-full.txt',
    'n_select_worst': 20,
    'optimization_bounds': [(1e-10, 100), (1e-10, 100), (1e-10, 100), (1.5, 20), (1e-10, 1000)],
    'optimization_maxiter': 100,
    'optimization_popsize': 15,
    'slsqp_maxiter': 500
}

# Set up logging
logging.basicConfig(
    filename='symbolic_cosmo_fit.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)

def D(n, beta, r=Float4096(1), k=Float4096(1), Omega=Float4096(1), base=Float4096(2), scale=Float4096(1), prime_interp=None):
    try:
        r_n = GRAElement(n + beta, Omega=Omega, base=base, prime_interp=prime_interp)
        val = scale * r_n._value
        if not val.is_finite() or val <= Float4096(0):
            logging.debug(f"D returned non-finite or non-positive value for n={n}, beta={beta}")
            return Float4096("1e-30")
        return val * (r ** k)
    except Exception as e:
        logging.debug(f"D failed for n={n}, beta={beta}: {e}")
        return Float4096("1e-30")

def parse_categorized_codata(filename):
    try:
        df = pd.read_csv(
            filename, sep='\t', header=0,
            names=['name', 'value', 'uncertainty', 'unit', 'category'],
            dtype={'name': str, 'value': float, 'uncertainty': float, 'unit': str, 'category': str},
            na_values=['exact']
        )
        df['uncertainty'] = df['uncertainty'].fillna(0.0)
        required_columns = ['name', 'value', 'uncertainty', 'unit']
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            raise ValueError(f"Missing required columns in {filename}: {missing}")
        logging.info(f"Successfully parsed {len(df)} constants from {filename}")
        return df
    except FileNotFoundError:
        logging.error(f"Input file {filename} not found")
        raise
    except Exception as e:
        logging.error(f"Error parsing {filename}: {e}")
        raise

def generate_emergent_constants(n_max, beta_steps, r_values, k_values, Omega, base, scale, prime_interp=None):
    candidates = []
    n_values = linspace(Float4096(0), Float4096(n_max), 500)
    beta_values = linspace(Float4096(0), Float4096(1), beta_steps)
    r_values = Float4096Array(r_values)
    k_values = Float4096Array(k_values)
    Omega = Float4096(Omega)
    base = Float4096(base)
    scale = Float4096(scale)
    
    # Use GoldenClassField for field-theoretic constants
    s_list = [sp.Rational(i, beta_steps) for i in range(1, beta_steps + 1)]
    x_list = n_values[:10]  # Limit for performance
    field = GoldenClassField(s_list, x_list, prime_interp=prime_interp)
    field_dict = field.as_dict()
    
    for n in tqdm(n_values, desc="Generating emergent constants"):
        for beta in beta_values:
            for r in r_values:
                for k in k_values:
                    # Direct D computation
                    val = D(n, beta, r, k, Omega, base, scale, prime_interp=prime_interp)
                    if val.is_finite():
                        candidates.append({
                            'n': float(n), 'beta': float(beta), 'value': float(val),
                            'r': float(r), 'k': float(k), 'scale': float(scale)
                        })
                    # Inverse via invert_D
                    n_est, beta_est, scale_est, _, r_est, k_est = invert_D(val, prime_interp=prime_interp)
                    if n_est is not None:
                        val_inv = D(n_est, beta_est, r_est, k_est, Omega, base, scale_est, prime_interp=prime_interp)
                        if val_inv.is_finite():
                            candidates.append({
                                'n': float(n_est), 'beta': float(beta_est), 'value': float(val_inv),
                                'r': float(r_est), 'k': float(k_est), 'scale': float(scale_est)
                            })
                    # Meta-operator enhancements
                    for op in [Spin, Splice]:
                        val_op = op(n, sp.Rational(1, 2), prime_interp=prime_interp)
                        if isinstance(val_op, ComplexFloat4096):
                            val_op = val_op.abs()
                        if val_op.is_finite():
                            candidates.append({
                                'n': float(n), 'beta': float(beta), 'value': float(val_op),
                                'r': float(r), 'k': float(k), 'scale': float(scale), 'source': op.__name__
                            })
    for (s, x), val in field_dict.items():
        if val.abs().is_finite():
            candidates.append({
                'n': float(x), 'beta': float(sp.Rational(s)), 'value': float(val.abs()),
                'r': float(r_values[0]), 'k': float(k_values[0]), 'scale': float(scale), 'source': 'GoldenClassField'
            })
    
    return pd.DataFrame(candidates)

def match_to_codata(df_emergent, df_codata, tolerance, batch_size, prime_interp=None):
    matches = []
    output_file = "emergent_constants.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        pd.DataFrame(columns=[
            'name', 'codata_value', 'emergent_value', 'n', 'beta', 'r', 'k', 'scale',
            'error', 'rel_error', 'codata_uncertainty', 'bad_data', 'bad_data_reason'
        ]).to_csv(f, sep="\t", index=False)
    
    for start in range(0, len(df_codata), batch_size):
        batch = df_codata.iloc[start:start + batch_size]
        for _, codata_row in tqdm(batch.iterrows(), total=len(batch), desc=f"Matching constants batch {start//batch_size + 1}"):
            value = Float4096(codata_row['value'])
            emergent_values = Float4096Array(df_emergent['value'])
            rel_error = abs(emergent_values - value) / max(abs(value), Float4096("1e-30"))
            mask = rel_error < Float4096(tolerance)
            matched = df_emergent[np.array(mask)]
            for _, emergent_row in matched.iterrows():
                error = abs(Float4096(emergent_row['value']) - value)
                rel_error = error / max(abs(value), Float4096("1e-30"))
                matches.append({
                    'name': codata_row['name'],
                    'codata_value': float(value),
                    'emergent_value': float(emergent_row['value']),
                    'n': emergent_row['n'],
                    'beta': emergent_row['beta'],
                    'r': emergent_row['r'],
                    'k': emergent_row['k'],
                    'scale': float(emergent_row['scale']),
                    'error': float(error),
                    'rel_error': float(rel_error),
                    'codata_uncertainty': codata_row['uncertainty'],
                    'bad_data': float(rel_error) > 0.5 or (
                        codata_row['uncertainty'] is not None and
                        float(abs(Float4096(codata_row['uncertainty']) - error)) > 10 * codata_row['uncertainty']
                    ),
                    'bad_data_reason': (
                        f"High rel_error ({float(rel_error):.2e})" if float(rel_error) > 0.5 else
                        f"Uncertainty deviation ({codata_row['uncertainty']:.2e} vs. {float(error):.2e})"
                        if (codata_row['uncertainty'] is not None and
                            float(abs(Float4096(codata_row['uncertainty']) - error)) > 10 * codata_row['uncertainty'])
                        else ""
                    )
                })
        try:
            with open(output_file, 'a', encoding='utf-8') as f:
                pd.DataFrame(matches).to_csv(f, sep="\t", index=False, header=False, lineterminator='\n')
                f.flush()
            matches = []
        except Exception as e:
            logging.error(f"Failed to save batch {start//batch_size + 1} to {output_file}: {e}")
    return pd.DataFrame(pd.read_csv(output_file, sep='\t'))

def check_physical_consistency(df_results):
    bad_data = []
    relations = [
        ('Planck constant', 'reduced Planck constant',
         lambda x, y: float(abs(Float4096(x['scale']) / Float4096(y['scale']) - Float4096(2) * pi_val())),
         0.1, 'scale ratio vs. 2π'),
        ('proton mass', 'proton-electron mass ratio',
         lambda x, y: float(abs(Float4096(x['n']) - Float4096(y['n']) - log10(Float4096(1836)))),
         0.5, 'n difference vs. log(proton-electron ratio)'),
        ('Fermi coupling constant', 'weak mixing angle',
         lambda x, y: float(abs(Float4096(x['scale']) - Float4096(y['scale']) / sqrt(Float4096(2)))),
         0.1, 'scale vs. sin²θ_W/√2'),
        ('tau energy equivalent', 'tau mass energy equivalent in MeV',
         lambda x, y: float(abs(Float4096(x['codata_value']) - Float4096(y['codata_value']))),
         0.01, 'value consistency'),
        ('proton mass', 'electron mass', 'proton-electron mass ratio',
         lambda x, y, z: float(abs(Float4096(z['n']) - abs(Float4096(x['n']) - Float4096(y['n'])))),
         10.0, 'n inconsistency for mass ratio'),
        ('fine-structure constant', 'elementary charge', 'Planck constant',
         lambda x, y, z: float(abs(Float4096(x['codata_value']) - (Float4096(y['codata_value']) ** Float4096(2)) /
                                   (Float4096(4) * pi_val() * Float4096(8.854187817e-12) * Float4096(z['codata_value']) * Float4096(299792458)))),
         0.01, 'fine-structure vs. e²/(4πε₀hc)'),
        ('Bohr magneton', 'elementary charge', 'Planck constant',
         lambda x, y, z: float(abs(Float4096(x['codata_value']) - Float4096(y['codata_value']) * Float4096(z['codata_value']) /
                                   (Float4096(2) * Float4096(9.1093837e-31)))),
         0.01, 'Bohr magneton vs. eh/(2m_e)'),
        ('speed of light in vacuum', None,
         lambda x: float(abs(Float4096(x['codata_value']) - Float4096(299792458))),
         0.01, 'speed of light deviation'),
        ('Newtonian constant of gravitation', None,
         lambda x: float(abs(Float4096(x['codata_value']) - Float4096(6.6743e-11))),
         1e-12, 'G deviation')
    ]
    for relation in relations:
        try:
            if len(relation) == 5:
                name1, name2, check_func, threshold, reason = relation
                if name1 in df_results['name'].values and name2 in df_results['name'].values:
                    row1 = df_results[df_results['name'] == name1].iloc[0]
                    row2 = df_results[df_results['name'] == name2].iloc[0]
                    if check_func(row1, row2) > threshold:
                        bad_data.append((name1, f"Physical inconsistency: {reason}"))
                        bad_data.append((name2, f"Physical inconsistency: {reason}"))
            elif len(relation) == 6:
                name1, name2, name3, check_func, threshold, reason = relation
                if all(name in df_results['name'].values for name in [name1, name2, name3]):
                    row1 = df_results[df_results['name'] == name1].iloc[0]
                    row2 = df_results[df_results['name'] == name2].iloc[0]
                    row3 = df_results[df_results['name'] == name3].iloc[0]
                    if check_func(row1, row2, row3) > threshold:
                        bad_data.append((name3, f"Physical inconsistency: {reason}"))
            elif len(relation) == 4:
                name, _, check_func, threshold, reason = relation
                if name in df_results['name'].values:
                    row = df_results[df_results['name'] == name].iloc[0]
                    if check_func(row) > threshold:
                        bad_data.append((name, f"Physical inconsistency: {reason}"))
        except Exception as e:
            logging.warning(f"Physical consistency check failed for {relation}: {e}")
            continue
    return bad_data

def total_error(params, df_subset, prime_interp=None):
    r, k, Omega, base, scale = Float4096Array(params)
    df_results = symbolic_fit_all_constants(df_subset, base=base, Omega=Omega, r=r, k=k, scale=scale, prime_interp=prime_interp)
    if df_results.empty:
        return float('inf')
    valid_errors = Float4096Array(df_results['rel_error'].dropna())
    return float(mean(valid_errors)) if len(valid_errors) > 0 else float('inf')

def process_constant(row, r, k, Omega, base, scale, prime_interp=None):
    try:
        name, value, uncertainty, unit = (
            row['name'],
            Float4096(row['value']),
            Float4096(row['uncertainty']) if pd.notnull(row['uncertainty']) else Float4096(0.0),
            row['unit']
        )
        abs_value = abs(value)
        sign = Float4096(1) if value >= Float4096(0) else Float4096(-1)
        result = invert_D(abs_value, r=r, k=k, Omega=Omega, base=base, scale=scale, prime_interp=prime_interp)
        if result[0] is None:
            logging.warning(f"No valid fit for {name}")
            return {
                'name': name, 'codata_value': float(value), 'unit': unit, 'n': None, 'beta': None,
                'emergent_value': None, 'error': None, 'rel_error': None, 'codata_uncertainty': float(uncertainty),
                'emergent_uncertainty': None, 'scale': None, 'bad_data': True,
                'bad_data_reason': 'No valid fit found', 'r': None, 'k': None
            }
        n, beta, dynamic_scale, emergent_uncertainty, r_local, k_local = result
        approx = D(n, beta, r_local, k_local, Omega, base, scale * dynamic_scale, prime_interp=prime_interp)
        if approx is None:
            logging.warning(f"D returned None for {name}")
            return {
                'name': name, 'codata_value': float(value), 'unit': unit, 'n': None, 'beta': None,
                'emergent_value': None, 'error': None, 'rel_error': None, 'codata_uncertainty': float(uncertainty),
                'emergent_uncertainty': None, 'scale': None, 'bad_data': True,
                'bad_data_reason': 'D function returned None', 'r': None, 'k': None
            }
        approx *= sign
        error = abs(approx - value)
        rel_error = error / max(abs(value), Float4096("1e-30")) if abs(value) > Float4096(0) else Float4096('inf')
        bad_data = False
        bad_data_reason = ""
        if rel_error > Float4096(0.5):
            bad_data = True
            bad_data_reason += f"High relative error ({float(rel_error):.2e} > 0.5); "
        if emergent_uncertainty is not None and uncertainty is not None:
            if emergent_uncertainty > uncertainty * Float4096(20) or emergent_uncertainty < uncertainty / Float4096(20):
                bad_data = True
                bad_data_reason += f"Uncertainty deviates from emergent ({float(emergent_uncertainty):.2e} vs. {float(uncertainty):.2e}); "
        return {
            'name': name, 'codata_value': float(value), 'unit': unit, 'n': float(n), 'beta': float(beta),
            'emergent_value': float(approx), 'error': float(error), 'rel_error': float(rel_error),
            'codata_uncertainty': float(uncertainty), 'emergent_uncertainty': float(emergent_uncertainty),
            'scale': float(scale * dynamic_scale), 'bad_data': bad_data, 'bad_data_reason': bad_data_reason,
            'r': float(r_local), 'k': float(k_local)
        }
    except Exception as e:
        logging.error(f"process_constant failed for {row['name']}: {e}")
        return {
            'name': row['name'], 'codata_value': row['value'], 'unit': row['unit'], 'n': None, 'beta': None,
            'emergent_value': None, 'error': None, 'rel_error': None, 'codata_uncertainty': row['uncertainty'],
            'emergent_uncertainty': None, 'scale': None, 'bad_data': True, 'bad_data_reason': f"Processing error: {str(e)}",
            'r': None, 'k': None
        }

def symbolic_fit_all_constants(df, base, Omega, r, k, scale, batch_size, prime_interp=None):
    logging.info(f"Starting symbolic fit with base={base}, Omega={Omega}, r={r}, k={k}, scale={scale}")
    start_time = time.time()
    base, Omega, r, k, scale = [Float4096(x) for x in (base, Omega, r, k, scale)]
    results = []
    output_file = "symbolic_fit_results_emergent_fixed.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        pd.DataFrame(columns=[
            'name', 'codata_value', 'unit', 'n', 'beta', 'emergent_value', 'error', 'rel_error',
            'codata_uncertainty', 'emergent_uncertainty', 'scale', 'bad_data', 'bad_data_reason', 'r', 'k'
        ]).to_csv(f, sep="\t", index=False)
    
    for start in range(0, len(df), batch_size):
        batch = df.iloc[start:start + batch_size]
        try:
            batch_results = Parallel(n_jobs=-1, timeout=120, backend='loky', maxtasksperchild=20)(
                delayed(process_constant)(row, r, k, Omega, base, scale, prime_interp)
                for row in tqdm(batch.to_dict('records'), total=len(batch), desc=f"Fitting constants batch {start//batch_size + 1}")
            )
            batch_results = [r for r in batch_results if r is not None]
            results.extend(batch_results)
            try:
                with open(output_file, 'a', encoding='utf-8') as f:
                    pd.DataFrame(batch_results).to_csv(f, sep="\t", index=False, header=False, lineterminator='\n')
                    f.flush()
            except Exception as e:
                logging.error(f"Failed to save batch {start//batch_size + 1} to {output_file}: {e}")
        except Exception as e:
            logging.error(f"Parallel processing failed for batch {start//batch_size + 1}: {e}")
            continue
    df_results = pd.DataFrame(results)
    if not df_results.empty:
        df_results['bad_data'] = df_results.get('bad_data', False)
        df_results['bad_data_reason'] = df_results.get('bad_data_reason', '')
        # Replace percentile with numpy.percentile
        for name in df_results['name'].unique():
            mask = df_results['name'] == name
            if df_results.loc[mask, 'codata_uncertainty'].notnull().any():
                uncertainties = np.array([float(x) for x in df_results.loc[mask, 'codata_uncertainty'].dropna()])
                if len(uncertainties) > 0:
                    Q1, Q3 = np.percentile(uncertainties, [25, 75])
                    IQR = Q3 - Q1
                    outlier_mask = (uncertainties < Q1 - 1.5 * IQR) | (uncertainties > Q3 + 1.5 * IQR)
                    if any(outlier_mask):
                        outlier_indices = df_results.loc[mask].index[np.where(outlier_mask)[0]]
                        df_results.loc[outlier_indices, 'bad_data'] = True
                        df_results.loc[outlier_indices, 'bad_data_reason'] += 'Uncertainty outlier; '
        high_rel_error_mask = df_results['rel_error'] > 0.5
        df_results.loc[high_rel_error_mask, 'bad_data'] = True
        df_results.loc[high_rel_error_mask, 'bad_data_reason'] += df_results.loc[high_rel_error_mask, 'rel_error'].apply(
            lambda x: f"High relative error ({x:.2e} > 0.5); "
        )
        high_uncertainty_mask = (df_results['emergent_uncertainty'].notnull()) & (
            (df_results['codata_uncertainty'] > 20 * df_results['emergent_uncertainty']) |
            (df_results['codata_uncertainty'] < 0.05 * df_results['emergent_uncertainty'])
        )
        df_results.loc[high_uncertainty_mask, 'bad_data'] = True
        df_results.loc[high_uncertainty_mask, 'bad_data_reason'] += df_results.loc[high_uncertainty_mask].apply(
            lambda row: f"Uncertainty deviates from emergent ({row['codata_uncertainty']:.2e} vs. {row['emergent_uncertainty']:.2e}); ",
            axis=1
        )
        bad_data = check_physical_consistency(df_results)
        for name, reason in bad_data:
            df_results.loc[df_results['name'] == name, 'bad_data'] = True
            df_results.loc[df_results['name'] == name, 'bad_data_reason'] += reason + '; '
    logging.info(f"Symbolic fit completed in {time.time() - start_time:.2f} seconds")
    return df_results

def select_worst_names(df, n_select):
    categories = df['category'].unique()
    n_per_category = max(1, n_select // len(categories))
    selected = []
    for category in categories:
        cat_df = df[df['category'] == category]
        if len(cat_df) > 0:
            n_to_select = min(n_per_category, len(cat_df))
            selected.extend(cat_df['name'].sample(n=n_to_select, replace=False).tolist())
    if len(selected) < n_select:
        remaining = df[~df['name'].isin(selected)]
        if len(remaining) > 0:
            selected.extend(remaining['name'].sample(n=n_select - len(selected), replace=False).tolist())
    return selected[:n_select]

def a_of_z(z):
    return Float4096(1) / (Float4096(1) + z)

def Omega(z, Omega0, alpha):
    a_z = a_of_z(z)
    return Omega0 / (a_z ** alpha)

def s(z, s0, beta):
    return s0 * (Float4096(1) + z) ** (-beta)

def G(z, k, r0, Omega0, s0, alpha, beta):
    return Omega(z, Omega0, alpha) * (k ** Float4096(2)) * r0 / s(z, s0, beta)

def H(z, k, r0, Omega0, s0, alpha, beta):
    Om_m = Float4096("0.3")
    Om_de = Float4096("0.7")
    Gz = G(z, k, r0, Omega0, s0, alpha, beta)
    Hz_sq = (Float4096(H0) ** Float4096(2)) * (Om_m * Gz * (Float4096(1) + z) ** Float4096(3) + Om_de)
    return sqrt(Hz_sq)

def emergent_c(z, Omega0, alpha, gamma):
    return Float4096(c0_emergent) * (Omega(z, Omega0, alpha) / Omega0) ** gamma * lambda_scale

def compute_luminosity_distance_grid(z_max, params, n=500):
    k, r0, Omega0, s0, alpha, beta, gamma = params
    z_grid = linspace(Float4096(0), z_max, n)
    c_z = emergent_c(z_grid, Omega0, alpha, gamma)
    H_z = H(z_grid, k, r0, Omega0, s0, alpha, beta)
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

def signal_handler(sig, frame):
    print("\nKeyboardInterrupt detected. Saving partial results...")
    logging.info("KeyboardInterrupt detected. Exiting gracefully.")
    for output_file in ["emergent_constants.txt", "symbolic_fit_results_emergent_fixed.txt", "cosmo_fit_results.txt"]:
        try:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.flush()
        except Exception as e:
            logging.error(f"Failed to flush {output_file} on interrupt: {e}")
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, signal_handler)
    start_time = time.time()
    stages = [
        'Parsing data', 'Generating emergent constants', 'Optimizing CODATA parameters',
        'Fitting CODATA constants', 'Fitting supernova data', 'Generating plots'
    ]
    progress = tqdm(stages, desc="Overall progress")
    
    prime_interp = prepare_prime_interpolation()
    
    # Stage 1: Parse CODATA
    if not os.path.exists(CONFIG['codata_file']):
        raise FileNotFoundError(f"{CONFIG['codata_file']} not found in the current directory")
    df = parse_categorized_codata(CONFIG['codata_file'])
    logging.info(f"Parsed {len(df)} constants")
    progress.update(1)
    
    # Stage 2: Generate emergent constants
    emergent_df = generate_emergent_constants(
        CONFIG['n_max'], CONFIG['beta_steps'], CONFIG['r_values'], CONFIG['k_values'],
        CONFIG['Omega'], CONFIG['base'], CONFIG['scale'], prime_interp=prime_interp
    )
    matched_df = match_to_codata(
        emergent_df, df, CONFIG['tolerance'], CONFIG['batch_size'], prime_interp=prime_interp
    )
    logging.info("Saved emergent constants to emergent_constants.txt")
    progress.update(1)
    
    # Stage 3: Optimize CODATA parameters
    worst_names = select_worst_names(df, CONFIG['n_select_worst'])
    print(f"Selected constants for optimization: {worst_names}")
    subset_df = df[df['name'].isin(worst_names)]
    if subset_df.empty:
        subset_df = df.head(50)
    init_params = [0.5, 0.5, 0.5, 2.0, 0.1]
    try:
        res = differential_evolution(
            total_error, CONFIG['optimization_bounds'], args=(subset_df, prime_interp),
            maxiter=CONFIG['optimization_maxiter'], popsize=CONFIG['optimization_popsize']
        )
        if res.success:
            res = minimize(
                total_error, res.x, args=(subset_df, prime_interp), bounds=CONFIG['optimization_bounds'],
                method='SLSQP', options={'maxiter': CONFIG['slsqp_maxiter']}
            )
        if not res.success:
            logging.warning(f"Optimization failed: {res.message}")
            r_opt, k_opt, Omega_opt, base_opt, scale_opt = init_params
        else:
            r_opt, k_opt, Omega_opt, base_opt, scale_opt = res.x
        print(f"CODATA Optimization complete. Found parameters:\nr = {r_opt:.6f}, k = {k_opt:.6f}, Omega = {Omega_opt:.6f}, base = {base_opt:.6f}, scale = {scale_opt:.6f}")
    except Exception as e:
        logging.error(f"CODATA Optimization failed: {e}")
        r_opt, k_opt, Omega_opt, base_opt, scale_opt = init_params
        print(f"CODATA Optimization failed: {e}. Using default parameters.")
    progress.update(1)
    
    # Stage 4: Fit CODATA constants
    df_results = symbolic_fit_all_constants(
        df, base=base_opt, Omega=Omega_opt, r=r_opt, k=k_opt, scale=scale_opt,
        batch_size=CONFIG['batch_size'], prime_interp=prime_interp
    )
    if not df_results.empty:
        with open("symbolic_fit_results.txt", 'w', encoding='utf-8') as f:
            df_results.to_csv(f, sep="\t", index=False)
            f.flush()
        logging.info(f"Saved CODATA results to symbolic_fit_results.txt")
    else:
        logging.error("No CODATA results to save")
    progress.update(1)
    
    # Stage 5: Fit supernova data
    if not os.path.exists(CONFIG['supernova_file']):
        raise FileNotFoundError(f"{CONFIG['supernova_file']} not found")
    lc_data = pd.read_csv(CONFIG['supernova_file'], delim_whitespace=True, comment='#', dtype={'zcmb': float, 'mb': float, 'dmb': float})
    z = Float4096Array(lc_data['zcmb'])
    mb = Float4096Array(lc_data['mb'])
    dmb = Float4096Array(lc_data['dmb'])
    
    # Reconstruct cosmological parameters
    fitted_params = {
        'k': 1.049342, 'r0': 1.049676, 'Omega0': 1.049675, 's0': 0.994533,
        'alpha': 0.340052, 'beta': 0.360942, 'gamma': 0.993975, 'H0': 70.0,
        'c0': float(sqrt(Float4096(2.5) * Float4096(6))), 'M': -19.3
    }
    params_reconstructed = {}
    for name, val in fitted_params.items():
        if name == 'M':
            params_reconstructed[name] = Float4096(val)
            continue
        n, beta, scale, _, r, k = invert_D(Float4096(val), prime_interp=prime_interp)
        params_reconstructed[name] = D(n, beta, r, k, prime_interp=prime_interp) if name != 'c0' else sqrt(Float4096(2.5) * n)
    
    global H0, c0_emergent, lambda_scale, lambda_G
    H0 = params_reconstructed['H0']
    c0_emergent = params_reconstructed['c0']
    lambda_scale = Float4096(299792.458) / c0_emergent
    lambda_G = Float4096(6.6743e-11) / G(
        Float4096(0), params_reconstructed['k'], params_reconstructed['r0'],
        params_reconstructed['Omega0'], params_reconstructed['s0'],
        params_reconstructed['alpha'], params_reconstructed['beta']
    )
    
    param_list = [
        params_reconstructed['k'], params_reconstructed['r0'], params_reconstructed['Omega0'],
        params_reconstructed['s0'], params_reconstructed['alpha'], params_reconstructed['beta'],
        params_reconstructed['gamma']
    ]
    mu_fit = model_mu(z, param_list)
    residuals = mb - params_reconstructed['M'] - mu_fit
    
    cosmo_results = pd.DataFrame({
        'z': z.__array__(),
        'mu_obs': (mb - params_reconstructed['M']).__array__(),
        'mu_fit': mu_fit.__array__(),
        'residuals': residuals.__array__(),
        'dmb': dmb.__array__()
    })
    with open("cosmo_fit_results.txt", 'w', encoding='utf-8') as f:
        cosmo_results.to_csv(f, sep="\t", index=False)
        f.flush()
    logging.info("Saved supernova fit results to cosmo_fit_results.txt")
    progress.update(1)
    
    # Stage 6: Generate plots
    df_results_sorted = df_results.sort_values("rel_error", na_position='last')
    print("\nTop 20 best CODATA fits:")
    print(df_results_sorted.head(20)[[
        'name', 'codata_value', 'unit', 'n', 'beta', 'emergent_value', 'error',
        'rel_error', 'codata_uncertainty', 'scale', 'bad_data', 'bad_data_reason'
    ]].to_string(index=False))
    print("\nTop 20 worst CODATA fits:")
    print(df_results_sorted.tail(20)[[
        'name', 'codata_value', 'unit', 'n', 'beta', 'emergent_value', 'error',
        'rel_error', 'codata_uncertainty', 'scale', 'bad_data', 'bad_data_reason'
    ]].to_string(index=False))
    print("\nPotentially bad data constants:")
    bad_data_df = df_results[df_results['bad_data'] == True][[
        'name', 'codata_value', 'error', 'rel_error', 'codata_uncertainty',
        'emergent_uncertainty', 'bad_data_reason'
    ]]
    print(bad_data_df.to_string(index=False))
    print("\nTop 20 emergent constants matches:")
    matched_df_sorted = matched_df.sort_values('rel_error', na_position='last')
    print(matched_df_sorted.head(20)[[
        'name', 'codata_value', 'emergent_value', 'n', 'beta', 'error', 'rel_error',
        'codata_uncertainty', 'bad_data', 'bad_data_reason'
    ]].to_string(index=False))
    
    plt.figure(figsize=(10, 5))
    plt.hist(df_results_sorted['rel_error'].dropna(), bins=50, color='skyblue', edgecolor='black')
    plt.title('Histogram of Relative Errors in CODATA Fit')
    plt.xlabel('Relative Error')
    plt.ylabel('Count')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('histogram_rel_errors.png')
    plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.scatter(df_results_sorted['n'], df_results_sorted['rel_error'], alpha=0.5, s=15, c='orange', edgecolors='black')
    plt.title('Relative Error vs Symbolic Dimension n (CODATA)')
    plt.xlabel('n')
    plt.ylabel('Relative Error')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('scatter_n_rel_error.png')
    plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.bar(matched_df_sorted.head(20)['name'], matched_df_sorted.head(20)['rel_error'], color='purple', edgecolor='black')
    plt.xticks(rotation=90)
    plt.title('Relative Errors for Top 20 Emergent Constants')
    plt.xlabel('Constant Name')
    plt.ylabel('Relative Error')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('bar_emergent_errors.png')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(z.__array__(), (mb - params_reconstructed['M']).__array__(), yerr=dmb.__array__(), fmt='.', alpha=0.5, label='Pan-STARRS1 SNe')
    plt.plot(z.__array__(), mu_fit.__array__(), 'r-', label='Symbolic Emergent Gravity Model')
    plt.xlabel('Redshift (z)')
    plt.ylabel('Distance Modulus (μ)')
    plt.title('Supernova Distance Modulus with Emergent G(z) and c(z)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('supernova_fit.png')
    plt.close()
    
    plt.figure(figsize=(10, 4))
    plt.errorbar(z.__array__(), residuals.__array__(), yerr=dmb.__array__(), fmt='.', alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Redshift (z)')
    plt.ylabel('Residuals (μ_data - μ_model)')
    plt.title('Residuals of Symbolic Model with Emergent G(z) and c(z)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('supernova_residuals.png')
    plt.close()
    
    z_grid = linspace(Float4096(0), max(z), 300)
    c_z = emergent_c(z_grid, params_reconstructed['Omega0'], params_reconstructed['alpha'], params_reconstructed['gamma'])
    G_z = G(z_grid, params_reconstructed['k'], params_reconstructed['r0'], params_reconstructed['Omega0'],
            params_reconstructed['s0'], params_reconstructed['alpha'], params_reconstructed['beta']) * lambda_G
    G_z_norm = G_z / G_z[0]
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(z_grid.__array__(), c_z.__array__(), label='c(z) (km/s)')
    plt.axhline(299792.458, color='red', linestyle='--', label='Local c')
    plt.xlabel('Redshift z')
    plt.ylabel('Speed of Light c(z) [km/s]')
    plt.title('Emergent Speed of Light Variation')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(z_grid.__array__(), G_z_norm.__array__(), label='G(z) / G_0')
    plt.axhline(1.0, color='red', linestyle='--', label='Local G')
    plt.xlabel('Redshift z')
    plt.ylabel('Normalized Gravitational Constant G(z)/G_0')
    plt.title('Emergent Gravitational Constant Variation')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('emergent_c_G.png')
    plt.close()
    
    logging.info(f"Total runtime: {time.time() - start_time:.2f} seconds")
    progress.update(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        signal_handler(None, None)
