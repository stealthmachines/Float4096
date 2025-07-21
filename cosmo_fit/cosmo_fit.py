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
from scipy.optimize import minimize
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
    'n_jobs': 4,
    'joblib_timeout': 300,
    'use_tqdm': True,
    'optimization_bounds': [(1e-5, 10), (1e-5, 10), (1e-5, 10), (1.5, 10), (1e-5, 100)],
    'optimization_maxiter': 100,
}

# Set up logging
logging.basicConfig(
    filename='symbolic_cosmo_fit.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)

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
        phi = (Float4096(1) + sqrt(Float4096(5))) / Float4096(2)
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

def total_error(params, df, prime_interp=None):
    r, k, Omega, base, scale = params
    df_fit = symbolic_fit_all_constants(df, r=r, k=k, Omega=Omega, base=base, scale=scale,
                                       max_n=CONFIG['max_n'], steps=CONFIG['beta_steps'], prime_interp=prime_interp)
    if df_fit.empty:
        return float('inf')
    errors = df_fit['rel_error'].dropna().values
    errors = np.array([float(e) for e in errors])
    threshold = np.percentile(errors, 95)
    filtered_errors = errors[errors <= threshold]
    return np.sum(filtered_errors ** 2) if len(filtered_errors) > 0 else float('inf')

def main():
    start_time = time.time()
    prime_interp = prepare_prime_interpolation()
    
    # Parse CODATA
    if not os.path.exists(CONFIG['codata_file']):
        raise FileNotFoundError(f"{CONFIG['codata_file']} not found")
    df_codata = parse_codata_ascii(CONFIG['codata_file'])
    logging.info(f"Parsed {len(df_codata)} constants")
    
    # Optimize parameters
    subset_df = df_codata.head(20)
    init_params = [1.0, 1.0, 1.0, 2.0, 1.0]
    try:
        res = minimize(
            total_error, init_params, args=(subset_df, prime_interp),
            bounds=CONFIG['optimization_bounds'], method='L-BFGS-B',
            options={'maxiter': CONFIG['optimization_maxiter']}
        )
        if res.success:
            r_opt, k_opt, Omega_opt, base_opt, scale_opt = res.x
            print(f"Optimization complete. Found parameters:\nr = {r_opt:.6f}, k = {k_opt:.6f}, "
                  f"Omega = {Omega_opt:.6f}, base = {base_opt:.6f}, scale = {scale_opt:.6f}")
        else:
            logging.warning(f"Optimization failed: {res.message}")
            r_opt, k_opt, Omega_opt, base_opt, scale_opt = init_params
    except Exception as e:
        logging.error(f"Optimization failed: {e}")
        r_opt, k_opt, Omega_opt, base_opt, scale_opt = init_params
    
    # Fit all constants
    fitted_df = symbolic_fit_all_constants(
        df_codata, r=r_opt, k=k_opt, Omega=Omega_opt, base=base_opt, scale=scale_opt,
        max_n=CONFIG['max_n'], steps=CONFIG['beta_steps'], prime_interp=prime_interp
    )
    fitted_df_sorted = fitted_df.sort_values("error")
    
    # Output results
    print("\nTop 20 best symbolic fits:")
    print(fitted_df_sorted.head(20).to_string(index=False))
    print("\nTop 20 worst symbolic fits:")
    print(fitted_df_sorted.tail(20).to_string(index=False))
    
    # Generate plots
    try:
        plt.figure(figsize=(10, 5))
        plt.hist(fitted_df_sorted['error'], bins=50, color='skyblue', edgecolor='black')
        plt.title('Histogram of Absolute Errors in Symbolic Fit')
        plt.xlabel('Absolute Error')
        plt.ylabel('Count')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('error_histogram.png')
        plt.close()
        
        plt.figure(figsize=(10, 5))
        plt.scatter(fitted_df_sorted['n'], fitted_df_sorted['error'], alpha=0.5, s=15, c='orange', edgecolors='black')
        plt.title('Absolute Error vs Symbolic Dimension n')
        plt.xlabel('n')
        plt.ylabel('Absolute Error')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('error_vs_n.png')
        plt.close()
        logging.info("Generated plots: error_histogram.png, error_vs_n.png")
    except Exception as e:
        logging.error(f"Plot generation failed: {e}")
    
    logging.info(f"Total execution time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
