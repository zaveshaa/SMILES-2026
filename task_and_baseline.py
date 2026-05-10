import numpy as np
from scipy.signal import convolve, firwin

CENTER = 1.9e6
BW = 0.6e6
MODEL_SUBSET = slice(20_000, 220_000)
MODEL_LAGS = tuple(range(-6, 7))
MIN_EXPLAIN_RATIO = 0.95
MAX_UNEXPLAINED_TO_RESIDUAL = 0.80

def make_bandpass(center_hz, bw_hz, fs_hz, n_taps=2047):
    lp = firwin(n_taps, bw_hz / 2, window="blackman", fs=fs_hz)
    return lp * np.exp(2j * np.pi * center_hz / fs_hz * np.arange(n_taps))

def shift_signal(x, k):
    y = np.zeros_like(x)
    if k >= 0: y[k:] = x[: len(x) - k]
    else: y[: len(x) + k] = x[-k:]
    return y

def shifted_window(x, k, start, stop):
    out = np.zeros(stop - start, dtype=np.complex128)
    src_start, src_stop = max(0, start - k), min(len(x), stop - k)
    if src_start < src_stop:
        out[src_start + k - start : src_stop + k - start] = x[src_start:src_stop]
    return out

def build_task_helpers(tx_n, fs_hz, n_samples):
    score_bp = make_bandpass(CENTER, BW, fs_hz)
    def score_filter(x): return convolve(x, score_bp, mode="same")
    
    terms = tuple(score_filter(tx_n[:, a]**2 * tx_n[:, b].conj()) for a, b in 
                 [(0,1),(1,0),(0,3),(3,0),(1,2),(2,1),(3,2),(2,3),(0,5),(5,0)])

    m_x = np.column_stack([shifted_window(t, l, MODEL_SUBSET.start, MODEL_SUBSET.stop) 
                           for t in terms for l in MODEL_LAGS])
    m_g = m_x.conj().T @ m_x + 1e-6 * np.eye(m_x.shape[1])

    def fit_tx_prediction(inp):
        p = np.zeros_like(inp)
        for c in range(inp.shape[1]):
            y = score_filter(inp[:, c])[MODEL_SUBSET]
            sol = np.linalg.solve(m_g, m_x.conj().T @ y).reshape(len(terms), len(MODEL_LAGS))
            for i, t in enumerate(terms):
                for cv, lag in zip(sol[i], MODEL_LAGS):
                    p[:, c] += cv * shift_signal(t, lag)
        return p

    def score(rx_b, rx_a, label=""):
        diff = rx_b - rx_a
        rem_b = np.column_stack([score_filter(diff[:, c]) for c in range(4)])
        tx_p = fit_tx_prediction(diff)
        res_b = rem_b - tx_p
        cov = res_b.conj().T @ res_b / res_b.shape[0]
        _, v = np.linalg.eigh(cov)
        sh = res_b @ v[:, -1]
        rk1 = np.column_stack([(np.vdot(sh, res_b[:, c]) / (np.vdot(sh, sh) + 1e-30)) * sh for c in range(4)])
        err = res_b - rk1
        
        exp = 1.0 - np.mean(np.abs(err)**2) / (np.mean(np.abs(rem_b)**2) + 1e-30)
        aft_b = np.column_stack([score_filter(rx_a[:, c]) for c in range(4)])
        err_p, res_p = np.mean(np.abs(err)**2, axis=0), np.mean(np.abs(aft_b)**2, axis=0) + 1e-30
        valid = exp >= MIN_EXPLAIN_RATIO and np.all(err_p <= MAX_UNEXPLAINED_TO_RESIDUAL * res_p)
        
        reds = [10 * np.log10(np.mean(np.abs(score_filter(rx_b[:, c]))**2) / 
                (np.mean(np.abs(score_filter(rx_a[:, c]))**2) + 1e-30)) if valid else 0.0 for c in range(4)]
        avg = np.mean(reds)
        print(f"  [{label}] Avg: {avg:.2f} dB {'(INVALID)' if not valid else ''}")
        return reds, avg

    return {"score_filter": score_filter, "fit_tx_prediction": fit_tx_prediction, "score": score}

def baseline(tx_n, rx, fit_p):
    return rx - fit_p(rx)
