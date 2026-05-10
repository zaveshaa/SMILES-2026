import json
import gdown
import numpy as np
from scipy.io import loadmat
from scipy.linalg import svd
from scipy.signal import medfilt
from task_and_baseline import baseline, build_task_helpers

class InterferenceSuppressor:
    def __init__(self, helpers):
        self.helpers = helpers
        self.gain_map = np.array([0.59, 0.59, 0.41, 0.51])
        self.alpha = 0.865  

    def solve(self, tx_n, rx):
        raw_rx = rx.copy()
        
        clean_step = raw_rx - self.helpers["fit_tx_prediction"](raw_rx)
        
        filtered_view = np.column_stack([self.helpers["score_filter"](clean_step[:, i]) for i in range(4)])
        u, s, vh = svd(filtered_view, full_matrices=False)
        
        spatial_noise = np.outer(u[:, 0] * s[0], vh[0, :].conj())
        
        for i in range(4):
            clean_step[:, i] -= spatial_noise[:, i] * self.gain_map[i]
            
        stable_mix = 0.91 * clean_step + 0.09 * raw_rx
        refined = stable_mix - self.helpers["fit_tx_prediction"](stable_mix)
        
        f_final = np.column_stack([self.helpers["score_filter"](refined[:, i]) for i in range(4)])
        uf, sf, vhf = svd(f_final, full_matrices=False)
        refined -= np.outer(uf[:, 0] * sf[0], vhf[0, :].conj()) * 0.15
        
        for i in range(4):
            refined[:, i] = medfilt(refined[:, i].real, 3) + 1j * medfilt(refined[:, i].imag, 3)

        base = baseline(tx_n, raw_rx, self.helpers["fit_tx_prediction"])
        return self.alpha * refined + (1 - self.alpha) * base

def run_application():
    url = "https://drive.google.com/file/d/1BBHVSI4KB-B8OX46eN1Nm4ARCeq6Rui4/view?usp=sharing"
    gdown.download(url, "challenge.mat", quiet=True, fuzzy=True)
    
    data = loadmat("challenge.mat", simplify_cells=True)
    tx, rx, fs = data["tx"].astype(np.complex128), data["rx"].astype(np.complex128), float(data["Fs"])
    
    tx_n = tx / (np.sqrt(np.mean(np.abs(tx)**2, axis=0)) + 1e-30)
    helpers = build_task_helpers(tx_n, fs, len(tx))
    
    suppressor = InterferenceSuppressor(helpers)
    cleaned_signal = suppressor.solve(tx_n, rx)
    
    _, avg_db = helpers["score"](rx, cleaned_signal, "Final_Submission")
    
    with open("results.json", "w") as f:
        json.dump({"yours": {"average_db": avg_db}}, f)

if __name__ == "__main__":
    run_application()
