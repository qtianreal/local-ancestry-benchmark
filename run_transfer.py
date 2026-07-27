"""Cross-divergence transfer: how badly does a CNN trained at one divergence
degrade when applied at another?

This is the model-misspecification experiment. Simulation-trained methods are
routinely criticised for learning the simulator rather than the biology; here
we quantify that directly by evaluating every trained model against every
held-out test set.
"""

import json
from pathlib import Path

import numpy as np
import torch

from lai.methods import DilatedCNN

OUT = Path("results")
CACHE = OUT / "cache"


@torch.no_grad()
def accuracy(model, x, y, device, batch=64):
    model.eval()
    correct = tot = 0
    for i in range(0, x.shape[0], batch):
        xb = torch.from_numpy(x[i : i + batch].astype(np.float32)).to(device)
        pred = (torch.sigmoid(model(xb)) > 0.5).cpu().numpy().astype(np.int8)
        correct += (pred == y[i : i + batch]).sum()
        tot += y[i : i + batch].size
    return float(correct / tot)


def main():
    main_results = json.loads((OUT / "main_results.json").read_text())
    split_times = [r["split_time"] for r in main_results]
    fst_by_t = {r["split_time"]: r["fst"] for r in main_results}

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    tests = {}
    for T in split_times:
        d = np.load(CACHE / f"test_T{T}.npz")
        tests[T] = (d["x"], d["y"])

    matrix = np.zeros((len(split_times), len(split_times)))
    for i, t_train in enumerate(split_times):
        model = DilatedCNN().to(device)
        model.load_state_dict(torch.load(CACHE / f"cnn_T{t_train}.pt", map_location=device))
        for j, t_test in enumerate(split_times):
            x, y = tests[t_test]
            matrix[i, j] = accuracy(model, x, y, device)
        print(f"trained T={t_train:5d} (Fst={fst_by_t[t_train]:.5f}): "
              + " ".join(f"{v:.3f}" for v in matrix[i]), flush=True)

    payload = {
        "split_times": split_times,
        "fst": [fst_by_t[t] for t in split_times],
        "matrix": matrix.tolist(),
        "note": "matrix[i][j] = accuracy of model trained at split_times[i], "
                "evaluated on held-out test data from split_times[j]",
    }
    (OUT / "transfer_results.json").write_text(json.dumps(payload, indent=2))
    print("\nwrote results/transfer_results.json")


if __name__ == "__main__":
    main()
