import sys, numpy as np, pandas as pd, pathlib

def mcc_placeholder(arr: np.ndarray) -> float:
    # TODO: replace with real MCC against ground truth adjacency
    return float(np.clip(arr.mean() / (arr.std() + 1e-6), -1.0, 1.0))

def main(inp, out):
    arr = np.load(inp)
    mcc = mcc_placeholder(arr)
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"metric": "MCC", "value": mcc}]).to_csv(out, index=False)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
