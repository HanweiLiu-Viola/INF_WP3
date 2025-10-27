import sys, numpy as np, pathlib

def compute_adtf(data: np.ndarray) -> np.ndarray:
    # TODO: implement real ADTF; placeholder uses rFFT magnitude
    return np.abs(np.fft.rfft(data, axis=1))

def main(inp, out):
    x = np.load(inp)
    res = compute_adtf(x)
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.save(out, res)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
