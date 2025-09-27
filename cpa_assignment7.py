#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Assignment 7 - CPA on T-Table Based AES (FPGA)
# Vectorized solver producing: key.txt, plots_k_corr/*, plots_nrtraces_corr/*, plots_points_corr/*, description.txt
import argparse, numpy as np, matplotlib.pyplot as plt
from pathlib import Path

SBOX = np.array([
  0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
  0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
  0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
  0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
  0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
  0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
  0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
  0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
  0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
  0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
  0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
  0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
  0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
  0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
  0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
  0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
], dtype=np.uint8)

def xtime(a):
    a = a.astype(np.uint16)
    res = (a << 1) & 0xFF
    res ^= ((a >> 7) & 1) * 0x1B
    return res.astype(np.uint8)

def mul(a, c):
    if c == 1: return a
    if c == 2: return xtime(a)
    if c == 3: return xtime(a) ^ a
    raise ValueError("mul only supports 1,2,3")

def ttable_words(s):
    s = s.astype(np.uint8)
    s2 = mul(s,2); s3 = mul(s,3)
    T0 = (s2.astype(np.uint32) << 24) | (s.astype(np.uint32) << 16) | (s.astype(np.uint32) << 8) | s3.astype(np.uint32)
    return T0  # T0 is enough to try tword/tbyte models

def HW8(x):  return np.unpackbits(x.reshape(-1,1), axis=1).sum(axis=1).astype(np.float32)
def HW32(x):
    b0 = ((x >> 24) & 0xFF).astype(np.uint8)
    b1 = ((x >> 16) & 0xFF).astype(np.uint8)
    b2 = ((x >> 8)  & 0xFF).astype(np.uint8)
    b3 = ( x        & 0xFF).astype(np.uint8)
    return (HW8(b0)+HW8(b1)+HW8(b2)+HW8(b3)).astype(np.float32)

def model_vectors(pt_b, model):
    N = pt_b.size
    kg = np.arange(256, dtype=np.uint16)
    PT = pt_b.reshape(1, N).astype(np.uint16)
    X = (PT ^ kg.reshape(256,1)).astype(np.uint8)
    S = SBOX[X]
    if model == "sbox_hw":
        return HW8(S).astype(np.float32)
    elif model == "tword_hw":
        T0 = ttable_words(S); return HW32(T0)
    elif model == "tbyte0_hw":
        T0 = ttable_words(S); return HW8(((T0>>24)&0xFF).astype(np.uint8))
    elif model == "tbyte1_hw":
        T0 = ttable_words(S); return HW8(((T0>>16)&0xFF).astype(np.uint8))
    elif model == "tbyte2_hw":
        T0 = ttable_words(S); return HW8(((T0>>8)&0xFF).astype(np.uint8))
    elif model == "tbyte3_hw":
        T0 = ttable_words(S); return HW8((T0&0xFF).astype(np.uint8))
    else:
        raise ValueError("Unknown model")

def pick_roi(traces, roi):
    var = traces.astype(np.float32).var(axis=0)
    if roi >= traces.shape[1]:
        idx = np.arange(traces.shape[1])
    else:
        idx = np.argpartition(-var, roi)[:roi]
        idx = idx[np.argsort(idx)]
    return idx.astype(np.int64)

def corr_all_keys(S, T):
    S_mean = S.mean(axis=1, keepdims=True)
    S_center = S - S_mean
    S_norm = np.sqrt((S_center**2).sum(axis=1, keepdims=True) + 1e-12)
    T_mean = T.mean(axis=0, keepdims=True)
    T_center = T - T_mean
    T_norm = np.sqrt((T_center**2).sum(axis=0, keepdims=True) + 1e-12)
    num = S_center @ T_center
    den = S_norm * T_norm
    return (num/den).astype(np.float32)

def progressive_corr_max(S, T, steps):
    N = S.shape[1]
    sizes = np.linspace(N/steps, N, steps, dtype=np.int64)
    out = np.zeros((steps, 256), dtype=np.float32)
    for i, n in enumerate(sizes):
        C = corr_all_keys(S[:, :n], T[:n, :])
        out[i] = np.max(np.abs(C), axis=1)
    return sizes, out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="Traces00000.dat")
    ap.add_argument("--plaintexts", default="plaintexts.dat")
    ap.add_argument("--ciphertexts", default="ciphertexts.dat")
    ap.add_argument("--outdir", default="assignment7_outputs")
    ap.add_argument("--model", default="sbox_hw",
                    choices=["sbox_hw","tword_hw","tbyte0_hw","tbyte1_hw","tbyte2_hw","tbyte3_hw"])
    ap.add_argument("--roi", type=int, default=800)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--lastname", default="OmidiZadeh")
    args = ap.parse_args()

    out = Path(args.outdir)
    (out / "plots_k_corr").mkdir(parents=True, exist_ok=True)
    (out / "plots_nrtraces_corr").mkdir(parents=True, exist_ok=True)
    (out / "plots_points_corr").mkdir(parents=True, exist_ok=True)

    N=10000; M=10000
    T = np.fromfile(args.traces, dtype=np.int8).reshape(N, M).astype(np.float32)
    P = np.fromfile(args.plaintexts, dtype=np.uint8).reshape(16, N)

    roi_idx = pick_roi(T, args.roi)
    T_roi = T[:, roi_idx]

    recovered = []; traces_needed = []

    for b in range(16):
        pt_b = P[b]
        S = model_vectors(pt_b, args.model)
        C = corr_all_keys(S, T_roi)

        max_abs = np.max(np.abs(C), axis=1)
        k_hat = int(np.argmax(max_abs))
        recovered.append(k_hat)

        xs = np.arange(256)
        plt.figure(figsize=(6,4), dpi=120)
        plt.plot(xs, max_abs, linewidth=1.0)
        plt.xlabel(f"key guess for byte {b}")
        plt.ylabel("max |corr| over ROI")
        plt.tight_layout()
        plt.savefig(out / "plots_k_corr" / f"plot_k_corr_{b:02d}.png"); plt.close()

        sizes, prog = progressive_corr_max(S, T_roi, args.steps)
        correct_curve = prog[:, k_hat]
        global_max = prog.max(axis=1)
        cross_idx = np.argmax(correct_curve >= global_max - 1e-9)
        n_needed = int(sizes[cross_idx]) if correct_curve[cross_idx] >= global_max[cross_idx]-1e-9 else int(sizes[-1])
        traces_needed.append(n_needed)

        plt.figure(figsize=(16,5), dpi=120)
        for k in range(256):
            lw = 2.0 if k == k_hat else 0.4
            plt.plot(sizes, prog[:, k], linewidth=lw)
        plt.xlabel("nr of traces"); plt.ylabel("max |corr| over ROI")
        plt.tight_layout()
        plt.savefig(out / "plots_nrtraces_corr" / f"plot_nrtraces_corr_{b:02d}.png"); plt.close()

        plt.figure(figsize=(12,4), dpi=120)
        plt.imshow(C, aspect="auto", interpolation="nearest")
        plt.xlabel("ROI sample index"); plt.ylabel("key guess (0..255)")
        plt.tight_layout()
        plt.savefig(out / "plots_points_corr" / f"plot_points_corr_{b:02d}.png"); plt.close()

    key_hex = " ".join(f"{k:02x}" for k in recovered)
    key_dec = " ".join(str(k) for k in recovered)
    (out / "key.txt").write_text("hex: "+key_hex+"
"+"dec: "+key_dec+"
", encoding="utf-8")
    (out / "description.txt").write_text(
        "CPA on T-Table AES (FPGA). Model: "+args.model+". ROI: "+str(args.roi)+". Steps: "+str(args.steps)+".
"+
        "Traces needed per byte: "+", ".join(str(n) for n in traces_needed)+"
"+
        "If some bytes fail: try different model (tword_hw / tbyte*_hw), adjust ROI, or align rounds.
", encoding="utf-8")

    print("Recovered key (hex):", key_hex)
    print("Outputs:", out.resolve())

if __name__ == "__main__":
    main()
