"""Benchmark Hamming distance: old loop vs new vectorized."""
import os, sys, time
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import numpy as np, yaml

with open('iris_recognition/configs/config.yaml') as f:
    config = yaml.safe_load(f)

from iris_recognition.matching.matcher import IrisMatcher
matcher = IrisMatcher(config)

H, C = 64, 4096
np.random.seed(42)
ca = np.random.randint(0, 2, (H, C), dtype=np.uint8)
ma = np.ones((H, C), dtype=np.uint8)
cb = np.random.randint(0, 2, (H, C), dtype=np.uint8)
mb = np.ones((H, C), dtype=np.uint8)

# Warmup
matcher.compute_hd(ca, ma, cb, mb)

# Benchmark
N = 100
t = time.time()
for _ in range(N):
    hd, sh = matcher.compute_hd(ca, ma, cb, mb)
elapsed = (time.time() - t) / N * 1000
print(f"Vectorized HD: {elapsed:.2f} ms/call  hd={hd:.4f}  shift={sh}")
print(f"Estimated genuine time  (~74k pairs): {elapsed*74000/1000:.0f}s")
print(f"Estimated impostor time (~50k pairs): {elapsed*50000/1000:.0f}s")
print(f"Estimated rank-1 time   (~2k x 2k ) : {elapsed*2000*2000/1000:.0f}s")
