import jax

BACKEND = jax.default_backend()

if BACKEND == "cpu":
  DEV_CNT = 1
  DEV_POOL = [None]
  PLATFORM = "CPU"
  PROPERTIES = None
else:
  DEV_CNT = len(jax.devices())
  DEV_POOL = list(range(DEV_CNT))
  PLATFORM = "CUDA" if BACKEND == "gpu" else "CPU"
  PROPERTIES = {"CudaDeviceIndex": ",".join(map(str, DEV_POOL))}
