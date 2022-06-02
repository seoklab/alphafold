import sys
import subprocess as sp


def memfraction():
  try:
    nvidia_smi_out = sp.run(["nvidia-smi", "--query-gpu=memory.total",
                             "--format=csv,noheader,nounits"],
                            stdout=sp.PIPE, text=True, check=True).stdout
    gpu_total = sum(map(int, nvidia_smi_out.splitlines()))
  except FileNotFoundError:
    # No gpu; just use 100% of cpu mem
    sys.stdout.write("1.0\n")
    return 0

  lsmem_out = sp.run(["lsmem", "--summary", "--bytes"],
                     stdout=sp.PIPE, text=True, check=True).stdout
  for line in lsmem_out.splitlines():
    if line.startswith("Total online"):
      cpu_total = int(line.split()[-1]) / 1024**2  # to MiB
      break
  else:
    raise RuntimeError("Could not find total RAM")

  sys.stdout.write(f"{(cpu_total / gpu_total) * 0.95 + 1:.2f}\n")
