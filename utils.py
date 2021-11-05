import os
import psutil

from alphafold.common import devices


def check_nvidia_cache():
  pass


if devices.BACKEND != "cpu":
  _NFS_CACHE = frozenset(
      os.stat(pi.mountpoint).st_dev
      for pi in psutil.disk_partitions(all=True)
      if pi.fstype == 'nfs')

  if _NFS_CACHE:

    def check_nvidia_cache():  # noqa: F811
      nvidia_cachedir = os.path.expanduser('~/.nv')
      nvidia_cachedir_non_nfs = os.path.expandvars("/tmp/$USER/nv")

      try:
        devid = os.stat(nvidia_cachedir, follow_symlinks=True).st_dev
      except FileNotFoundError:
        # Maybe broken symlink, test it
        if os.path.islink(nvidia_cachedir):
          os.makedirs(nvidia_cachedir_non_nfs, exist_ok=True)
          assert os.path.isdir(nvidia_cachedir)
          return

      if devid in _NFS_CACHE:
        raise RuntimeError("NVIDIA cache dir must be on non-nfs mountpoint, "
                           "Please remove ~/.nv and retry.")
      elif os.path.exists(nvidia_cachedir):
        return

      os.makedirs(nvidia_cachedir_non_nfs, exist_ok=True)
      os.symlink(nvidia_cachedir_non_nfs, nvidia_cachedir)
