import json
from time import perf_counter
from datetime import timedelta
from contextlib import AbstractContextManager
from functools import wraps, partial
from collections import defaultdict

from typing import DefaultDict, List


class profiler(AbstractContextManager):
  timings: DefaultDict[str, List[timedelta]] = defaultdict(list)

  def __init__(self, name: str, printer=None, store=True, **printer_kwargs):
    self.name = name
    self.store = store
    self.delta = None

    if printer is None:
      self.printer = None
    else:
      self.printer = partial(printer, **printer_kwargs)

  @classmethod
  def clear(cls):
    cls.timings = defaultdict(list)

  @classmethod
  def _json_default(cls, obj):
    if isinstance(obj, timedelta):
      return str(obj)
    raise TypeError

  @classmethod
  def dump(cls, output_path, indent=4):
    with open(output_path, 'w') as f:
      json.dump(cls.timings, f, indent=indent, default=cls._json_default)

  @classmethod
  def profile(cls, printer=print, **printer_kwargs):
    def wrapper(func):
      prf = cls(func.__name__, printer=printer, **printer_kwargs)

      @wraps(func)
      def profiled_func(*args, **kwargs):
        with prf:
          return func(*args, **kwargs)

      return profiled_func
    return wrapper

  def __enter__(self):
    self.begin = perf_counter()
    return self

  def __exit__(self, *exc_info):
    if any(e is not None for e in exc_info):
      self.delta = None
      return

    self.delta = delta = timedelta(seconds=perf_counter() - self.begin)
    if self.store:
      self.timings[self.name].append(delta)
    if self.printer is not None:
      self.printer(f"Time for {self.name}:\t{delta}")
