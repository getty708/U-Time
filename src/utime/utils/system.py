"""

References:
- https://github.com/perslev/MultiPlanarUNet/blob/master/mpunet/utils/system.py
- https://github.com/perslev/MultiPlanarUNet/blob/master/mpunet/utils/utils.py

"""
from multiprocessing import Process, Event, Queue
import time
import os
import re

import glob
import contextlib
import numpy as np

from utime.logging import ScreenLogger


def _get_system_wide_set_gpus():
    allowed_gpus = os.environ.get("CUDA_VISIBLE_DEVICES")
    if allowed_gpus:
        allowed_gpus = allowed_gpus.replace(" ", "").split(",")
    return allowed_gpus


def get_free_gpus(max_allowed_mem_usage=400):
    # Check if allowed GPUs are set in CUDA_VIS_DEV.
    allowed_gpus = _get_system_wide_set_gpus()
    if allowed_gpus:
        print("[OBS] Considering only system-wise allowed GPUs: {} (set in"
              " CUDA_VISIBLE_DEVICES env variable).".format(allowed_gpus))
        return allowed_gpus
    # Else, check GPUs on the system and assume all non-used (mem. use less
    # than max_allowed_mem_usage) is fair game.
    from subprocess import check_output
    try:
        # Get list of GPUs
        gpu_list = check_output(["nvidia-smi", "-L"], universal_newlines=True)
        gpu_ids = np.array(re.findall(r"GPU[ ]+(\d+)", gpu_list), dtype=np.int)

        # Query memory usage stats from nvidia-smi
        output = check_output(["nvidia-smi", "-q", "-d", "MEMORY"],
                              universal_newlines=True)

        # Fetch the memory usage of each GPU
        mem_usage = re.findall(r"FB Memory Usage.*?Used[ ]+:[ ]+(\d+)",
                               output, flags=re.DOTALL)
        assert len(gpu_ids) == len(mem_usage)

        # Return all GPU ids for which the memory usage is exactly 0
        free = list(map(lambda x: int(x) <= max_allowed_mem_usage, mem_usage))
        return list(gpu_ids[free])
    except FileNotFoundError as e:
        raise FileNotFoundError("[ERROR] nvidia-smi is not installed. "
                                "Consider setting the --num_GPUs=0 flag.") from e


def _get_free_gpu(free_GPUs, N=1):
    try:
        free_gpu = ",".join(map(str, free_GPUs[0:N]))
    except IndexError as e:
        raise OSError("No GPU available.") from e
    return free_gpu


def get_free_gpu(N=1):
    free = get_free_gpus()
    return _get_free_gpu(free, N=N)


def await_and_set_free_gpu(N=1, sleep_seconds=60, logger=None):
    gpu = ""
    if N != 0:
        from time import sleep
        logger = logger or print
        logger("Waiting for free GPU.")
        found_gpu = False
        while not found_gpu:
            gpu = get_free_gpu(N=N)
            if gpu:
                logger("Found free GPU: %s" % gpu)
                found_gpu = True
            else:
                logger("No available GPUs... Sleeping %i seconds." % sleep_seconds)
                sleep(sleep_seconds)
    set_gpu(gpu)


def set_gpu(gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)



class GPUMonitor(Process):
    def __init__(self, logger=None):
        self.logger = logger or ScreenLogger()

        # Prepare signal
        self.stop_signal = Event()
        self.run_signal = Event()
        self.set_signal = Event()

        # Stores list of available GPUs
        self._free_GPUs = Queue()

        super(GPUMonitor, self).__init__(target=self._monitor)
        self.start()

    def stop(self):
        self.stop_signal.set()

    def _monitor(self):
        while not self.stop_signal.is_set():
            if self.run_signal.is_set():
                # Empty queue, just in case...?
                self._free_GPUs.empty()

                # Get free GPUs
                free = get_free_gpus()

                # Add number of elements that will be put in the queue as first
                # element to be read from main process
                self._free_GPUs.put(len(free))

                # Add available GPUs to queue
                for g in free:
                    self._free_GPUs.put(g)

                # Set the signal that main process can start reading queue
                self.set_signal.set()

                # Stop run signal for this process
                self.run_signal.clear()
            else:
                time.sleep(0.5)
                self.set_signal.clear()

    @property
    def free_GPUs(self):
        self.run_signal.set()
        while not self.set_signal.is_set():
            time.sleep(0.2)

        free = []
        for i in range(self._free_GPUs.get()):
            free.append(self._free_GPUs.get())
        return free

    def get_free_GPUs(self, N):
        return _get_free_gpu(self.free_GPUs, N)

    def await_and_set_free_GPU(self, N=0, sleep_seconds=60, stop_after=False):
        cuda_visible_dev = ""
        if N != 0:
            self.logger("Waiting for free GPU.")
            found_gpu = False
            while not found_gpu:
                cuda_visible_dev = self.get_free_GPUs(N=N)
                if cuda_visible_dev:
                    self.logger("Found free GPU: %s" % cuda_visible_dev)
                    found_gpu = True
                else:
                    self.logger("No available GPUs... Sleeping %i seconds." % sleep_seconds)
                    time.sleep(sleep_seconds)
        else:
            self.logger("Using CPU based computations only!")
        self.set_GPUs = cuda_visible_dev
        if stop_after:
            self.stop()

    @property
    def num_currently_visible(self):
        return len(self.set_GPUs.strip().split(","))

    @property
    def set_GPUs(self):
        try:
            return os.environ["CUDA_VISIBLE_DEVICES"]
        except KeyError:
            return ""

    @set_GPUs.setter
    def set_GPUs(self, GPUs):
        set_gpu(GPUs)

    def set_and_stop(self, GPUs):
        self.set_GPUs = GPUs
        self.stop()
