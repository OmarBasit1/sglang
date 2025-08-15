from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict
import threading
import atexit
import logging
from typing import List
import pandas as pd
from dataclasses import asdict
logger = logging.getLogger(__name__)

class CSVLogger():
    """
    Logs to CSV. Writes are incremental to avoid blocking.
    """

    def __init__(self,
                 filename: str,
                 disable_periodic_persist_to_disk: bool = False,
                 persist_to_disk_every: int = 50) -> None:
        self.filename = Path(filename)
        self.disable_periodic_persist_to_disk = disable_periodic_persist_to_disk
        self.persist_to_disk_every = persist_to_disk_every

        self.filename.parent.mkdir(parents=True, exist_ok=True)
        if self.filename.exists():
            self.filename.unlink()
        self.iter = 0
        self.csv_buf: list[Dict] = []
        self.buf_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1)

        atexit.register(self._flush_remaining_sync)

    def increment_counter_and_maybe_persist_to_disk(self):
        self.iter += 1
        if not self.disable_periodic_persist_to_disk and \
           self.iter % self.persist_to_disk_every == 0:
            self.persist_to_disk()

    def persist_to_disk(self):
        with self.buf_lock:
            data_to_write = self.csv_buf.copy()
            self.csv_buf.clear()

        if not data_to_write:
            return

        self.executor.submit(self._write_to_csv, data_to_write)

    def _write_to_csv(self, data: List[Dict]):
        file_exists = self.filename.exists()
        pd.DataFrame(data).to_csv(self.filename,
                                  mode='a',
                                  header=not file_exists,
                                  index=False)
        logger.info("CSVLogger persisted %d entries to disk", len(data))

    def log(self, stats) -> None:
        with self.buf_lock:
            self.csv_buf.append(asdict(stats))

    def _flush_remaining_sync(self):
        with self.buf_lock:
            data_to_write = self.csv_buf.copy()
            self.csv_buf.clear()

        if data_to_write:
            self._write_to_csv(data_to_write)
        self.executor.shutdown(wait=True)

class PrefillStats:
    def __init__(self):
        # time at which log was created
        self.now = 0
        # batch size of previous batch
        self.num_running_sys_last_batch = 0
        # batch size of batch to be scheduled
        self.num_running_sys = 0
        # num reqs left in wait queue
        self.num_waiting_sys = 0
        # prefill budget used in batch to be scheduled, can be max chunked-prefill-size
        self.num_prefilled_tokens = 0
        # request IDs that will be in the batch to be scheduled
        self.req_ids_iter = []
        # precomputed tokens for each request in the batch to be scheduled
        self.req_precomputed_tokens_iter = []
        # input tokens +1 (generated during prefill), for each request
        self.req_total_prefilled_tokens = []
        # time entered the queue
        self.req_queue_start_time = []
        # time last batch finished
        self.last_batch_finished_time = []

class PrefillCSVLogger(CSVLogger):
    """
    Each row is the engine metrics at time of logging. Each log() adds one row.
    """

    # For now, we log all tokens of primitive types (int, float).
    PREFILL_FIELDS = [
        'now',
        'num_running_sys_last_batch',
        'num_running_sys',
        'num_waiting_sys',
        'num_prefilled_tokens',
        'req_ids_iter',
        'req_precomputed_tokens_iter',
        'req_total_prefilled_tokens',
        'req_queue_start_time',
        'last_batch_finished_time'
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Cumulative number of completed requests
        self.num_completed_reqs = 0

    def log_prefill(self, stats) -> None:
        perf_dict = {field: getattr(stats, field) for field in self.PREFILL_FIELDS}

        self.csv_buf.append(perf_dict)
        self.increment_counter_and_maybe_persist_to_disk()