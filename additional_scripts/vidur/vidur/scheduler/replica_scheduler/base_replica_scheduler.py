from abc import ABC, abstractmethod
from collections import deque
from typing import List
import numpy as np
from typing import Optional

from vidur.config import (
    BaseReplicaSchedulerConfig,
    BaseRequestGeneratorConfig,
    ReplicaConfig,
)
from vidur.entities import Batch, Replica, Request
from vidur.execution_time_predictor import BaseExecutionTimePredictor
from vidur.logger import init_logger
from vidur.scheduler.replica_stage_scheduler import ReplicaStageScheduler
from vidur.scheduler.utils.memory_planner import MemoryPlanner

logger = init_logger(__name__)


class BaseReplicaScheduler(ABC):
    def __init__(
        self,
        replica_config: ReplicaConfig,
        replica_scheduler_config: BaseReplicaSchedulerConfig,
        request_generator_config: BaseRequestGeneratorConfig,
        replica: Replica,
        num_stages: int,
        execution_time_predictor: BaseExecutionTimePredictor,
    ) -> None:
        self._config = replica_scheduler_config
        self._replica_config = replica_config
        self._request_generator_config = request_generator_config
        self._replica_id = replica.id
        self._num_stages = num_stages

        self._max_blocks_per_sequence = (
            self._request_generator_config.max_tokens // self._config.block_size
        )

        memory_planner = MemoryPlanner(self._replica_config, replica)

        if not self._config.num_blocks:
            self._config.num_blocks = (
                self._max_blocks_per_sequence * memory_planner.get_max_request_slots()
            )
        self._max_batch_size = min(
            memory_planner.get_max_batch_size(),
            self._config.batch_size_cap,
        )

        logger.debug(
            f"Obtained max batch size of {self._max_batch_size} for replica {self._replica_id}"
        )

        self._request_queue = []
        self._num_allocated_blocks = 0
        self._allocation_map = {}

        self._replica_stage_schedulers = {
            stage_id: ReplicaStageScheduler(
                replica.id,
                stage_id,
                stage_id == num_stages - 1,
                execution_time_predictor,
            )
            for stage_id in range(num_stages)
        }

        self.outstanding_batches: deque[Batch] = deque()
        self._scheduler_is_running = False
        self._scheduler_last_stopped = 0.0
        self._last_returned_batch: Optional[Batch] = None

    @property
    def num_pending_requests(self) -> int:
        return len(self._request_queue)

    @property
    def replica_id(self) -> int:
        return self._replica_id

    @property
    def num_allocated_blocks(self) -> int:
        return self._num_allocated_blocks

    @property
    def memory_usage_percent(self) -> int:
        return (self._num_allocated_blocks * 100) / self._config.num_blocks

    def is_empty(self) -> bool:
        return (
            self.num_pending_requests == 0
            and len(self._allocation_map) == 0
            and all(
                stage_scheduler.is_empty()
                for stage_scheduler in self._replica_stage_schedulers.values()
            )
        )

    def _get_request_next_num_tokens(self, request: Request) -> int:
        assert not request.completed

        if request.is_prefill_complete:
            return 1

        return request.num_prefill_tokens

    def add_request(self, request: Request) -> None:
        self._request_queue.append(request)

    def get_replica_stage_scheduler(self, stage_id: int):
        return self._replica_stage_schedulers[stage_id]

    def can_allocate(self, num_blocks: int) -> bool:
        return self._config.num_blocks - self._num_allocated_blocks >= num_blocks

    def allocate(self, request_id: int, num_blocks: int) -> None:
        self._num_allocated_blocks += num_blocks
        if request_id not in self._allocation_map:
            self._allocation_map[request_id] = num_blocks
        else:
            self._allocation_map[request_id] += num_blocks

        assert self._num_allocated_blocks <= self._config.num_blocks

    def free(self, *request_ids: List[int]) -> None:
        for request_id in request_ids:
            num_blocks = self._allocation_map.pop(request_id)
            self._num_allocated_blocks -= num_blocks

        assert self._num_allocated_blocks >= 0

    def free_batch(self, batch: Batch) -> None:
        self.free(*batch.request_ids)

    @abstractmethod
    def on_batch_end(self, batch: Batch) -> None:
        pass

    @abstractmethod
    def _get_next_batch(self) -> Batch:
        pass

    def on_schedule(self, current_time) -> List[Batch]:
        if self._num_running_batches == self._num_stages:
            # Scheduler is excessively triggered by new request arrival
            assert self._scheduler_is_running
            return []

        self._scheduler_is_running = True

        # Record num running requests before actually performing the
        # scheduling, otherwise requests will be taken out of the running queue
        if hasattr(self, 'num_running_requests'):
            running_queue_len = self.num_running_requests
        else:
            running_queue_len = 0

        # If true, it means there's nothing to schedule, in which case the
        # scheduler will stop running until the next request arrives
        scheduler_will_stop = True

        scheduled_batches: List[Batch] = []
        while self._num_running_batches < self._num_stages:
            scheduler_has_ticked = True
            batch = self._get_next_batch()
            if not batch:
                break
            scheduler_will_stop = False
            scheduled_batches.append(batch)
            self.outstanding_batches.append(batch)
            self._num_running_batches += 1

        # If scheduler_will_stop, meaning the scheduled batch is empty, don't
        # update last batch stats, so that the stats can be picked up on the
        # first iteration the scheduler resumes running
        if scheduler_will_stop:
            self._scheduler_is_running = False
            self._scheduler_last_stopped = current_time

        return scheduled_batches

    def get_states(self, current_time):
        """
        Get states of: 
            - The scheduler
            - The latest returned batch
        """
        while len(self.outstanding_batches) > 0 and self.outstanding_batches[-1].completed:
            self._last_returned_batch = self.outstanding_batches.popleft()

        # Avg power. It is weighted avg over busy + idle
        if self._last_returned_batch:
            b = self._last_returned_batch   # shorthand
            busy_duration = b.completed_at - b.inference_started_at

            # TTFT
            ttft_arr = [r.prefill_completed_at - r.arrived_at
                        for r in b.requests
                        if r.num_processed_tokens == r.num_prefill_tokens + 1]
            ttft = np.mean(ttft_arr).item() if len(ttft_arr) > 0 else 0.0
        else:
            # Will reach here before any batch returns
            busy_duration = 0.0
            ttft = 0.0

        ret = {
            'waiting_queue_len': self.num_pending_requests,
            'waiting_queue_num_tokens': sum(
                [r.num_decode_tokens for r in self._request_queue]),
            'wait_queue_requests_mean': np.mean(
                [r.num_decode_tokens for r in self._request_queue]),
            'wait_queue_requests_std': np.std(
                [r.num_decode_tokens for r in self._request_queue]),
            'wait_queue_requests_max': max(
                [r.num_decode_tokens for r in self._request_queue], default=0),
            'memory_usage_percent': self.memory_usage_percent,
            'last_batch_busy_duration': busy_duration,
            'last_batch_ttft': ttft,
            'wait_queue_num_prefill_tokens_per_req': [r.num_prefill_tokens for r in self._request_queue],
            'wait_queue_num_processed_tokens_per_req': [r.num_processed_tokens for r in self._request_queue],
            'wait_queue_waiting_time_per_req': [current_time - r.arrived_at for r in self._request_queue],
        }
        if hasattr(self, 'num_running_requests'):
            ret['running_queue_len'] = self.num_running_requests
        return ret