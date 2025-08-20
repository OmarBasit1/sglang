from typing import List

from vidur.entities import Batch
from vidur.events import BaseEvent
from vidur.logger import init_logger
from vidur.metrics import MetricsStore
from vidur.scheduler import BaseGlobalScheduler
from vidur.types import EventType

logger = init_logger(__name__)


class BatchEndEvent(BaseEvent):
    def __init__(self, time: float, replica_id: int, batch: Batch):
        super().__init__(time, EventType.BATCH_END)

        self._replica_id = replica_id
        self._batch = batch

    def handle_event(
        self, scheduler: BaseGlobalScheduler, metrics_store: MetricsStore
    ) -> List[BaseEvent]:
        from vidur.events.replica_schedule_event import ReplicaScheduleEvent

        self._batch.on_batch_end(self.time)
        replica_scheduler = scheduler.get_replica_scheduler(self._replica_id)
        replica_scheduler.on_batch_end(self._batch)

        memory_usage_percent = replica_scheduler.memory_usage_percent
        metrics_store.on_batch_end(
            self.time, self._batch, self._replica_id, memory_usage_percent
        )

        return [ReplicaScheduleEvent(self.time, self._replica_id)]

    def to_dict(self):
        return {
            "time": self.time,
            "event_type": self.event_type,
            "batch_id": self._batch.id,
        }
    
    def to_chrome_trace(self) -> List[dict]:
        chrome_traces = []
        for req in self._batch.completed_requests:
            chrome_traces.append({
                "name": "request_end",
                "ph": "i",
                "ts": self.time * 1e6,
                "pid": 0,
                "tid": 0,
                "args": {
                    'request_id': req.id,
                    'arrived_at': req.arrived_at,
                    'prefill_completed_at': req.prefill_completed_at,
                    'completed_at': req.completed_at,
                    'num_prefill_tokens': req.num_prefill_tokens,
                    'num_decode_tokens': req.num_decode_tokens,
                }
            })
        return chrome_traces if len(chrome_traces) > 0 else None
