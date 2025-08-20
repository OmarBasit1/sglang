from typing import List

from vidur.entities.batch import Batch
from vidur.events import BaseEvent
from vidur.events.batch_stage_arrival_event import BatchStageArrivalEvent
from vidur.metrics import MetricsStore
from vidur.scheduler import BaseGlobalScheduler
from vidur.types import EventType


class BatchStageArrivalDelayEvent(BaseEvent):
    """
    Adds a delay before scheduling each `BatchStageArrivalEvent`, which includes:
        - CPU overhead
        - An artificially injected delay
    """

    def __init__(self, time: float, replica_id: int, stage_id: int, batch: Batch):
        super().__init__(time, EventType.BATCH_STAGE_ARRIVAL_DELAY)

        self.replica_id = replica_id
        self.stage_id = stage_id
        self.batch = batch

    def handle_event(
        self, scheduler: BaseGlobalScheduler, metrics_store: MetricsStore
    ) -> List[BaseEvent]:
        return [
            BatchStageArrivalEvent(
                (self.time),
                self.replica_id,
                self.stage_id,
                self.batch,
            )
        ]