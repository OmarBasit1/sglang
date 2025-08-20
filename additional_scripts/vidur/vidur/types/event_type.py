from vidur.types.base_int_enum import BaseIntEnum


class EventType(BaseIntEnum):
    # at any given time step, call the schedule event at the last
    # to ensure that all the requests are processed
    BATCH_STAGE_ARRIVAL_DELAY = 1
    BATCH_STAGE_ARRIVAL = 2
    REQUEST_ARRIVAL = 3
    BATCH_STAGE_END = 4
    BATCH_END = 5
    GLOBAL_SCHEDULE = 6
    REPLICA_SCHEDULE = 7
    REPLICA_STAGE_SCHEDULE = 8
