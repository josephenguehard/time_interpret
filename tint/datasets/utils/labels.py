import numpy as np
import torch as th

from typing import List, Optional

EPS = 1e-3


def create_labels(
    events: List[List[str]],
    event_times: List[List[float]],
    labels: list,
    time_to_task: Optional[float] = 24 * 365.25,
    std_time_to_task: Optional[float] = 0.0,
    maximum_time: Optional[float] = 24 * 365.25 * 120,
    seed: Optional[int] = 0,
):
    """
    Utility function to create code classification task.
    Args:
        events (list): List of events.
        event_times (list): List of events times.
        labels (list): Which labels to predict.
        time_to_task (float): When to stop recording before an event, for event
            classification. Default to a year: ``24 * 365.25``
        std_time_to_task (float): Add randomness into when to stop recording
            before the event. Default to 0.
        maximum_time (float): Maximum time to record. Default to 120 years:
            ``24 * 365.25 * 120``
        seed (int): Seed to control randomness. Default to 0
    Returns:
        - **end_of_events** (*Tensor*) -- When to stop recording.
        - **labels** (*Tensor*) -- If a patient got the disease.
    """
    # Set random seed
    np.random.seed(seed)

    # Create labels (1. if event happens, 0. otherwise)
    # Create times of event randomly between 0 and maximum time if the event
    # does not happen
    times = list()
    labels_ = list()
    for e, t in zip(events, event_times):
        a = list()
        b = list()
        for e_, t_ in zip(e, t):
            if len(set(labels).intersection(set(e_))) > 0:
                a.append(t_)
                b.append(1)
                break
        a += [np.random.uniform(high=maximum_time)]
        times.append(a)
        b.append(0)
        labels_.append(b)
    labels_ = th.Tensor([x[0] for x in labels_]).type(th.float32).unsqueeze(-1)
    times = th.Tensor([x[0] for x in times]).type(th.float32)

    # Remove some of the record
    # If std_time_to_task is positive, sample time_to_task around its average
    # value, and clamp it to be strictly positive
    if std_time_to_task > 0:
        time_to_task = th.clamp(
            th.from_numpy(
                np.random.normal(
                    loc=time_to_task, scale=std_time_to_task, size=len(times)
                )
            ),
            min=EPS,
        )
    end_of_records = th.clamp(times - time_to_task, min=0.0, max=maximum_time)

    return labels_, end_of_records
