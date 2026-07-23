"""Utils.
"""

import time

def time_since(start_time_s: float) -> float:
    """
    Calculates the time since a given start time. Uses and expects a time.perf_counter() as an input. 
    Returns time in seconds as a float. 
    """

    if start_time_s is None:
        raise TypeError("start_time_s cannot be None type")
    return time.perf_counter() - start_time_s
