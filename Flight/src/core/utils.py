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

def time_since_ns(start_time_ns: int) -> int:
    """
    Calculates the time since a given start time. Uses and expects a time.perf_counter_ns() as an input. 
    Returns time in nanoseconds as a float. 
    """

    if start_time_ns is None:
        raise TypeError("start_time_ns cannot be None type")
    
    return time.perf_counter_ns() - start_time_ns
