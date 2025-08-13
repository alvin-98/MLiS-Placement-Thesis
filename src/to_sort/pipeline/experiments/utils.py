import time
import functools

def timing_decorator(func):
    """
    A decorator that prints the execution time of the function it wraps.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # Start the clock
        result = func(*args, **kwargs)    # Call the original function
        end_time = time.perf_counter()    # Stop the clock
        
        execution_time = end_time - start_time
        print(f"Execution time for '{func.__name__}': {execution_time:.4f} seconds")
        
        return result
    return wrapper