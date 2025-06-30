import time
from functools import wraps
from typing import BinaryIO
import os

# Global dictionary to track execution statistics for each function
execution_stats = {}


def timeit(func):
    """Decorator to measure execution time of a function and track cumulative statistics."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        execution_time = end_time - start_time

        # Initialize stats for this function if not exists
        if func.__name__ not in execution_stats:
            execution_stats[func.__name__] = {
                "total_time": 0.0,
                "call_count": 0,
                "min_time": float("inf"),
                "max_time": 0.0,
            }

        # Update statistics
        stats = execution_stats[func.__name__]
        stats["total_time"] += execution_time
        stats["call_count"] += 1
        stats["min_time"] = min(stats["min_time"], execution_time)
        stats["max_time"] = max(stats["max_time"], execution_time)

        # print(f"{func.__name__} took {execution_time:.4f} seconds to execute")
        return result

    return wrapper


def print_execution_summary():
    """Print a summary of all function execution statistics."""
    print("\n" + "=" * 60)
    print("EXECUTION SUMMARY")
    print("=" * 60)

    for func_name, stats in execution_stats.items():
        avg_time = stats["total_time"] / stats["call_count"] if stats["call_count"] > 0 else 0
        print(f"{func_name}:")
        print(f"  Total calls: {stats['call_count']}")
        print(f"  Total time: {stats['total_time']:.4f}s")
        print(f"  Average time: {avg_time:.4f}s")
        print(f"  Min time: {stats['min_time']:.4f}s")
        print(f"  Max time: {stats['max_time']:.4f}s")
        print()


def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def init_vocabulary():
    vocab = {i: bytes([i]) for i in range(256)}
    return vocab, set(vocab.values())
