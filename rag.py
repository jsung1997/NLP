import psutil

def get_memory_usage_mb():
    """
    Returns the memory usage of the current process in megabytes.
    """
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / (1024 * 1024)  # Convert bytes to megabytes

import time
import faiss
from faiss.contrib.datasets import DatasetSIFT1M

ds = DatasetSIFT1M()

xq = ds.get_queries()
xb = ds.get_database()
gt = ds.get_groundtruth()

k=1
d = xq.shape[1]
nq = 1000
xq = xq[:nq]

for i in range(1, 10, 2):
    start_memory = get_memory_usage_mb()
    start_indexing = time.time()  # <-- fix here
    index = faiss.IndexFlatL2(d)
    index.add(xb[:(i+1)*100000])
    end_indexing = time.time()
    end_memory = get_memory_usage_mb()

    t0 = time.time()
    D, I = index.search(xq, k)
    t1 = time.time()
    print(f"Data count {(i+1)*100000}: ")
    print(f"Index: {(end_indexing - start_indexing)*1000:.3f} ms "
          f"({end_memory - start_memory:.3f} MB)\n"
          f"Search: {(t1 - t0)*1000/nq :.3f} ms")

