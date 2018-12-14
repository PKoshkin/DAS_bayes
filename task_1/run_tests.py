from test import run_tests
import os
import psutil

if __name__ == "__main__":
    run_tests()
    process = psutil.Process(os.getpid())
    print(float(process.memory_info().rss) / 1024 / 1024)  # in Mbytes 
