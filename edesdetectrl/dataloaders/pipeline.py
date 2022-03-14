import time
import streamz
import numpy as np

if __name__ == "__main__":

    def do_hard_work(s):
        start = time.time()
        while time.time() < start + s:
            pass

    def load_video(data):
        filename, time_started = data
        for _ in range(10):
            time.sleep(0.1)
            do_hard_work(0.1)
        #print(f"{filename} loaded")
        return filename, np.array([0, 1, 2, 3, 4, 5]), time_started

    def further_process(data):
        filename, video, time_started = data
        do_hard_work(1)
        #print(f"{filename} processed")
        return filename, video + 1, time_started

    finished_files = {}

    def register_finished(data):
        global finished_files
        now = time.time()
        filename, video, time_started = data
        elapsed = now - time_started
        print(f"{filename}: {elapsed:.2f}")
        #finished_files[filename] = (elapsed, now)

    pipeline_source = streamz.Stream()
    pipeline_source.map(load_video).map(further_process).sink(register_finished)

    from distributed import Client

    client = Client()
    pipeline_source = streamz.Stream()
    (
        pipeline_source.scatter()  # Dask stream
        .map(load_video)
        .map(further_process)
        .buffer(2)
        .gather()
        .sink(register_finished)
    )

    time_started = time.time()
    for filename in [f"file_{i}" for i in range(10)]:
        pipeline_source.emit((filename, time_started))

    

    #print(
    #    f"{np.mean([v[0] for k, v in finished_files.items()]):.2f} mean processing time."
    #)


