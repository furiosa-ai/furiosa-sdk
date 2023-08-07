import numpy as np
import random

from furiosa.runtime import session

submitter, queue = session.create_async("mnist.onnx",
                                        worker_num=2,
                                        # Determine how many asynchronous requests you can submit
                                        # without blocking.
                                        input_queue_size=100,
                                        output_queue_size=100)

for i in range(0, 5):
    idx = random.randint(0, 59999)
    input = np.random.rand(1, 1, 28, 28).astype(np.float32)
    submitter.submit(input, context=idx) # non blocking call

for i in range(0, 5):
    context, outputs = queue.recv(100) # 100 ms for timeout. If None, queue.recv() will be blocking.
    print(outputs[0].numpy())

if queue:
    queue.close()
if submitter:
    submitter.close()