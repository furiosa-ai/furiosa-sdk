#!/usr/bin/env python3

"""Image classification example"""
import logging
import os
import sys
from pathlib import Path

import numpy as np
from furiosa.runtime.tensor import DataType

LOGLEVEL = os.environ.get('FURIOSA_LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=LOGLEVEL)


def random_input_inference(model_path, num_inf):
    from furiosa.runtime import session

    print(f"Loading and compiling the model {model_path}")
    with session.create(str(model_path)) as sess:
        print(f"Model has been compiled successfully")

        print("Model input and output:")
        print(sess.print_summary())

        for idx in range(num_inf):
            print(f'Iteration {idx}...')

            inputs = []
            for session_input in sess.inputs():
                if session_input.dtype() == DataType.UINT8:
                    inputs.append(np.random.randint(0, 255, session_input.shape(), dtype=np.uint8))
                elif session_input.dtype() == DataType.INT8:
                    inputs.append(np.random.randint(-128, 127, session_input.shape(), dtype=np.int8))
                elif session_input.dtype() == DataType.FLOAT32:
                    inputs.append(np.random.random(session_input.shape()).astype(np.float32))
                else:
                    raise Exception(f"Unsupported DataType({session_input.dtype()}) of input.")
            sess.run(inputs)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.stderr.write("./random_input_inference.py <modee_path> <num_inferences>\n")
        sys.exit(-1)

    model_path = Path(sys.argv[1])
    if not model_path.exists():
        sys.stderr.write(f"{model_path} not found")
        sys.exit(-1)

    if len(sys.argv) >= 3:
        num_inf = int(sys.argv[2])
    else:
        num_inf = 5

    random_input_inference(model_path, num_inf=num_inf)
