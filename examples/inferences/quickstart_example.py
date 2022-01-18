import logging
import os
from pathlib import Path
from furiosa import runtime
from furiosa.runtime import session
import numpy as np

LOGLEVEL = os.environ.get('FURIOSA_LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=LOGLEVEL)


def run_example():
  runtime.__full_version__

  current_path = Path(__file__).absolute().parent
  path = current_path.joinpath("../assets/quantized_models/MNISTnet_uint8_quant_without_softmax.tflite")
 
  # Load a model and compile the model
  sess = session.create(str(path))
 
  # Print the model summary
  sess.print_summary()
  
  # Print the first input tensor shape and dimensions
  input_meta = sess.inputs()[0]
  print(input_meta)
  
  # Generate the random input tensor according to the input shape
  input = np.random.randint(-128, 127, input_meta.shape(), dtype=np.int8)
  
  # Run the inference
  outputs = sess.run(input)
  
  print("== Output ==")
  print(outputs)
  print(outputs[0].numpy())


if __name__ == "__main__":
  run_example()
