# C API of NPU Engine
Nux is the C API library of the NPU Engine.
Nux provides two methods to run inference tasks through the NPU.
The following demonstrates the examples of two execution ways.
If you want to see more details about API types and functions, please refer to \ref nux.h.

### Sync Model
The sync model provides APIs to run a single inference task with a single input batch at a time.

#### Example of Sync Model
```c
// Get the size of ENF file
const char *enf_path = "<a path of ENF file>";
struct stat st;
stat(enf_path, &st);
size_t enf_size = st.st_size;

// Read the ENF binary from the ENF file
void *enf_buf = malloc(enf_size);
FILE *enf_file = fopen(enf_path, "rb");
size_t read_item = fread(enf_buf, enf_size, 1, enf_file);
assert(read_item == 1);
fclose(enf_file);

// Initialize nux handle
nux_handle_t nux = NULL;
assert(create_nux(&nux) == nux_error_t_success);

// Create a sync model
nux_sync_model_t sync_model = NULL;
assert(nux_create_sync_model(nux, binary_buf, enf_size, &sync_model) == nux_error_t_success);

// Get the 0th input tensor from this model, and 
// A model can have more than one input tensor depending on its model design.
// You can specify an index of input tensor to get your desired input tensor. 
nux_tensor_t input_tensor = NULL;
assert(model_input_tensor(sync_model, 0, &input_tensor) == nux_error_t_success);

// Fill the buffer of the input tensor with your data
// Here, the input shape of the 0th input tensor is 1x28x28x1 in NxHxWxC, 
// and it has totally 784 elements. In this example, the buffer is filled with a fake data. 
uint8_t input_buf [784] = {[0 ... 783] = 1};
tensor_set_buffer(input_tensor, input_buf, 784);

// Get the 0th output tensor from the model
nux_tensor_t output_tensor = NULL;
assert(model_output_tensor(sync_model, 0, &output_tensor) == nux_error_t_success);

// Run an inference task with the input data filled as above.
assert(model_run(sync_model) == nux_error_t_success);

// Get the buffer of the 0th's output tensor.
// output_buf will receive a pointer to the output tensor's buffer.
// output_len will receive the length of the output buffer.
uint8_t *output_buf = NULL;
uintptr_t output_len = NULL;
tensor_get_buffer(output_tensor, &output_buf, &output_len);

// Release resources
destroy_sync_model(sync_model);
destroy_nux(nux);
```

## Task Model
The task model provides APIs to run multiple inference tasks asynchronously and simultaneously.
The callback functions defined by users will be called when each task is completed.
When a user creates a task model, the user can configure the concurrency of running tasks 
depending on HW capacity.
 
### Example of Task Model
```c
// Define three callback functions for a task model
void my_nux_output_cb(nux_request_id_t id,
                      nux_output_index_t out_id,
                      nux_buffer_t buf,
                      nux_buffer_len_t buf_len) {
    // fill your logic
}
void my_error_cb(nux_request_id_t id, nux_error_t err) {
    // fill your logic
}
void my_finish_cb(nux_request_id_t id) {
    // fill your logic
}


void run() {
    // Get the size of ENF file
    const char *enf_path = "<a path of ENF file>";
    struct stat st;
    stat(enf_path, &st);
    size_t enf_size = st.st_size;
    
    // Read the ENF binary from the ENF file
    void *enf_buf = malloc(enf_size);
    FILE *enf_file = fopen(enf_path, "rb");
    size_t read_item = fread(enf_buf, enf_size, 1, enf_file);
    assert(read_item == 1);
    fclose(enf_file);
    
    // Create a nux handle
    nux_handle_t nux = NULL;
    assert(create_nux(&nux) == nux_error_t_success);
    
    // Create a task model with 3 max concurrency
    nux_task_model_t task_model = NULL;
    const uint32_t max_concurrency = 3;
    assert(nux_create_task_model(nux, enf_buf, enf_size, max_concurrency,
           my_nux_output_cb, my_error_cb, my_finish_cb, &task_model) == nux_error_t_success);
    
    // Get a single task which allows your code to request a single asynchronous inference task.
    // Depending on the concurrency, you can get multiple tasks at the same time. 
    // In this case, you can request multiple tasks simultaneously.
    // If there's no available task slot, calling task_model_get_task will be blocked 
    // until new available task comes out.
    nux_task_t task1 = NULL;
    assert(task_model_get_task(task_model, &task1) == nux_error_t_success);
    
    // Fill the buffer of the 0th input tensor with your data.
    // Here, the input shape of the 0th input tensor is 1x28x28x1 in NxHxWxC, 
    // and it has totally 784 elements.
    const uintptr_t INPUT_SIZE = 784;
    assert(task_input_size(task1, 0) == INPUT_SIZE);
    // It's a fake data. Please fill this buffer with yours.
    uint8_t input_buf [784] = {[0 ... 783] = 1};
    // Get a mutable buffer of the 0th input tensor from task1
    uint8_t* buf = task_input(task1, 0);
    // Copy your data to the buffer
    memcpy(buf, input_buf, INPUT_SIZE);
    
    // Submit the task1 with an arbitrary req_id, 
    // which will be passed to callback functions to identify some context.
    RequestId req_id = 1;
    task_execute(task1, req_id);
    
    // Wait for until all submitted tasks are completed
    while(!task_model_is_all_task_done(task_model)) {
        sleep(1);
    }
    
    // Release resources
    destroy_task_model(task_model);
    destroy_nux(nux);
    free(enf_buf);
}
```

## How to build source codes with Nux
To build your source code with Nux, please follow:
```sh
export NPU_TOOLS_DIR="<npu-tools src dir>"
export NUX_DIR=${NPU_TOOLS_DIR}/crates/nux
cargo build --release --target x86_64-unknown-linux-gnu
cc sync_example.c -I${NUX_DIR}/include -L${NPU_TOOLS_DIR}/target/x86_64-unknown-linux-gnu/release -lnux
```

## How to generate Doxygen
The following commands will generate a Doxygen document at `$NUX_TOOLS_DIR/target/doxygen/html`.
```
cd <nux src root>
doxygen crates/nux/Doxyfile
```
