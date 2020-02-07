#include <torch/script.h> // One-stop header.
#include <unsupported/Eigen/FFT>

#include <iostream>
#include <memory>

using namespace std;
#define NUM_THREADS 3

torch::jit::script::Module gmodule;

void *hello(void *threadid) {
  long tid;
  tid = (long)threadid;
  cout << "Hello World! Thread ID, " << tid << endl;
  return nullptr;
}

void *forward(void *threadid) {

  // Create a vector of inputs.
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(torch::rand({1, 3, 800, 600}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)));
  // inputs.push_back(torch::rand({1, 3, 800, 600}, torch::dtype(torch::kFloat32).device(torch::kCPU, 0)));

  // Execute the model and turn its output into a tensor.
  for (int i=0; i<20; ++i) {
    at::Tensor output = gmodule.forward(inputs).toTensor().cpu();
    // std::cout << output[0][0][0][0] << '\n';
    // std::cout << output.sizes() << '\n';
  }

  return nullptr;
}

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }
  
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    gmodule = torch::jit::load(argv[1]);
    // gmodule = torch::jit::load("../model.ts");
    std::cout << "model " << argv[1] << " loaded\n";

  } catch (const c10::Error &e) {
    std::cerr << "error loading the model\n";
  }

  pthread_t threads[NUM_THREADS];
  int rc;
  int i;
  
  for( i = 0; i < NUM_THREADS; i++ ) {
    cout << "main() : creating thread, " << i << endl;
    rc = pthread_create(&threads[i], NULL, forward, (void *)i);
    
    if (rc) {
        cout << "Error:unable to create thread," << rc << endl;
        exit(-1);
    }
  }

  for( i = 0; i < NUM_THREADS; i++ ) {
    (void) pthread_join(threads[i], NULL);
  }
}