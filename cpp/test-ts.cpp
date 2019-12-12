#include <torch/script.h> // One-stop header.
#include <unsupported/Eigen/FFT>

#include <iostream>
#include <memory>

using namespace std;

namespace Array {
  typedef Eigen::ArrayXXf array_xxf;
}

Array::array_xxf rebin(const Array::array_xxf &in, const unsigned int k) {
  Array::array_xxf out = Array::array_xxf::Zero(in.rows()/k, in.cols());
  for(unsigned int i=0; i<in.rows(); ++i) {
    out.row(i/k) = out.row(i/k) + in.row(i);
  }
  return out/k;
}

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }


  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }


  const int nwire = 3;
  const int ntick = 2;

  Eigen::ArrayXXf a(2,3);
  Eigen::ArrayXXf b(2,3);
  a << 1,2,3,
       4,5,6;
  b << 0.1,0.2,0.3,
       0.4,0.5,0.6;

  // Eigen::ArrayXXf a = Eigen::ArrayXXf::Random(ntick, nwire);
  // Eigen::ArrayXXf b = Eigen::ArrayXXf::Constant(ntick, nwire, 0.1);

  torch::Tensor ta = torch::from_blob(a.data(), {nwire,ntick});
  torch::Tensor tb = torch::from_blob(b.data(), {nwire,ntick});

  auto img = torch::stack({ta, tb, ta}, 0);
  auto batch = torch::stack({img}, 0);

  // cout << batch[0][0] << endl;
  // Eigen::Map<Eigen::ArrayXXf> E(batch[0][0].data_ptr<float>(), batch.size(3), batch.size(2));
  // std::cout << "EigenMat:\n" << E << std::endl;

  cout << a << endl;
  cout << rebin(a,2) << endl;
  cout << b << endl;
  // cout << ta << endl;
  // cout << tb << endl;
  // cout << img << endl;
  // cout << batch << endl;

  return 0;

  // Create a vector of inputs.
  std::vector<torch::jit::IValue> inputs;
  // inputs.push_back(torch::rand({1, 3, 800, 600}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)));
  inputs.push_back(batch.cuda());

  // Execute the model and turn its output into a tensor.
  at::Tensor output = module.forward(inputs).toTensor().cpu();
  std::cout << output[0][0][0][0] << '\n';
  std::cout << output[0][0][0][1] << '\n';
  std::cout << output[0][0][0][2] << '\n';
  std::cout << output[0][0][0][3] << '\n';
  std::cout << output[0][0][0][4] << '\n';
  std::cout << output.sizes() << '\n';

  Eigen::Map<Eigen::ArrayXXf> out_e(output[0][0].data_ptr<float>(), output.size(3), output.size(2));
  std::cout << "EigenMat:\n" << out_e.cols() << ", " << out_e.rows() << std::endl;
  std::cout << out_e(0,0) << std::endl;
  std::cout << out_e(1,0) << std::endl;
  std::cout << out_e(2,0) << std::endl;
  std::cout << out_e(3,0) << std::endl;
  std::cout << out_e(4,0) << std::endl;

  std::cout << "ok\n";

  return 0;
}