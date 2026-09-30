#include <c10/core/SymInt.h>
#include <torch/library.h>
#include <tuple>
#include <utility>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/from_blob.h>
#include <ATen/ops/scalar_tensor.h>
#include <ATen/ops/zeros.h>
#endif

#include <ATen/xpu/XPUGeneratorImpl.h>

namespace torch::inductor {
using namespace at;

// XPU counterpart of inductor_reserve_rng_state in inductor_ops_gpu.cpp.
//
// Reserves `increment` Philox samples on an XPU generator, exactly as the
// eager XPU random kernels do via XPUGeneratorImpl::philox_xpu_state, so that
// Inductor's eager-aligned random kernels (config.align_random_eager) consume
// the same stream and advance the generator by the same amount.
//
// -param gen The XPU generator to use.
// -param increment The number of RNG values to reserve.
// -return A tuple of (Seed Tensor, Offset Tensor, Intragraph Offset CPU
// Tensor).
static std::tuple<Tensor, Tensor, Tensor> inductor_reserve_rng_state_xpu_impl(
    const Generator& generator,
    c10::SymInt increment) {
  auto* gen_impl = at::check_generator<at::XPUGeneratorImpl>(generator);

  const auto dev_opts =
      at::TensorOptions().dtype(at::kLong).device(generator.device());
  const auto cpu_opts = at::TensorOptions().dtype(at::kLong).device(at::kCPU);

  int64_t inc = increment.expect_int();
  at::PhiloxXpuState st;
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen_impl->mutex_);
    st = gen_impl->philox_xpu_state(static_cast<uint64_t>(inc));
  }

  if (st.captured_) {
    auto seed_t = at::from_blob(
        static_cast<void*>(st.seed_.ptr), {1}, [](void*) {}, dev_opts);
    auto off_t = at::from_blob(
        static_cast<void*>(st.offset_.ptr), {1}, [](void*) {}, dev_opts);
    auto intra_t =
        at::scalar_tensor(static_cast<int64_t>(st.offset_intragraph_), cpu_opts)
            .unsqueeze(0);
    return {std::move(seed_t), std::move(off_t), std::move(intra_t)};
  }

  auto seed_t = at::scalar_tensor(static_cast<int64_t>(st.seed_.val), dev_opts)
                    .unsqueeze(0);
  auto off_t = at::scalar_tensor(static_cast<int64_t>(st.offset_.val), dev_opts)
                   .unsqueeze(0);
  auto intra_t = at::zeros({1}, cpu_opts);
  return {std::move(seed_t), std::move(off_t), std::move(intra_t)};
}

TORCH_LIBRARY_IMPL(inductor_prims, XPU, m) {
  m.impl(
      "inductor_reserve_rng_state",
      TORCH_FN(inductor_reserve_rng_state_xpu_impl));
}

} // namespace torch::inductor
