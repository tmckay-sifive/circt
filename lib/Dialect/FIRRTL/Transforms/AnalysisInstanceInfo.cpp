//===- AnalysisInstanceInfo.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception//
//
//===----------------------------------------------------------------------===//
//
// Run FIRRTL's InstanceInfo analysis and cache it in MLIR's infrastructure.
// The effect of running this pass is that passes nested under this will have
// the cached analysis available.
//
//===----------------------------------------------------------------------===//

#include "circt/Analysis/FIRRTLInstanceInfo.h"
#include "circt/Dialect/FIRRTL/Passes.h"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_ANALYSISINSTANCEINFO
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

class AnalysisInstanceInfo
    : public circt::firrtl::impl::AnalysisInstanceInfoBase<
          AnalysisInstanceInfo> {

  void runOnOperation() override {
    getAnalysis<InstanceInfo>();
    return markAllAnalysesPreserved();
  }
};

//===----------------------------------------------------------------------===//
// Pass Creation
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::Pass> circt::firrtl::createAnalysisInstanceInfo() {
  auto pass = std::make_unique<AnalysisInstanceInfo>();
  return pass;
}
