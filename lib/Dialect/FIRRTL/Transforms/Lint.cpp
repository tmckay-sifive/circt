//===- Lint.cpp -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Analysis/FIRRTLInstanceInfo.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/APSInt.h"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_LINT
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace firrtl;

namespace {
/// Linter configuration.
struct Config {
  /// If true, then assertions that are statically false (and will trivially
  /// fail simulation) will result in an error.
  bool lintStaticAsserts;
};

/// Class that stores state related to linting.  This exists to avoid needing to
/// clear members of `LintPass` and instead just rely on `Linter` objects being
/// deleted.
class Linter {

public:
  Linter(FModuleOp fModule, InstanceInfo &instanceInfo, const Config &config)
      : fModule(fModule), instanceInfo(instanceInfo), config(config){};

  /// Lint the specified module.
  LogicalResult lint() {
    bool failed = false;
    fModule.walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (isa<WhenOp>(op))
        return WalkResult::skip();
      if (isa<AssertOp, VerifAssertIntrinsicOp>(op))
        if (config.lintStaticAsserts && checkAssert(op).failed())
          failed = true;

      if (auto xmrDerefOp = dyn_cast<XMRDerefOp>(op))
        if (checkXmr(xmrDerefOp).failed())
          failed = true;

      return WalkResult::advance();
    });

    if (failed)
      return failure();

    return success();
  }

private:
  FModuleOp fModule;
  InstanceInfo &instanceInfo;
  const Config &config;

  LogicalResult checkAssert(Operation *op) {
    Value predicate;
    if (auto a = dyn_cast<AssertOp>(op)) {
      if (auto constant = a.getEnable().getDefiningOp<firrtl::ConstantOp>())
        if (constant.getValue().isOne()) {
          predicate = a.getPredicate();
        }
    } else if (auto a = dyn_cast<VerifAssertIntrinsicOp>(op))
      predicate = a.getProperty();

    if (!predicate)
      return success();
    if (auto constant = predicate.getDefiningOp<firrtl::ConstantOp>())
      if (constant.getValue().isZero())
        return op->emitOpError(
                     "is guaranteed to fail simulation, as the predicate is "
                     "constant false")
                   .attachNote(constant.getLoc())
               << "constant defined here";

    if (auto reset = predicate.getDefiningOp<firrtl::AsUIntPrimOp>())
      if (firrtl::type_isa<ResetType, AsyncResetType>(
              reset.getInput().getType()))
        return op->emitOpError("is guaranteed to fail simulation, as the "
                               "predicate is a reset signal")
                   .attachNote(reset.getInput().getLoc())
               << "reset signal defined here";

    return success();
  }

  LogicalResult checkXmr(XMRDerefOp op) {
    // XMRs under layers are okay.
    if (op->getParentOfType<LayerBlockOp>() ||
        op->getParentOfType<sv::IfDefOp>())
      return success();

    // The XMR is not under a layer.  This module must never be instantiated in
    // the design.  Intentionally do NOT use "effective" design as this could
    // lead to false positives.
    if (!instanceInfo.allInstancesInDesign(fModule))
      return success();

    // If the op has a single user, that user is a connect, and the connect is
    // to an instance which is marked `lowerToBind`, then this is a pattern for
    // inlining the XMR into the bound instance site.  This pattern is used by
    // Grand Central, but not elsewhere.
    if (op.getResult().hasOneUse()) {
      auto *user = *op.getResult().getUsers().begin();
      auto connect = dyn_cast<MatchingConnectOp>(user);
      if (connect && connect.getSrc() == op.getResult())
        if (auto *definingOp = connect.getDest().getDefiningOp())
          if (auto instanceOp = dyn_cast<InstanceOp>(definingOp))
            if (instanceOp->hasAttr("lowerToBind"))
              return success();
    }

    auto diag =
        op.emitOpError()
        << "is in the design. (Did you forget to put it under a layer?)";
    diag.attachNote(fModule.getLoc()) << "op is instantiated in this module";

    return failure();
  }
};

struct LintPass : public circt::firrtl::impl::LintBase<LintPass> {
  using LintBase::lintStaticAsserts;

  void runOnOperation() override {
    auto instanceInfo = getCachedParentAnalysis<InstanceInfo>();
    if (!instanceInfo) {
      llvm::errs() << "Unable to run the LintPass as no cached "
                      "InstanceInfoAnalysis is available";
      return signalPassFailure();
    }
    if (failed(
            Linter(getOperation(), *instanceInfo, {lintStaticAsserts}).lint()))
      return signalPassFailure();

    markAllAnalysesPreserved();
  };
};
} // namespace

std::unique_ptr<Pass> firrtl::createLintingPass(bool lintStaticAsserts) {
  auto pass = std::make_unique<LintPass>();
  pass->lintStaticAsserts = lintStaticAsserts;
  return pass;
}
