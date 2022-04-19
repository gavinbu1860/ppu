// Copyright 2021 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#include "ppu/mpc/beaver/beaver_cheetah.h"

#include <random>

#include "ppu/core/array_ref_util.h"
#include "ppu/link/link.h"
#include "ppu/mpc/beaver/prg_tensor.h"
#include "ppu/mpc/util/ring_ops.h"
#include "ppu/utils/serialize.h"

namespace ppu::mpc {
namespace {

uint128_t GetHardwareRandom128() {
  std::random_device rd;
  // call random_device four times, make sure uint128 is random in 2^128 set.
  uint64_t lhs = static_cast<uint64_t>(rd()) << 32 | rd();
  uint64_t rhs = static_cast<uint64_t>(rd()) << 32 | rd();
  return MakeUint128(lhs, rhs);
}

}  // namespace

BeaverCheetah::BeaverCheetah(std::shared_ptr<link::Context> lctx)
    : lctx_(lctx), seed_(GetHardwareRandom128()), counter_(0) {
  auto buf = utils::SerializeUint128(seed_);
  std::vector<Buffer> all_bufs =
      link::Gather(lctx_, buf, 0, "BEAVER_TFP:SYNC_SEEDS");

  if (lctx_->Rank() == 0) {
    // Collects seeds from all parties.
    for (size_t rank = 0; rank < lctx_->WorldSize(); ++rank) {
      PrgSeed seed = utils::DeserializeUint128(all_bufs[rank]);
      tp_.setSeed(rank, lctx_->WorldSize(), seed);
    }
  }

  // Setup silent ot. Map rank to party.
  cheetah_party_ = lctx_->Rank() == 0 ? emp::ALICE : emp::BOB;
}

void BeaverCheetah::set_primitives(
    std::shared_ptr<ppu::CheetahPrimitives> cheetah_primitives) {
  this->cheetah_primitives_ = cheetah_primitives;
}

Beaver::Triple BeaverCheetah::Mul(FieldType field, size_t size) {
  std::vector<PrgArrayDesc> descs(3);

  auto a = prgCreateArray(field, size, seed_, &counter_, &descs[0]);
  auto b = prgCreateArray(field, size, seed_, &counter_, &descs[1]);
  auto c = prgCreateArray(field, size, seed_, &counter_, &descs[2]);

  if (lctx_->Rank() == 0) {
    c = tp_.adjustMul(descs);
  }

  return {a, b, c};
}

Beaver::Triple BeaverCheetah::Dot(FieldType field, size_t M, size_t N,
                                  size_t K) {
  std::vector<PrgArrayDesc> descs(3);

  auto a = prgCreateArray(field, M * K, seed_, &counter_, &descs[0]);
  auto b = prgCreateArray(field, K * N, seed_, &counter_, &descs[1]);
  auto c = prgCreateArray(field, M * N, seed_, &counter_, &descs[2]);

  if (lctx_->Rank() == 0) {
    c = tp_.adjustDot(descs, M, N, K);
  }

  return {a, b, c};
}

Beaver::Triple BeaverCheetah::And(FieldType field, size_t size) {
  ArrayRef a(makeType<RingTy>(field), size);
  ArrayRef b(makeType<RingTy>(field), size);
  ArrayRef c(makeType<RingTy>(field), size);

  cheetah_primitives_->nonlinear()->beaver_triple(
      (uint8_t*)a.data(), (uint8_t*)b.data(), (uint8_t*)c.data(),
      size * a.elsize() * 8, true);

  return {a, b, c};
}

Beaver::Pair BeaverCheetah::Trunc(FieldType field, size_t size, size_t bits) {
  std::vector<PrgArrayDesc> descs(2);

  auto a = prgCreateArray(field, size, seed_, &counter_, &descs[0]);
  auto b = prgCreateArray(field, size, seed_, &counter_, &descs[1]);

  if (lctx_->Rank() == 0) {
    b = tp_.adjustTrunc(descs, bits);
  }

  return {a, b};
}

ArrayRef BeaverCheetah::RandBit(FieldType field, size_t size) {
  PrgArrayDesc desc{};
  auto a = prgCreateArray(field, size, seed_, &counter_, &desc);

  if (lctx_->Rank() == 0) {
    a = tp_.adjustRandBit(desc);
  }

  return a;
}

}  // namespace ppu::mpc
