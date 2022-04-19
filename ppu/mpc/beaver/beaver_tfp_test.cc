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


#include "ppu/mpc/beaver/beaver_tfp.h"

#include "ppu/mpc/beaver/beaver_test.h"

namespace ppu::mpc {

INSTANTIATE_TEST_SUITE_P(
    BeaverTfpTest, BeaverTest,
    testing::Combine(
        testing::Values([](const std::shared_ptr<link::Context>& lctx) {
          return std::make_unique<BeaverTfp>(lctx);
        }),
        testing::Values(4, 3, 2),
        testing::Values(FieldType::FM32, FieldType::FM64, FieldType::FM128)),
    [](const testing::TestParamInfo<BeaverTest::ParamType>& info) {
      return fmt::format("{}x{}", std::get<1>(info.param),
                         std::get<2>(info.param));
    });

}  // namespace ppu::mpc
