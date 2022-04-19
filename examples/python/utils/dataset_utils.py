
# Copyright 2021 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np


def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def standardize(data):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(data)
    return scaler.transform(data)


def add_constant_col(data):
    return np.c_[data, np.ones((data.shape[0], 1))]


def load_full_dataset():
    from sklearn.datasets import load_breast_cancer
    ds = load_breast_cancer()
    x, y = ds['data'], ds['target']
    x = normalize(x)
    y = y.astype(dtype=np.float64)
    return x, y


def load_feature(rank: int):
    x, _ = load_full_dataset()
    if rank == 0:
        return x[:, :15]
    elif rank == 1:
        return x[:, 15:]
    else:
        raise Exception(f'only two party supported, got={rank}')


def load_label():
    _, y = load_full_dataset()
    return y
