"""
Modified from https://github.com/gdlg/simple-waymo-open-dataset-reader
"""

# Copyright (c) 2019, Grégoire Payen de La Garanderie, Durham University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import struct
from waymo_open_dataset import dataset_pb2

def read_record(f, header_only = False):
    header = f.read(12)
    if header == b'':
        raise StopIteration()
    length, lengthcrc = struct.unpack("QI", header)
    if header_only:
        # Skip length+4 bytes ahead
        f.seek(length+4,1)
        return None
    else:
        data = f.read(length)
        datacrc = struct.unpack("I",f.read(4))

        frame = dataset_pb2.Frame()
        frame.ParseFromString(data)
        return frame

class WaymoDataFileReader:
    def __init__(self, filename):
        self.filename = filename
        with open(self.filename, 'rb') as f:
            f.seek(0,0)
            table = []
            while f:
                offset = f.tell()
                try:
                    read_record(f, header_only=True)
                    table.append(offset)
                except StopIteration:
                    break
            f.seek(0,0)

        self.table = table

    def __len__(self):
        return len(self.table)

    def __iter__(self):
        with open(self.filename, 'rb') as f:
            f.seek(0,0)
            while f:
                try:
                    yield read_record(f)
                except StopIteration:
                    break