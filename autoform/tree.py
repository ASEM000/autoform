# Copyright 2026 The autoform Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pytree utilities"""

from autoform.src.tree import NAMESPACE as NAMESPACE
from autoform.src.tree import PyTreeSpec as PyTreeSpec
from autoform.src.tree import Tree as Tree
from autoform.src.tree import broadcast_prefix as broadcast_prefix
from autoform.src.tree import dataclass as dataclass
from autoform.src.tree import equal as equal
from autoform.src.tree import field as field
from autoform.src.tree import flatten as flatten
from autoform.src.tree import is_leaf as is_leaf
from autoform.src.tree import leaves as leaves
from autoform.src.tree import map as map
from autoform.src.tree import mask as mask
from autoform.src.tree import partition as partition
from autoform.src.tree import register_node as register_node
from autoform.src.tree import stack as stack
from autoform.src.tree import structure as structure
from autoform.src.tree import unstack as unstack
