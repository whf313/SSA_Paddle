# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ReftArgument:
    layers: str = field(default="all", metadata={"help": "Layer configuration for the model."})
    position: str = field(default="f7+l7", metadata={"help": "Position parameter for model."})
    intervention_type: str = field(default="LoreftIntervention", metadata={"help": "Type of intervention."})
    rank: int = field(default=8, metadata={"help": "Rank parameter for model."})
    act_fn: str = field(default="linear", metadata={"help": "Activation function."})
    add_bias: bool = field(default=False, metadata={"help": "Flag indicating whether to add bias."})
    dropout: float = field(default=0.0, metadata={"help": "Dropout rate."})


@dataclass
class GenerateArgument:
    top_k: int = field(
        default=1,
        metadata={
            "help": "The number of highest probability tokens to keep for top-k-filtering in the sampling strategy"
        },
    )
    top_p: float = field(
        default=1.0, metadata={"help": "The cumulative probability for top-p-filtering in the sampling strategy."}
    )


@dataclass
class EmbeddingArgument:
    max_query_len: int = field(
        default=1,
        metadata={
            "help": "The number of highest probability tokens to keep for top-k-filtering in the sampling strategy"
        },
    )
    max_passage_len: int = field(
        default=1.0, metadata={"help": "The cumulative probability for top-p-filtering in the sampling strategy."}
    )
    group_size: int = field(
        default=8,
        metadata={
            "help": (
                "Number of total positive and negative samples associated with " "each query for embedding training."
            )
        },
    )
    query_template: str = field(
        default="Query: {text}\nUse one word to summarize the query's relevant information. The word is: \"",
        metadata={
            "help": (
                "Query template. Ensure the template includes the placeholder "
                "'{text}' to insert the actual query text."
            )
        },
    )
    passage_template: str = field(
        default="Text: {text}\nUse one word to summarize the text's content. The word is: \"",
        metadata={
            "help": (
                "Passage template. Ensure the template includes the placeholder "
                "'{text}' to insert the actual passage text."
            )
        },
    )
    embedding_temperature: float = field(
        default=0.02,
        metadata={"help": "The temperature used in embedding learning."},
    )
    embedding_negatives_cross_device: bool = field(
        default=True,
        metadata={"help": "Whether to share the negatives across all GPUs."},
    )
    embedding_matryoshka_dims: Optional[List[int]] = field(
        default=None,
        metadata={"help": "The dims for matryoshka training."},
    )
    loss_type: str = field(
        default="contrastive",
        metadata={"help": "The type of loss computation."},
    )
    inf_cl_head_dim: int = field(
        default=64,
        metadata={"help": "The size of the head dimension when gpu ops are set as 'inf_cl'."},
    )


@dataclass
class DataConfig:

    dataset_name_or_path: str = field(default=None, metadata={"help": "Name or path for dataset"})
    task_name: str = field(default=None, metadata={"help": "Additional name to select a more specific task."})
    zero_padding: bool = field(default=False, metadata={"help": "Whether to use Zero Padding data stream"})
    greedy_zero_padding: bool = field(
        default=False,
        metadata={
            "help": "Whether to use Greedy Zero Padding data stream, should be used together with `zero_padding=True`."
        },
    )
    pad_to_multiple_of: int = field(
        default=None, metadata={"help": "If set will pad the sequence to a multiple of the provided value."}
    )
    src_length: int = field(default=1024, metadata={"help": "The maximum length of source(context) tokens."})
    max_length: int = field(
        default=2048,
        metadata={
            "help": "The maximum length that model input tokens can have. When Zero Padding is set to True, it's also the maximum length for Zero Padding data stream"
        },
    )
    eval_with_do_generation: bool = field(default=False, metadata={"help": "Whether to do generation for evaluation"})
    save_generation_output: bool = field(
        default=False,
        metadata={"help": "Whether to save generated text to file when eval_with_do_generation set to True."},
    )
    lazy: bool = field(
        default=False,
        metadata={
            "help": "Weather to return `MapDataset` or an `IterDataset`.True for `IterDataset`. False for `MapDataset`."
        },
    )
    chat_template: str = field(
        default=None,
        metadata={
            "help": "the path of `chat_template.json` file to handle multi-rounds conversation. If is None, it will not use `chat_template.json`; If is equal with `model_name_or_path`, it will use the default loading; If is directory, it will find the `chat_template.json` under the directory; If is file, it will load it."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={"help": "Pad the input sequence to `max_length`."},
    )
    autoregressive: bool = field(
        default=False,
        metadata={"help": "Whether to use autoregressive mode."},
    )
    # Pose related parameters
    use_pose_convert: bool = field(default=False, metadata={"help": "Whether to use PoSE data conversion function"})
    
    # SSA related parameters
    use_ssa_convert: bool = field(default=False, metadata={"help": "Whether to use SSA data conversion function"})
    block_min_length: int = field(default=False, metadata={"help": "Block min length"})
    num_blocks: int = field(default=False, metadata={"help": "Number of blocks"})