from dataclasses import dataclass, field
from typing import Optional
from transformers import (
    MODEL_FOR_CAUSAL_LM_MAPPING,
)
from transformers.utils.versions import require_version

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    lora_rank: Optional[int] = field(
        default=8,
        metadata={
            "help": "lora rank"
        },
    )
    lora_alpha: Optional[int] = field(
        default=32,
        metadata={
            "help": "lora rank"
        },
    )

    use_vector_num: Optional[int] = field(
        default=0,
        metadata={
            "help": "number of used vectors"
        }
    )

    stride: Optional[int] = field(
        default=1,
        metadata={
            "help": "stride to sample token embeddings of a document(passage context)"
        }
    )

    online: Optional[int] = field(
        default=0,
        metadata={
            "help": "Deprecated."
        }
    )

    train_encoder: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether fine-tune encoder"
        }
    )

    train_decoder: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether fine-tune decoder"
        }
    )


    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    enforce_truncation_left: Optional[int] = field(
        default=0,
        metadata={
            "help": (
                'enfore truncation at left'
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    shuffle_train: bool = field(default=False, metadata={"help": "shuffle training data"})
    mask_prompt: bool = field(default=False, metadata={"help": "if mask prompts for loss"})
    gen_tok_as_special: bool = field(default=False, metadata={"help": "gentok as special, otherwise normal words"})
    use_lora: bool = field(default=False, metadata={"help": "use lora"})
    prompt_max_length: Optional[int] = field(
        default=512,
        metadata={
            "help": (
                "Optional input prompt [token] length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
            )
        },
    )
    answer_max_length: Optional[int] = field(
        default=512,
        metadata={
            "help": (
                "Optional input prompt [token] length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    retrieval_vec_save_mode: str = field(
        default='value',
        metadata={"help": "what format the encoded vectors are saved in the data file.",
                  "choices": ["index", "value"],
                  }
    )

    vec_database_file: str = field(
        default=None, metadata={"help": "Which file contains the encoded vectors of retrieved data."}
    )

    vec_dim: int = field(
        default=768, metadata={"help": "The dimension of encoded vector."}
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")


@dataclass
class GLMModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    ptuning_checkpoint: str = field(
        default=None, metadata={"help": "Path to p-tuning v2 checkpoints"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                "the model's position embeddings."
            )
        },
    )
    quantization_bit: Optional[int] = field(
        default=None
    )
    pre_seq_len: Optional[int] = field(
        default=None
    )
    prefix_projection: bool = field(
        default=False
    )
    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
