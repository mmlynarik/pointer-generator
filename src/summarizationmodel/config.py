from pathlib import Path

ROOT_DIR = Path().resolve()
TOKENIZER_DIR = ROOT_DIR / "src" / "summarizationmodel" / "trained_tokenizer"
DATA_DIR = ROOT_DIR / "data"

UNK_TOKEN = "[UNK]"
PAD_TOKEN = "[PAD]"
START_TOKEN = "[START]"
END_TOKEN = "[STOP]"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, START_TOKEN, END_TOKEN]

VOCAB_SIZE = 50000
MAX_ENCODER_STEPS = 400
MAX_DECODER_STEPS = 100
MODEL_MAX_LENGTH = 2048

PADDABLE_FEATURES = [
    "encoder_input_ids",
    "encoder_padding_mask",
    "decoder_input_ids",
    "decoder_padding_mask",
    "decoder_target_ids",
    "encoder_inputs_extvoc",
]

NON_PADDABLE_FEATURES = ["oovs"]
