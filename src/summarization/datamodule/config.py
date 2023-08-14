from pathlib import Path


ROOT_DIR = Path().resolve()
DATA_DIR = ROOT_DIR / "data"
TOKENIZER_DIR = ROOT_DIR / "src" / "summarizationmodel" / "trained_tokenizer"

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
START_TOKEN = "[START]"
END_TOKEN = "[STOP]"
VOCAB_SIZE = 50000
MAX_ENCODER_STEPS = 400
MAX_DECODER_STEPS = 100
MODEL_MAX_LENGTH = 2048
PAD_TOKEN_ID = 0
