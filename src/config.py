from pathlib import Path


ROOT = Path().resolve()
TOKENIZER_DIR = str(ROOT / "src" / "trained_tokenizer")

UNK_TOKEN = "[UNK]"
PAD_TOKEN = "[PAD]"
START_TOKEN = "[START]"
END_TOKEN = "[STOP]"

SPECIAL_TOKENS = [UNK_TOKEN, PAD_TOKEN, START_TOKEN, END_TOKEN]
