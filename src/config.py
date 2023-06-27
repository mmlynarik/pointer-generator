from pathlib import Path


ROOT_DIR = Path().resolve()
TOKENIZER_DIR = ROOT_DIR / "src" / "trained_tokenizer"

UNK_TOKEN = "[UNK]"
PAD_TOKEN = "[PAD]"
START_TOKEN = "[START]"
END_TOKEN = "[STOP]"

SPECIAL_TOKENS = [UNK_TOKEN, PAD_TOKEN, START_TOKEN, END_TOKEN]
