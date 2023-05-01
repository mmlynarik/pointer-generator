import tokenizers
from transformers import PreTrainedTokenizerFast
from tokenizers import models, normalizers, pre_tokenizers, trainers
from datasets import load_dataset

from config import UNK_TOKEN, SPECIAL_TOKENS, TOKENIZER_DIR

dataset = load_dataset("cnn_dailymail", name="3.0.0")
train_dataset = dataset["train"]
train_dataset = train_dataset.map(lambda x: {"string": x["article"] + " " + x["highlights"]})

normalizers_list = [normalizers.NFD(), normalizers.StripAccents(), normalizers.NFC(), normalizers.Lowercase()]
normalizer = normalizers.Sequence(normalizers_list)
pre_tokenizer = pre_tokenizers.Whitespace()
trainer = trainers.WordLevelTrainer(vocab_size=50000, special_tokens=SPECIAL_TOKENS)

tokenizer = tokenizers.Tokenizer(models.WordLevel(unk_token=UNK_TOKEN))
tokenizer.normalizer = normalizer
tokenizer.pre_tokenizer = pre_tokenizer
tokenizer.train_from_iterator(train_dataset["string"], trainer=trainer)
tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)

tokenizer.save_pretrained(TOKENIZER_DIR)
