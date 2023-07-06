from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from summarization.config import TOKENIZER_DIR
from summarization.datamodule.dataset import load_cnn_dailymail_dataset
from summarization.datamodule.tokenizer import SummarizationTokenizerFast, tokenize

raw_dataset = load_cnn_dailymail_dataset()
encoded_dataset = raw_dataset.map(tokenize, batched=True)

tokenizer = SummarizationTokenizerFast.from_pretrained(TOKENIZER_DIR)
data_collator = DataCollatorWithPadding(tokenizer)

train_dataloader = DataLoader(raw_dataset["train"], batch_size=32, shuffle=True, collate_fn=data_collator)

for step, batch in enumerate(train_dataloader):
    print(batch)
    if step > 5:
        break
