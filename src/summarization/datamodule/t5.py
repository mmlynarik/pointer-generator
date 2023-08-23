from typing import Callable, Sequence, Union

from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers import pipeline
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.models.t5.tokenization_t5_fast import T5TokenizerFast
from transformers.tokenization_utils_base import BatchEncoding

from summarization.datamodule.config import (
    END_TOKEN,
    START_TOKEN,
)


TEXT = Union[str, Sequence[str]]
TokenizerFunction = Callable[[TEXT], BatchEncoding]
TruncationChecker = Callable[[TEXT], bool]
START_OR_END_TOKEN = Union[START_TOKEN, END_TOKEN]


def main():
    # text = "Here is a short article."
    # text_2 = "Here is a long article, which exceeds model max length."
    # texts = [text, text_2]

    # model: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained("t5-small")
    # tokenizer: T5TokenizerFast = T5TokenizerFast.from_pretrained("t5-small")

    # inputs = tokenizer(texts, padding=True, truncation=True, max_length=10, return_tensors="pt")
    # labels = inputs["input_ids"]
    # decoder_input_ids = model.prepare_decoder_input_ids_from_labels(labels)
    # tokenizer.model_max_length
    # print(labels)
    # print(decoder_input_ids)

    text = "summarize: The Inflation Reduction Act lowers prescription drug costs, health care costs, and energy costs. It's the most aggressive action on tackling the climate crisis in American history, which will lift up American workers and create good-paying, union jobs across the country. It'll lower the deficit and ask the ultra-wealthy and corporations to pay their fair share. And no one making under $400,000 per year will pay a penny more in taxes."

    summarizer = pipeline("summarization", model="stevhliu/my_awesome_billsum_model", max_length=70)
    print(summarizer(text)[0])

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_billsum_model")
    inputs = tokenizer(text, return_tensors="pt").input_ids

    from transformers import AutoModelForSeq2SeqLM

    model = AutoModelForSeq2SeqLM.from_pretrained("stevhliu/my_awesome_billsum_model")
    outputs = model.generate(inputs, max_new_tokens=100, do_sample=False)

    print(tokenizer.decode(outputs[0], skip_special_tokens=False))


if __name__ == "__main__":
    main()
