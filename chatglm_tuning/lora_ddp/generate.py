import sys
import os

grandparent_path = os.path.abspath("..")
sys.path.insert(0, grandparent_path)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from torch.cuda.amp import autocast
import transformers
from transformers.generation.utils import LogitsProcessorList
from peft import PeftModel
from chatglm import ChatGLMForConditionalGeneration, ChatGLMTokenizer, InvalidScoreLogitsProcessor

from config import Config


class Generator:

    def __init__(self):

        self.tokenizer = ChatGLMTokenizer.from_pretrained(Config.base_model)
        model = ChatGLMForConditionalGeneration.from_pretrained(
            Config.base_model,
            torch_dtype=torch.float16
        )
        self.model = PeftModel.from_pretrained(
            model,
            Config.lora_model + "/2000",
            torch_dtype=torch.float16
        )

        # 推断时直接转换成float16
        # self.model.half()
        self.model.cuda()
        self.model.eval()

    def tokenize(self, prompt, output):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        a_ids = self.tokenizer.encode(text=prompt, add_special_tokens=False)
        b_ids = self.tokenizer.encode(text=output, add_special_tokens=False)

        if len(a_ids) > self.max_input_len - 1:
            a_ids = a_ids[: self.max_input_len - 1]

        if len(b_ids) > self.max_output_len - 2:
            b_ids = b_ids[: self.max_output_len - 2]

        input_ids = self.tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)

        context_length = input_ids.index(self.tokenizer.bos_token_id)
        mask_position = context_length - 1
        labels = [self.label_pad_token_id] * context_length + input_ids[mask_position + 1:]

        pad_len = (self.max_input_len + self.max_output_len) - len(input_ids)
        input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
        labels = labels + [self.label_pad_token_id] * pad_len

        return input_ids, labels

    @torch.no_grad()
    def evaluate(
            self,
            title
    ):

        title += "。净含量是："
        logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())
        gen_kwargs = {"max_length": 2048, "num_beams": 1, "do_sample": True, "top_p": 0.7,
                      "temperature": 0.95, "logits_processor": logits_processor}

        inputs = self.tokenizer([title], return_tensors="pt")
        input_ids = inputs["input_ids"].cuda()
        outputs = self.model.generate(input_ids=input_ids, **gen_kwargs)
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
        response = self.tokenizer.decode(outputs)

        return response


if __name__ == "__main__":
    generator = Generator()

    pred = generator.evaluate("西红柿500g")
    print(pred)