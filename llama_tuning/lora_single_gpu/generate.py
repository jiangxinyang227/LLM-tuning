import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from config import Config


class Generator:

    def __init__(self):

        self.tokenizer = LlamaTokenizer.from_pretrained(Config.base_model)
        model = LlamaForCausalLM.from_pretrained(
            Config.base_model,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model = PeftModel.from_pretrained(
            model,
            Config.lora_model,
            torch_dtype=torch.float16,
        )

        self.model.config.pad_token_id = self.tokenizer.pad_token_id = 0  # unk
        self.model.config.bos_token_id = 1
        self.model.config.eos_token_id = 2

        # 推断时直接转换成float16
        self.model.half()
        self.model.eval()

    def generate_prompt(self, instruction, input=None):
        # sorry about the formatting disaster gotta move fast
        if input:
            return f"""给定任务的描述和输入的问题，请返回结果。
    描述:
    {instruction}
    输入:
    {input}
    回答:
    """
        else:
            return f"""给定问题的描述，请返回结果。
    描述:
    {instruction}
    回答:
    """

    def evaluate(
            self,
            instruction,
            input=None,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=4,
            max_new_tokens=128,
            **kwargs,
    ):

        prompt = self.generate_prompt(instruction, input)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].cuda()
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )

        s = generation_output.sequences[0]
        output = self.tokenizer.decode(s)
        print(output)
        return output.strip("回答:")


if __name__ == "__main__":
    generator = Generator()
    instruction = "标注下面商品标题的实体词，乐事薯片500克"
    inp = ""

    res = generator.evaluate(instruction, inp)
    print(res)