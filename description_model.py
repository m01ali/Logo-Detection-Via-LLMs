import torch
from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports
from PIL import Image, ImageDraw, ImageFont
from itertools import cycle
import numpy as np
import os
import getpass

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)


def fixed_get_imports(filename) -> list[str]:
    if not str(filename).endswith("modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    imports.remove("flash_attn")
    return imports


colormap = ['blue','orange','green','purple','brown','pink','gray','olive','cyan','red',
            'lime','indigo','violet','aqua','magenta','coral','gold','tan','skyblue']

class GemmaModel:
    def __init__(self, model_id="google/gemma-2-2b"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=quantization_config).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        
    def run_example(self, text, prompt=None):
        if not prompt:
            prompt = f"{text}\nThe name of the logo given in the description (not OCR) is:\n"
            
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**input_ids, max_new_tokens=32)
        outputs = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        logo_name = outputs[len(prompt):].split('\n')[0].strip()

        return logo_name

class FlorenceModel:
    def __init__(self, model_id="microsoft/Florence-2-base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model_id = model_id
        
        # with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=self.torch_dtype, trust_remote_code=True).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model.eval()
        
    def run_example(self, task_prompt: str, text_input:str = None, image: Image =None):
        if text_input is None:
            prompt = task_prompt
        else:
            prompt = task_prompt + text_input
        
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device, self.torch_dtype)
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=256,
            # early_stopping=False,
            do_sample=False,
            num_beams=3)
        
    
        
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(generated_text,
                                                      task=task_prompt,
                                                      image_size=(image.width, image.height))
        
        # print(generated_ids)
        # print(generated_text)
        # print(inputs)
        
        return parsed_answer
    
    
    def describe_image(self, image: Image):
        prompt = "<MORE_DETAILED_CAPTION>"
        return self.run_example(prompt, image=image)[prompt]
    
    def extract_text(self, image: Image):
        prompt = "<OCR_WITH_REGION>"
        results = self.run_example(prompt, image=image)[prompt]
        labels = results['labels']
        labels = [str.strip(str.strip(label, "</s>")) for label in labels]
        text = " ".join(labels)
        return text
    
    def generate_description(self, image: Image):
        ocr = self.extract_text(image)
        desc = self.describe_image(image)
        result = "OCR: " + ocr + "\n" + "Description: " + desc
        return result
    

def main():
    os.environ["HF_TOKEN"] = getpass.getpass()
    model = FlorenceModel()
    gemma = GemmaModel()
    logo = Image.open(".\Logo-Detection\test-logo-images\1.png")

    desc = model.generate_description(logo)
    print(desc)
    logo_name = gemma.run_example(desc)
    print(logo_name)

    
if __name__ == "__main__":
    main()