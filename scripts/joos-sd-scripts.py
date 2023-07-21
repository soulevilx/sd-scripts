import copy
import os
import random
from pathlib import Path

import gradio as gr

import modules.scripts as scripts
from modules.processing import Processed, process_images

class Generator:
    def __init__(self, sdp):
        self.sdp = sdp

    def load_file_content(self, filename):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        joos_dir = base_dir + '/joos'
        file_path = Path(joos_dir).with_name(filename)
        content = open(file_path).read()

        return content

    def prompt(self):
        prompt = self.sdp.prompt
        prompt = prompt.strip()
        prompt = prompt.strip(',')

        return prompt.split(',') + self.load_file_content('prompt').strip().strip(',').split(',')

    def negative_prompt(self):
        negative_prompt = self.sdp.negative_prompt
        negative_prompt = negative_prompt.strip()
        negative_prompt = negative_prompt.strip(',')

        return negative_prompt.split(',') + self.load_file_content('negative_prompt').strip().strip(',').split(',')

    def lora(self, lora, weight_from, weight_to, step):
        weight = weight_from
        loras = []

        while weight <= weight_to:
            loras.append('<lora:' + lora + ':' + str(weight) + '>')
            weight = round(weight + step, 1)

        return loras

    def generate(self, settings):
        loras = self.lora(settings.lora_weight_lora,settings.lora_weight_from,settings.lora_weight_to ,settings.lora_weight_step )

        prompt = ','.join(self.prompt())
        negative_prompt = ','.join(self.negative_prompt())

        spds = []
        for i_lora in loras:
            csdp = copy.copy(self.sdp)
            setattr(csdp, 'prompt', prompt + ',' + i_lora)
            setattr(csdp, 'negative_prompt', negative_prompt)

            if (settings.random_seed == 1):
                setattr(csdp, 'seed', int(random.randrange(4294967294)))

            spds.append(csdp)

        return spds

class Settings:
    pass

class Script(scripts.Script):
    def title(self):
        return "Dynamic massive generate"

    def ui(self, is_img2img):
        lora_txt = gr.Textbox(label="Specific lora", lines=1, elem_id=self.elem_id("lora_txt"))
        checkbox_random_seed = gr.Checkbox(label="Random seed for each prompt", value=False, elem_id=self.elem_id("checkbox_random_seed"))

        return [lora_txt, checkbox_random_seed]

    def run(self, p, lora_txt: str, checkbox_random_seed):
        settings = Settings
        settings.prompt = p.prompt
        settings.negative_prompt = p.negative_prompt
        settings.random_seed = checkbox_random_seed
        settings.lora_weight_lora = lora_txt
        settings.lora_weight_from = -0.1
        settings.lora_weight_to = 1
        settings.lora_weight_step = 0.1

        generator = Generator(p)

        sdps = (generator.generate(settings))
        images = []
        all_prompts = []
        infotexts = []

        for sdp in sdps:
            print('Prompt: ' + sdp.prompt)
            print('Negative Prompt: ' + sdp.negative_prompt)
            proc = process_images(sdp)
            images += proc.images
            all_prompts += proc.all_prompts
            infotexts += proc.infotexts

        return Processed(p, images, p.seed, "", all_prompts=all_prompts, infotexts=infotexts)