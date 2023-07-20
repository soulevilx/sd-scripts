import copy
import random
import shlex

import modules.scripts as scripts
import gradio as gr

import os
import logging

from modules import sd_samplers, errors
from modules.processing import Processed, process_images
from modules.shared import state

from pathlib import Path

def process_string_tag(tag):
    return tag

def process_int_tag(tag):
    return int(tag)

def process_float_tag(tag):
    return float(tag)

def process_boolean_tag(tag):
    return True if (tag == "true") else False

prompt_tags = {
    "sd_model": None,
    "outpath_samples": process_string_tag,
    "outpath_grids": process_string_tag,
    "prompt_for_display": process_string_tag,
    "prompt": process_string_tag,
    "negative_prompt": process_string_tag,
    "styles": process_string_tag,
    "seed": process_int_tag,
    "subseed_strength": process_float_tag,
    "subseed": process_int_tag,
    "seed_resize_from_h": process_int_tag,
    "seed_resize_from_w": process_int_tag,
    "sampler_index": process_int_tag,
    "sampler_name": process_string_tag,
    "batch_size": process_int_tag,
    "n_iter": process_int_tag,
    "steps": process_int_tag,
    "cfg_scale": process_float_tag,
    "width": process_int_tag,
    "height": process_int_tag,
    "restore_faces": process_boolean_tag,
    "tiling": process_boolean_tag,
    "do_not_save_samples": process_boolean_tag,
    "do_not_save_grid": process_boolean_tag
}

def cmdargs(line):
    args = shlex.split(line)
    pos = 0
    res = {}

    while pos < len(args):
        arg = args[pos]

        assert arg.startswith("--"), f'must start with "--": {arg}'
        assert pos + 1 < len(args), f'missing argument for command line option {arg}'

        tag = arg[2:]

        if tag == "prompt" or tag == "negative_prompt":
            pos += 1
            prompt = args[pos]
            pos += 1
            while pos < len(args) and not args[pos].startswith("--"):
                prompt += " "
                prompt += args[pos]
                pos += 1
            res[tag] = prompt
            continue

        func = prompt_tags.get(tag, None)
        assert func, f'unknown commandline option: {arg}'

        val = args[pos + 1]
        if tag == "sampler_name":
            val = sd_samplers.samplers_map.get(val.lower(), None)

        res[tag] = func(val)

        pos += 2

    return res

def load_prompt_file(file):
    if file is None:
        return None, gr.update(), gr.update(lines=7)
    else:
        lines = [x.strip() for x in file.decode('utf8', errors='ignore').split("\n")]
        return None, "\n".join(lines), gr.update(lines=7)

class Script(scripts.Script):
    def title(self):
        return "Dynamic massive generate"

    def ui(self, is_img2img):
        prompt_txt = gr.Textbox(label="Custom prompt", lines=1, elem_id=self.elem_id("prompt_txt"))
        lora_txt = gr.Textbox(label="Specific lora", lines=1, elem_id=self.elem_id("lora_txt"))
        checkbox_random_seed = gr.Checkbox(label="Random seed for each prompt", value=False, elem_id=self.elem_id("checkbox_random_seed"))

        return [prompt_txt, lora_txt, checkbox_random_seed]

    def run(self, p, prompt_txt: str, lora_txt: str, checkbox_random_seed):
        logging.basicConfig(filename='logging.log', level=logging.INFO)
        baseDir = os.path.dirname(os.path.abspath(__file__))
        joosDir = baseDir + '/joos'

        logging.info('Dir: ' + joosDir)

        p.do_not_save_grid = True

        loraWeightBegin = -0.9
        logging.info('Start weight: ' + str(loraWeightBegin))

        job_count = 0
        jobs = []
        args = {}
        images = []
        all_prompts = []
        infotexts = []

        # negative prompt
        negativePromptFile = Path(joosDir).with_name("negative_prompt")
        negativePrompt = open(negativePromptFile).read()
        logging.info('Loaded negative prompt: ' + str(negativePrompt))

        # prompt
        promptFile = Path(joosDir).with_name("prompt")
        prompt = open(promptFile).read()
        logging.info('Loaded prompt: ' + str(prompt))

        # build prompt & lora
        while (loraWeightBegin <= 1.0):
            lora = "<lora:" + lora_txt + ":" + str(loraWeightBegin) + ">"
            # main prompt + custom prompt + lora + fixed prompt
            # main negative prompt + fixed negative prompt
            args['prompt'] = p.prompt + "," + prompt_txt + "," + lora + "," + prompt
            args['negative_prompt'] = p.negative_prompt + "," + negativePrompt

            if (checkbox_random_seed) and p.seed == -1:
                args['seed'] = int(random.randrange(4294967294))

            job_count += args.get("n_iter", p.n_iter)
            jobs.append(args)

            loraWeightBegin = round(loraWeightBegin + 0.1, 1)

            copy_p = copy.copy(p)
            for k, v in args.items():
                setattr(copy_p, k, v)

            print('Prompt: ' + copy_p.prompt + "\n")
            print('Negative prompt: ' + copy_p.negative_prompt + "\n")
            print('Seed: ' + str(copy_p.seed) + "\n")

            proc = process_images(copy_p)
            images += proc.images

            all_prompts += proc.all_prompts
            infotexts += proc.infotexts

        return Processed(p, images, p.seed, "", all_prompts=all_prompts, infotexts=infotexts)
