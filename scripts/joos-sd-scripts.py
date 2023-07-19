import copy
import random
import shlex

import modules.scripts as scripts
import gradio as gr

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
        return "Lora weights"

    def ui(self, is_img2img):
        prompt_txt = gr.Textbox(label="List of prompt inputs", lines=1, elem_id=self.elem_id("prompt_txt"))
        lora_txt = gr.Textbox(label="Specific lora", lines=1, elem_id=self.elem_id("lora_txt"))

        # We start at one line. When the text changes, we jump to seven lines, or two lines if no \n.
        # We don't shrink back to 1, because that causes the control to ignore [enter], and it may
        # be unclear to the user that shift-enter is needed.
        prompt_txt.change(lambda tb: gr.update(lines=7) if ("\n" in tb) else gr.update(lines=2), inputs=[prompt_txt],
                          outputs=[prompt_txt], show_progress=False)
        return [prompt_txt, lora_txt]

    def run(self, p, prompt_txt: str, lora_txt: str):
        p.do_not_save_grid = True

        loraWeightBegin = -0.9

        negativePromptFile = Path(__file__ + "/joos").with_name("negative_prompt")
        negativePrompt = open(negativePromptFile).read()

        promptFile = Path(__file__ + "/joos").with_name("prompt")
        prompt = open(promptFile).read()

        job_count = 0
        jobs = []

        while (loraWeightBegin <= 1.0):
            lora = "<lora:" + lora_txt + ":" + str(loraWeightBegin) + ">"
            newPrompt = prompt_txt + lora + prompt
            args = {"prompt": p.prompt + newPrompt, "negative_prompt": p.negative_prompt + negativePrompt}

            job_count += args.get("n_iter", p.n_iter)
            jobs.append(args)

            print(lora + "\n")
            loraWeightBegin = round(loraWeightBegin + 0.1, 1)

        print(f"Will process {job_count} jobs.")

        state.job_count = job_count

        images = []
        all_prompts = []
        infotexts = []

        for args in jobs:

            state.job = f"{state.job_no + 1} out of {state.job_count}"

            copy_p = copy.copy(p)
            for k, v in args.items():
                setattr(copy_p, k, v)

            print('Prompts: ' + copy_p.prompt + "\n")
            print('Negative prompts: ' + copy_p.negative_prompt + "\n")

            proc = process_images(copy_p)
            images += proc.images

            all_prompts += proc.all_prompts
            infotexts += proc.infotexts

        return Processed(p, images, p.seed, "", all_prompts=all_prompts, infotexts=infotexts)
