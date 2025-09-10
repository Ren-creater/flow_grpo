import logging
import os
import pickle

import gradio as gr
import matplotlib.pyplot as plt
import torch
from safetensors.torch import load_file

from src.models.stable_diffusion_3.modeling_sd3_pnt import SD3PredictNextTimeStepModel
from src.models.stable_diffusion_v1_5.modeling_sd_v1_5 import SD15PredictNextTimeStepModel


os.environ["GRADIO_TEMP_DIR"] = "./gradio_temp"

logger = logging.getLogger(__name__)

model = SD3PredictNextTimeStepModel("models/stabilityai/stable-diffusion-3-medium-diffusers").eval().to("cuda")

ckpt = load_file("checkpoint/sd3/model.safetensors")
model.load_state_dict(ckpt, strict=False)

# inference flux
# model = FluxPredictNextTimeStepModel("models/black-forest-labs/FLUX.1-dev").eval().to("cuda")

# ckpt = load_file("checkpoint/flux/model.safetensors")
# ckpt = {k.replace("agent_model.", ""): v for k, v in ckpt.items()}
# model.load_state_dict(ckpt, strict=False)



def generate_image(simple_prompt, medium_prompt, complex_prompt, complexity, seed):
    if complexity == "Simple":
        prompt = simple_prompt
    elif complexity == "Medium":
        prompt = medium_prompt
    else:
        prompt = complex_prompt

    seed = int(seed)
    outputs = model.forward(
        prompt=prompt,
        predict=True,
        max_inference_steps=35,
        generator=torch.Generator("cuda").manual_seed(seed),
        return_full_process_images=False,
    )
    images = outputs.images[0]
    sigmas = outputs.sigmas.tolist()[0]
    return images[-1], f"Inference steps: {len(sigmas)}"


iface = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.components.Textbox(label="Simple Prompt", value="Snowy peak, soaring eagle, icy winds, blue sky."),
        gr.components.Textbox(
            label="Medium Prompt",
            value="An ornate, golden invitation letter with intricate calligraphy. The text reads 'Your Presence is Requested at the Royal Feast' in elegant, swirling script. The letter is illuminated by soft candlelight and rests on a royal velvet cushion. The background features a grand palace with towering spires and lush gardens, with a small scroll tucked inside the envelope.",
        ),
        gr.components.Textbox(
            label="Complex Prompt",
            value="An ancient, leather-bound book with embossed mystical symbols, open on a wooden desk in a dimly lit library. The pages are yellowed, filled with arcane illustrations and handwritten notes in an unknown script. A faint, ethereal glow emanates from the book, casting soft light on a quill and ink pot nearby. A crystal ball sits on the desk, reflecting the glow. Through an arched window, a full moon shines in a starry night sky, and a raven perches on the windowsill, its eyes fixed on the book. In the background, shelves are lined with other mysterious tomes, some with spines adorned with gems or strange markings.",
        ),
        gr.components.Radio(choices=["Simple", "Medium", "Complex"], label="Select Complexity"),
        gr.components.Textbox(label="Seed", value="0"),
    ],
    outputs=[gr.components.Image(label="Image"), gr.components.Textbox(label="Steps")],
    title="Schedule On the Fly: Diffusion Time Prediction for Faster and Better Image Generation",
    description="<div style='text-align: center;'></div>\n\n \
                In this project, we argue that the optimal noise schedule should \
                adapt to each inference instance, and introduce the Time \
                Prediction Diffusion Model (TPDM) to accomplish this. \
                TPDM employs a plug-and-play Time Prediction Module \
                (TPM) that predicts the next noise level based on current \
                latent features at each denoising step. We train the TPM \
                using reinforcement learning, aiming to maximize a reward \
                that discounts the final image quality by the number of denoising steps. With such an adaptive scheduler, TPDM not \
                only generates high-quality images that are aligned closely \
                with human preferences but also adjusts the number of denoising steps and time on the fly, enhancing both performance and efficiency.",
)

if __name__ == "__main__":
    iface.launch(share=True)
