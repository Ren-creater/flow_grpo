import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed.checkpoint
from accelerate import Accelerator
from accelerate.utils import gather_object
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
from torch.distributed.fsdp import FullyShardedDataParallel
from transformers.integrations import WandbCallback

import wandb
import wandb.plot


logger = logging.getLogger(__name__)

mscoco_val_captions = [
    "A woman stands in the dining area at the table.",
    "A big burly grizzly bear is shown with grass in the background.",
    "A bedroom with a neatly made bed, a window, and a bookshelf.",
    "A stop sign installed upside down on a street corner.",
    "Three teddy bears, each a different color, snuggling together.",
]

shagpt4v_gpt4v_val_captions = [
    "In the center of the image, a vibrant blue lunch tray holds four containers, each brimming with a variety of food items. The containers, two in pink and two in yellow, are arranged in a 2x2 grid.\n\nIn the top left pink container, a slice of bread rests, lightly spread with butter and sprinkled with a handful of almonds. The bread is cut into a rectangle, and the almonds are scattered across its buttery surface.\n\nAdjacent to it in the top right corner, another pink container houses a mix of fruit. Sliced apples with their fresh white interiors exposed share the space with juicy chunks of pineapple. The colors of the apple slices and pineapple chunks contrast beautifully against the pink container.\n\nBelow these, in the bottom left corner of the tray, a yellow container holds a single meatball alongside some broccoli. The meatball, round and browned, sits next to the vibrant green broccoli florets.\n\nFinally, in the bottom right yellow container, there's a sweet treat - a chocolate chip cookie. The golden-brown cookie is dotted with chocolate chips, their dark color standing out against the cookie's lighter surface.\n\nThe arrangement of these containers on the blue tray creates a visually appealing and balanced meal, with each component neatly separated yet part of a cohesive whole.",
    "This image captures a serene moment in a zoo enclosure, where two majestic giraffes are seen in their natural behavior. The giraffes, adorned in their distinctive brown and white patterns, stand tall against the backdrop of lush green trees.\n\nOn the left, one giraffe is actively engaged in a meal, its long neck extended towards the tree as it munches on the verdant leaves. Its companion on the right stands leisurely next to a tree trunk, perhaps taking a break from its own leafy feast.\n\nThe enclosure they inhabit is grassy and spacious, providing them with ample room to roam and forage. The trees dotting the enclosure not only offer a source of food but also create a naturalistic habitat for these towering creatures.\n\nIn summary, this image is a snapshot of life in a zoo, showcasing the grace and beauty of giraffes in an environment designed to mimic their wild habitats.",
    "The image presents a serene garden scene, centered around a white vase with a bouquet of flowers. The vase, exhibiting a fluted design and resting on a pedestal base, is placed on a white railing. It holds a bouquet of flowers, a harmonious blend of white and pink blossoms, interspersed with touches of greenery. \n\nThe vase and its floral contents are the focal point of the image, captured from a slight angle that places them in the center of the frame. The background reveals a lush garden, populated with verdant plants and enclosed by a white fence. The overall composition exudes an atmosphere of tranquility and natural beauty.",
    "This is a detailed description of the image:\n\n- The image captures a **single zebra** in its natural habitat, a lush green field.\n- The zebra, known for its distinctive **black and white stripes**, is the main subject of the photograph.\n- The zebra is positioned towards the **right side** of the image, giving a side profile view to the observer.\n- It is standing on all **four legs**, indicating a state of calm and ease.\n- The zebra is engaged in a common activity for its species - grazing. Its neck is bent downwards as it feeds on the **green grass** beneath it.\n- The field in which the zebra is grazing is not just green but also dotted with **small white flowers**, adding a touch of contrast and beauty to the scene.\n- There are no other zebras or animals visible in the image, suggesting that this particular zebra might be alone at this moment.\n\nThis description is based on what can be confidently determined from the image. It provides a comprehensive understanding of the image content, including object types, colors, actions, and locations.",
    'In the image, there is a woman standing in front of a white sign. The sign has blue text that reads "WELCOME TO THE LAKE". The woman is holding a pink umbrella in her hand and is wearing a colorful swimsuit. She is smiling, adding a cheerful vibe to the scene. The background of the image features a serene lake surrounded by lush green trees under a clear blue sky. The woman appears to be the only person in the image, and her position in front of the sign suggests she might be at the entrance of the lake area. The pink umbrella she\'s holding contrasts nicely with her colorful swimsuit and the natural colors of the background.',
]


class DiffusionWandbCallback(WandbCallback):
    def __init__(self, trainer, prompts=None):
        # if wandb have not init, the base class will init it
        super().__init__()
        self._trainer = trainer
        self.accelerator: Accelerator = self._trainer.accelerator
        self.prompts = mscoco_val_captions + shagpt4v_gpt4v_val_captions if prompts is None else prompts

    @torch.inference_mode()
    def on_step_end(self, args, state, control, **kwargs):
        super().on_step_end(args, state, control, **kwargs)
        eval_steps = args.eval_steps
        global_steps = state.global_step
        model = kwargs["model"]
        if global_steps % eval_steps == 0:
            logger.info(f"Evaluating at step {global_steps}")
            with self.accelerator.split_between_processes(self.prompts) as prompt:
                inputs = {
                    "prompt": prompt,
                    "return_full_process_images": True,
                    "predict": True,
                    "max_inference_steps": 40,
                }
                if len(args.fsdp) > 0:
                    with FullyShardedDataParallel.summon_full_params(model):
                        outputs = model.sample(inputs)
                else:
                    outputs = model.sample(inputs)

                reward, last_reward = model.reward(
                    inputs,
                    outputs,
                    self._trainer.reward_model,
                    gamma=self._trainer.args.gamma,
                    return_last_reward=True,
                )
                reward = reward.tolist()
                last_reward = last_reward.tolist()

                img = outputs["images"]
                sig = outputs["sigmas"].tolist()
                alpha = outputs["alphas"].tolist()
                beta = outputs["betas"].tolist()

            rewards = gather_object(reward)
            last_rewards = gather_object(last_reward)
            images = gather_object(img)
            sigmas = gather_object(sig)
            alphas = gather_object(alpha)
            betas = gather_object(beta)
            image_plots = [[] for _ in range(len(sigmas))]

            if self.accelerator.is_main_process:
                # 使用 matplotlib 绘制折线图并上传到 wandb
                for i in range(len(sigmas)):
                    # 其他属性只保留sigmas[i]中大于0.01的
                    images[i] = [images[i][j] for j in range(len(sigmas[i])) if sigmas[i][j] > 0.01]
                    alphas[i] = [alphas[i][j] for j in range(len(sigmas[i])) if sigmas[i][j] > 0.01]
                    betas[i] = [betas[i][j] for j in range(len(sigmas[i])) if sigmas[i][j] > 0.01]
                    # 只保留sigmas[i]中大于0.01的值
                    sigmas[i] = [x for x in sigmas[i] if x > 0.01]

                    concentration = [(alpha - 1) / (alpha + beta - 2) for alpha, beta in zip(alphas[i], betas[i])]

                    # 绘制并保存 sigmas
                    fig1, ax1 = plt.subplots()
                    # add first sigma
                    sigmas[i] = [1.0] + sigmas[i]
                    ax1.plot(sigmas[i], label="sigma", color="blue")
                    ax1.set_xlabel("Step")
                    ax1.set_ylabel("Value")
                    ax1.legend()
                    canvas1 = FigureCanvas(fig1)
                    canvas1.draw()
                    image_plot1 = Image.frombytes("RGB", canvas1.get_width_height(), canvas1.tostring_rgb())
                    image_plots[i].append(wandb.Image(image_plot1))

                    # 绘制并保存 alphas
                    fig2, ax2 = plt.subplots()
                    ax2.plot(alphas[i], label="alpha", color="red")
                    ax2.set_xlabel("Step")
                    ax2.set_ylabel("Value")
                    ax2.legend()
                    canvas2 = FigureCanvas(fig2)
                    canvas2.draw()
                    image_plot2 = Image.frombytes("RGB", canvas2.get_width_height(), canvas2.tostring_rgb())
                    image_plots[i].append(wandb.Image(image_plot2))

                    # 绘制并保存 betas
                    fig3, ax3 = plt.subplots()
                    ax3.plot(betas[i], label="beta", color="green")
                    ax3.set_xlabel("Step")
                    ax3.set_ylabel("Value")
                    ax3.legend()
                    canvas3 = FigureCanvas(fig3)
                    canvas3.draw()
                    image_plot3 = Image.frombytes("RGB", canvas3.get_width_height(), canvas3.tostring_rgb())
                    image_plots[i].append(wandb.Image(image_plot3))

                    # 绘制并保存 concentration
                    fig4, ax4 = plt.subplots()
                    ax4.plot(concentration, label="concentration", color="purple")
                    ax4.set_xlabel("Step")
                    ax4.set_ylabel("Value")
                    ax4.legend()
                    canvas4 = FigureCanvas(fig4)
                    canvas4.draw()
                    image_plot4 = Image.frombytes("RGB", canvas4.get_width_height(), canvas4.tostring_rgb())
                    image_plots[i].append(wandb.Image(image_plot4))

                # 将图像和 sigma 折线图一一配对并上传到 wandb
                for i, (image, image_plot) in enumerate(zip(images, image_plots)):
                    plots = [
                        wandb.Image(img.resize((256, 256)), caption=self.prompts[i]) for img in image
                    ] + image_plot
                    self._wandb.log({f"eval/image_sigma_{i}": plots})
                    self._wandb.log({f"eval/reward_{i}": rewards[i]})
                    self._wandb.log({f"eval/last_image_reward_{i}": last_rewards[i]})
