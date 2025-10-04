# import
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from typing import List, Sequence, Union
from transformers import AutoModel, AutoProcessor


# load model
class PickScoreModel(nn.Module):
    def __init__(
        self,
        device="cuda",
        processor_name_or_path="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        model_pretrained_name_or_path="yuvalkirstain/PickScore_v1",
    ):
        super(PickScoreModel, self).__init__()
        self.device = device
        self.processor = AutoProcessor.from_pretrained(processor_name_or_path)
        self.model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(device)
        self.eval()

    def _prepare_prompts(self, prompt: Union[str, Sequence[str]], num_images: int) -> List[str]:
        if isinstance(prompt, (list, tuple)):
            prompts = list(prompt)
            if len(prompts) == 1 and num_images > 1:
                prompts = prompts * num_images
            elif len(prompts) != num_images:
                raise ValueError(
                    f"Expected 1 or {num_images} prompts, but received {len(prompts)} entries."
                )
        else:
            prompts = [prompt] * num_images
        return [str(p) for p in prompts]

    def _prepare_images(self, images: Union[Image.Image, torch.Tensor, np.ndarray, Sequence]) -> List[Image.Image]:
        if isinstance(images, (Image.Image, torch.Tensor, np.ndarray)):
            image_list = [images]
        else:
            image_list = list(images)
            if not image_list:
                raise ValueError("At least one image must be provided for PickScore evaluation.")
        return [self._to_pil(img) for img in image_list]

    @staticmethod
    def _to_pil(image: Union[Image.Image, torch.Tensor, np.ndarray]) -> Image.Image:
        if isinstance(image, Image.Image):
            return image

        if isinstance(image, torch.Tensor):
            tensor = image.detach().cpu()
            if tensor.ndim == 3 and tensor.shape[0] in {1, 3}:
                if tensor.shape[0] == 1:
                    tensor = tensor.repeat(3, 1, 1)
                tensor = tensor.permute(1, 2, 0)
            elif tensor.ndim == 2:
                tensor = tensor.unsqueeze(-1)
            tensor = tensor.numpy()
        elif isinstance(image, np.ndarray):
            tensor = image
        else:
            raise TypeError(f"Unsupported image type for PickScoreModel: {type(image)}")

        if tensor.ndim == 3 and tensor.shape[0] in {1, 3} and tensor.shape[-1] not in {1, 3}:
            tensor = np.moveaxis(tensor, 0, -1)

        if tensor.ndim == 3 and tensor.shape[-1] == 1:
            tensor = np.repeat(tensor, 3, axis=-1)

        if tensor.dtype != np.uint8:
            tensor = tensor.astype(np.float32)
            if tensor.max() <= 1.0:
                tensor = tensor * 255.0
            tensor = np.clip(tensor, 0, 255).astype(np.uint8)

        return Image.fromarray(tensor)

    def forward(self, prompt, images):
        pil_images = self._prepare_images(images)
        prompts = self._prepare_prompts(prompt, len(pil_images))

        image_inputs = self.processor(
            images=pil_images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        text_inputs = self.processor(
            text=prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            image_embs = self.model.get_image_features(**image_inputs)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

            text_embs = self.model.get_text_features(**text_inputs)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

            scale = self.model.logit_scale.exp()
            scores = (text_embs * image_embs).sum(dim=-1) * scale

        return scores.detach().cpu().tolist()

    def score(self, prompt, image) -> float:
        scores = self.forward(prompt, image)
        if isinstance(scores, list):
            return float(scores[0])
        return float(scores)

    def score_many(self, prompt, images) -> List[float]:
        scores = self.forward(prompt, images)
        if isinstance(scores, list):
            return [float(s) for s in scores]
        if isinstance(scores, torch.Tensor):
            return [float(s) for s in scores.detach().cpu().tolist()]
        return [float(scores)]


if __name__ == "__main__":
    model = PickScoreModel(
        device="cuda",
        processor_name_or_path="/mnt/sda/models/laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        model_pretrained_name_or_path="/mnt/sda/models/yuvalkirstain/PickScore_v1",
    )
    pil_images = [Image.open("demo/misaka.png")]
    prompt = "fantastic, incredible prompt"
    print(model.score(prompt, pil_images[0]))
