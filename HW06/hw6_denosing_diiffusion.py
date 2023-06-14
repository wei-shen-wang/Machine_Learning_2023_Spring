import os
import torch
import torchvision
from denoising_diffusion_pytorch import Trainer, GaussianDiffusion, Unet


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


base_dir = "./"
output_path = "submission"
if not os.path.exists(output_path):
    os.mkdir(output_path)
data_path = "/mnt/data/r112_whitebear/ML2023/HW6_data/diffusion/faces/faces"
model = Unet(dim=64, dim_mults=(1, 2, 4, 8))
diffusion = GaussianDiffusion(model=model,
                              image_size=64,
                              sampling_timesteps=200,
                              beta_schedule="cosine")
trainer = Trainer(diffusion_model=diffusion,
                  train_batch_size=16,
                  train_lr=1e-5,
                  folder=data_path,
                  gradient_accumulate_every=1,
                  calculate_fid=True,
                  amp=False,
                  train_num_steps=1000000)
trainer.train()
num = 1000
n_iter = 5
output_path = './submission'
if not os.path.exists(output_path):
    os.mkdir(output_path)
with torch.no_grad():
    for i in range(n_iter):
        batches = num_to_groups(num // n_iter, 200)
        all_images = list(
            map(lambda n: trainer.ema.ema_model.sample(batch_size=n),
                batches))[0]
        for j in range(all_images.size(0)):
            torchvision.utils.save_image(
                all_images[j], f'{output_path}/{i * 200 + j + 1}.jpg')

os.system("cd submission && tar zcvf ../images.tgz *.jpg")
# reference: https://github.com/lucidrains/denoising-diffusion-pytorch