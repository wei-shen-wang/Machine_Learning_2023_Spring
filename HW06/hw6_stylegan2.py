import os
import torch
from torchvision.utils import save_image
from torchvision import transforms
from stylegan2_pytorch import ModelLoader, Trainer

base_dir = "./"
output_path = "submission"
if not os.path.exists(output_path):
    os.mkdir(output_path)
data_path = "/mnt/data1/r112_whitebear/ML2023/HW6_data/diffusion/faces/faces"
trainer = Trainer(trunc_psi=0.75, base_dir=base_dir, num_workers=2)
trainer.set_data_src(data_path)
step = 0
total_step = 80000

while (step < total_step):
    step += 1
    trainer.train()
    if step % trainer.save_every == 0:
        trainer.print_log()
loader = ModelLoader(base_dir="./")

topil = transforms.ToPILImage()
totensor = transforms.ToTensor()

def inference(num=1000, output_path='./submission'):
    for j in range(num):
        noise = torch.randn(1, 512).cuda()
        styles = loader.noise_to_styles(noise, trunc_psi=0.75)
        images = loader.styles_to_images(styles)
        pilimages = topil(images[0])
        pilimages = pilimages.resize((64, 64))
        compressedimages = totensor(pilimages)
        save_image(compressedimages, f'{output_path}/{j + 1}.jpg')


inference()
os.system("cd submission && tar zcvf ../images.tgz *.jpg")

# reference: https://github.com/lucidrains/stylegan2-pytorch