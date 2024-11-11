from ram.models import ram
from ram import inference_ram
from ram import get_transform
import torch

ram_path = '/root/Matterport3DSimulator/pretrained/ckpt/ram_swin_large_14m.pth'
ram_model = ram(pretrained=ram_path, image_size=384, vit='swin_l').eval().cuda()