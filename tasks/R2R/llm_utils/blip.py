import torch
from PIL import Image
from lavis.models import load_model_and_preprocess

# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
# load sample image
# raw_image.show()

# loads InstructBLIP model
model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5_instruct", model_type="flant5xl", is_eval=True, device=device)
# prepare the image

def get_blip_response(raw_img):
    image = vis_processors["eval"](raw_img).unsqueeze(0).cuda()
    output = model.generate({"image": image, "prompt": "Describe in detail the indoor scene in this image with position information like 'left, right, middle, top, bottom' from our view."})
    return output[0]

def get_blip_response_ins(raw_img, instruction):
    image = vis_processors["eval"](raw_img).unsqueeze(0).cuda()
    output = model.generate({"image": image, "prompt": f"Describe in detail the indoor scene in this image with position information like 'left, right, middle, top, bottom' from our view."})
    return output[0]

if __name__ == "__main__":
    raw_image = Image.open("/root/Matterport3DSimulator/output/images/rgb.jpg").convert("RGB")
    instruction = 'Go down the path on the left towards the large dining room. \
                    Enter and go down the red carpet on the right side of the room, stopping at the door leading into another room.'
    print(get_blip_response_ins(raw_image, instruction))
    pass