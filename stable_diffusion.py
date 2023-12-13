from diffusers import StableDiffusionPipeline
import torch
model = 'runwayml/stable-diffusion-v1-5'
pipe = StableDiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16)
if torch.cuda.is_available():
    pipe = pipe.to("cuda")
    print("CUDA is availale. Using GPU.")
else:
    pipe = torch.device("cuda")
    print("CUDA is not available. Using CPU.")

prompt = "A dog riding a bike"
image = pipe(prompt).images[0]

image.save("dog_driving_bike.png")