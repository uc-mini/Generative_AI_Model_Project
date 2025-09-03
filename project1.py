from transformers import pipeline
from diffusers import StableDiffusionPipeline
import torch

# Load text generator
story_gen = pipeline("text-generation", model="gpt2")

# Load image generator (Stable Diffusion - lightweight version)
device = "cuda" if torch.cuda.is_available() else "cpu"
image_gen = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5"
).to(device)

# User prompt
prompt = input("Enter your story idea: ")

# Generate story
story = story_gen(prompt, max_length=200, num_return_sequences=1)[0]['generated_text']

print("\n✨ Your AI-Generated Story ✨\n")
print(story)

# Generate image for the story
image = image_gen(prompt).images[0]
image.save("story_image.png")
print("\n🎨 An illustration has been saved as story_image.png")
