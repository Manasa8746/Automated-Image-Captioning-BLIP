from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from tkinter import Tk, filedialog
import torch
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

print("🔄 Loading BLIP model... (first time may take 1–2 minutes)")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

Tk().withdraw()  
print("📂 Please select an image file to caption...")
image_path = filedialog.askopenfilename(
    title="Select an Image",
    filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")]
)

if not image_path:
    print("❌ No file selected. Exiting.")
    exit()

try:
    image = Image.open(image_path).convert("RGB")
except Exception as e:
    print("❌ Error loading image:", e)
    exit()

print("✨ Generating caption...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

inputs = processor(image, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

output = model.generate(**inputs, max_new_tokens=30)
caption = processor.decode(output[0], skip_special_tokens=True)

print(f"\n🖼 Generated Caption: {caption}")
image.show(title=caption)