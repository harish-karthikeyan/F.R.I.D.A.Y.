
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch

MODEL_NAME = "therealcyberlord/vit-indian-food"

print("Loading pretrained food recognition model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
model = model.to(device)
model.eval()

print("Model loaded successfully!")

image_path = input("Enter the full path of a food image: ").strip().strip('"')

try:
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)[0]
        top_results = torch.topk(probabilities, k=5)

    print("\nTop 5 predictions:")
    for score, index in zip(top_results.values, top_results.indices):
        label = model.config.id2label[index.item()]
        print(f"{label}: {score.item() * 100:.2f}%")

except Exception as error:
    print("Could not process image:", error)
