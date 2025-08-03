import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel
from PIL import Image
import requests
import evaluate
from datasets import Dataset

# --- 1. CONFIGURATION ---

# Define the base model and the path to the saved adapters
MODEL_NAME = "Salesforce/blip-image-captioning-large"
ADAPTER_PATH = "./blip-medical-captioning-qlora"

# QLoRA configuration for 4-bit quantization
# This must match the configuration used during fine-tuning
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

# --- 2. MODEL AND ADAPTER LOADING ---

# Load the base model with 4-bit quantization
print("Loading base model...")
base_model = BlipForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load the processor (tokenizer and image processor)
processor = BlipProcessor.from_pretrained(MODEL_NAME)

# Load the fine-tuned adapters
print("Loading and merging QLoRA adapters...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

# Merge the adapters into the base model for a consolidated model for inference
model = model.merge_and_unload()
model.eval()

print("Model and adapters loaded successfully.")

# --- 3. INFERENCE DATA PREPARATION ---

def create_inference_data():
    """
    Generates a dummy dataset with images and ground-truth captions.
    For CIDEr calculation, each example must have a list of reference captions.
    """
    # Real, publicly accessible chest X-ray image URLs.
    # Note: These URLs are for demonstration purposes.
    image_url_1 = "https://i.imgur.com/W205S6L.jpeg"
    image_url_2 = "https://i.imgur.com/GzI8y4C.jpeg"

    data = {
        "image_url": [image_url_1, image_url_2],
        "image_text": [
            "The chest x-ray shows a small, non-displaced fracture of the right fifth rib. There is no evidence of pneumothorax or pleural effusion. The lungs are clear.",
            "Normal chest x-ray. The heart size is within normal limits. The mediastinal and hilar contours are unremarkable. No pulmonary opacities or effusions are seen.",
        ],
        "image_id": ["fractured_rib_001", "normal_chest_002"]
    }
    return Dataset.from_dict(data)

inference_dataset = create_inference_data()

# --- 4. INFERENCE FUNCTION ---

def generate_caption(image_path):
    """
    Generates a caption for a given image path.
    Args:
        image_path (str): The local path or URL of the image.
    Returns:
        str: The generated caption.
    """
    try:
        if image_path.startswith("http"):
            image = Image.open(requests.get(image_path, stream=True).raw).convert("RGB")
        else:
            image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        return "Failed to load image."

    # Preprocess the image
    inputs = processor(images=image, return_tensors="pt").to(model.device, model.dtype)

    # Generate the caption using the fine-tuned model
    # The `pixel_values` are the preprocessed image data
    out = model.generate(**inputs, max_length=50)

    # Decode the generated tokens back to a string
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# --- 5. RUN INFERENCE AND EVALUATION ---

generated_captions = []
reference_captions = []
image_ids = []

print("\nStarting inference and evaluation...")
for i, example in enumerate(inference_dataset):
    # Generate a caption for the image
    generated_caption = generate_caption(example["image_url"])
    print(f"\nExample {i+1}:")
    print(f"  Generated Caption: {generated_caption}")
    print(f"  Reference Caption: {example['image_text']}")
    
    generated_captions.append(generated_caption)
    reference_captions.append(example["image_text"])
    image_ids.append(example["image_id"])

# Initialize CIDEr metric
cider_metric = evaluate.load("cider")

# CIDEr requires a specific format:
# predictions = [{"prediction": "generated text", "id": "image_id"}]
# references = [[{"reference": "reference text", "id": "image_id"}]]
predictions = [{"prediction": c, "id": i} for c, i in zip(generated_captions, image_ids)]
references = [[{"reference": r, "id": i}] for r, i in zip(reference_captions, image_ids)]

# Compute the CIDEr score
cider_score = cider_metric.compute(predictions=predictions, references=references)

print("\n--- Final Results ---")
print(f"Total examples processed: {len(generated_captions)}")
print(f"CIDEr Score: {cider_score['cider']:.4f}")
