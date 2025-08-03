# inference_with_cider.py
# This script uses a fine-tuned BLIP model to generate a caption for a single image,
# and then calculates the CIDEr score against a provided reference caption.

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel
from PIL import Image
import evaluate
import argparse
import os

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

# --- 2. ARGUMENT PARSER ---

# Setup an argument parser to take the image path and reference caption from the command line.
parser = argparse.ArgumentParser(description="Generate a caption and calculate CIDEr for a single image.")
parser.add_argument("--image_path", type=str, required=True, help="Path to the image file to be captioned.")
parser.add_argument("--reference_caption", type=str, required=True, help="The ground-truth caption for the image.")
args = parser.parse_args()

# --- 3. MODEL AND ADAPTER LOADING ---

# Load the base model with 4-bit quantization
print("Loading base model...")
try:
    base_model = BlipForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
except Exception as e:
    print(f"Error loading base model: {e}")
    exit()

# Load the processor (tokenizer and image processor)
processor = BlipProcessor.from_pretrained(MODEL_NAME)

# Load the fine-tuned adapters
print("Loading and merging QLoRA adapters...")
try:
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    # Merge the adapters into the base model for a consolidated model for inference
    model = model.merge_and_unload()
    model.eval()
except Exception as e:
    print(f"Error loading adapters: {e}")
    print(f"Please ensure the adapter is saved at '{ADAPTER_PATH}' and is compatible with the base model.")
    exit()

print("Model and adapters loaded successfully.")

# --- 4. INFERENCE FUNCTION ---

def generate_caption(image_path, model, processor):
    """
    Generates a caption for a given image path.
    Args:
        image_path (str): The local path of the image.
    Returns:
        str: The generated caption.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at '{image_path}'.")
        return None
        
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

    # Preprocess the image
    inputs = processor(images=image, return_tensors="pt").to(model.device, model.dtype)

    # Generate the caption using the fine-tuned model
    with torch.no_grad():
        out = model.generate(**inputs, max_length=50)

    # Decode the generated tokens back to a string
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# --- 5. RUN INFERENCE AND EVALUATION ---

print("\nStarting inference and evaluation...")

# Generate a caption for the single image
generated_caption = generate_caption(args.image_path, model, processor)

if generated_caption:
    print(f"\nImage Path: {args.image_path}")
    print(f"  Generated Caption: {generated_caption}")
    print(f"  Reference Caption: {args.reference_caption}")

    # Initialize CIDEr metric
    cider_metric = evaluate.load("cider")

    # CIDEr requires a specific format for predictions and references.
    # We will create single-item lists for the single image.
    # For CIDEr, the reference must be a list of lists of dictionaries.
    predictions = [{"prediction": generated_caption, "id": "user_image"}]
    references = [[{"reference": args.reference_caption, "id": "user_image"}]]
    
    # Compute the CIDEr score
    cider_score = cider_metric.compute(predictions=predictions, references=references)

    print("\n--- Final Results ---")
    print(f"CIDEr Score: {cider_score['cider']:.4f}")
else:
    print("\nInference failed. CIDEr score cannot be calculated.")
