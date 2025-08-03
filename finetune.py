import torch
from transformers import BitsAndBytesConfig, BlipProcessor, BlipForConditionalGeneration, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import Dataset

# --- 1. CONFIGURATION ---

# Define the base model and dataset information
MODEL_NAME = "Salesforce/blip-image-captioning-large"
DATASET_NAME = "iu_x_ray"  # The name of the dataset to be loaded
OUTPUT_DIR = "./blip-medical-captioning-qlora"

# QLoRA configuration for 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

# LoRA configuration for adapters
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["query", "value"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# --- 2. DATASET PREPARATION ---

# Note: This is a dummy dataset for demonstration.
# You will need to replace this with the actual IU X-Ray dataset
# and load it from Hugging Face Hub or a local path.
def create_dummy_data():
    """Generates a small, dummy dataset to make the script runnable."""
    data = {
        "image_path": [
            "path/to/image1.jpg",
            "path/to/image2.jpg",
        ],
        "text": [
            "The chest x-ray shows a nodule in the right upper lobe.",
            "Normal chest x-ray, with no evidence of acute cardiopulmonary disease.",
        ],
    }
    return Dataset.from_dict(data)

# Load the processor and model
processor = BlipProcessor.from_pretrained(MODEL_NAME)
model = BlipForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Function to preprocess the dataset
def preprocess_function(examples):
    """
    Tokenizes and preprocesses image-text pairs for model training.
    This function will need to be adapted for your specific dataset structure
    and image loading mechanism.
    """
    # Placeholder for image loading and processing
    # In a real scenario, you would load the images from disk here
    # and then pass them to the processor.
    images = [torch.randn(3, 384, 384) for _ in examples["text"]]

    # Process images and text
    inputs = processor(
        images=images,
        text=examples["text"],
        padding="max_length",
        return_tensors="pt"
    )
    # The labels are the same as the input text
    inputs["labels"] = inputs["input_ids"]
    return inputs

# Create and preprocess the dummy dataset
dummy_dataset = create_dummy_data()
processed_dataset = dummy_dataset.map(preprocess_function, batched=True)

# --- 3. PEFT MODEL SETUP ---

# Get the PEFT model with the LoRA adapters
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model.config.use_cache = False

# --- 4. TRAINING ARGUMENTS AND TRAINER ---

# Define training arguments based on the README.md
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,      # VRAM-friendly batch size
    gradient_accumulation_steps=4,      # Emulate a larger batch size of 4
    learning_rate=2e-4,
    num_train_epochs=5,
    logging_steps=50,
    save_steps=100,
    save_total_limit=1,
    fp16=True,                          # Use mixed precision training
    report_to="none",
    gradient_checkpointing=True,        # Further VRAM optimization
    lr_scheduler_type="cosine",
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
    # You would also include a `eval_dataset` here for validation
)

# --- 5. START TRAINING ---

print("Starting fine-tuning...")
trainer.train()
print("Fine-tuning completed!")

# --- 6. SAVE THE ADAPTERS ---

# Save only the LoRA adapters, not the entire base model
model.save_pretrained(OUTPUT_DIR)
print(f"LoRA adapters saved to {OUTPUT_DIR}")
