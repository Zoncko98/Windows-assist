import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig
from datasets import Dataset

# Load GPT-Neo model and tokenizer
model_name = "EleutherAI/gpt-neo-1.3B"
model = GPTNeoForCausalLM.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Check device and move model accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


### Function 1: Generate a Response ###
def generate_response(prompt, external_data=""):
    """
    Generate GPT-Neo responses augmented with additional live data content.

    :param prompt: User command or query as a string.
    :param external_data: Optional external data (e.g., live web data, Wikipedia).
    :return: Generated response as a string.
    """
    # Create the input context for GPT-Neo
    input_text = (
        f"User Query: {prompt}\n\n"
        f"Live Data: {external_data}\n\n"
        f"Assistant Response:"
    )
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    # Generate the response using GPT-Neo
    output = model.generate(
        input_ids,
        max_length=300,  # Maximum length of generated tokens
        num_return_sequences=1,
        temperature=0.7,  # Sampling diversity
        no_repeat_ngram_size=2,  # Avoid repetitive text
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response.split("Assistant Response:")[-1].strip()  # Remove prompt context


### Function 2: Prepare Training Data ###
def prepare_training_data(feedback_log):
    """
    Prepare runtime feedback into Hugging Face `Dataset` format for training.

    :param feedback_log: List of dictionaries containing feedback (prompt, response, correction).
    :return: Hugging Face Dataset object.
    """
    # Separate inputs and expected outputs for tokenization
    input_texts = [entry["input_text"] for entry in feedback_log]
    target_outputs = [entry["expected_output"] for entry in feedback_log]

    # Tokenize inputs and labels
    inputs = tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt")
    outputs = tokenizer(target_outputs, padding=True, truncation=True, return_tensors="pt")

    # Return dataset dictionary
    return Dataset.from_dict({
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": outputs["input_ids"],
    })


### Function 3: Fine-Tune the Model ###
def fine_tune_model(dataset):
    """
    Fine-tune GPT-Neo based on user feedback dataset.

    :param dataset: A Hugging Face Dataset object prepared via `prepare_training_data`.
    """
    # Set up standard training arguments for fine-tuning
    training_args = TrainingArguments(
        output_dir="./fine_tuned_model",  # Directory for the fine-tuned model
        num_train_epochs=1,  # Number of epochs (can adjust as needed)
        per_device_train_batch_size=1,  # Small batch size (for memory efficiency)
        save_steps=10,  # Save progress during training
        save_total_limit=2,  # Number of model checkpoints to keep
        logging_dir="./logs",
        logging_steps=5,  # Log every 5 steps
        learning_rate=5e-5,  # Standard fine-tuning rate
        evaluation_strategy="no",  # No explicit evaluation (can be added if needed)
    )

    # Initialize the Hugging Face Trainer class
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,  # Prepared dataset
    )

    # Train and save the fine-tuned model
    print("Starting fine-tuning...")
    trainer.train()
    model.save_pretrained("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")
    print("Fine-tuning complete! Model saved to './fine_tuned_model'.")


### Function 4: LoRA Fine-Tuning ###
def apply_lora_optimization(dataset):
    """
    Apply LoRA (Low-Rank Adaptation) to optimize GPT-Neo for lightweight fine-tuning.

    :param dataset: A Hugging Face Dataset created via `prepare_training_data`.
    """
    # Configure LoRA
    lora_config = LoraConfig(
        r=8,  # Low-rank update dimension
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],  # Parts of the transformer to target
        lora_dropout=0.1,  # Dropout for LoRA-specific weights
    )

    # Wrap model with LoRA
    lora_model = get_peft_model(model, lora_config)
    print("LoRA model configured for lightweight updates.")

    # Training arguments specific to LoRA
    training_args = TrainingArguments(
        output_dir="./lora_optimized_model",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        save_steps=10,
        save_total_limit=2,
        logging_dir="./logs",
    )

    # Train and save LoRA-fine-tuned model
    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=dataset,
    )
    print("Starting LoRA optimization...")
    trainer.train()
    lora_model.save_pretrained("./lora_optimized_model")
    print("LoRA optimization complete! Lightweight model saved to './lora_optimized_model'.")


### Function 5: Load an Improved Model ###
def load_improved_model(model_path="./fine_tuned_model"):
    """
    Dynamically load an improved fine-tuned or LoRA-optimized model.

    :param model_path: Directory containing the improved GPT-Neo model.
    :return: A fine-tuned or optimized model if available, otherwise the original model.
    """
    try:
        if os.path.exists(model_path):
            # Load the improved or fine-tuned model
            improved_model = GPTNeoForCausalLM.from_pretrained(model_path)
            improved_model.to(device)
            print(f"Loaded improved model from {model_path}.")
            return improved_model
        else:
            print(f"No fine-tuned model found at {model_path}. Using the original model.")
            return model
    except Exception as e:
        print(f"Error loading improved model: {str(e)}. Using the original model.")
        return model