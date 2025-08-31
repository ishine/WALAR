import torch
import numpy as np
import tqdm
from tqdm import *
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def load_flores(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]


def load_model_and_tokenizer(model_path):
    try:
        # Load the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map='auto', # Automatically select device (GPU if available)
            torch_dtype=torch.float16 # Use half-precision for memory efficiency
        )
        # Set padding token if it's not already set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id

    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have accepted the Llama-2 license on Hugging Face and are logged in.")
        exit()
    model.eval()
    print("Model loaded successfully.")
    return tokenizer, model

def load_instructions(language_list):
    flores_path = "/mnt/gemini/data1/yifengliu/data/flores101_dataset/devtest"
    instructions, languages = [], []
    for lang in language_list:
        flores = load_flores(f"{flores_path}/{lang}.devtest")
        instructions.extend(flores)
        languages.extend([lang]*len(flores))
    return instructions, languages

def extract_representation(tokenizer, model, instructions):
    sentence_representations = []

    print("\nExtracting sentence representations...")
    with torch.no_grad(): # Disable gradient calculation for inference
        for i in tqdm(range(len(instructions))):
            instruction = instructions[i]
            # Tokenize the instruction
            inputs = tokenizer(instruction, return_tensors="pt", padding=True, truncation=True).to(model.device)
            
            # Get model outputs, including hidden states
            outputs = model(**inputs, output_hidden_states=True)
            
            # Get the last hidden state from the final layer
            last_hidden_states = outputs.hidden_states[-1]
            
            # Following the described method, use the hidden state of the LAST token
            # Shape: [batch_size, sequence_length, hidden_size]
            # We select the last token's representation: [:, -1, :]
            sentence_vector = last_hidden_states[:, -1, :].cpu().numpy()
            
            sentence_representations.append(sentence_vector)
    high_dim_vectors = np.concatenate(sentence_representations, axis=0)
    print(f"Extraction complete. Shape of representations: {high_dim_vectors.shape}")
    return high_dim_vectors

def apply_tsne(high_dim_vectors):
    print("\nApplying t-SNE to reduce dimensions to 2D...")
    tsne = TSNE(
        n_components=2,      # Reduce to 2 dimensions
        perplexity=50,        # A good value for small datasets
        random_state=42,     # For reproducibility
        max_iter=1000,         # Number of iterations for optimization
        init='pca',          # Initialize with PCA for stability
        learning_rate='auto'
    )
    low_dim_vectors = tsne.fit_transform(high_dim_vectors)
    print(f"t-SNE complete. Shape of reduced representations: {low_dim_vectors.shape}")
    return low_dim_vectors

if __name__ == "__main__":
    model_name = "checkpoint"
    path_dict = {
        "Qwen3-4B": "/mnt/gemini/data1/yifengliu/model/Qwen3-4B",
        "checkpoint": "/mnt/gemini/data1/yifengliu/checkpoints/New-Align-Rule-Detect-MetricX-Qwen3-4B-en-mix-mid2-1M-bsz128/global_step580_hf",
    }
    language_list = ["eng", "ltz", "mkd","pol","srp","slk","slv","ben","guj","hin", "mar", "pan", "hye", "ell", "lav", "lit", "fas", "tgl", "jav", "tur", "tam", "fin"]
    tokenizer, model = load_model_and_tokenizer(path_dict[model_name])
    instructions, languages = load_instructions(language_list)
    language_map = {lang: i for i, lang in enumerate(language_list)}
    # language_map = {src_lang: 0, tgt_lang: 1}
    colors = [language_map[lang] for lang in languages]
    high_dim_vectors = extract_representation(tokenizer, model, instructions)
    low_dim_vectors = apply_tsne(high_dim_vectors)
    
    print("\nGenerating plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create a scatter plot
    scatter = ax.scatter(
        low_dim_vectors[:, 0], 
        low_dim_vectors[:, 1], 
        c=colors, 
        cmap='viridis', # Color map for different languages
        alpha=0.8,
        s=100 # Marker size
    )

    # Add a legend
    legend_labels = list(language_map.keys())
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w',
                                markerfacecolor=scatter.cmap(scatter.norm(language_map[label])), 
                                markersize=10) for label in legend_labels]
    ax.legend(legend_handles, legend_labels, title="Languages")

    # Add titles and labels
    ax.set_title(f"t-SNE Visualization of {model_name} Sentence Representations", fontsize=16)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.grid(True)

    # Save and show the plot
    plt.savefig(f"tsne_{model_name}_visualization.png", dpi=300)
    plt.show()
    
    print(f"\nPlot saved as 'tsne_{model_name}_visualization.png'")