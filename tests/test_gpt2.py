import torch
import tiktoken

from models.gpt2 import GPT2Config, GPT2


config = GPT2Config()
# model = GPT2.from_pretrained('gpt2')
model = GPT2.load_checkpoint('checkpoints/gpt2/scratch/run1/best.pt', map_location='cpu')[0]
sample_test = "Homarus gammarus is a large crustacean , with a body length up to 60"
tokenizer = tiktoken.get_encoding("gpt2")
input_ids = tokenizer.encode(sample_test)
input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
output = model.generate(input_tensor, max_new_tokens=10)
tokenized_output = tokenizer.decode(output.squeeze().tolist())
print(f"Input: {sample_test}")
print(f"Output: {tokenized_output}")