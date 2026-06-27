import torch

def generate_text(model, tokenizer, text, block_size, max_length=100):
    model.eval()

    device = next(model.parameters()).device

    # Encode input
    generated_indices = tokenizer.encode(text)

    with torch.no_grad():
        for _ in range(max_length):
            # Keep only the last block_size tokens
            input_ids = generated_indices[-block_size:]

            input_tensor = torch.tensor(
                input_ids,
                dtype=torch.long,
                device=device
            ).unsqueeze(0)

            # (1, seq_len, vocab_size)
            logits = model(input_tensor)

            # (vocab_size,)
            next_token_logits = logits[:, -1, :]

            # Greedy decoding
            next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), num_samples=1).item()

            # Append the predicted token to the generated sequence
            generated_indices.append(next_token)

    return tokenizer.decode(generated_indices)