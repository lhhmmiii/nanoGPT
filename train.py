def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    for epoch in range(num_epochs):
        model.train()

        total_loss = 0.0

        for x, y in train_loader:
            # Move data to the specified device
            x = x.to(device)
            y = y.to(device)
            # Forward
            logits = model(x)

            # logits: (B, T, V)
            # y:      (B, T)
            B, T, V = logits.shape

            loss = criterion(logits.view(-1, V), y.view(-1))

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Loss: {avg_loss:.4f}"
        )
        