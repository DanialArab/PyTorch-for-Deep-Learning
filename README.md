# PyTorch for Deep Learning

## Course 1: PyTorch: Fundamentals

## Course 2: PyTorch: Techniques and Ecosystem Tools

## Course 3: PyTorch: Advanced Architectures and Deployment

## PyTorch: Fundamentals

![](https://github.com/DanialArab/images/blob/main/PyTorch-for-Deep-Learning/single_neuron.png)

![](https://github.com/DanialArab/images/blob/main/PyTorch-for-Deep-Learning/import_libraries.png)

Some notes on the training in PyTorch:

    # Training loop
    for epoch in range(500):
        # Reset the optimizer's gradients
        optimizer.zero_grad()
        # Make predictions (forward pass)
        outputs = model(distances)
        # Calculate the loss
        loss = loss_function(outputs, times)
        # Calculate adjustments (backward pass)
        loss.backward()
        # Update the model's parameters
        optimizer.step()
        # Print loss every 50 epochs
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}: Loss = {loss.item()}")
        
- optimizer.zero_grad(): Clears gradients from the previous round. Without this, PyTorch would accumulate adjustments, which could break the learning process.
- outputs = model(distances): Performs the "forward pass", where the model makes predictions based on the input distances.
- loss = loss_function(outputs, times): Calculates how wrong the predicted outputs are by comparing them to the actual delivery times.
- loss.backward(): The "backward pass" (backpropagation) is performed, which calculates exactly how to adjust the weight and bias to reduce the error.
- optimizer.step(): Updates the model's parameters using those calculated adjustments.
- The loss is printed every 50 epochs to allow you to track the model's learning progress as the error decreases.
