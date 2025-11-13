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

Some notes on the inference in PyTorch:

        # Use the torch.no_grad() context manager for efficient predictions
        with torch.no_grad():
            # Convert the Python variable into a 2D PyTorch tensor that the model expects
            new_distance = torch.tensor([[distance_to_predict]], dtype=torch.float32)
            
            # Pass the new data to the trained model to get a prediction
            predicted_time = model(new_distance)
            
            # Use .item() to extract the scalar value from the tensor for printing
            print(f"Prediction for a {distance_to_predict}-mile delivery: {predicted_time.item():.1f} minutes")
        
            # Use the scalar value in a conditional statement to make the final decision
            if predicted_time.item() > 30:
                print("\nDecision: Do NOT take the job. You will likely be late.")
            else:
                print("\nDecision: Take the job. You can make it!")

- The entire prediction process is wrapped in a with torch.no_grad() block.
    - This tells PyTorch you're not training anymore, just making a prediction. This makes the process faster and more efficient.
- A new input tensor is created using the distance_to_predict variable.
    - This must be formatted as a 2D tensor ([[7.0]]), as the model expects this specific structure, not a simple number.
- Your trained model is called with this new tensor to generate a predicted_time.
- After getting the prediction (which is also a tensor), the code extracts the actual numerical value from it using .item().

![](https://github.com/DanialArab/images/blob/main/PyTorch-for-Deep-Learning/activation_functions.png)

#### Tensors: The Core of PyTorch

You've seen that the journey of building a neural network begins with data. Before you can design a model or start the training process, you must gather your information and prepare it in a format the model can understand. In PyTorch, that fundamental format is the tensor. Tensors are more than just data **containers**; they are optimized for the mathematical operations that power deep learning.

Mastering tensors is a vital step. Many of the most common errors encountered when building models are related to tensor shapes, types, or dimensions. 

