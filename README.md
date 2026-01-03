# ANAIS_2025
I will upload my daily pratice Notebook practiced during 11-Days program conducted by ANAIS_2025

### DAY_1
We use a GIN-based message passing network to learn node embeddings. These embeddings are then combined in a decoder to predict whether an edge exists between two nodes. Negative sampling converts link prediction into a binary classification task. GIN is chosen because it matches the expressive power of the Weisfeiler–Lehman test.

### DAY_2
##### anais_gdl_p2_transformers
This notebook provides a practical introduction to Transformers, covering both foundational concepts and modern architectures. It begins with a theoretical refresh and setup, followed by a hands-on implementation of a vanilla Transformer based on the 'Attention Is All You Need' paper, demonstrating its ability to perform a sequence copying task using Multi-head Self-Attention and Positional Embeddings. The notebook then progresses to building a modern GPT-2 style Transformer from scratch, focusing on its decoder-only architecture, pre-LayerNorm for stability, and causal masking for autoregressive generation, which is applied to a text generation task.

#### Generative Models Lab
This notebook provided a practical introduction to generative models, focusing on two main areas: Flow Matching with TorchCFM and the mathematical foundations of generative processes. In the first part, we debugged and implemented a flow-matching model to learn a vector field mapping a source distribution (e.g., 8-component Gaussian mixture) to a target distribution (e.g., two moons or Swiss roll), generating new samples through training a neural network to approximate the conditional flow. The second part delved into the mathematical underpinnings, exploring practical implementations of Gaussian sampling using the Box-Muller transform, understanding uniform random number generation, and investigating Stochastic Differential Equations (SDEs). We learned about the Euler-Maruyama integrator, its components like drift and diffusion, and observed how noise affects particle trajectories and long-time behavior in SDE simulations. Finally, we touched upon score matching basics, understanding the necessity and methods for estimating score functions in generative models.


### Day_3
No lab

### Day_4 
Hike Day

### Day_5
#### Agents lab-ANAIS
This notebook introduces the concept of LLM-powered agents by building a flight booking agent. It covers setting up the environment, defining data classes for Date, UserProfile, and Flight, populating databases with user and flight information, and then implementing a set of tools (get_user_info, fetch_flight_info, book_flight, file_ticket). Finally, it demonstrates an agent loop that uses these tools to process user queries related to flight booking.

#### nanoGPT-implementation-ANAIS25
This notebook explores the fundamentals of building a GPT-like language model, starting with data preparation and moving into the core architectural components. Here's a brief summary:
Data Preparation: It begins by downloading and exploring a text dataset (Tiny Shakespeare). It then creates a character-level tokenizer, encodes the text into numerical tensors, and splits the data into training and validation sets.
Bigram Language Model: A simple Bigram Language Model is implemented using nn.Embedding in PyTorch. This model predicts the next character based only on the current character. It demonstrates the training loop and text generation from this basic model.
Self-Attention Mechanism: The notebook then dives into the mathematical concepts behind self-attention, explaining how weighted aggregation is performed using matrix multiplication. It illustrates the steps to build a self-attention head, including query, key, and value vectors, masked attention, and softmax. It also discusses the importance of 'scaled attention'.
Layer Normalization: Finally, it includes a simple implementation of Layer Normalization, a crucial component often used in transformer architectures to stabilize training.

### Day_6
#### LAB_combined_final
This notebook focuses on Panoptic Segmentation, a task that combines instance and semantic segmentation. It starts by visualizing dataset samples and then proceeds to load and set up a Mask2Former model for panoptic segmentation.
The notebook defines custom functions for:
Panoptic Inference: run_panoptic_inference which processes images through the model and applies proper post-processing to obtain panoptic segmentations, including handling 'thing' and 'stuff' classes.
PQ Computation: compute_pq_detailed which calculates Panoptic Quality (PQ) with a breakdown for 'thing' and 'stuff' classes.
Evaluation: evaluate_panoptic_pq which automates the evaluation of the panoptic model on a dataset using the detailed PQ metric.
Finally, it sets up the training environment for the panoptic model, including data loaders, optimizer, and learning rate scheduler, and then runs a training loop with periodic evaluation and saving of the best model based on validation PQ. The notebook also includes code to plot the training and validation curves.

