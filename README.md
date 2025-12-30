# ANAIS_2025
I will upload my daily pratice Notebook practiced during 11-Days program conducted by ANAIS_2025

### DAY_1
We use a GIN-based message passing network to learn node embeddings. These embeddings are then combined in a decoder to predict whether an edge exists between two nodes. Negative sampling converts link prediction into a binary classification task. GIN is chosen because it matches the expressive power of the Weisfeiler–Lehman test.

### DAY_2
##### anais_gdl_p2_transformers
This notebook provides a practical introduction to Transformers, covering both foundational concepts and modern architectures. It begins with a theoretical refresh and setup, followed by a hands-on implementation of a vanilla Transformer based on the 'Attention Is All You Need' paper, demonstrating its ability to perform a sequence copying task using Multi-head Self-Attention and Positional Embeddings. The notebook then progresses to building a modern GPT-2 style Transformer from scratch, focusing on its decoder-only architecture, pre-LayerNorm for stability, and causal masking for autoregressive generation, which is applied to a text generation task.

#### Generative Models Lab
This notebook provided a practical introduction to generative models, focusing on two main areas: Flow Matching with TorchCFM and the mathematical foundations of generative processes. In the first part, we debugged and implemented a flow-matching model to learn a vector field mapping a source distribution (e.g., 8-component Gaussian mixture) to a target distribution (e.g., two moons or Swiss roll), generating new samples through training a neural network to approximate the conditional flow. The second part delved into the mathematical underpinnings, exploring practical implementations of Gaussian sampling using the Box-Muller transform, understanding uniform random number generation, and investigating Stochastic Differential Equations (SDEs). We learned about the Euler-Maruyama integrator, its components like drift and diffusion, and observed how noise affects particle trajectories and long-time behavior in SDE simulations. Finally, we touched upon score matching basics, understanding the necessity and methods for estimating score functions in generative models.
