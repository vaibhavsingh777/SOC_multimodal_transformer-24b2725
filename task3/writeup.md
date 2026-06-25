Task 3: From Pixels to Patches (Conceptual Writeup)

7. Compare your CNN baseline and your ViT on CIFAR-10.

Observation: Both models achieved a roughly similar validation accuracy (around 65-72%), but the CNN reached its peak much faster (in just 10 epochs) compared to the ViT (which required 30 epochs and a longer training time).

Explanation: CNNs possess strong inductive biases specifically designed for images. The sliding convolutional filters inherently assume translation equivariance (a feature is a feature regardless of where it is) and locality (nearby pixels form cohesive patterns). This allows the CNN to generalize efficiently even on small datasets like CIFAR-10 (50k images). ViTs have none of these built-in assumptions. The transformer treats the image as an un-ordered sequence of patches and has to learn spatial relations and translation invariance purely from data. Because CIFAR-10 is a relatively tiny dataset, the ViT is fundamentally bottlenecked by the lack of data to overcome its lack of inductive bias.

8. Why patching is necessary instead of feeding pixels directly

Attention has a computational complexity of $O(T^2)$ with respect to sequence length. Feeding a tiny 32x32 image pixel-by-pixel yields a sequence of $T = 1,024$. The attention matrix would require over a million pairwise scores per head, per layer. For a standard 224x224 image, it would be $T = 50,176$, requiring 2.5 billion operations, which is completely computationally intractable.

Furthermore, individual pixels lack semantic meaning. Extracting a $4\times4$ patch clusters 48 raw pixel values (16 pixels $\times$ 3 channels) into a single "token." Just like a word in a sentence, an image patch carries meaningful low-level structures (like edges, gradients, and color blobs) that the network can actually reason about and compose.

9. The role of the CLS token

The [CLS] token acts as a learned, global summary container for the entire image. Because it is prepended to the sequence and passed through the bidirectional self-attention blocks, it interacts equally with every patch in the image at every layer. By the final layer, its vector representation contains a highly condensed, global aggregation of the visual features.

The classifier reads only from the [CLS] token because it forces the network to route all class-relevant information into this single slot, preventing the model from overly relying on or biasing toward specific spatial patches (which might happen if we fed patch tokens directly into the linear head).

10. Removing the causal mask in ViT

In Task 2, we built an autoregressive language model where the goal was predicting the next token. The causal mask was mathematically required to prevent the model from "cheating" by looking at future words during training.

In Task 3, our goal is image classification. An image is fully observed all at once, and there is no strict temporal sequence (the bottom right of an image can provide vital context for the top left, and vice versa). Removing the mask allows bidirectional attention, enabling a global receptive field. If we accidentally kept the causal mask, the ViT would be artificially handicapped: patch 16 could look at patches 1-15, but patch 1 could only look at itself, completely destroying the model's ability to understand the image holistically.

11. What position embeddings for images encode

Because self-attention is purely a set operation (it is permutation invariant), a ViT without position embeddings would see the patches as a jumbled bag of features rather than a structured grid. Scrambling the image patches would yield the exact same output.

Position embeddings provide the vital spatial coordinate metadata (e.g., "I am patch #1 in the top left"). The model learns these embeddings during training, effectively learning to reconstruct the 2D geometry of the image inside the transformer's latent space, allowing it to recognize shapes and global structures.

12. Mentee Reflection

Hardest part: Wrapping my head around the Conv2d "trick" for patch embedding. Setting the kernel_size and stride both to patch_size effectively creates non-overlapping projections, collapsing extraction and linear embedding into a single elegant operation. Resolving the tensor shape transformations (flatten and transpose) to convert (B, C, H, W) to (B, Tokens, Embedding) was tricky.
What clicked unexpectedly: Seeing how mathematically identical the ViT block is to the language model block. Once the image is chopped into patches and flattened into (Batch, Sequence, Channels), the Transformer genuinely does not care that the data came from a JPEG instead of a text file.
