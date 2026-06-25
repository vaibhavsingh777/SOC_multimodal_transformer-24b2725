## Task 2: Transformer Architecture and Ablations

### Part E: Final Writeup

24. The Role of the MLP

While Multi-Head Attention acts as a communication mechanism (routing and mixing information across different token positions), the MLP acts as a computation mechanism. The MLP processes the gathered information independently within each individual token. It allows the model to "think" about the new context it just absorbed and transform those features into a higher-level representation before passing it to the next block.

25. Pre-norm vs. Post-norm

We used Pre-norm in our architecture, meaning LayerNorm is applied to the inputs before they go into the Attention and MLP blocks (x = x + self.attn(self.ln1(x))).
Pre-norm is significantly easier to train for deep networks because the residual connections (x + ...) remain completely unobstructed. This provides a clean, direct "highway" for gradients to flow straight from the final loss all the way back to the initial embeddings. Post-norm puts the normalization operation inside the residual path, which dampens the gradients at every layer and makes deep networks notoriously unstable to train.

26. Generated Text & Task 1 Comparison

Generated Text (300 Characters):

Till me withalks may mustle gentlemong be.

BENVOLIO:
In price is is love vourk a passions to day to he:
Fass his no more ingeldring death means:
Good some more breat leary of the sonse make out
On thire your receives deso words me to disconse,
And husber his no very for she stand munt man!

POMINIU

Qualitative Comparison:
Compared to the Bigram model from Task 1, the leap in quality is massive. Task 1 only had a context window of 1 character, so it produced structureless gibberish with roughly correct word lengths. This Transformer has a 64-character context window. As a result, it successfully learned structural formatting (capitalizing character names like BENVOLIO: followed by a newline), generated perfectly valid English words (passions, death, words, love), and strung them together in syntax patterns that closely mimic actual Shakespearean dialogue.

27. Ablation Catastrophes & Gradient Flow

Numerically, removing Residuals was far more catastrophic (loss stalled at ~2.56). Removing LayerNorm surprisingly matched the baseline (~1.72).

Removing Residuals breaks the gradient highway. Gradients are forced to multiply through every single weight matrix and non-linearity sequentially during backpropagation. This immediately causes the classic vanishing gradient problem, starving the early layers of any useful learning signal.

Removing LayerNorm normally breaks activation scaling, which saturates the softmax and kills gradients. However, because our network is only 4 layers deep, the activation variance did not have enough time to compound to fatal levels, allowing the model to survive.

28. Mentee Reflection

Hardest part: Deriving the Softmax Jacobian and attention gradients by hand. Wrangling the indices for the Kronecker delta and executing the summation expansions required a lot of careful matrix calculus.
What clicked unexpectedly: Analyzing the results of Variant 3. I expected removing LayerNorm to break the model entirely. Seeing it survive perfectly proved to me that architectural requirements are contextual—shallow networks can get away with missing stabilizers that would instantly break a massive LLM!
