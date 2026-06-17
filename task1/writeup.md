### 12. Why do we divide attention scores by $\sqrt{d_k}$? Connect your answer to the variance of dot products and to the shape of the softmax.

We divide by $\sqrt{d_k}$ to stabilize the gradients during training. When we calculate the attention scores by taking the dot product of the Query ($Q$) and Key ($K$) vectors, the variance of that dot product scales linearly with the dimension $d_k$. If $Q$ and $K$ have a mean of 0 and a variance of 1, their dot product will have a variance of $d_k$.

For large dimensions, this results in attention scores with extreme magnitudes (very large positive or negative numbers). When passed through the softmax function, these large values cause the distribution to become extremely "peaked" (where one position gets a probability of almost 1.0, and the rest get 0.0). A heavily peaked softmax has extremely small gradients, which causes learning to stall (the vanishing gradient problem). Dividing the scores by $\sqrt{d_k}$ normalizes the variance back to roughly 1, ensuring the softmax outputs a smoother distribution and gradients flow properly.

---

### 13. The causal mask is applied before softmax, by setting future-position scores to negative infinity. What would happen if instead you applied a mask after softmax by zeroing out those entries? Why is the before-softmax approach correct?

If we applied the mask _after_ the softmax by simply zeroing out future probabilities, our probability distribution would break. The softmax function ensures that all outputs sum to exactly 1.0. If we zero out certain elements after the fact, the remaining probabilities will sum to less than 1.0, violating the fundamental properties of a probability distribution.

Furthermore, applying the mask after softmax means that information from future tokens would "leak" into the denominator of the softmax calculation. The probabilities of the valid past tokens would be artificially squashed because the softmax had already accounted for the future tokens when calculating the total sum. By applying the mask _before_ softmax (setting future scores to $-\infty$), $e^{-\infty}$ evaluates to exactly 0. This completely removes the future tokens from the softmax equation, ensuring the remaining valid tokens correctly sum to 1.0 without any forward-looking leakage.

---

### 14. Describe Q, K, and V in your own words. An analogy is fine — but also give the linear-algebra view. What is each linear projection doing?

**The Analogy:**
Imagine you are searching a database. The **Query (Q)** is what you type into the search bar (what information the current token is looking for). The **Key (K)** is the title and tags of every document in the database (what information each past token contains). The model computes the similarity between your Query and all the Keys. Once it finds the best matches, it returns the actual text of those documents — this is the **Value (V)**.

**The Linear Algebra View:**
In code, $Q$, $K$, and $V$ are created by taking the input embedding matrix $X$ (which contains the raw token representations) and multiplying it by three separate, learned weight matrices ($W_q$, $W_k$, $W_v$).

- The $Q$ projection maps the token into an "attention space" optimized for asking questions.
- The $K$ projection maps the token into the same space, but optimized for advertising its contents.
- The $V$ projection maps the token into a space representing the actual informational content it should pass forward to the next layer if it is selected by the attention mechanism.

---

### 15. Your single-head attention model only marginally outperforms the bigram. Why? What is the bottleneck — capacity, context length, depth, or something else?

The bottleneck here is a combination of **capacity** and **depth**, with a strong emphasis on the lack of a Feed-Forward Network (MLP).

While a self-attention head allows tokens to communicate and gather context from previous positions (up to the context length of 8), attention is fundamentally just a routing mechanism — it moves information around. On its own, a single attention head with an embedding dimension of 32 simply lacks the parameters (capacity) to deeply understand or "compute" on the aggregated information. In a real Transformer, the attention mechanism gathers the context, but the dense multi-layer perceptrons (MLPs) that follow it are what actually process that context into complex grammatical understanding. Without depth (stacking multiple blocks) and without MLPs, the model hits a severe performance ceiling.

---

### 16. Paste 200 characters of generated text from the bigram model and 200 from the attention model. Describe the qualitative difference.

**Bigram Output:**

> LIZAntaitoupis!
> BENIngt,
> N tiel, serhe hill: h wous sal ayolf sthereeyowoulour: horgonof m
> sunicour,
> ANLOurak anominfaind oul bond f DIC:
> O g.
> Gr IOLouspold se.
> Dotamy t Y mioke om, d a he ates,

**Attention Output:**

> Ano' ou sene od wistwin, wildalle adery wheerel cro ove.
> K:
> OMInciove dan uts wat fo stu bur,
> Wer
> my seng
> Becthikure
> Pe.
> INCECLIUS: MId miombellavo focen soun ers.
> A:
> Way berup he past arr:
> A sg wo

**Qualitative Difference:**
Both outputs are essentially gibberish, but the Attention model demonstrates a slightly better grasp of English word structure. The Bigram model frequently outputs long, unpronounceable strings of characters (like `sthereeyowoulour` or `anominfaind`) because it only plans one letter at a time without looking at the whole word. The Attention model's output features character groupings that look closer to actual English syllables and words (`wistwin`, `wildalle`, `wheerel`). It is beginning to learn that certain combinations of vowels and consonants belong together over a slightly longer context window.
