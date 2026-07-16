"""Theory registry: plain-language + math explanations for every operation.

Extensible: register new entries with register_theory(); lookup falls back
through class-name aliases to a generic entry, so unknown layer types still
get a sensible panel. Formulas are written in a lightweight math notation
the frontend renders as-is (unicode), keeping the app fully self-contained.
"""

from __future__ import annotations

_REGISTRY: dict[str, dict] = {}
_ALIASES: dict[str, str] = {}


def register_theory(key: str, entry: dict, aliases: list = ()):
    entry = {"key": key, **entry}
    _REGISTRY[key] = entry
    for a in aliases:
        _ALIASES[a] = key


def get_theory(class_name: str) -> dict:
    key = class_name if class_name in _REGISTRY else _ALIASES.get(class_name)
    if key is None:
        # substring fallback: "Conv2d" matches "LazyConv2d", GPT2's "Conv1D" etc.
        for k in _REGISTRY:
            if k.lower() in class_name.lower():
                key = k
                break
    if key is None:
        return {**_REGISTRY["_generic"], "requested": class_name}
    return _REGISTRY[key]


def all_theory() -> dict:
    return _REGISTRY


register_theory("_generic", {
    "title": "Neural network module",
    "formula": "y = f(x; θ)",
    "what": "A parameterized function: it takes an input tensor x, applies a "
            "transformation using its learned parameters θ, and produces an "
            "output tensor y. This specific layer type isn't in the theory "
            "library yet, but the same rules apply: forward computes y, and "
            "backward computes how the loss changes with respect to both x "
            "(passed upstream) and θ (used to update the weights).",
    "why": "Deep networks are compositions of many simple differentiable "
           "functions. Any module — however exotic — fits the same contract: "
           "differentiable in, differentiable out.",
    "gradient": "By the chain rule, ∂L/∂x = ∂L/∂y · ∂y/∂x and "
                "∂L/∂θ = ∂L/∂y · ∂y/∂θ. Autograd derives both automatically "
                "from the operations used in forward().",
})

register_theory("Linear", {
    "title": "Linear (fully-connected) layer",
    "formula": "y = xWᵀ + b        W: [out_features × in_features]",
    "what": "Every output value is a weighted sum of ALL input values plus a "
            "bias. The weight matrix W holds one row per output neuron; "
            "entry W[i,j] says how much input feature j contributes to "
            "output feature i.",
    "why": "This is the basic 'mixing' operation of neural networks — it can "
           "represent any linear map between feature spaces. Stacked with "
           "nonlinearities, linear layers are universal function "
           "approximators.",
    "gradient": "∂L/∂W = (∂L/∂y)ᵀ x — the gradient for a weight is (upstream "
                "gradient at its output) × (the input it saw). Inputs that "
                "were large get large weight updates. ∂L/∂x = (∂L/∂y) W "
                "passes the signal to earlier layers.",
}, aliases=["LazyLinear", "Conv1D"])  # HF GPT-2 uses Conv1D as a linear layer

register_theory("Conv2d", {
    "title": "2D convolution",
    "formula": "y[o, i, j] = Σ_c Σ_u Σ_v  W[o, c, u, v] · x[c, i+u, j+v] + b[o]",
    "what": "A small learned filter (kernel) slides across the image and "
            "computes a dot product at every position. Each output channel "
            "has its own set of filters — one per input channel — so the "
            "layer detects the same local pattern everywhere in the image.",
    "why": "Images have translation structure: an edge is an edge wherever "
           "it appears. Convolution shares weights across positions, which "
           "cuts parameters enormously versus a Linear layer and builds in "
           "translation equivariance.",
    "gradient": "∂L/∂W is itself a convolution: each filter's gradient is "
                "the correlation between the input patch it saw and the "
                "upstream gradient at the positions it produced. ∂L/∂x is a "
                "'transposed' convolution that routes gradients back to "
                "every input pixel each filter touched.",
}, aliases=["Conv1d", "Conv3d", "LazyConv2d"])

register_theory("BatchNorm2d", {
    "title": "Batch normalization",
    "formula": "y = γ · (x − μ_batch) / √(σ²_batch + ε) + β",
    "what": "Normalizes each channel to zero mean / unit variance using "
            "statistics of the current batch, then re-scales with learned "
            "γ and shifts with learned β. At eval time it uses running "
            "averages of μ and σ² collected during training.",
    "why": "Keeps activation distributions stable as earlier layers change "
           "during training ('internal covariate shift'), allowing higher "
           "learning rates and acting as a mild regularizer.",
    "gradient": "Gradients flow through both the normalization (coupling "
                "every sample in the batch, since each affects μ and σ²) "
                "and directly to γ, β: ∂L/∂γ = Σ ∂L/∂y · x̂, ∂L/∂β = Σ ∂L/∂y.",
}, aliases=["BatchNorm1d", "BatchNorm3d", "SyncBatchNorm"])

register_theory("LayerNorm", {
    "title": "Layer normalization",
    "formula": "y = γ ⊙ (x − μ) / √(σ² + ε) + β     (μ, σ² over the feature dim)",
    "what": "Like BatchNorm but statistics are computed per sample across "
            "the feature dimension, not across the batch. Every token/row "
            "is normalized independently, then scaled by learned γ and β.",
    "why": "Transformers process variable-length sequences with small or "
           "streaming batches, where batch statistics are unreliable. "
           "LayerNorm depends only on the sample itself, making it the "
           "standard normalizer in attention architectures.",
    "gradient": "∂L/∂γ = Σ ∂L/∂y ⊙ x̂ and ∂L/∂β = Σ ∂L/∂y. The gradient "
                "through x̂ subtracts its projections onto the mean and "
                "variance directions — LayerNorm gradients are 'centered'.",
}, aliases=["RMSNorm", "GroupNorm", "LlamaRMSNorm", "T5LayerNorm"])

register_theory("ReLU", {
    "title": "ReLU activation",
    "formula": "y = max(0, x)",
    "what": "Passes positive values through unchanged and clamps negatives "
            "to zero. The output is sparse: typically ~50% of units are "
            "'off' for a given input.",
    "why": "The nonlinearity is what lets stacked layers represent more "
           "than a single linear map. ReLU is cheap, and its gradient is "
           "exactly 1 for active units — no shrinking — which is why deep "
           "ReLU nets train much better than sigmoid/tanh nets ever did.",
    "gradient": "∂y/∂x = 1 if x > 0 else 0. Gradients pass untouched through "
                "active units and are completely blocked at inactive ones "
                "('dead ReLU' if a unit is never active).",
}, aliases=["ReLU6", "LeakyReLU", "ELU"])

register_theory("GELU", {
    "title": "GELU activation",
    "formula": "y = x · Φ(x)      Φ = standard normal CDF",
    "what": "A smooth version of ReLU: instead of a hard cut at 0, each "
            "value is scaled by the probability that a standard normal is "
            "below it. Small negatives get slightly through; large "
            "positives pass unchanged.",
    "why": "The smooth curve gives non-zero gradient everywhere (no dead "
           "units) and empirically trains better in transformers — GPT and "
           "BERT both use it.",
    "gradient": "∂y/∂x = Φ(x) + x·φ(x). Near zero this is ≈ 0.5, for large "
                "positive x it approaches 1, for large negative x it "
                "approaches 0 — a soft gate on the gradient.",
}, aliases=["GELUActivation", "NewGELUActivation", "SiLU", "Mish", "QuickGELUActivation"])

register_theory("Sigmoid", {
    "title": "Sigmoid activation",
    "formula": "y = 1 / (1 + e⁻ˣ)",
    "what": "Squashes any real number into (0, 1). Historically the default "
            "neuron activation; today mostly used to produce probabilities "
            "and gates (e.g. in LSTMs).",
    "why": "Its output is interpretable as a probability, which is exactly "
           "what you want at a binary-classification output or a gate that "
           "decides 'how much of this signal passes'.",
    "gradient": "∂y/∂x = y(1−y), maximum 0.25 at x=0 and near zero for |x| "
                "large. Stacking sigmoids multiplies these small factors — "
                "the classic cause of vanishing gradients.",
})

register_theory("Tanh", {
    "title": "Tanh activation",
    "formula": "y = (eˣ − e⁻ˣ) / (eˣ + e⁻ˣ)",
    "what": "Squashes values into (−1, 1), zero-centered unlike sigmoid.",
    "why": "Zero-centered outputs keep the next layer's inputs balanced "
           "around 0, which conditions optimization better than sigmoid.",
    "gradient": "∂y/∂x = 1 − y². At most 1 (at x=0) and vanishing for "
                "saturated units — deep tanh stacks still suffer vanishing "
                "gradients.",
})

register_theory("Softmax", {
    "title": "Softmax",
    "formula": "yᵢ = eˣⁱ / Σⱼ eˣʲ",
    "what": "Turns a vector of raw scores (logits) into a probability "
            "distribution: all outputs positive, summing to 1. "
            "Exponentiation amplifies differences — the largest logit "
            "dominates.",
    "why": "Needed wherever the network must express 'a distribution over "
           "choices': the output of a classifier, next-token probabilities "
           "in an LM, and the attention weights inside every transformer "
           "block.",
    "gradient": "∂yᵢ/∂xⱼ = yᵢ(δᵢⱼ − yⱼ). Combined with cross-entropy loss "
                "this collapses to the famously clean ∂L/∂x = y − target.",
}, aliases=["LogSoftmax"])

register_theory("Embedding", {
    "title": "Embedding lookup",
    "formula": "y = W[index]      W: [num_embeddings × embedding_dim]",
    "what": "A learned lookup table: each discrete token ID selects one row "
            "of the weight matrix. The row IS the token's dense vector "
            "representation, learned end-to-end.",
    "why": "Neural networks operate on continuous vectors, not symbols. "
           "Embeddings let the model place tokens in a geometric space "
           "where similar meanings end up nearby — the foundation of every "
           "language model.",
    "gradient": "Only the rows that were actually looked up receive "
                "gradient: ∂L/∂W[i] = Σ over positions where token i "
                "appeared of the upstream gradient. All other rows are "
                "untouched this step.",
})

register_theory("MultiheadAttention", {
    "title": "Multi-head (self-)attention",
    "formula": "Attention(Q,K,V) = softmax(QKᵀ/√d_k)V,  with Q=xW_Q, K=xW_K, V=xW_V per head",
    "what": "Every position emits a query (what am I looking for?), a key "
            "(what do I contain?), and a value (what do I offer?). Each "
            "query is dotted against all keys; the softmax of those scores "
            "becomes the attention weights, and the output is the "
            "weight-averaged values. Multiple heads run this in parallel "
            "on different learned projections, then concatenate.",
    "why": "This lets every token gather information from any other token "
           "in one step, with data-dependent routing — unlike convolution "
           "(fixed local window) or recurrence (sequential bottleneck). "
           "The √d_k scaling keeps dot products from saturating softmax. "
           "Multiple heads let the model attend to different relationships "
           "(syntax, coreference, position) simultaneously.",
    "gradient": "Gradients flow through V directly (weighted by attention), "
                "and through the softmax into Q and K — so the model learns "
                "both what to fetch (values) and where to look (the "
                "query/key geometry).",
}, aliases=["GPT2Attention", "BertSelfAttention", "LlamaAttention", "Attention",
            "GPT2SdpaAttention", "BertSdpaSelfAttention", "SelfAttention"])

register_theory("Dropout", {
    "title": "Dropout",
    "formula": "y = x ⊙ m / (1−p),   m ~ Bernoulli(1−p)   (training only)",
    "what": "During training, randomly zeroes each element with probability "
            "p and rescales the rest so the expected value is unchanged. "
            "At eval time it is the identity — which is what you see here "
            "unless the model is in train mode.",
    "why": "Prevents co-adaptation: no unit can rely on a specific other "
           "unit being present, so the network learns redundant, robust "
           "features. Acts like training an ensemble of subnetworks.",
    "gradient": "The same mask applies on the way back: dropped units get "
                "zero gradient this step, surviving units get theirs "
                "scaled by 1/(1−p).",
})

register_theory("MaxPool2d", {
    "title": "Max pooling",
    "formula": "y[i,j] = max over the k×k window of x",
    "what": "Downsamples a feature map by keeping only the strongest "
            "activation in each window.",
    "why": "Shrinks spatial size (less compute downstream), and gives a "
           "small amount of translation invariance — the exact position "
           "within the window stops mattering.",
    "gradient": "Winner-take-all: the gradient flows only to the input "
                "element that was the maximum; every other element in the "
                "window gets zero.",
}, aliases=["AvgPool2d", "AdaptiveAvgPool2d", "AdaptiveMaxPool2d", "MaxPool1d"])

register_theory("Flatten", {
    "title": "Flatten",
    "formula": "y = reshape(x, [batch, −1])",
    "what": "Rearranges a multi-dimensional tensor into a flat vector per "
            "sample. No parameters, no arithmetic — just a view change.",
    "why": "Bridges convolutional feature maps (channels × height × width) "
           "to fully-connected layers, which expect flat vectors.",
    "gradient": "The gradient is reshaped back — numerically it passes "
                "through unchanged.",
}, aliases=["Unflatten", "Identity"])

register_theory("CrossEntropyLoss", {
    "title": "Cross-entropy loss",
    "formula": "L = −log softmax(logits)[target]",
    "what": "Measures how much probability the model assigned to the "
            "correct class/token. Perfect confidence in the right answer "
            "gives loss 0; assigning it probability p gives loss −log p.",
    "why": "It is the maximum-likelihood objective for classification: "
           "minimizing it makes the model's predicted distribution match "
           "the true one. Its interaction with softmax produces an "
           "exceptionally clean, well-scaled gradient.",
    "gradient": "∂L/∂logits = softmax(logits) − onehot(target) — literally "
                "'predicted probabilities minus truth'. Confidently wrong "
                "predictions get the largest gradients.",
}, aliases=["NLLLoss", "MSELoss"])

register_theory("_backprop", {
    "title": "Backpropagation",
    "formula": "∂L/∂θₗ = ∂L/∂yₙ · ∂yₙ/∂yₙ₋₁ ⋯ ∂yₗ₊₁/∂yₗ · ∂yₗ/∂θₗ   (chain rule)",
    "what": "One application of the chain rule, organized efficiently: "
            "starting from ∂L/∂output = 1, gradients flow backwards "
            "through every operation in reverse execution order. Each "
            "layer receives 'how the loss changes with my output', "
            "computes 'how the loss changes with my weights' (stored in "
            ".grad) and 'how the loss changes with my input' (passed "
            "further back).",
    "why": "Computing all N parameter gradients naively would cost N "
           "forward passes; backprop gets every gradient in roughly one "
           "backward pass — the algorithmic trick that makes deep "
           "learning trainable at all.",
    "gradient": "Watch the per-layer gradient magnitudes: if they shrink "
                "steadily toward the early layers, you are seeing "
                "vanishing gradients; if they blow up, exploding "
                "gradients. Both are visible in the gradient view here.",
})

register_theory("_gradient_descent", {
    "title": "Gradient descent & optimizers",
    "formula": "SGD: θ ← θ − η·∇L      Adam: θ ← θ − η·m̂/(√v̂ + ε)",
    "what": "The gradient points in the direction of steepest loss "
            "increase, so stepping against it decreases the loss. SGD "
            "applies the raw gradient scaled by the learning rate η. Adam "
            "keeps running averages of the gradient (m, momentum) and its "
            "square (v), so each parameter gets its own adaptive step "
            "size — parameters with consistently large gradients step "
            "smaller, rare-but-important ones step larger.",
    "why": "The loss surface of a deep net is high-dimensional and "
           "non-convex; following the local downhill direction, one small "
           "step at a time over many batches, is the only tractable "
           "strategy — and empirically it finds excellent minima.",
    "gradient": "In the before/after diff, SGD updates are exactly "
                "−η·grad, so update size is proportional to gradient "
                "size. With Adam the first step is ≈ −η·sign(grad): "
                "roughly the SAME magnitude for every parameter — compare "
                "the two and you can see the normalization directly.",
})
