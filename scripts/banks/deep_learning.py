r"""Deep Learning — the ML depth round.

Weighted towards *why training goes wrong* rather than architecture trivia,
because that is what an Applied Scientist interview probes: anyone can name the
layers in a transformer, far fewer can say what they would check first when the
loss goes to NaN.
"""

from __future__ import annotations

ENTRIES = [

    # ------------------------------------------------------------------ 1 --- #

    (1, "what a neuron computes",
     "What does a single neuron in a neural network compute?",
     "A weighted sum of its inputs plus a bias, passed through a non-linear "
     "activation function. The weights and bias are learned; the activation is "
     "what stops the whole network collapsing into a single linear function.",
     ["a weighted sum of the inputs",
      "plus a bias term",
      "passed through an activation function",
      "the weights and bias are learned from data"],
     []),

    (1, "why non-linearity is needed",
     "Why do neural networks need non-linear activation functions?",
     "Because stacking linear layers gives you another linear function, no "
     "matter how many you stack. Without a non-linearity a hundred-layer network "
     "has exactly the representational power of a single layer. The "
     "non-linearity is what lets depth buy anything.",
     ["composing linear layers yields a linear function",
      "depth would add no representational power",
      "the network could only fit linear relationships",
      "non-linearity is what makes depth useful"],
     []),

    (1, "activation functions",
     "Why is ReLU the usual default activation?",
     "It is cheap to compute, and its gradient is one for positive inputs, so it "
     "does not squash gradients the way sigmoid and tanh do. That makes deep "
     "networks trainable. The cost is that neurons stuck on the negative side "
     "get no gradient at all and can die.",
     ["it is computationally cheap",
      "its gradient does not saturate for positive inputs",
      "it avoids the vanishing gradient problem of sigmoid and tanh",
      "units can die when stuck in the negative region"],
     ["What would you switch to if you saw dying units?"]),

    (1, "gradient descent variants",
     "What is gradient descent doing during training?",
     "It measures how the loss would change with a small change to each "
     "parameter, then nudges every parameter a little way in the direction that "
     "reduces the loss. Repeating that over many batches walks the parameters "
     "downhill towards a minimum.",
     ["it computes the gradient of the loss for each parameter",
      "parameters move in the direction that reduces loss",
      "the step size is the learning rate",
      "repeated over batches until the loss stops improving"],
     []),

    (1, "loss functions for classification",
     "What is a loss function?",
     "A single number measuring how wrong the model's predictions are on the "
     "data it has just seen. Training is the process of minimising it, so the "
     "loss is what actually defines what the model is trying to do — the "
     "objective is the loss, not your intention.",
     ["it quantifies how wrong the predictions are",
      "it produces a single number to minimise",
      "training minimises it via gradient descent",
      "it defines what the model actually optimises for"],
     []),

    (1, "batch size effects",
     "What's the difference between an epoch and a batch?",
     "A batch is the group of examples processed before one parameter update. "
     "An epoch is one complete pass over the whole training set, which is many "
     "batches. You update weights once per batch, not once per epoch.",
     ["a batch is the examples used for one update",
      "an epoch is one full pass over the training set",
      "an epoch contains many batches",
      "weights update once per batch"],
     []),

    (1, "overfitting in deep networks",
     "How can you tell a deep network is overfitting?",
     "The training loss keeps falling while the validation loss flattens and "
     "then starts to rise. That divergence is the signal. The point where "
     "validation loss turns is where you should have stopped.",
     ["training loss continues to fall",
      "validation loss stops improving or rises",
      "the gap between them widens",
      "the turning point is where training should stop"],
     ["What would you try first to fix it?"]),

    (1, "dropout",
     "What does dropout do?",
     "During training it randomly switches off a fraction of the units in a "
     "layer on every step, so the network cannot rely on any single unit and has "
     "to spread the representation out. It is turned off at inference time, when "
     "the full network is used.",
     ["it randomly disables units during training",
      "it stops the network relying on any single unit",
      "it acts as a regulariser",
      "it is disabled at inference time"],
     []),

    # ------------------------------------------------------------------ 2 --- #

    (2, "backpropagation",
     "What is backpropagation?",
     "It is the chain rule applied backwards through the network. A forward pass "
     "computes the loss, then gradients are propagated from the output back "
     "through each layer, reusing the downstream gradient at every step so the "
     "whole thing costs about as much as one forward pass.",
     ["it applies the chain rule backwards through the layers",
      "it computes the gradient of the loss for every parameter",
      "it reuses downstream gradients rather than recomputing",
      "it follows a forward pass that computes the loss"],
     []),

    (2, "learning rate and schedules",
     "What happens if the learning rate is too high, or too low?",
     "Too high and the updates overshoot the minimum, so the loss oscillates or "
     "diverges to NaN. Too low and training crawls, and can settle in a poor "
     "region because it never takes a step large enough to escape. It is the "
     "hyperparameter most worth tuning first.",
     ["too high causes oscillation or divergence",
      "too low makes training extremely slow",
      "a low rate can stall in a poor region",
      "it is usually the most important hyperparameter"],
     ["What does a learning rate schedule buy you?"]),

    (2, "cross-entropy versus mean squared error",
     "Why use cross-entropy rather than squared error for classification?",
     "Cross-entropy penalises confident wrong answers very heavily, and paired "
     "with a softmax its gradient is simply the predicted probability minus the "
     "target. Squared error gives tiny gradients when the model is confidently "
     "wrong, which is exactly when you need a large correction.",
     ["cross-entropy heavily penalises confident errors",
      "its gradient with softmax is simple and well behaved",
      "squared error gives small gradients when confidently wrong",
      "that slows or stalls learning"],
     []),

    (2, "the softmax function",
     "What does softmax do?",
     "It turns a vector of raw scores into a probability distribution — every "
     "value positive and the whole thing summing to one. It exponentiates each "
     "score and normalises, so larger scores dominate and the relative ordering "
     "is preserved.",
     ["it converts raw scores into a probability distribution",
      "outputs are positive and sum to one",
      "it exponentiates then normalises",
      "the ordering of the scores is preserved"],
     []),

    (2, "batch normalisation",
     "What does batch normalisation do, and why does it help?",
     "It normalises each layer's inputs using the statistics of the current "
     "batch, then rescales with learned parameters. It stabilises the "
     "distribution each layer sees, which lets you use higher learning rates and "
     "makes training less sensitive to initialisation.",
     ["it normalises activations using batch statistics",
      "it applies a learned scale and shift afterwards",
      "it allows higher learning rates",
      "it reduces sensitivity to weight initialisation"],
     ["Why does it behave differently at inference time?"]),

    (2, "the vanishing gradient problem",
     "What is the vanishing gradient problem?",
     "Gradients are multiplied together as they propagate backwards. If each "
     "factor is smaller than one, the product shrinks exponentially with depth, "
     "so early layers receive almost no gradient and stop learning. Saturating "
     "activations like sigmoid make it much worse.",
     ["gradients shrink multiplicatively going backwards",
      "early layers receive almost no gradient",
      "those layers stop learning",
      "saturating activations such as sigmoid worsen it"],
     ["What are the standard fixes?"]),

    (2, "transfer learning",
     "What is transfer learning, and why does it work?",
     "You start from a model already trained on a large dataset and adapt it to "
     "your task. It works because the early layers learn general features — "
     "edges, textures, basic language structure — that transfer across tasks, so "
     "you only have to learn what is specific to yours.",
     ["start from a model pretrained on a large dataset",
      "adapt it to a new, usually smaller task",
      "early layers learn general reusable features",
      "it needs far less data and compute than training from scratch"],
     ["When would transfer learning not help?"]),

    (2, "data augmentation",
     "What is data augmentation and when does it help?",
     "It expands the training set by applying transformations that change the "
     "input but not the label — flips and crops for images, paraphrasing for "
     "text. It helps most when data is limited, and it teaches the model which "
     "variations it is supposed to ignore.",
     ["it creates new training examples by transforming existing ones",
      "the transformation must preserve the label",
      "it helps most when data is limited",
      "it encodes the invariances you want the model to learn"],
     []),

    (2, "convolutional layers",
     "Why use a convolutional layer instead of a fully connected one on images?",
     "Because it shares one small set of weights across every position, which "
     "cuts the parameter count enormously and builds in the assumption that a "
     "pattern means the same thing wherever it appears. A fully connected layer "
     "would have to learn that separately at every pixel.",
     ["weights are shared across spatial positions",
      "far fewer parameters than a fully connected layer",
      "it builds in translation invariance",
      "each filter looks at a local receptive field"],
     []),

    # ------------------------------------------------------------------ 3 --- #

    (3, "gradient descent variants",
     "How does Adam differ from plain stochastic gradient descent?",
     "Adam keeps running averages of both the gradient and its square, and uses "
     "them to give every parameter its own effective step size. That makes it "
     "converge quickly with little tuning. Plain SGD with momentum often "
     "generalises slightly better but needs a carefully chosen schedule.",
     ["Adam keeps running averages of the gradient and its square",
      "it adapts the step size per parameter",
      "it converges fast with little tuning",
      "SGD with momentum can generalise better but needs more tuning"],
     []),

    (3, "residual connections",
     "What problem do residual connections solve?",
     "They give the gradient a path that skips the layers entirely, so it "
     "reaches early layers without being repeatedly multiplied down. That is "
     "what makes very deep networks trainable — before them, adding layers past "
     "a point made both training and test error worse.",
     ["they add a shortcut path around a block of layers",
      "gradients flow backwards without repeated multiplication",
      "they make very deep networks trainable",
      "each block learns a residual rather than a whole mapping"],
     []),

    (3, "attention",
     "What does an attention mechanism compute?",
     "For each position it scores how relevant every other position is, turns "
     "those scores into weights with a softmax, and returns a weighted sum of "
     "their values. So each position builds its representation from whichever "
     "parts of the input matter to it, rather than from a fixed window.",
     ["it scores relevance between positions",
      "scores become weights via softmax",
      "the output is a weighted sum of values",
      "each position can draw on any other position"],
     ["What are the queries, keys and values?"]),

    (3, "transformers",
     "Why did transformers displace recurrent networks for sequences?",
     "A recurrent network processes tokens one at a time, so training cannot be "
     "parallelised across the sequence and long-range information degrades over "
     "many steps. Self-attention connects every position to every other in one "
     "step and processes the whole sequence at once.",
     ["recurrence is inherently sequential and hard to parallelise",
      "attention processes all positions simultaneously",
      "any two positions are one step apart",
      "long-range dependencies degrade less"],
     ["What does that parallelism cost you?"]),

    (3, "layer normalisation",
     "When would you use layer normalisation instead of batch normalisation?",
     "When the batch statistics are unreliable or unavailable — small batches, "
     "variable-length sequences, or anything recurrent. Layer normalisation "
     "normalises across the features of a single example, so it does not depend "
     "on the batch at all and behaves identically at training and inference.",
     ["layer norm normalises across features within one example",
      "it does not depend on batch size",
      "it suits sequence models and small batches",
      "it behaves identically at training and inference time"],
     []),

    (3, "LSTM and GRU gating",
     "What problem does the gating in an LSTM solve?",
     "A plain recurrent network overwrites its hidden state at every step, so "
     "information from far back is lost and gradients vanish. Gates let the "
     "network decide what to keep, what to discard and what to expose, giving "
     "the cell state a nearly additive path through time.",
     ["a plain RNN overwrites its state each step",
      "gates control what is kept, forgotten and output",
      "the cell state carries information over long spans",
      "it mitigates vanishing gradients through time"],
     []),

    (3, "weight initialisation",
     "Why does weight initialisation matter?",
     "The scale of the initial weights determines whether activations and "
     "gradients grow or shrink as they pass through the layers. Too small and "
     "the signal vanishes with depth, too large and it explodes. Schemes like "
     "Xavier and He pick the variance so it stays roughly constant.",
     ["initial scale controls how signals propagate through depth",
      "too small a scale makes activations vanish",
      "too large a scale makes them explode",
      "Xavier and He set the variance to keep it stable"],
     []),

    (3, "exploding gradients and clipping",
     "Your loss suddenly becomes NaN during training. What do you check?",
     "Usually an exploding gradient or a numerical problem. I would lower the "
     "learning rate, add gradient clipping, and check for a log or division that "
     "can hit zero. I would also check the input data for NaNs and infinities "
     "before blaming the model.",
     ["suspect exploding gradients first",
      "reduce the learning rate",
      "apply gradient clipping",
      "check the input data and the loss for numerical issues"],
     []),

    # ------------------------------------------------------------------ 4 --- #

    (4, "batch size effects",
     "What changes when you increase the batch size?",
     "Gradient estimates get less noisy and the hardware is used better, so each "
     "epoch is faster. But fewer updates happen per epoch, so the learning rate "
     "usually has to rise to compensate, and very large batches tend to "
     "generalise slightly worse because the gradient noise itself was helping.",
     ["gradient estimates become less noisy",
      "throughput improves but there are fewer updates per epoch",
      "the learning rate usually needs to increase",
      "very large batches can generalise slightly worse"],
     []),

    (4, "batch normalisation",
     "Why does batch normalisation behave differently at inference?",
     "At training time it normalises using the current batch, which is only "
     "possible when you have a batch. At inference it uses running averages "
     "collected during training, so a single prediction is deterministic and "
     "does not depend on whatever else happened to be batched with it.",
     ["training uses statistics from the current batch",
      "inference uses running averages from training",
      "predictions must not depend on other batch members",
      "forgetting to switch modes causes wrong predictions"],
     []),

    (4, "distributed training",
     "How does data-parallel training work across several GPUs?",
     "Each device holds a full copy of the model and processes a different slice "
     "of the batch. Gradients are averaged across devices before the update, so "
     "every copy stays identical. The effective batch size is the sum, so the "
     "learning rate normally has to be scaled up.",
     ["each device holds a replica of the model",
      "each processes a different slice of the batch",
      "gradients are averaged across devices before updating",
      "the effective batch size grows, so the learning rate is scaled"],
     ["What do you do when the model itself does not fit on one device?"]),

    (4, "fine-tuning versus feature extraction",
     "When would you freeze the pretrained layers instead of fine-tuning them?",
     "When the new dataset is small. Fine-tuning everything on a few thousand "
     "examples overfits and destroys the pretrained features. Freezing the "
     "backbone and training only a head is the safe default, and you unfreeze "
     "progressively as you get more data.",
     ["freeze when the new dataset is small",
      "full fine-tuning on little data overfits",
      "it can destroy useful pretrained features",
      "unfreeze progressively as more data becomes available"],
     []),

    (4, "catastrophic forgetting",
     "What is catastrophic forgetting?",
     "When a network trained on a new task loses what it knew about an earlier "
     "one, because nothing in the new objective preserves the old behaviour and "
     "the weights simply move. It is the reason sequential fine-tuning degrades "
     "the original capabilities of a model.",
     ["a model loses earlier capabilities when trained on new data",
      "the new objective does not constrain old behaviour",
      "it affects sequential or continual fine-tuning",
      "mitigated by replaying old data or constraining weight change"],
     []),

    (4, "mixed precision training",
     "What does mixed precision training buy you, and what is the risk?",
     "Half precision halves memory and runs much faster on tensor cores, so you "
     "can fit bigger batches. The risk is numerical: small gradients underflow "
     "to zero in half precision, which is why loss scaling exists and why master "
     "weights are kept in full precision.",
     ["lower precision reduces memory and increases speed",
      "it allows larger batches or models",
      "small gradients can underflow to zero",
      "loss scaling and full-precision master weights address this"],
     []),

    # ------------------------------------------------------------------ 5 --- #

    (5, "overfitting in deep networks",
     "Your model trains perfectly but generalises badly. Walk me through it.",
     "I would confirm it is generalisation and not a broken evaluation split "
     "first. Then work down the cheap options: more data or augmentation, then "
     "regularisation through dropout and weight decay, then early stopping, then "
     "reducing capacity. I would also check the validation set actually "
     "represents production data.",
     ["first verify the evaluation split is sound",
      "add data or augmentation",
      "add regularisation such as dropout or weight decay",
      "use early stopping or reduce model capacity"],
     []),

    (5, "distributed training",
     "How would you train a model too large to fit on a single GPU?",
     "Start with the cheap wins — gradient checkpointing, mixed precision, "
     "gradient accumulation for the effective batch size. If the weights "
     "themselves still do not fit, shard the optimiser state and parameters "
     "across devices, or split the model by layer or by tensor across devices.",
     ["use gradient checkpointing to trade compute for memory",
      "use mixed precision and gradient accumulation",
      "shard optimiser state and parameters across devices",
      "or split the model itself with pipeline or tensor parallelism"],
     []),

    (5, "attention",
     "Why is self-attention expensive for long sequences, and what can be done?",
     "Every position attends to every other, so both compute and memory grow "
     "with the square of the sequence length. The usual answers are to restrict "
     "attention to a local window, to attend to a sparse or learned subset, or "
     "to use a kernel-based approximation that avoids forming the full matrix.",
     ["cost grows quadratically with sequence length",
      "both compute and memory are affected",
      "sparse or windowed attention limits the pairs considered",
      "linear or kernel approximations avoid the full matrix"],
     []),
    # --------------------------------------------------------------------- #
    # Added in the expansion pass: more of the ground floor and lower middle.
    # --------------------------------------------------------------------- #

    (1, "tensors",
     "What is a tensor?",
     "An array with any number of dimensions - a number, a vector, a matrix, or "
     "something with more axes than that. Everything a network handles is one, "
     "and most bugs in deep learning code are two tensors whose shapes do not "
     "line up.",
     ["an array with an arbitrary number of dimensions",
      "scalars, vectors and matrices are special cases",
      "the shape describes the size of each dimension",
      "shape mismatches are the common source of errors"],
     []),

    (1, "hidden layers",
     "What makes a layer a hidden layer?",
     "That it sits between the input and the output, so nothing outside the "
     "network sees its values. Those intermediate representations are where the "
     "useful learned features live - the network builds up from simple patterns "
     "to complex ones as depth increases.",
     ["it lies between the input and output layers",
      "its activations are not directly observed",
      "it holds learned intermediate representations",
      "deeper layers build on features from earlier ones"],
     []),

    (1, "training versus inference",
     "What's different about running a network at inference time?",
     "No gradients are computed and no weights change, so it needs far less "
     "memory and runs faster. Layers that behave differently in the two modes - "
     "dropout and batch normalisation - switch behaviour, and forgetting to "
     "switch is a classic bug.",
     ["no gradients are computed and no weights update",
      "memory use is much lower without activations stored",
      "dropout is disabled at inference",
      "batch normalisation uses stored running statistics"],
     []),

    (1, "why GPUs",
     "Why are GPUs used to train neural networks?",
     "Because training is mostly large matrix multiplications, and those are "
     "thousands of independent multiply-and-add operations. A GPU has thousands "
     "of simple cores that do exactly that in parallel, where a CPU has a few "
     "fast general ones.",
     ["training is dominated by large matrix operations",
      "those operations are highly parallel",
      "GPUs have many cores suited to that work",
      "high memory bandwidth also matters"],
     []),

    (2, "weight decay",
     "What does weight decay do?",
     "It adds a penalty proportional to the size of the weights, so training "
     "prefers smaller ones unless the data justifies otherwise. That limits how "
     "sharply the network can fit, which is the standard regulariser for deep "
     "models alongside dropout.",
     ["it penalises large weight values",
      "it pushes weights towards zero during training",
      "it reduces overfitting",
      "it is equivalent to L2 regularisation for plain gradient descent"],
     []),

    (2, "early stopping",
     "How does early stopping work?",
     "You watch validation loss and stop when it stops improving, keeping the "
     "weights from the best point rather than the last. It is regularisation "
     "that costs nothing, because the training you skip is the training that was "
     "overfitting.",
     ["monitor validation loss during training",
      "stop when it stops improving",
      "restore the weights from the best epoch",
      "it prevents the overfitting phase of training"],
     ["How long do you wait before deciding it has stopped improving?"]),

    (2, "input normalisation",
     "Why normalise the inputs before feeding them to a network?",
     "Because features on wildly different scales make the loss surface stretched "
     "in some directions and flat in others, so no single learning rate suits "
     "all of them. Normalising makes the surface better conditioned and training "
     "converges faster and more stably.",
     ["features on different scales distort the loss surface",
      "one learning rate cannot suit all directions",
      "normalising speeds up and stabilises convergence",
      "use the training statistics on validation and test data too"],
     []),

    (2, "learning rate schedules",
     "What does a learning rate schedule give you?",
     "Large steps early, when you are far from a minimum and want to cover "
     "ground, and small ones later, when you need to settle rather than bounce "
     "around it. A fixed rate has to compromise between the two, and decaying "
     "usually beats any single value.",
     ["large steps early to make progress quickly",
      "smaller steps later to converge precisely",
      "a fixed rate compromises between the two",
      "warmup avoids instability in the first steps"],
     []),

    (2, "loss plateaus",
     "The loss drops fast then flattens well above zero. What do you check?",
     "Whether it is the model or the setup. Too little capacity underfits, and a "
     "learning rate now too large bounces around the minimum. But I would also "
     "check the label noise floor and whether the task is even learnable from "
     "those features, because zero loss is often not achievable.",
     ["the model may lack capacity for the task",
      "the learning rate may now be too large",
      "there is an irreducible error floor from label noise",
      "check the features actually contain the signal"],
     []),

    (3, "gradient accumulation",
     "What is gradient accumulation and when do you need it?",
     "Running several small batches and summing their gradients before updating, "
     "so the update behaves like one large batch. You need it when the batch size "
     "you want does not fit in memory - it buys effective batch size at the cost "
     "of wall-clock time.",
     ["gradients from several small batches are summed",
      "the update happens once per accumulation cycle",
      "it simulates a larger effective batch size",
      "it trades time for memory"],
     []),

    (3, "embedding layers",
     "What does an embedding layer actually do?",
     "It is a lookup table mapping each discrete item to a learned dense vector. "
     "Mathematically it is a one-hot vector times a weight matrix, but "
     "implemented as an index lookup because everything else in that product is "
     "zero. The vectors are learned with the rest of the network.",
     ["it maps discrete items to dense learned vectors",
      "it is equivalent to a one-hot vector times a matrix",
      "implemented as a lookup for efficiency",
      "the vectors are trained along with the model"],
     []),

    (3, "pooling",
     "What is pooling for in a convolutional network?",
     "Shrinking the spatial size while keeping the strongest signal, which cuts "
     "compute for later layers and widens what each unit effectively sees. It "
     "also buys a little tolerance to a feature shifting by a pixel or two.",
     ["it reduces the spatial dimensions",
      "it lowers compute and parameters downstream",
      "it enlarges the effective receptive field",
      "it adds small-shift invariance"],
     []),

    (3, "label smoothing",
     "What does label smoothing do?",
     "Instead of training towards a target of exactly one for the true class, it "
     "targets something slightly less and spreads the remainder over the others. "
     "That stops the network driving logits arbitrarily high to chase a "
     "probability it can never reach, which improves calibration.",
     ["the target probability is softened from one",
      "the remainder is spread across other classes",
      "it stops the model becoming over-confident",
      "it usually improves calibration"],
     []),

    (3, "shuffling and SGD",
     "Why does shuffling matter for stochastic gradient descent?",
     "Because each step follows the gradient of one batch, and if batches are "
     "ordered by class or by time the model is pulled towards whatever that batch "
     "contains and then pulled back. Shuffling makes each batch an unbiased "
     "sample, so the noisy steps average towards the true gradient.",
     ["each update follows one batch's gradient",
      "ordered batches give biased gradient estimates",
      "shuffling makes batches representative",
      "the noise then averages out across steps"],
     []),
]
