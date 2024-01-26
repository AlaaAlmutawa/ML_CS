import trax
from trax import layers as tl
from trax.fastmath import numpy as jnp
from trax.supervised import training

## Refernece: 
## This was completed as a part of following tutorial provided by Coursera Deep Learning Specialization
## https://www.coursera.org/specializations/natural-language-processing
## https://github.com/LaurentVeyssier/TRAX_transformer_abstractive_summarization_model/blob/main/TRAX_transformer_summarizer_model.ipynb

def DotProductAttention(query, key, value, mask):
    """Dot product self-attention.
    Args:
        query (jax.interpreters.xla.DeviceArray): array of query representations with shape (L_q by d)
        key (jax.interpreters.xla.DeviceArray): array of key representations with shape (L_k by d)
        value (jax.interpreters.xla.DeviceArray): array of value representations with shape (L_k by d) where L_v = L_k
        mask (jax.interpreters.xla.DeviceArray): attention-mask, gates attention with shape (L_q by L_k)

    Returns:
        jax.interpreters.xla.DeviceArray: Self-attention array for q, k, v arrays. (L_q by L_k)
    """

    assert query.shape[-1] == key.shape[-1] == value.shape[-1], "Embedding dimensions of q, k, v aren't all the same"

    depth = query.shape[-1]
    dots = jnp.matmul(query, jnp.swapaxes(key, -2, -1)) / jnp.sqrt(depth)
    if mask is not None: 
        dots = jnp.where(mask, dots, jnp.full_like(dots, -1e9))

    logsumexp = trax.fastmath.logsumexp(dots, axis=-1, keepdims=True)

    dots = jnp.exp(dots - logsumexp)

    attention = jnp.matmul(dots, value)

    return attention

# compute_attention_heads_closure
def compute_attention_heads_closure(n_heads, d_head):
    """ Function that simulates environment inside CausalAttention function.
    Args:
        d_head (int):  dimensionality of heads.
        n_heads (int): number of attention heads.
    Returns:
        function: compute_attention_heads function
    """

    def compute_attention_heads(x):
        """ Compute the attention heads.
        Args:
            x (jax.interpreters.xla.DeviceArray): tensor with shape (batch_size, seqlen, n_heads X d_head).
        Returns:
            jax.interpreters.xla.DeviceArray: reshaped tensor with shape (batch_size X n_heads, seqlen, d_head).
        """
        batch_size = x.shape[0]

        seqlen = x.shape[1]

        x = jnp.reshape(x,(batch_size, seqlen, n_heads, d_head))

        x = jnp.transpose(x, (0, 2, 1, 3))

        x = jnp.reshape(x, (batch_size*n_heads, seqlen, d_head))

        return x

    return compute_attention_heads

# dot_product_self_attention
def dot_product_self_attention(q, k, v):
    """ Masked dot product self attention.
    Args:
        q (jax.interpreters.xla.DeviceArray): queries.
        k (jax.interpreters.xla.DeviceArray): keys.
        v (jax.interpreters.xla.DeviceArray): values.
    Returns:
        jax.interpreters.xla.DeviceArray: masked dot product self attention tensor.
    """
    mask_size = q.shape[-2]

    mask = jnp.tril(jnp.ones((1, mask_size, mask_size), dtype=jnp.bool_), k=0)
    return DotProductAttention(q, k, v, mask)

# compute_attention_output_closure
def compute_attention_output_closure(n_heads, d_head):
    """ Function that simulates environment inside CausalAttention function.
    Args:
        d_head (int):  dimensionality of heads.
        n_heads (int): number of attention heads.
    Returns:
        function: compute_attention_output function
    """

    def compute_attention_output(x):
        """ Compute the attention output.
        Args:
            x (jax.interpreters.xla.DeviceArray): tensor with shape (batch_size X n_heads, seqlen, d_head).
        Returns:
            jax.interpreters.xla.DeviceArray: reshaped tensor with shape (batch_size, seqlen, n_heads X d_head).
        """
        seqlen = x.shape[1]
        x = jnp.reshape(x, (-1, n_heads, seqlen, d_head))
        x = jnp.transpose(x, (0, 2, 1, 3))

        return jnp.reshape(x, (-1, seqlen, n_heads * d_head))

    return compute_attention_output
def CausalAttention(d_feature,
                    n_heads,
                    compute_attention_heads_closure=compute_attention_heads_closure,
                    dot_product_self_attention=dot_product_self_attention,
                    compute_attention_output_closure=compute_attention_output_closure,
                    mode='train'):
    """Transformer-style multi-headed causal attention.

    Args:
        d_feature (int):  dimensionality of feature embedding.
        n_heads (int): number of attention heads.
        compute_attention_heads_closure (function): Closure around compute_attention heads.
        dot_product_self_attention (function): dot_product_self_attention function.
        compute_attention_output_closure (function): Closure around compute_attention_output.
        mode (str): 'train' or 'eval'.

    Returns:
        trax.layers.combinators.Serial: Multi-headed self-attention model.
    """

    assert d_feature % n_heads == 0
    d_head = d_feature // n_heads

    ComputeAttentionHeads = tl.Fn('AttnHeads', compute_attention_heads_closure(n_heads, d_head), n_out=1)


    return tl.Serial(
        tl.Branch(
            [tl.Dense(d_feature), ComputeAttentionHeads], # queries
            [tl.Dense(d_feature), ComputeAttentionHeads], # keys
            [tl.Dense(d_feature), ComputeAttentionHeads], # values
        ),

        tl.Fn('DotProductAttn', dot_product_self_attention, n_out=1), 

        tl.Fn('AttnOutput', compute_attention_output_closure(n_heads, d_head), n_out=1), 
        tl.Dense(d_feature), 
    )
# DecoderBlock
def DecoderBlock(d_model, d_ff, n_heads,
                 dropout, mode, ff_activation):
    """Returns a list of layers that implements a Transformer decoder block.

    The input is an activation tensor.

    Args:
        d_model (int):  depth of embedding.
        d_ff (int): depth of feed-forward layer.
        n_heads (int): number of attention heads.
        dropout (float): dropout rate (how much to drop out).
        mode (str): 'train' or 'eval'.
        ff_activation (function): the non-linearity in feed-forward layer.

    Returns:
        list: list of trax.layers.combinators.Serial that maps an activation tensor to an activation tensor.
    """
    causal_attention = CausalAttention(
                        d_model,
                        n_heads=n_heads,
                        mode=mode
                        )

    feed_forward = [
        tl.LayerNorm(),
        tl.Dense(d_ff),
        ff_activation(), 
        tl.Dropout(rate=dropout, mode=mode),
        tl.Dense(d_model),
        tl.Dropout(rate=dropout, mode=mode)
    ]

    return [
      tl.Residual(
          tl.LayerNorm(),
          causal_attention,
          tl.Dropout(rate=dropout, mode=mode),
        ),
      tl.Residual(
          feed_forward
        ),
      ]

# Function to build the complete transfoemr model with default parameters passed. 
def TransformerLM(vocab_size=32000,
                  d_model=512,
                  d_ff=2048,
                  n_layers=6,
                  n_heads=8,
                  dropout=0.1,
                  max_len=4096,
                  mode='train',
                  ff_activation=tl.Relu):
    """Returns a Transformer language model.

    The input to the model is a tensor of tokens. (This model uses only the
    decoder part of the overall Transformer.)

    Args:
        vocab_size (int): vocab size.
        d_model (int):  depth of embedding.
        d_ff (int): depth of feed-forward layer.
        n_layers (int): number of decoder layers.
        n_heads (int): number of attention heads.
        dropout (float): dropout rate (how much to drop out).
        max_len (int): maximum symbol length for positional encoding.
        mode (str): 'train', 'eval' or 'predict', predict mode is for fast inference.
        ff_activation (function): the non-linearity in feed-forward layer.

    Returns:
        trax.layers.combinators.Serial: A Transformer language model as a layer that maps from a tensor of tokens
        to activations over a vocab set.
    """

    positional_encoder = [
        tl.Embedding(vocab_size, d_model),
        tl.Dropout(rate=dropout, mode=mode),
        tl.PositionalEncoding(max_len=max_len, mode=mode)]
    decoder_blocks = [DecoderBlock(d_model, d_ff, n_heads, dropout, mode, ff_activation) for _ in range(n_layers)]

    return tl.Serial(
        tl.ShiftRight(mode=mode), 
        positional_encoder,
        decoder_blocks,
        tl.LayerNorm(),

        tl.Dense(vocab_size),
        tl.LogSoftmax()
    )

# define training loop function. 
def training_loop(TransformerLM, train_gen, eval_gen, output_dir = "./model"):
    '''
    Input:
        TransformerLM (trax.layers.combinators.Serial): The model you are building.
        train_gen (generator): Training stream of data.
        eval_gen (generator): Evaluation stream of data.
        output_dir (str): folder to save your file.

    Returns:
        trax.supervised.training.Loop: Training loop.
    '''
    lr_schedule = trax.lr.warmup_and_rsqrt_decay(n_warmup_steps=1000, max_value=0.01)

    train_task = training.TrainTask(
      labeled_data= train_gen, 
      loss_layer= tl.CrossEntropyLoss(), 
      optimizer= trax.optimizers.Adam(0.01), 
      lr_schedule= lr_schedule,
      n_steps_per_checkpoint= 10,
    )

    eval_task = training.EvalTask(
      labeled_data= eval_gen, 
      metrics=[tl.CrossEntropyLoss(), tl.Accuracy()] 
    )


    loop = training.Loop(TransformerLM(d_model=512,
                                       d_ff=2048,
                                       n_layers=6,
                                       n_heads=8,
                                       mode='train'),
                         train_task,
                         eval_tasks=[eval_task],
                         output_dir=output_dir)

    return loop
