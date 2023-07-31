<h1>The Illustrated Transformer - Attention is all you need</h1>

The Transformer was proposed in the paper Attention is All You Need [[1]](#ref1). The impact of this paper has been tremendous. It paved the way for a new generation of models such as BERT, GPT-2/3/4, T5, and others, which have achieved state-of-the-art results on a wide range of NLP tasks, including machine translation, text summarization, and sentiment analysis. These models, based on the Transformer architecture, have pushed the boundaries of what's possible with NLP, making applications like real-time translation, automatic content generation, and advanced chatbots a reality.

In this notebook we will attempt to recreate the “annotated” version of the paper in the form of a line-by-line implementation. We will be using several resources to aid our task:
 - The Illustrated Transformer by Jay Alammar [[2]](#ref2)
 - Harvard’s NLP group created a guide annotating the paper with PyTorch implementation [[3]](#ref3)


The key idea behind the Transformer is the self-attention mechanism, which computes a weighted sum of all words in a sentence for each word, where the weights are determined by the compatibility of the word with all others.

This model eliminated the need for recurrence and convolution, which were the dominant paradigms in the field up until then, and it allowed for much more parallelizable training, significantly reducing training times. It also improved the handling of long-term dependencies in text, a notorious difficulty with previous models.

<h2>Challenges in RNN that Transformer models help overcome</h2>

<table>
  <tr>
    <th><b>Challenges</b></th>
    <th><b>RNNs</b></th>
    <th><b>Transformers</b></th>
  </tr>
  <tr>
    <td>Long-Range Dependencies</td>
    <td>RNNs struggls with long-distance dependencies. RNNs typically falter when handling long text documents</td>
    <td>Transformer architectures primarily rely on attention mechanisms. These mechanisms allow the model to establish relationships between any parts of a sequence, making it adept at handling long-distance dependencies. With transformers, long-distance dependencies are as likely to be addressed as any shorter-distance ones.</td>
  </tr>
  <tr>
    <td>Vanishing and exploding gradients</td>
    <td>Suffers from gradient vanishing and gradient explosion.</td>
    <td> Transformers experience minimal vanishing or exploding gradient issues. The complete sequence is trained simultaneously in Transformer networks, supplemented by only a few additional layers. Therefore, gradient problems are seldom encountered</td>
  </tr>
    <tr>
        <td>Training steps required to reach a local/global minima</td>
        <td>RNNs require more training steps to reach local or global minima. When visualized, RNNs appear as deep, unrolled networks with the network's depth determined by the sequence's length. This leads to a high number of parameters, many of which are interconnected, resulting in longer training times and a need for multiple steps</td>
        <td>Transformers require less training steps compared to RNNs</td>
    </tr>
    <tr>
        <td>Parallel Computation</td>
        <td>RNNs are unable to support parallel computation. Despite the benefits of GPUs in enabling parallel computation, RNNs function as sequence models, meaning all network computations occur sequentially, prohibiting parallelization</td>
        <td>The absence of recurrence in transformer networks facilitates parallel computation. This means computations can occur concurrently at every step</td>
    </tr>
</table>

<h2>Attention Mechanism<h2>
Attention Mechanism consists of following parts:
<h3>1. Self-Attention Module<h3>
Sentence is fed to a computer, and it considers each word as a token t, and each token has a word embedding V with shape [vocub_size, model_dimension]. But these word embeddings have no context. So the idea is to apply some kind of weighing or similarity to obtain final word embedding Y, which has more context than the initial embedding V.

- first find the weights by multiplying (dot product) the initial embedding of the first word with the embedding of all other words in the sentence. 
- nomralize these weights to have a sum of 1.
- multiply the normalized weights with the initial embeddings of all the words in the sentence

All weights will have the context of the first word. So when we are multiplying these weights to each word, we are essentially reweighing all the other words towards the first word. This is repeated for all words so that each word gets some context from every other word in the sentence.

Weights are not trained, the order or proximity of the words have no influence on each other. Also, the process has no dependency on the length of the sentence, that is, more or fewer words in a sentence do not matter. This approach of adding some context to the words in a sentence is known as Self-Attention.

- Multihead Attention Module
- Position-Wise Feed-Forward Module