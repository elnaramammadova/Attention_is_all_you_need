<h1>The Illustrated Transformer - Attention is all you need</h1>

The Transformer was proposed in the paper Attention is All You Need [[1]](#ref1). The impact of this paper has been tremendous. It paved the way for a new generation of models such as BERT, GPT-2/3/4, T5, and others, which have achieved state-of-the-art results on a wide range of NLP tasks, including machine translation, text summarization, and sentiment analysis. These models, based on the Transformer architecture, have pushed the boundaries of what's possible with NLP, making applications like real-time translation, automatic content generation, and advanced chatbots a reality.

In this notebook we will attempt to recreate the “annotated” version of the paper in the form of a line-by-line implementation. We will be using several resources to aid our task:
 - The Illustrated Transformer by Jay Alammar [[2]](#ref2)
 - Harvard’s NLP group created a guide annotating the paper with PyTorch implementation [[3]](#ref3)


The key idea behind the Transformer is the self-attention mechanism, which computes a weighted sum of all words in a sentence for each word, where the weights are determined by the compatibility of the word with all others.

This model eliminated the need for recurrence and convolution, which were the dominant paradigms in the field up until then, and it allowed for much more parallelizable training, significantly reducing training times. It also improved the handling of long-term dependencies in text, a notorious difficulty with previous models.