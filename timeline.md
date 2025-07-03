# 01/07/2025

- Make use of Docling library
- Docling receives many types of format: **md html docx pptx png pdf asciidoc **
- Then let's docstring perform parsing --> markdown format + extract images as PNG + extract tables as MD format
- Reason for choosing Docling instead of Unstructured is that Unstructured provide much "complicated code" + image extraction is not as good as docling (You can see this when you perform parsing the document pdf of MDocAgent)
- Try to implement a Text-based RAG first based on this kind of implementation.

---

# 02/07/2025

- Currently for the ImageRAGAgent, we use the Qwen-2.5-VL-Instruct to perform describing and analyzing the image to provide some captions. Then put that captions into the vectorstore. Then just process it like the TextRAGAgent.

---

# 03/07/2025 

- Now for Image-RAG pipeline, we use Colpali `vidore/colqwen2-v1.0`. I have tried to use the minimalist config of bitsandbytes of int4 but the performance for querying on those embeddings is not quite good. BUT i am limited by my VRAM (4 gb vram and 16 gb shared memory). 
- Currently not just need to depend on Qwen-2.5-VL-Instruct for generating caption for retrieved images. We can switch to use OpenAI API or Gemini API by changing the value in the `config.py` of `IMAGE_CAPTIONING = "gemini" # "openai" or "qwen" (Qwen2.5-VL)`.

```
INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
🧠 Output from TextRAGAgent:
The Transformer architecture consists of an encoder and a decoder, each made up of a stack of identical layers. The encoder has six layers, each containing two sub-layers: a multi-head self-attention mechanism and a position-wise fully connected feed-forward network. The decoder also has a similar structure but includes an additional layer for "encoder-decoder attention" (source: [store/arxiv_org_pdf_1706_03762_pdf.pdf]).

There are three kinds of attention mechanisms used in the Transformer architecture:

1. **Encoder-Decoder Attention**: In this layer, the queries come from the previous decoder layer, while the memory keys and values come from the encoder's output. This allows the decoder to attend to all positions in the input sequence.   

2. **Self-Attention in the Encoder**: In this layer, all keys, values, and queries come from the same source, specifically the output of the previous layer in the encoder. Each position in the encoder can attend to all positions in the previous layer.

3. **Self-Attention in the Decoder**: Similar to the encoder's self-attention, this layer allows each position in the decoder to attend to all positions in the decoder up to and including that position, preventing leftward attention (source: [store/arxiv_org_pdf_1706_03762_pdf.pdf]).
--------------------------------------------------------------------------------
🤖 Running ImageRAGAgent...
INFO:rag_image.loader:🧠 Embedding query: tell me about the Transformer architecture. How many kinds o...
INFO:agents.image_agent:🔍 Retrieved 5 relevant documents for question: tell me about the Transformer architecture. How many kinds of attention mechanism are there using in transformer architecture
INFO:agents.image_agent:[Doc 0] source=arxiv_org_pdf_1706_03762_pdf.pdf#page=5 | content preview=tmp\pdf_pages\arxiv_org_pdf_1706_03762_pdf-page-5.png
INFO:agents.image_agent:[Doc 1] source=arxiv_org_pdf_1706_03762_pdf.pdf#page=2 | content preview=tmp\pdf_pages\arxiv_org_pdf_1706_03762_pdf-page-2.png
INFO:agents.image_agent:[Doc 2] source=arxiv_org_pdf_1706_03762_pdf.pdf#page=13 | content preview=tmp\pdf_pages\arxiv_org_pdf_1706_03762_pdf-page-13.png
INFO:agents.image_agent:[Doc 3] source=arxiv_org_pdf_2501_06322_pdf.pdf#page=32 | content preview=tmp\pdf_pages\arxiv_org_pdf_2501_06322_pdf-page-32.png
INFO:agents.image_agent:[Doc 4] source=arxiv_org_pdf_2503_13964_pdf.pdf#page=4 | content preview=tmp\pdf_pages\arxiv_org_pdf_2503_13964_pdf-page-4.png
INFO:rag_image.caption.gemini_runner:📥 Received query: tell me about the Transformer architecture. How many kinds of attention mechanism are there using in transformer architecture
INFO:rag_image.caption.gemini_runner:🔍 Retrieved 5 image(s):
INFO:rag_image.caption.gemini_runner:  - tmp\pdf_pages\arxiv_org_pdf_1706_03762_pdf-page-5.png
INFO:rag_image.caption.gemini_runner:  - tmp\pdf_pages\arxiv_org_pdf_1706_03762_pdf-page-2.png
INFO:rag_image.caption.gemini_runner:  - tmp\pdf_pages\arxiv_org_pdf_1706_03762_pdf-page-13.png
INFO:rag_image.caption.gemini_runner:  - tmp\pdf_pages\arxiv_org_pdf_2501_06322_pdf-page-32.png
INFO:rag_image.caption.gemini_runner:  - tmp\pdf_pages\arxiv_org_pdf_2503_13964_pdf-page-4.png
INFO:google_genai.models:AFC remote call 1 is done.
INFO:rag_image.caption.gemini_runner:✅ Gemini caption generated.
🧠 Output from ImageRAGAgent:
Okay, let's break down the Transformer architecture and the attention mechanisms it employs.

**The Transformer Architecture: A High-Level Overview**

The Transformer, introduced in the paper "Attention is All You Need" by Vaswani et al. (2017), revolutionized sequence-to-sequence tasks (like machine translation) and has since become the foundation for many state-of-the-art natural language processing (NLP) models, including BERT, GPT, and many others.  It's a departure from recurrent neural networks (RNNs) and convolutional neural networks (CNNs) by relying entirely on attention mechanisms.

Here's a breakdown of the key components:

1.  **Encoder-Decoder Structure:** The Transformer follows the encoder-decoder structure common in sequence-to-sequence models:
    *   **Encoder:** Processes the input sequence and produces a context-aware representation.  This is essentially a stack of identical encoder layers.
    *   **Decoder:**  Generates the output sequence, conditioned on the encoder's output and the previously generated tokens.  It's also a stack of identical decoder layers.

2.  **Encoder Layers:** Each encoder layer contains two main sub-layers:
    *   **Multi-Head Self-Attention:**  This is the core of the Transformer. It allows the encoder to attend to different parts of the input sequence when processing each word.  It helps the model understand relationships between words in the input.
    *   **Feed-Forward Network:** A fully connected feed-forward network is applied to each position independently. This adds non-linearity and helps the model learn more complex representations.

    Each of these sub-layers has a residual connection and layer normalization around it. (Add & Norm)

3.  **Decoder Layers:** Each decoder layer has three main sub-layers:
    *   **Masked Multi-Head Self-Attention:** Similar to the encoder, this allows the decoder to attend to the previously generated tokens in the output sequence.  The "masking" prevents the decoder from "cheating" by looking at future tokens during training.
    *   **Multi-Head Encoder-Decoder Attention:** This allows the decoder to attend to the output of the encoder.  It helps the decoder focus on the relevant parts of the input sequence when generating each token.
    *   **Feed-Forward Network:**  Same as in the encoder, a fully connected feed-forward network is applied to each position.

    Each of these sub-layers has a residual connection and layer normalization around it. (Add & Norm)

4.  **Attention Mechanism:** The heart of the Transformer is the attention mechanism, specifically the scaled dot-product attention, which allows the model to weigh the importance of different parts of the input sequence. The formula for attention is:

    ```
    Attention(Q, K, V) = softmax( (Q K^T) / sqrt(d_k) ) V
    ```

    Where:
    *   **Q** (Queries): Represents the query vectors.
    *   **K** (Keys): Represents the key vectors.
    *   **V** (Values): Represents the value vectors.
    *   `d_k`: Is the dimension of the keys. Used to scale the dot products.
    *   `softmax`:  Normalizes the weights, producing probabilities.

5.  **Multi-Head Attention:**  Instead of performing a single attention calculation, the Transformer uses "multi-head attention." This means that the queries, keys, and values are linearly projected into multiple subspaces, and attention is calculated in each subspace independently.  The results are then concatenated and linearly transformed to produce the final output.  This allows the model to capture different types of relationships between words.

    ```
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    ```

6.  **Positional Encoding:** Because the Transformer doesn't have any inherent understanding of the order of words (unlike RNNs), positional encodings are added to the input embeddings. These encodings provide information about the position of each word in the sequence.  Various forms of positional encoding exist; one common approach is to use sine and cosine functions of different frequencies.

**Types of Attention Mechanisms Used in the Transformer**

The Transformer architecture uses (at least) three primary types of attention mechanisms, distinguished by *where* they attend:

1.  **Self-Attention in the Encoder:**

    *   The encoder uses self-attention layers.
    *   This allows the encoder to attend to all positions in the input sequence when processing each word.
    *   In this case, the queries, keys, and values all come from the output of the previous layer in the encoder.
    *   This helps the encoder to understand relationships between different words in the input sequence.

2.  **Masked Self-Attention in the Decoder:**

    *   Used in the decoder.
    *   It's similar to encoder self-attention, but it's *masked*.
    *   The masking prevents the decoder from attending to future tokens in the output sequence.  This is crucial during training to ensure that the decoder only uses information from previously generated tokens.
    *   Each position in the decoder is allowed to attend to all positions up to and including that position. This is implemented by masking all values in the input of the softmax which correspond to illegal connections.

3.  **Encoder-Decoder Attention (or Cross-Attention):**

    *   Used in the decoder.
    *   This allows the decoder to attend to the output of the encoder.
    *   In this case, the queries come from the output of the previous layer in the decoder, while the keys and values come from the output of the encoder.
    *   This helps the decoder to focus on the relevant parts of the input sequence when generating each token in the output sequence.

In summary, the three types of attention are distinguished by *where* the queries, keys, and values come from within the encoder-decoder architecture. They enable the Transformer to effectively model relationships within the input sequence, within the output sequence, and between the input and output sequences.
--------------------------------------------------------------------------------
🤖 Running GeneralizeAgent...
INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
🧠 Output from GeneralizeAgent:
The Transformer architecture, introduced in the paper "Attention is All You Need" by Vaswani et al. (2017), consists of an encoder and a decoder, each made up of a stack of identical layers. The encoder processes the input sequence and produces a context-aware representation, while the decoder generates the output sequence based on the encoder's output and previously generated tokens.

### Key Components of the Transformer Architecture:

1. **Encoder Layers**:
   - **Multi-Head Self-Attention**: Allows the encoder to attend to different parts of the input sequence, helping it understand relationships between words.
   - **Feed-Forward Network**: A fully connected network applied to each position independently, adding non-linearity and enabling the model to learn complex representations.
   - Each sub-layer has a residual connection and layer normalization.

2. **Decoder Layers**:
   - **Masked Multi-Head Self-Attention**: Similar to the encoder's self-attention but prevents the decoder from attending to future tokens during training.
   - **Multi-Head Encoder-Decoder Attention**: Allows the decoder to focus on relevant parts of the encoder's output when generating each token.
   - **Feed-Forward Network**: Same as in the encoder, applied to each position independently.
   - Each sub-layer also has a residual connection and layer normalization.

3. **Attention Mechanism**: The core of the Transformer is the attention mechanism, specifically the scaled dot-product attention, which weighs the importance of different parts of the input sequence. The formula for attention is:
   \[
   \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
   \]
   where \(Q\) (queries), \(K\) (keys), and \(V\) (values) are vectors derived from the input.

4. **Multi-Head Attention**: Instead of a single attention calculation, the Transformer uses multiple attention heads to capture different types of relationships between words. The results from each head are concatenated and linearly transformed.

5. **Positional Encoding**: Since the Transformer lacks inherent understanding of word order, positional encodings are added to input embeddings to provide information about the position of each word in the sequence.

### Types of Attention Mechanisms:

1. **Self-Attention in the Encoder**: All queries, keys, and values come from the encoder's output, allowing it to attend to all positions in the input sequence.

2. **Masked Self-Attention in the Decoder**: Similar to encoder self-attention but masked to prevent attending to future tokens, ensuring the decoder only uses information from previously generated tokens.

3. **Encoder-Decoder Attention**: In the decoder, queries come from the previous decoder layer, while keys and values come from the encoder's output, enabling the decoder to focus on relevant parts of the input sequence.

In summary, the Transformer architecture effectively models relationships within the input and output sequences through its innovative use of attention mechanisms, making it foundational for many state-of-the-art natural language processing models.
--------------------------------------------------------------------------------
🤖 Running FinalizeAgent...
INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
🧠 Output from FinalizeAgent:
The Transformer architecture, introduced in the paper "Attention is All You Need" by Vaswani et al. (2017), consists of an encoder and a decoder, each made up of a stack of identical layers. The encoder processes the input sequence to produce a context-aware representation, while the decoder generates the output sequence based on the encoder's output and previously generated tokens.

### Key Components:
1. **Encoder Layers**:
   - **Multi-Head Self-Attention**: Attends to different parts of the input sequence to understand relationships between words.
   - **Feed-Forward Network**: Applies a fully connected network to each position independently.
   - Each sub-layer includes a residual connection and layer normalization.

2. **Decoder Layers**:
   - **Masked Multi-Head Self-Attention**: Prevents the decoder from attending to future tokens during training.
   - **Multi-Head Encoder-Decoder Attention**: Allows the decoder to focus on relevant parts of the encoder's output.
   - **Feed-Forward Network**: Similar to the encoder's feed-forward network.
   - Each sub-layer also has a residual connection and layer normalization.

3. **Attention Mechanism**: The core of the Transformer is the scaled dot-product attention, which weighs the importance of different parts of the input sequence.

4. **Multi-Head Attention**: Utilizes multiple attention heads to capture various relationships between words, with results concatenated and linearly transformed.

5. **Positional Encoding**: Adds information about the position of each word in the sequence since the Transformer does not inherently understand word order.

### Types of Attention Mechanisms:
1. **Self-Attention in the Encoder**: All queries, keys, and values come from the encoder's output.
2. **Masked Self-Attention in the Decoder**: Similar to self-attention but masked to prevent attending to future tokens.
3. **Encoder-Decoder Attention**: Queries come from the decoder, while keys and values come from the encoder's output.

In summary, the Transformer architecture employs innovative attention mechanisms to effectively model relationships within input and output sequences, making it foundational for many state-of-the-art natural language processing models.     
--------------------------------------------------------------------------------

💬 Final Answer:
The Transformer architecture, introduced in the paper "Attention is All You Need" by Vaswani et al. (2017), consists of an encoder and a decoder, each made up of a stack of identical layers. The encoder processes the input sequence to produce a context-aware representation, while the decoder generates the output sequence based on the encoder's output and previously generated tokens.

### Key Components:
1. **Encoder Layers**:
   - **Multi-Head Self-Attention**: Attends to different parts of the input sequence to understand relationships between words.
   - **Feed-Forward Network**: Applies a fully connected network to each position independently.
   - Each sub-layer includes a residual connection and layer normalization.

2. **Decoder Layers**:
   - **Masked Multi-Head Self-Attention**: Prevents the decoder from attending to future tokens during training.
   - **Multi-Head Encoder-Decoder Attention**: Allows the decoder to focus on relevant parts of the encoder's output.
   - **Feed-Forward Network**: Similar to the encoder's feed-forward network.
   - Each sub-layer also has a residual connection and layer normalization.

3. **Attention Mechanism**: The core of the Transformer is the scaled dot-product attention, which weighs the importance of different parts of the input sequence.

4. **Multi-Head Attention**: Utilizes multiple attention heads to capture various relationships between words, with results concatenated and linearly transformed.

5. **Positional Encoding**: Adds information about the position of each word in the sequence since the Transformer does not inherently understand word order.

### Types of Attention Mechanisms:
1. **Self-Attention in the Encoder**: All queries, keys, and values come from the encoder's output.
2. **Masked Self-Attention in the Decoder**: Similar to self-attention but masked to prevent attending to future tokens.
3. **Encoder-Decoder Attention**: Queries come from the decoder, while keys and values come from the encoder's output.

In summary, the Transformer architecture employs innovative attention mechanisms to effectively model relationships within input and output sequences, making it foundational for many state-of-the-art natural language processing models.     

```

---

# 04/07/2025 

- ... 
- 



