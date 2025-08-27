# Extractive Summarization with LLMs and Heuristics

<p align="center">
  <a href="#introduction">Introduction</a> - 
  <a href="#dataset">Dataset</a> - 
  <a href="#track-a">Track A</a> - 
  <a href="#track-b">Track B</a> - 
  <a href="#track-c">Track C</a>
</p>

---
### Introduction
While abstractive summarization has dominated recent research—driven by the rapid advancement of large language models (LLMs) and generative **AI—extractive summarization remains a vital component** in real-world applications. Unlike its abstractive counterpart, extractive summarization offers several critical advantages:
-  **Factual reliability**: Since extracted sentences come directly from the source, the risk of hallucinations is eliminated;
-  **Interpretability and traceability**: Especially in high-stakes domains (e.g., legal, biomedical, scientific), being able to reference exact source sentences improves user trust and auditability;
-  **Computational efficiency**: Extractive models are generally less resource-intensive.
Despite its importance, modern research in extractive summarization continues to rely heavily on traditional encoder-only architectures (e.g., BERT-based extractors), often ignoring the potential of prompting with modern LLMs. This leaves an open research space for exploring more flexible extractive approaches powered by prompting.

### Dataset 
The used dataset is [**ACLSum**](https://huggingface.co/datasets/sobamchan/aclsum/viewer/extractive) and the relative [**paper**](https://aclanthology.org/2024.naacl-long.371.pdf) published at [NAACL24](https://2024.naacl.org/).
ACLSum is structured as follows:
-  A document is the concatenation of the **Abstract**, **Introduction** and **Conclusion** sections of NLP papers published at ACL from 1974 to 2022.
-  For each document, **6 summaries are provided** (3 extractive and 3 abstractive).
    -  **Extractive summaries** are just sentences selected and labeled according to the aspect they refer to (Challenge, Approach, Outcome).
    -  **Abstractive summaries** are new, single sentence summaries written by humans like reviewers; one per aspect.

### Track A
#### Objectives
The overarching goal of this project is to systematically **investigate prompting-based strategies** for extractive summarization using LLMs, moving beyond static architectures and toward more dynamic, context-aware reasoning patterns.
Specific objectives include:
-  Implementing and benchmarking a wide range of prompting paradigms tailored for extractive summarization;
-  Evaluating combinations of prompting strategies to discover synergistic effects;
-  Leveraging oracle-based supervision to quantitatively assess the quality of extracted summaries.

#### Methodology: Prompting Strategies for Extraction
A suite of prompting techniques has been explored, each designed to guide the model through different reasoning patterns for identifying key sentences. A baseline vanilla prompt will serve as a control across experiments:

 ```text
Vanilla Prompt Template:
System message: "You are an expert in extractive summarization. Your task is to select the most important sentences from the document."
 
Input:
Sentences are presented in a numbered list.
Sentence 1: "Text_S1"
Sentence 2: "Text_S2"
…
Sentence n: "Text_Sn”
 
Output:
Return a list of selected sentence indices in JSON, for example:
{"selected_sentences": [1, 3, 5]}
```
#### Prompting Techniques
| Technique                     | How it works                                                                                  | Prompt Example                                                                                   |
|--------------------------------|----------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|
| Chain-of-Thought (CoT)         | The model explicitly reasons about key facts or events before selecting sentences that convey them. | Step 1: What are the most important facts in the article?<br>Step 2: Select the sentences that express these facts. |
| Least-to-Most Prompting (LtM)  | The document is broken into thematic sections (e.g., problem, method, results), and representative sentences are extracted from each. | 1. Divide the article into sections: [Event, Context, Consequences].<br>2. For each section, extract the most relevant sentence.<br>3. Combine selected sentences into a final summary. |
| Tool-Augmented Prompting        | Each sentence is accompanied by a precomputed importance score (e.g., TF-IDF, BERTScore, LexRank), guiding selection. | Here is the article with sentences accompanied by centrality scores. Extract the most relevant sentences.<br>- 'Sentence 1' [centrality: 0.89]<br>- 'Sentence 2' [centrality: 0.21] |
| Self-Ask Prompting              | The model generates internal questions like “What is this document about?” and selects the sentences that answer them. | Q: What is the purpose of this document? Q: Which sentences answer it? |
| Scoring-Based Prompting         | The model assigns scores to each sentence (e.g., 1–5 or 1–10) and selects those with the highest ratings. | "Score each sentence for importance. Select those ≥ 4." |
| Explanation-Based Selection     | The model explains why a sentence might be important; strong justifications warrant inclusion. | "Why is this sentence important? → If rationale is strong, keep it." |
| Salience Inference Prompting    | The model simulates what a reader would remember and extracts sentences accordingly. | Which sentences would a reader remember after reading this article? Select those. |


#### Used Data
This track focuses on extractive summarization benchmarking (no training), we limited to the *test set* and to *extractive targets* only. In it there are 100 documents and, so, 300 summaries (100 per aspect). 
Thanks to ACLSum's aspect-based organisation we can check performances on both, at the **aspect level** and in an **aspect-agnostic way**. Both have been implemented. 

#### Evaluation metrics
Each strategy has been evaluated with exact-match metrics including:
-  **Precision**, **Recall** and **F1-Score**, computed w.r.t. the gold sentence indices.

These metrics are particularly well-suited for extractive summarization tasks where the reference summaries consist of exact source sentences.
 
Additionally, **ROUGE** scores (i.e., ROUGE-1, ROUGE-2, ROUGE-L F1) may be computed as a secondary evaluation, comparing the model's extractive output against the abstractive reference summaries also provided by the ACLSum dataset. 
While ROUGE is less indicative of extractive accuracy per se, it can offer complementary insights into how much content overlap exists between extracted sentences and human-written summaries.


#### Models
To carry out our experiments, we will employ modern open-weight LLMs with strong capabilities in reasoning and instruction following, easy to run in Colab. Specifically:
**LLaMA 3.2 (3B)**: A balanced and widely adopted open model, suitable for general-purpose prompting.
**Qwen 3 (4B)**: A reasoning-oriented model, particularly effective for multi-step and structured prompting strategies.
**Qwen 2.5 (3B)**: A reasoning-oriented model.


### Track B
Here the focus changes from benchmarking to **Dataset Construction**. 

#### Inspired By 
Our work is primarily inspired by the paper [DYLE: Dynamic Latent Extraction for Abstractive Long-Input Summarization](https://aclanthology.org/2022.acl-long.118/) published in [NAACL24](https://2024.naacl.org/).
The authors explore four heuristics, falling into two categories. Both categories involve determining candidate extractive elements and ranking them, but the key difference lies in whether the candidate is an entire extractive summary (i.e., a set of sentences from the source document) or a single sentence from the source.
 
1. **Summary-level**:<br>
  (Note: The following names have nothing to do with decoding strategies.)
    -  **Greedy Search**: the extractive summary starts empty and is incrementally built up. At each step, one sentence from the source document is appended to the current extractive summary. Among all possible additions, the sentence that results in the highest similarity between the updated extractive summary and the generated abstractive summary is retained. If no addition increases the similarity, the process stops. This approach has been used by [DyLE (ACL 2022)](https://aclanthology.org/2022.acl-long.118/), a milestone in the field. 
    -  **Beam Search**: instead of evolving just one extractive summary at a time, a list of the top-K candidates is maintained. In their experiments, the beam width is set to 4.
3. **Sentence-level**:
    -  **Local Scorer**: each sentence from the source document is evaluated individually for similarity with the generated abstractive summary;
    -  **Global Scorer**: in this case, sentence ranking is done with broader context in mind—namely, the extractive summaries in which those sentences might appear. First, method (1.b) is used to generate a set of high-quality candidate extractive summaries. Then, all sentences appearing in those summaries are considered (a single sentence may appear in multiple summaries). Each such sentence is assigned a score equal to the sum of the similarity scores between each summary it appears in and the generated abstractive summary. The idea is that if a sentence contributes meaningfully to multiple strong extractive summaries, then it is more relevant.

In both categories, the final extractive summary is constructed by selecting the top-K ranked sentences. K is determined via grid search in the range [1, 32].

#### Objectives
Since there is a lack of Datasets for extractive summarization, but we have tons of abstractive summarization datasets, we focus on try to **transform abstractive summaries into their respective extractive verison**. 
The four heuristics described above have been implemented in order to predict the extractive summaries starting from the abstractive version, their results have then been analysed. 

#### Used Data
In this case both, **Extractive** and **Abstractive** summaries have been considered. The Abstractive Summaries are used as input for our heuristics in order to get the Extractive version, then the Extractive Gold labels have been used to compare the results.

#### Evaluation metrics
The *similarity* has been computed using **ROUGE** metrics. In order to have a wider overview about the performances of our heuristics, also reference-free metrics has been used (such as **SummaC-Conv**). 


### Track C
In this track we took the best heuristic according to (B) results on ACLSum and uses it to actually generate extractive versions of popular abstractive datasets (e.g., CNN/DM, XSUM).
Once done, we took the code from (A) and use it to assess LLM extractive summarization capabilities on these novel silver datasets, extending (A) benchmark, originally limited to one dataset only.
