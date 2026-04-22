# LLM Pretraining Data Pipeline

This syllabus covers the full data pipeline for building a high-quality pretraining corpus in a multilingual setting, from extracting raw text out of PDF documents, through deduplication, heuristic filtering, toxicity removal, and PII redaction, to model-based quality classification. Each stage is treated not as an isolated step but as part of a coherent engineering system, where decisions made early in the pipeline (such as deduplication ordering or language detection thresholds) have downstream consequences on corpus quality and model behavior. The primary languages covered are English (EN) and Bahasa Indonesia (ID). English serves as the baseline language for most open-source pipeline tooling and research, while Indonesian is used as a representative Southeast Asian language to ground the discussion in real multilingual challenges — including code-switching, morphological richness, script variation, and the relative scarcity of high-quality Indonesian web text compared to English. Participants who complete this syllabus should be equipped to adapt the same pipeline to other SEA languages such as Thai, Vietnamese, or Tagalog with minimal additional effort.

## Table of Contents
- [Rationale](#rationale)
- [Overview of Pipeline](#overview)
- [1.1 PDF Extraction](#1_1-pdf-extraction)
- [1.2 Language Detection](#1_2-language-detection)
- [1.3 Deduplication](#1_3-deduplication)
- [1.4 Filtering](#1_4-filtering)
- [1.5 Toxicity Removal](#1_5-toxicity-removal)
- [1.6 Quality Classification](#1_6-quality-classification)
- [1.7 PII Protection](#1_7-pii-protection)

## Rationale

Building a high-quality pretraining corpus is one of the most important steps in LLM development. The pretraining corpus is the foundation upon which everything else is built: the model's world knowledge, its linguistic fluency, its reasoning patterns, and even its safety-relevant behaviors all emerge from statistical patterns in that raw data. Poor data quality compounds silently across billions of gradient updates, and by the time the damage is visible in evaluation benchmarks or downstream deployment, it is deeply baked into the model weights. No amount of post-training whether through RLHF, DPO, or instruction fine-tuning can fully recover knowledge that was never there, or reliably suppress behaviors that were rehearsed at trillion-token scale. This is not merely theoretical, empirical results from FineWeb, DCLM, and the Phi series have shown that a smaller model trained on a carefully curated corpus routinely outperforms a larger model trained on noisier data. Ultimately, data quality is paramount in model development, and the pipeline stages covered in this syllabus are the engineering discipline through which that quality is achieved.


**Recommended background reading:**
- [GneissWeb: Preparing High Quality Data for LLMs at Scale](https://arxiv.org/pdf/2502.14907) - IBM,2025
- [Ultra-FineWeb: Efficient Data Filtering and Verification for High-Quality LLM Training Data](https://arxiv.org/pdf/2505.05427)
- [Sailor: Open Language Models for South-East Asia](https://arxiv.org/abs/2404.03608) — Sea AI Lab, 2024
- [Nemotron-CC: Transforming Common Crawl into a Refined Long-Horizon Pretraining Dataset](https://arxiv.org/pdf/2412.02595) - NVIDIA,2024
- [Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research](https://arxiv.org/abs/2402.00159) — AI2, 2024
- [The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale](https://arxiv.org/abs/2406.17557) — Hugging Face, 2024
- [RedPajama-Data-V2: An Open Dataset with 30 Trillion Tokens for Training Large Language Models](https://www.together.ai/blog/redpajama-data-v2) — Together AI, 2023
- [CulturaX: A Cleaned, Enormous, and Multilingual Dataset for Large Language Models in 167 Languages](https://arxiv.org/pdf/2309.09400) - Adobe Research, 2023
- [MADLAD-400: A Multilingual And Document-Level Large Audited Dataset](https://arxiv.org/abs/2309.04662) — Google, 2023
- [mC4: mT5: A Massively Multilingual Pre-Trained Text-to-Text Transformer](https://arxiv.org/pdf/2103.12028) — Google, 2021
- [Trafilatura: A Web Scraping Library and Command-Line Tool for Text Discovery and Extraction](https://github.com/adbar/trafilatura) — Adrien Barbaresi, 2021
- [The Pile: An 800GB Dataset of Diverse Text for Language Modeling](https://arxiv.org/abs/2101.00027) — EleutherAI, 2020
- [CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data](https://arxiv.org/abs/1911.00359) — Facebook AI Research, 2019


## Overview of Pipeline

The typical pretraining data pipeline transforms raw, heterogeneous web/document data into a clean, filtered, deduplicated corpus. Each stage below addresses a specific quality or safety concern.

```
Raw Source (Common Crawl / PDFs / Web Scraped HTML)
        │
        ▼
  [1.1] Extraction (PDF, HTML → plain text)
        │
        ▼   
  [1.2] Language Detection    
        │
        ▼
  [1.3] Deduplication (exact + fuzzy)
        │
        ▼
  [1.4] Filtering (heuristic rules)
        │
        ▼
  [1.5] Toxicity Removal
        │
        ▼
  [1.6] Quality Classification (perplexity / model score)
        │
        ▼
  [1.7] PII Removal
        │
        ▼
   Final Corpus
```

---

### 1.1 PDF Extraction

**Goal:** Convert unstructured PDF documents into clean, machine-readable plain text suitable for language model training.

PDFs are among the richest sources of high-quality written text — academic papers, government reports, legal documents, and books exist in this format. However, the PDF specification was designed for visual presentation, not text extraction. This fundamental mismatch makes reliable extraction difficult, and naive extraction tools might produce garbled, incomplete, or structurally wrong text.

#### Key Challenges

- **Two-column and multi-column layouts**
    - Most PDF extraction tools read text in left-to-right, top-to-bottom order across the full page width. For a two-column academic paper, this means the tool will interleave the left and right columns line by line, producing incoherent text.
- **Scanned PDFs (image-only)**
    -Many older documents — especially government archives, older academic papers were digitized by scanning physical pages. These PDFs contain no embedded text at all. Each page is simply an image. They require a separate OCR (Optical Character Recognition) step, typically using Tesseract or a deep learning-based model, which rasterizes each page and predicts the character sequence from the image. OCR introduces its own error rate, which is significantly higher for non-English scripts and low-resolution scans.
- **Mathematical equations and tables**
    - Equations in academic papers are frequently embedded as vector graphics or encoded using ligature fonts that don't map cleanly to Unicode characters. Tables present a similar problem: the spatial grid structure that makes a table readable is invisible to a text-layer extractor, which will often flatten all cell values into a single undifferentiated stream of numbers and words.
- **Boilerplate noise**
    - PDFs contain a large amount of text that is visually useful but semantically useless for language model training: page numbers, running headers and footers (e.g., "Journal of Computational Linguistics, Vol. 12"), watermarks, footnote markers, and figure captions. If not removed, these fragments bleed into the extracted text and create incoherent sentence boundaries, spurious repetitions, and noisy n-grams.

#### Post-Extraction Cleaning

Raw extracted text is rarely usable without cleaning. Even from a clean digital PDF, the extracted text will typically contain artifacts that need to be resolved before the document enters the rest of the pipeline.

After text is extracted:
- Strip headers and footers (heuristics: short lines near page boundaries, repeated strings)
- Remove or normalize hyphenation (e.g., `en-\nvironment` → `environment`)
- Normalize Unicode (NFC normalization; handle ligatures like `ﬁ` → `fi`)
- Remove lines with high symbol/number ratio (often figure captions or reference lists)


#### Other Tools & Libraries

- [`pdfminer.six`](https://github.com/pdfminer/pdfminer.six)
- [`pdfplumber`](https://github.com/jsvine/pdfplumber)
- [`PyMuPDF`](https://pymupdf.readthedocs.io/)
- [`Nougat`](https://github.com/facebookresearch/nougat)
- [`Tesseract OCR`](https://github.com/tesseract-ocr/tesseract)

---
### 1.2 Language Detection

**Goal:** Identify the primary language of each document so that language-specific processing (filters, quality classifiers) can be applied correctly.

Language detection is a prerequisite for any multilingual pipeline. Every downstream stage in this pipeline is language-sensitive: heuristic filters use language-specific stopword lists, KenLM quality models are trained per language, and toxicity classifiers are often monolingual. Routing a document to the wrong language processor is not a benign error — an Indonesian document scored by an English KenLM model will receive meaningless perplexity values, and an English document filtered by Indonesian stopword thresholds will be incorrectly penalized. Getting language detection right, cheaply and at scale, is therefore one of the most foundational decisions in the pipeline.

#### How Language Detection Works
There are three approaches for language detection:

1. N-gram frequency
    - Each language is represented as a frequency distribution over character n-grams (typically 1–3 characters). At inference time, the n-gram profile of the input document is compared against each language's reference profile using a distance metric, and the closest match wins. This approach is fast, requires no GPU, and works well for long documents in common languages, but degrades significantly on short texts (< 50 characters) and on closely related languages that share large portions of their n-gram vocabulary (e.g Bahasa Indonesia and Bahasa Melayu)
2. Supervised classification (fastText) 
    — fastText LID and GlotLID train a linear classifier over n-gram features using the fastText framework. The model is trained on hundreds of millions of examples across many languages and learns discriminative character-level features rather than relying on simple frequency comparisons. This is significantly more accurate than n-gram profiles, especially for short texts and closely related languages, and runs at hundreds of thousands of documents per second on CPU — making it suitable for billion-document pipelines.
3. Neural classifiers 
    — cld3 (Google's Compact Language Detector 3) uses a small feedforward neural network over character n-gram embeddings. It offers higher accuracy than fastText on some edge cases (especially code-switched text) but is slower and harder to deploy at scale.

#### Tools Comparison

| Tool | Method |
|------|--------|
| `langdetect` | Naive Bayes (Google) |
| `langid.py` | Logistic regression |
| `fastText LID` | fastText classifier (176 langs) |
| `lingua-py` | Ensemble + n-gram |
| `cld3` (Google) | Neural Network |
| `GlotLID` | fastText (1600+ langs) |

#### Best Practices
- Applying a single language label to an entire document works well for monolingual content, but Indonesian web text frequently mixes languages within a single document. English technical terms, Malay phrases, and Javanese words may all appear in an otherwise Indonesian article. Document-level detection will correctly label this as Indonesian, but the mixed content may confuse downstream filters. For higher precision, apply detection at the paragraph level using a sliding window, then aggregate. Essentially, keep the document if more than a threshold fraction of paragraphs are in the target language.
- Both fastText and GlotLID return a confidence score alongside the predicted label. Documents near the decision boundary e.g. very short documents, heavily code-switched text, or documents with many proper nouns will receive low confidence scores. Setting a minimum confidence threshold (typically 0.5–0.7) and routing low-confidence documents to a "unknown" bucket rather than forcing a label prevents systematic misclassification
- Indonesian documents — especially from blogs, news comments, and social media — frequently embed Javanese (jv), Sundanese (su), Balinese (ban), or Minangkabau (min) phrases within otherwise Indonesian text. These are not errors; they reflect natural code-switching in Indonesian discourse. However, a document-level detector may label such documents as Javanese or Sundanese if the regional language content is dense enough, causing valid Indonesian documents to be dropped.

```python
import fasttext
model = fasttext.load_model("lid.176.bin")

def detect_lang(text: str, threshold: float = 0.5):
    text_clean = text.replace("\n", " ")
    labels, scores = model.predict(text_clean, k=1)
    lang = labels[0].replace("__label__", "")
    return lang if scores[0] >= threshold else "unknown"
```

#### Tools & Libraries

- [`fastText LID`](https://fasttext.cc/docs/en/language-identification.html)
- [`GlotLID`](https://github.com/cisnlp/GlotLID) 
- [`lingua-py`](https://github.com/pemistahl/lingua-py)
- [`langdetect`](https://github.com/Mimino666/langdetect)

#### Optional Reads

- [GlotLID: Language Identification for Low-Resource Languages](https://arxiv.org/abs/2310.16248) — Kargaran et al., 2023 
- [A Robust Self-Learning Method for Fully Unsupervised Cross-Lingual Mappings of Word Embeddings](https://arxiv.org/abs/1805.06297) — Conneau et al., 2018
- [No Language Left Behind: Scaling Human-Centered Machine Translation](https://arxiv.org/abs/2207.04672) — NLLB Team, Meta, 2022

---
### 1.3 Deduplication

**Goal:** Remove duplicate and near-duplicate documents to prevent memorization, reduce training data imbalance, and improve generalization.

Deduplication is the single most impactful data cleaning operation relative to its implementation cost. The intuition is straightforward: if the same document (or a near-identical variant of it) appears thousands of times in your training corpus, the model will see it thousands of times during training and will disproportionately memorize it. This manifests in two concrete failure modes. First, the model becomes more likely to reproduce that content verbatim when prompted, which is both a quality problem (the model is reciting rather than generalizing) and a privacy/copyright risk if the duplicated content contains PII or proprietary text. Second, duplicated content distorts the effective data distribution — a piece of content that represents 0.001% of the real world's written text but 5% of your training corpus will have an outsized influence on the model's learned representations.Empirical evidence for the importance of deduplication is strong. Research showed that training on deduplicated data improves perplexity and few-shot task performance, and that models trained on deduplicated data memorize significantly less training content. The FineWeb dataset found that aggressive near-deduplication was one of the single largest quality improvements in their pipeline. Critically, deduplication must happen before quality filtering, otherwise high-quality documents that happen to be widely syndicated or republished would be over-represented in the final corpus after filtering, precisely because they pass quality filters more reliably.

#### Types of Deduplication

| Level | Granularity | Method |
|-------|-------------|--------|
| Exact deduplication | Full document | MD5/SHA hash of normalized text |
| Near-deduplication | Document-level | MinHash + LSH (Locality-Sensitive Hashing) |
| Semantic deduplication | Meaning-level | Embedding similarity (e.g., SemDeDup) |

#### MinHash LSH (Most Common Approach)

MinHash converts a document into a fixed-size sketch (a set of hash values) that approximates Jaccard similarity between sets of shingles (n-grams). MinHash solves this by approximating the Jaccard similarity using a set of hash functions: for each of k hash functions, take the minimum hash value over all shingles in the document. The resulting vector of k minimum hash values is the MinHash signature. The key property is that the probability that two documents have the same minimum hash under a given function equals their Jaccard similarity, so the fraction of matching values across k functions gives an unbiased estimate of their Jaccard similarity.LSH then groups documents with similar signatures into candidate buckets for pairwise comparison.

**Key hyperparameters:**
- Shingle size (typically 5-13 character n-grams)
- Number of hash functions / permutations (typically 128–256)
- Jaccard similarity threshold (typically 0.7–0.85)

#### Tools & Libraries

- [`datasketch`](https://github.com/ekzhu/datasketch) — MinHash LSH
- [`text-dedup`](https://github.com/ChenghaoMou/text-dedup) — Comprehensive dedup toolkit (MinHash, SimHash)
- [`datatrove`](https://github.com/huggingface/datatrove) — HuggingFace's industrial pipeline with built-in dedup

#### Optional Reads
- [SemDeDup: Data-efficient learning at web-scale through semantic deduplication](https://arxiv.org/abs/2303.09540) — Abbas et al., 2023
---

### 1.4 Filtering

**Goal:** Remove low-quality, malformed, or irrelevant documents using fast heuristic rules before applying computationally expensive model-based classifiers.

Heuristic filtering is the first quality gate in the pipeline, and its design reflects a fundamental engineering tradeoff: model-based quality classifiers are far more accurate at identifying high-quality text, but they require running inference on every document in the corpus, i.e an operation that can cost thousands of GPU-hours at billion-document scale. Heuristic filters, by contrast, are pure CPU operations that run in microseconds per document. The goal of heuristic filtering is therefore not to perfectly identify quality, but to cheaply eliminate the long tail of documents that are obviously malformed, structurally broken, or linguistically incoherent, shrinking the corpus enough that model-based filtering becomes computationally tractable. A well-designed heuristic filter should have high recall on bad documents (it must catch obviously low-quality content reliably) and tolerable precision (it is acceptable to occasionally discard a borderline document, but catastrophic to discard large fractions of genuinely good content). The filters described below are drawn from the published pipelines of Gopher (DeepMind), C4 (Google), CCNet (Facebook), and RedPajama (Together AI), which collectively represent the most widely cited and replicated approaches in the field.

The intuition behind heuristic filters is that high quality data has predictable statistical properties. For example, a well-written news article will have sentences that are adequately long and typically does not contain too many symbols. Documents that deviate significantly form these norms (e.g advertisement, boilerplate content) are highly likely to be considered low quality. These structural anomalies are taken as proxies for contents that we want to exclude
#### Common Heuristic Filters

**Document-level statistics:**
- **Length filters** 
    — minimum/maximum word count or character count (e.g., discard docs < 100 words)
- **Mean word length** 
    — very low (< 3 chars) or very high (> 10 chars) often signals garbled text
- **Symbol-to-word ratio** 
    — high ratio of `#`, `|`, `{`, `}` suggests code or structured data leaking in

**Line-level statistics:**
- **Bullet ratio** 
    — too many bullet-point lines suggests lists, not prose
- **Short line ratio** 
    — too many very short lines indicates formatting artifacts
- **Ellipsis ratio** 
    — excessive `...` often signals truncated or auto-generated content

**N-gram and vocabulary checks:**
- Ratio of stopwords to total tokens (very low ratio = likely code or garbled text)
- Presence of common boilerplate phrases ("click here", "terms and conditions", "cookie policy")
- Repeated n-gram detection within a single document (copy-pasted blocks)

#### Reference Implementations

- **CCNet pipeline** — applies perplexity + length/character filters
- **C4 filters** — list sentence endings, remove lines with "javascript", phone numbers, etc.
- **Gopher rules** — comprehensive heuristic set: word count, mean word length, bullet ratio, symbol ratio
- **RedPajama** — extends Gopher rules with additional checks

```python
# Example: Gopher-style filters
def passes_gopher_filters(doc: str) -> bool:
    words = doc.split()
    if not (50 <= len(words) <= 100_000):
        return False
    mean_word_len = sum(len(w) for w in words) / len(words)
    if not (3 <= mean_word_len <= 10):
        return False
    if sum(1 for w in words if w.endswith(("...",))) / len(words) > 0.3:
        return False
    return True
```

An important thing to note is that when applying filters to Indonesian text, the thresholds derived from English data does not apply for Indonesian text due to linguistic differences and there is a high chance of misclassification. 

#### Tools & Libraries

- [`datatrove`](https://github.com/huggingface/datatrove) — includes `GopherQualityFilter`, `GopherRepetitionFilter`, `C4QualityFilter`
- [`dolma`](https://github.com/allenai/dolma) — AllenAI's data curation toolkit
- [`data-juicer`](https://github.com/modelscope/data-juicer) - Alibaba's pipeline 

#### Optional Reads

- [Scaling Language Models: Methods, Analysis & Insights from Training Gopher](https://arxiv.org/abs/2112.11446) — Rae et al., DeepMind, 2021 
- [Documenting the English Colossal Clean Crawled Corpus (C4)](https://arxiv.org/abs/2104.08758) — Dodge et al., 2021
- [Data-Juicer: A One-Stop Data Processing System for Large Language Models](https://arxiv.org/abs/2309.02033) — Chen et al., Alibaba, 2023
---

### 1.5 Toxicity Removal

**Goal:** Detect and remove documents containing hate speech, explicit content, violent or offensive material, and other harmful content before it enters the training corpus.

Pretraining on toxic content leads to models that are more likely to produce harmful outputs. Unlike factual errors or low-quality prose which degrade model performance in ways that are visible in benchmarks. Toxic content baked into pretraining weights manifests as latent behaviors that only surface under specific prompting conditions. A model pretrained on a corpus containing extremist content may pass all standard safety evaluations and yet produce harmful outputs when prompted in particular ways, because the underlying representations were shaped by that content long before safety fine-tuning was applied. Post-training alignment techniques like RLHF and DPO can suppress the expression of these behaviors but cannot fully erase the underlying representations. They work by steering the model's outputs, not by rewriting its weights. This is why toxicity removal at the data level is not merely an ethical checkbox but a foundational engineering requirement: it is far cheaper and more reliable to exclude harmful content from training than to attempt to suppress its effects afterward.
Toxicity removal is also distinct from quality filtering in an important way. A document can be high-quality by every linguistic and informational metric: well-written, coherent, factually dense and still be deeply toxic. A well-articulated extremist manifesto or a persuasively written piece of radicalization material would pass every heuristic quality filter and score highly under a KenLM or FineWeb-Edu classifier. Toxicity removal requires its own dedicated pipeline stages that are explicitly designed to detect harmful content regardless of its linguistic quality.

#### Threat Model

| Category | Examples |
|----------|---------|
| Hate speech | Slurs, content targeting protected groups |
| Sexually explicit content | Pornographic text |
| Violence/gore | Graphic descriptions of violence |
| Extremist content | Terrorist manifestos, radicalization material |
| Self-harm | Detailed instructions for self-harm or suicide |
| Spam / SEO abuse | Keyword-stuffed, manipulative content |

#### Approaches

**Keyword/n-gram filters:**
- Keyword filters maintain a list of high-toxicity terms and phrases and discard or flag documents that contain them above a density threshold. They are fast (a simple string search), require no model inference, and are completely transparent and auditable. Applying a filter that discards any document containing a toxic term will discard large amounts of legitimate content: news articles reporting on hate crimes, academic papers studying extremist rhetoric, historical documents, and fiction that depicts violence or discrimination. A density threshold (e.g., discard documents where toxic terms exceed 1% of total tokens) is more appropriate for most categories, reserving zero-tolerance filtering for the most severe categories
- A keyword list curated for English will have near-zero recall on Indonesian toxic content, which uses a completely different vocabulary of slurs, obscenities, and hate speech terms. Indonesian-specific wordlists must be curated separately, and they require ongoing maintenance as slang evolves rapidly on Indonesian social media.

**Classifier-based:**
- Classifiers trained on labeled toxic content datasets offer significantly higher precision and recall than keyword filters, particularly for implicit hate speech, coded language, and context-dependent toxicity. The tradeoff is inference cost, running a BERT-class classifier over a billion-document corpus requires substantial GPU time.
- `Perspective API` (Google Jigsaw) — REST API, English-focused
- `detoxify` — lightweight open-source Unitary AI model, multilingual variants
- `HateBERT` — BERT fine-tuned on Reddit hate speech
- `toxic-comment-classifier` — multi-label, available on HuggingFace

**LLM-based:**
- Using an aligned LLM (GPT-4o, Claude, or an open-source equivalent) as a toxicity judge offers the highest accuracy, particularly for nuanced cases — context-dependent hate speech, coded language, dog whistles, and content that requires cultural knowledge to interpret. An LLM judge can be given a detailed rubric and can provide reasoning for its decisions, making it auditable in a way that a classifier score is not.The practical limitation is cost. At $0.01–0.15 per thousand tokens for frontier model APIs, scoring a billion documents is simply not economically feasible. LLM-based scoring is therefore used selectively: for validation of a classifier's decisions on a sampled subset, for auditing the final corpus, or for annotating a dataset that will be used to train a cheaper dedicated classifier.

#### Tools & Libraries

- [`detoxify`](https://github.com/unitaryai/detoxify)
- [`Perspective API`](https://perspectiveapi.com/)
- [`datatrove`](https://github.com/huggingface/datatrove) — includes URL-based adult content filtering

#### Optional Reads

- [ToxiGen: A Large-Scale Machine-Generated Dataset for Adversarial and Implicit Hate Speech Detection](https://arxiv.org/abs/2203.09509) — Hartvigsen et al., 2022
- [HateBERT: Retraining BERT for Abusive Language Detection](https://arxiv.org/abs/2010.12472) — Caselli et al., 2021
- [A Survey on Hate Speech Detection using Natural Language Processing](https://aclanthology.org/W17-1101/) — Schmidt and Wiegand, 2017

---
### 1.6 Quality Classification

**Goal:** Score documents by their linguistic and informational quality, keeping only documents that contribute positively to language model training.

The heuristic filters that we saw earlier are designed to catch documents that are structurally broken: garbled OCR, navigation menus, symbol-heavy markup artifacts. Quality classification addresses a fundamentally harder problem: distinguishing between documents that are linguistically valid but still not worth training on. A coherent, well-formatted blog post about celebrity gossip, a keyword-stuffed SEO article that reads fluently, and a Wikipedia article on thermodynamics are all grammatically correct and would pass every heuristic filter — but their value as pretraining data differs enormously. Quality classification is the mechanism that makes these distinctions. There is no formal definition for "good/high quality" dataset, however they can identified based on the following characteristics:

- Linguistic fluency — Is the text grammatically correct, coherent, and well-formed?
- Informational density — Does the text contain facts, reasoning, and knowledge, or is it mostly filler?
- Educational value — Could a reader learn something from this document?
- Coherence and structure — Does the document develop an argument or narrative, or is it a random collection of sentences?


> **Note:** This section provides an overview. Day 2A covers quality classification in depth with hands-on implementations.

#### Approaches

1. **Perplexity-based (KenLM)** 
    - Perplexity-based quality filtering uses a language model trained on a high-quality reference corpus to score documents in the larger noisy corpus. The core assumption is that high-quality text is predictable to a model trained on high-quality text — it uses familiar vocabulary in familiar patterns, follows expected syntactic structures, and produces low surprisal scores. Low-quality text — garbled content, spammy keyword stuffing, machine-generated boilerplate — is more "surprising" to the reference model and receives higher perplexity scores.
2. **Classifier-based**
    - Neural classifiers represent the current practical state of the art for quality classification at scale. The approach treats quality classification as a supervised learning problem: collect a labeled dataset of (document, quality score) pairs, train a classification or regression head on top of a pretrained language model backbone, and use the trained classifier to score the full corpus. The quality labels can come from human annotation, heuristic proxies or llm annotations
3. **LLM-as-judge**
    - Using a frontier LLM (GPT-4o, Claude Opus, or a large open-source model like Llama-3-70B) as a quality judge provides the highest-fidelity quality signal available. An LLM judge can evaluate quality along multiple dimensions simultaneously, apply nuanced reasoning about context and purpose, handle edge cases that stump rule-based systems, and provide explicit reasoning for its decisions that can be audited and used to improve downstream classifiers

#### Optional Reads

- [The RefinedWeb Dataset for Falcon LLM](https://arxiv.org/abs/2306.01116) — Penedo et al., 2023
- [FineWeb: Decanting the Web for the Finest Text Data at Scale](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) — HuggingFace, 2024 
- [DCLM: Data Curation for Language Models at Scale](https://arxiv.org/abs/2406.11794) — Li et al., 2024
- [Textbooks Are All You Need (Phi-1)](https://arxiv.org/abs/2306.11644)
---

### 1.7 PII Protection

**Goal:** Detect and remove or redact Personally Identifiable Information (PII) from pretraining data to prevent models from memorizing and regurgitating private information.

PII protection is the final safety gate in the pipeline before the corpus enters training, and its importance is grounded in a concrete and well-documented empirical phenomenon: large language models memorize their training data. The memorization risk scales with model size, training duration, and crucially with the frequency of occurrence in the training data: PII that appears many times (due to insufficient deduplication or high natural frequency) is memorized with near certainty. A phone number that appears in a leaked credential dump scraped into Common Crawl will be reproduced by a model trained on that data whenever prompted with the associated name.The legal exposure compounds the ethical risk. In Singapore, the Personal Data Protection Act (PDPA) restricts the collection and processing of personal data without consent. Building a pipeline that systematically removes PII is a best practice for data curation pipeline

#### PII Categories

| Category | Examples | Detection Method |
|----------|---------|-----------------|
| Email addresses | `user@example.com` | Regex |
| Phone numbers | `+65 1234 5678` | Regex (locale-aware) |
| IP addresses | `192.168.1.1` | Regex |
| Social security / ID / NRIC numbers | SXXXXXXXX | Regex |
| Credit card numbers | `4111 1111 1111 1111` | Regex |
| Names (person) | "John Smith" | NER model |
| Addresses | "Blk 123, Tanglin Road" | NER model |
| Dates of birth | `01/01/1990` | Regex + context |

#### Approaches

**Regex-based:**
- Regex patterns are the first and most reliable tool for structured PII — information that follows a predictable format regardless of context. Email addresses, phone numbers, IP addresses, and national ID numbers all have well-defined structural patterns that can be matched with high precision.It requires locale specific patterns for it to be useful

**NER-based:**
- Named Entity Recognition is required for unstructured PII information that does not follow a fixed format and requires linguistic context to identify. Person names, physical addresses, organization names (when linked to individuals), and medical conditions are the primary targets.NER-based PII detection introduces a fundamental precision-recall tradeoff that regex does not face. A person's name ("Ahmad", "Siti", "Budi") is indistinguishable at the token level from a common word used in a different context. The NER model must use surrounding context to determine whether "Ahmad" in a given sentence is a person's name being referenced specifically or a generic mention. False positives would be redacting common words that happen to be names which degrade corpus quality and create incoherent text. False negatives would be missing actual names and leaving the PII in the corpus. Microsoft Presidio is the most production-ready open-source PII detection framework. It combines regex recognizers for structured PII with NER-based recognizers for unstructured PII, supports custom recognizer plugins, and provides both redaction and anonymization output modes:

**LLM-based:**
LLM-based PII detection provides the highest recall on ambiguous, contextual, and culturally specific PII, cases where whether something constitutes PII depends on reasoning about the surrounding context rather than pattern matching. An LLM can correctly identify that "the patient in Room 3B" constitutes PII when combined with a named hospital and date, where a regex or NER model would miss the connection.

#### Redaction vs. Removal

Two strategies exist:
- Redaction (masking) replaces detected PII spans with typed placeholder tokens ([PERSON], [EMAIL], [PHONE], [NIK]). This preserves the document's linguistic structure, narrative flow, and all non-PII content. A redacted sentence like "Contact [PERSON] at [EMAIL] for more information" remains coherent and informative for language model training. The model learns the surrounding language patterns without learning the specific PII values. Redaction is the preferred strategy for documents that have isolated PII instances embedded in otherwise valuable content.
- Document removal discards the entire document when PII density is high enough that redaction would destroy the document's coherence and value. A leaked credential database, a directory of phone numbers, or a document that is primarily a list of names and addresses has no residual training value after redaction, removing the document entirely is more appropriate. A practical threshold is to remove documents where more than 20–30% of tokens would be redacted, or where more than a threshold number of distinct PII instances are detected per 1000 tokens.

Redaction is generally preferred as it preserves document context and fluency. However, if a document is primarily PII (e.g., a leaked credential dump), full removal is appropriate.


#### Tools & Libraries

- [`Microsoft Presidio`](https://github.com/microsoft/presidio) — modular PII detection + redaction ⭐
- [`scrubadub`](https://github.com/LeapBeyond/scrubadub) — Python PII scrubber
- [`spaCy`](https://spacy.io/) — NLP pipeline with NER

#### Optional Reads

- [Extracting Training Data from Large Language Models](https://arxiv.org/abs/2012.07805) — Carlini et al., 2021 
- [Deduplicating Training Data Mitigates Privacy Risks in Language Models](https://arxiv.org/abs/2202.06539) — Kandpal et al., 2022

---