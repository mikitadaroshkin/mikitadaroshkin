# Mikita Daroshkin

AI/ML engineer building production LLM systems and the ML underneath. Nine years, 30+ systems shipped, working forward deployed: one seat covering engineer, architect, and delivery lead.

## Agents

The largest so far is a multi-agent genomics mesh for a pharma research organisation, running on Vertex AI Agent Engine: specialist agents over an internal literature corpus that answer in 1.4 minutes what used to take a research team weeks. It now runs entirely on the client's own seven engineers, which was the point. Elsewhere, a fleet of Claude agents inside Workday, and a voice and chat booking agent on Bedrock over airline APIs behind guardrails.

That work is LangGraph and Google ADK, communicating over A2A and MCP. Tool calls go through function calling and grammar-constrained decoding. Planning is ReAct or Plan-and-Execute. The reliability layer carries the weight: maker-checker and critic-verifier loops, self-consistency, log-prob confidence gating with deterministic fallback. Tools stay idempotent behind bounded retries and circuit breakers. Agents authenticate by token exchange, execute sandboxed, and egress is controlled.

## Evaluation

I built the LLM evaluation program a Fortune 100 division standardised on, and I hold everything else to it.

Behavior is gated on evaluation in CI. Golden datasets scored by LLM-as-judge, with position-bias correction and calibrated rubrics. Faithfulness and groundedness report bootstrap confidence intervals. Runs replay deterministically; OpenTelemetry carries span-level token accounting. Regression thresholds block the merge. A prompt is code and should be able to fail the build.

## Retrieval

Twelve terabytes of construction documents behind a RAG lake in pgvector. A document-intelligence pipeline at 99.2% extraction accuracy that moved mortgage files from nine days to same-day. An admissions screening model calibrated to 94% precision at 90% recall across 250+ facilities and two million patient-days a month.

Retrieval starts hybrid: dense and BM25 fused by reciprocal rank, reranked by cross-encoder or ColBERT late interaction. SPLADE where the match must stay lexical but learned. Queries get HyDE, step-back rewriting, and decomposition. Chunking is contextual; embeddings come from contrastive training and hard-negative mining, Matryoshka where vectors shorten without retraining. Indexes are HNSW or IVF-PQ under scalar or binary quantization. GraphRAG or self-correcting text-to-SQL where a graph or a schema fits better than embeddings.

## Serving

An 8B multimodal assistant compressed 27x to run offline on a mid-tier Android phone, for field technicians working without signal. On the fleet side, 3 to 4x throughput per dollar against seven-figure inference budgets.

Serving goes down to the kernel. Continuous batching over paged and radix KV-cache, chunked prefill, prefill/decode disaggregation. Speculative and assisted decoding. FlashAttention, CUDA graphs, torch.compile, FP8 and INT4 against a quantized KV-cache, custom Triton kernels where the framework falls short. Model routing, multi-LoRA, and SLA-aware scheduling on vLLM, TensorRT-LLM, and SGLang.

## Training

Back when the frontier models were not good enough yet, I fine-tuned a GPT-2 that beat GPT-3 few-shot on the task at an eighteenth of the cost. That habit stuck: the smaller model that holds quality is usually the one that ships.

Training is FSDP or DeepSpeed ZeRO-3, sharded across tensor, pipeline, and sequence parallelism. Activation recomputation, NCCL collectives tuned to the interconnect, BF16 or FP8. Post-training is LoRA and QLoRA, SFT, DPO, reward modeling, and distillation.

## Foundations

Two years of senior ML research under FDA regulation came before any of the LLM work, and the layer under that is older still: tiled CUDA with shared-memory tiling and warp-level primitives, MPI over InfiniBand, computer vision on ViT, 3D U-Net, Mask R-CNN, and CLIP. Then classical ML with statistical discipline: gradient boosting, probability calibration, conformal prediction, causal inference, over lock-free streaming and distributed systems. It is why a calibration curve still comes before a bigger model.

## Delivery

All of the above shipped in regulated industries: pharma, healthcare, lending, aviation. Work starts in a repo and ends under audit.

GCP, AWS, and Azure. Data on Databricks, Snowflake, and BigQuery; vectors in pgvector, FAISS, or Qdrant; integrations into Palantir Foundry, Salesforce Agentforce, ServiceNow, and FHIR. Regulated domains ship under NIST CSF, ISO 42001, the EU AI Act, HIPAA, and GxP.

## Repositories

The client work is closed, so these are the public equivalents: reference implementations of the methods above, small enough to clone and run. Each one builds and tests in CI.

- [agentic-rag-evals](https://github.com/mikitadaroshkin/agentic-rag-evals): LangGraph agentic RAG, hybrid retrieval, LLM-as-judge eval suite, CI gate on eval-metric regression, OpenTelemetry, FastAPI, Docker.
- [gpu-mpi-search](https://github.com/mikitadaroshkin/gpu-mpi-search): BM25, cross-encoder rerank, tiled CUDA kernels, data-parallel scoring over MPI and multiprocessing.
- [mri-segmentation](https://github.com/mikitadaroshkin/mri-segmentation): 3D U-Net, soft-Dice/BCE loss, patch-based volumetric inference.
- [image-captioning-vit](https://github.com/mikitadaroshkin/image-captioning-vit): ViT encoder, Transformer decoder, image captioning.
- [nutrizy](https://github.com/mikitadaroshkin/nutrizy): cross-modal retrieval over a shared text and image embedding space, CLIP.
- [greenbox](https://github.com/mikitadaroshkin/greenbox): CNN disease classifier, LightGBM growth model, MQTT edge pipeline.
- [warehouse-patterns](https://github.com/mikitadaroshkin/warehouse-patterns): dimensional modeling, star schema, SCD-2, runnable DuckDB demo.

## Contact

I write about the engineering at [mikitadaroshkin.com](https://mikitadaroshkin.com/). Email is the fastest way to reach me: mikitadaroshkin@gmail.com.
