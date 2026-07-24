# Mikita Daroshkin

AI/ML engineer. I build production LLM systems and the ML underneath them.

Currently forward deployed on enterprise agentic AI. The parts I find most
interesting are the ones where an LLM system meets a real constraint, which is
usually evaluation, retrieval quality, or inference economics rather than the
model itself.

## Areas

What I work in and what I am interested in. Which of these a given project
actually needs varies a lot.

- Agentic systems: LangGraph, LangChain, Google ADK, CrewAI, DSPy, A2A, MCP, Vertex AI Agent Engine, Amazon Bedrock, multi-agent orchestration, function calling, grammar-constrained decoding, JSON-schema tool contracts, ReAct, Plan-and-Execute, graphs as checkpointed state machines, human-in-the-loop interrupts, episodic and semantic memory, maker-checker and critic-verifier loops, self-consistency, log-prob confidence gating, deterministic fallbacks, bounded retries, circuit breakers, idempotent tools, token-exchange identity, sandboxed execution, egress control, prompt-injection testing

- Evaluation: golden datasets, LLM-as-judge, position-bias correction, calibrated rubrics, pairwise comparison, faithfulness and groundedness, bootstrap confidence intervals, recall@k, nDCG, MRR, deterministic replay from recorded fixtures, OpenTelemetry tracing, span-level token accounting, CI regression gates, shadow traffic, canaries, drift monitoring

- Retrieval: hybrid dense and BM25, reciprocal-rank fusion, cross-encoder and ColBERT late-interaction reranking, SPLADE, HyDE, step-back rewriting, query decomposition and routing, contextual chunking, parent-document and sentence-window expansion, metadata filtering, hard-negative mining, contrastive and Matryoshka embeddings, HNSW, IVF-PQ, scalar and binary quantization, incremental indexing, pgvector, FAISS, Qdrant, Elasticsearch, GraphRAG, knowledge graphs in Neo4j, self-correcting text-to-SQL

- Serving: vLLM, TensorRT-LLM, SGLang, continuous batching, paged and radix KV-cache, prefix reuse, chunked prefill, prefill/decode disaggregation, speculative and assisted decoding, FlashAttention, CUDA graphs, torch.compile, FP8 and INT4, KV-cache quantization, custom Triton kernels, model routing, multi-LoRA, SLA-aware scheduling, time-to-first-token, inter-token latency, p99 under load

- Training: FSDP, DeepSpeed ZeRO-3, tensor, pipeline and sequence parallelism, gradient accumulation, activation recomputation, sharded checkpointing, sequence packing, NCCL-tuned collectives, BF16 and FP8, LoRA and QLoRA, SFT, DPO, RLHF, reward modeling, distillation

- Foundations: Python, PyTorch, TensorFlow, Hugging Face Transformers, scikit-learn, XGBoost, tiled CUDA with shared-memory tiling and warp-level primitives, MPI over InfiniBand, ViT, 3D U-Net, Mask R-CNN, CLIP, recommender systems, gradient boosting, probability calibration, conformal prediction, causal inference, lock-free streaming and distributed systems

- Platform: GCP, AWS, Azure, Vertex AI, SageMaker, Azure OpenAI, Docker, Kubernetes, Terraform, CI/CD, Airflow, model monitoring, A/B testing, Databricks, Snowflake, BigQuery, Spark and PySpark, Kafka, dbt, Postgres, MongoDB, Palantir Foundry, Salesforce Agentforce, ServiceNow, Workday, FHIR, NIST CSF, ISO 42001, EU AI Act, HIPAA, GxP

## Repositories

Client work is closed, so these are the public equivalents: reference
implementations small enough to clone and run. Each one builds and tests in CI.

- [agentic-rag-evals](https://github.com/mikitadaroshkin/agentic-rag-evals): agentic RAG in LangGraph, gated in CI on eval-metric regression rather than on tests alone.
- [gpu-mpi-search](https://github.com/mikitadaroshkin/gpu-mpi-search): BM25 with cross-encoder rerank, tiled CUDA kernels, data-parallel scoring over MPI.
- [mri-segmentation](https://github.com/mikitadaroshkin/mri-segmentation): 3D U-Net for volumetric segmentation, soft-Dice/BCE, patch-based inference.
- [warehouse-patterns](https://github.com/mikitadaroshkin/warehouse-patterns): dimensional modeling write-up with a runnable DuckDB star schema and SCD-2.
- [nutrizy](https://github.com/mikitadaroshkin/nutrizy): cross-modal retrieval over a shared text and image embedding space.
- [image-captioning-vit](https://github.com/mikitadaroshkin/image-captioning-vit): ViT encoder with a Transformer decoder.
- [greenbox](https://github.com/mikitadaroshkin/greenbox): CNN disease classifier and LightGBM growth model behind an MQTT edge pipeline.

## Contact

I write about the engineering at [mikitadaroshkin.com](https://mikitadaroshkin.com/), and email is the fastest way to reach me: mikitadaroshkin@gmail.com.
