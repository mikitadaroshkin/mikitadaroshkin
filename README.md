# Mikita Daroshkin

AI/ML engineer. Production LLM systems and the ML underneath.

```
agents         multi-agent orchestration, A2A, MCP, function calling, grammar-constrained decoding, ReAct, Plan-and-Execute, maker-checker, critic-verifier, self-consistency, log-prob confidence gating, deterministic fallbacks, bounded retries, circuit breakers, idempotent tools, token-exchange identity, sandboxed execution, egress control
evaluation     golden datasets, LLM-as-judge, position-bias correction, calibrated rubrics, faithfulness, groundedness, bootstrap confidence intervals, deterministic replay, span-level token accounting, regression gates in CI, OpenTelemetry
retrieval      hybrid dense + BM25, reciprocal-rank fusion, cross-encoder and ColBERT late-interaction rerank, SPLADE, HyDE, step-back, query decomposition, contextual chunking, hard-negative mining, contrastive and Matryoshka embeddings, HNSW, IVF-PQ, scalar and binary quantization, GraphRAG, self-correcting text-to-SQL
serving        continuous batching, paged and radix KV-cache, chunked prefill, disaggregated prefill/decode, speculative and assisted decoding, FlashAttention, CUDA graphs, torch.compile, FP8, INT4, KV-cache quantization, custom Triton kernels, model routing, multi-LoRA, SLA-aware scheduling
training       FSDP, DeepSpeed ZeRO-3, tensor/pipeline/sequence parallelism, activation recomputation, NCCL tuning, BF16, FP8, LoRA/QLoRA, SFT, DPO, reward modeling, distillation
vision + ml    ViT, 3D U-Net, Mask R-CNN, ResNet, CLIP, segmentation, detection, gradient boosting, probability calibration, conformal prediction, causal inference, uplift modeling, survival analysis, LSTMs, time series
systems + hpc  tiled CUDA (shared-memory tiling, warp primitives, coalesced access), MPI, InfiniBand, lock-free concurrency, Kafka, backpressure, idempotency, blue-green/shadow/canary, feature stores, model registry
```

## Stack

```
frameworks    PyTorch, JAX, LangGraph, LangChain, Google ADK, DSPy, Semantic Kernel, CrewAI, AutoGen, Megatron-LM, NeMo, scikit-learn, XGBoost, LightGBM
serving       vLLM, TensorRT-LLM, SGLang, Triton, TGI, LiteRT, ONNX Runtime; open-weight (Llama, Qwen, DeepSeek, Gemma, Mistral, Phi)
retrieval     pgvector, FAISS, Qdrant, Milvus, OpenSearch, Vertex AI Vector Search, Azure AI Search, Neo4j, Neptune
eval + obs    RAGAS, DeepEval, Promptfoo, Braintrust, LangSmith, Langfuse, Arize Phoenix, LlamaGuard, NeMo Guardrails
data          BigQuery, Snowflake, Databricks (Unity Catalog, DBRX, Delta Live Tables), Spark, dbt, Dagster, Airflow, DuckDB, Iceberg, Kafka, Trino
cloud         GCP (Vertex AI, GKE), AWS (SageMaker, Bedrock, Trainium), Azure (Azure OpenAI, ML), Docker, Kubernetes, Terraform, Argo, MLflow, Weights and Biases
enterprise    Palantir Foundry/AIP, Salesforce Agentforce, ServiceNow, SAP, Microsoft Power Platform, Epic/FHIR
governance    NIST CSF, ISO 42001, EU AI Act, GDPR, HIPAA, GxP, on-prem and sovereign, multi-tenancy, PII redaction, RBAC
languages     Python, C++/CUDA, SQL, TypeScript, Go, Java, C#, R
```

## Repositories

- [agentic-rag-evals](https://github.com/mikitadaroshkin/agentic-rag-evals): LangGraph agentic RAG, hybrid retrieval, LLM-as-judge eval suite, CI gate on eval-metric regression, OpenTelemetry, FastAPI, Docker.
- [gpu-mpi-search](https://github.com/mikitadaroshkin/gpu-mpi-search): BM25, cross-encoder rerank, tiled CUDA kernels, data-parallel scoring over MPI and multiprocessing.
- [mri-segmentation](https://github.com/mikitadaroshkin/mri-segmentation): 3D U-Net, soft-Dice/BCE loss, patch-based volumetric inference.
- [image-captioning-vit](https://github.com/mikitadaroshkin/image-captioning-vit): ViT encoder, Transformer decoder, image captioning.
- [nutrizy](https://github.com/mikitadaroshkin/nutrizy): cross-modal retrieval over a shared text and image embedding space, CLIP.
- [greenbox](https://github.com/mikitadaroshkin/greenbox): CNN disease classifier, LightGBM growth model, MQTT edge pipeline.
- [warehouse-patterns](https://github.com/mikitadaroshkin/warehouse-patterns): dimensional modeling, star schema, SCD-2, runnable DuckDB demo.

[mikitadaroshkin.com](https://mikitadaroshkin.com/) | mikitadaroshkin@gmail.com
