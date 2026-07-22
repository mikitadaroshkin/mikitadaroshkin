# Mikita Daroshkin

Production LLM systems, evaluation-gated, from CUDA kernels to multi-agent orchestration. Depth by area:

Agents. Reliability and identity over the happy path: maker-checker and critic-verifier loops, self-consistency, log-prob confidence gating, deterministic fallbacks, bounded retries with jitter, circuit breakers, idempotent tool calls, token-exchange identity between agents, sandboxed execution, egress control, grammar-constrained structured output.

Evaluation. Load-bearing, in CI: golden datasets, LLM-as-judge with position-bias correction and calibrated rubrics, faithfulness and groundedness scoring, bootstrap confidence intervals, deterministic replay, span-level trace and token accounting, merge-blocking regression gates, OpenTelemetry.

Retrieval. Where RAG quality is won: hybrid dense and BM25, reciprocal-rank fusion, cross-encoder and ColBERT late-interaction reranking, SPLADE learned-sparse, HyDE and step-back rewriting, query decomposition, contextual and semantic chunking, hard-negative mining, contrastive and Matryoshka embeddings, HNSW vs IVF-PQ, scalar and binary vector quantization, GraphRAG, self-correcting text-to-SQL.

Serving. Down to the kernel: continuous batching, paged and radix KV-cache, chunked prefill, disaggregated prefill and decode, speculative and assisted decoding, FlashAttention, CUDA graphs, torch.compile, FP8 and INT4 with KV-cache quantization, custom Triton kernels, model routing, multi-LoRA, SLA-aware scheduling. 8B stacks compressed 27x to run offline on-device; multi-GPU fleets at 3 to 4x throughput per dollar.

Training. FSDP, DeepSpeed ZeRO-3, tensor, pipeline, and sequence parallelism, activation recomputation, NCCL-tuned collectives, mixed precision BF16/FP8, LoRA/QLoRA, SFT, DPO, reward modeling, distillation, quantization-aware and speculative-draft training.

Foundations. Tiled CUDA with shared-memory tiling and warp-level primitives, MPI over InfiniBand, computer vision (ViT, 3D U-Net, Mask R-CNN, CLIP), classical ML (gradient boosting, calibration, conformal prediction, causal inference, uplift, survival analysis), lock-free concurrency, streaming and distributed systems.

## Repositories

- [agentic-rag-evals](https://github.com/mikitadaroshkin/agentic-rag-evals): LangGraph agentic RAG, hybrid retrieval, LLM-as-judge eval suite, CI gate on eval-metric regression, OpenTelemetry, FastAPI, Docker.
- [gpu-mpi-search](https://github.com/mikitadaroshkin/gpu-mpi-search): BM25, cross-encoder rerank, tiled CUDA kernels with shared-memory tiling, data-parallel scoring over MPI and multiprocessing.
- [mri-segmentation](https://github.com/mikitadaroshkin/mri-segmentation): 3D U-Net, soft-Dice/BCE loss, patch-based volumetric inference.
- [image-captioning-vit](https://github.com/mikitadaroshkin/image-captioning-vit): ViT encoder, Transformer decoder, image captioning.
- [nutrizy](https://github.com/mikitadaroshkin/nutrizy): cross-modal retrieval over a shared text and image embedding space, CLIP.
- [greenbox](https://github.com/mikitadaroshkin/greenbox): CNN disease classifier, LightGBM growth model, MQTT edge pipeline.
- [warehouse-patterns](https://github.com/mikitadaroshkin/warehouse-patterns): dimensional modeling, star schema, SCD-2, runnable DuckDB demo.

## Stack

```
frameworks    PyTorch, JAX, LangGraph, LangChain, Google ADK, DSPy, Semantic Kernel, CrewAI, AutoGen, Megatron-LM, NeMo, scikit-learn, XGBoost, LightGBM
serving       vLLM, TensorRT-LLM, SGLang, Triton, TGI, LiteRT, ONNX Runtime; open-weight (Llama, Qwen, DeepSeek, Gemma, Mistral, Phi)
retrieval     pgvector, FAISS, Qdrant, Milvus, OpenSearch, Vertex AI Vector Search, Azure AI Search, Neo4j, Neptune
eval + obs    RAGAS, DeepEval, Promptfoo, Braintrust, LangSmith, Langfuse, Arize Phoenix, LlamaGuard, NeMo Guardrails
data          BigQuery, Snowflake, Databricks (Unity Catalog, DBRX, Delta Live Tables), Spark, dbt, Dagster, Airflow, DuckDB, Iceberg, Kafka, Trino
cloud         GCP (Vertex AI, GKE), AWS (SageMaker, Bedrock, Trainium), Azure (Azure OpenAI, ML); Docker, Kubernetes, Terraform, Argo, MLflow, Weights and Biases
enterprise    Palantir Foundry/AIP, Salesforce Agentforce, ServiceNow, SAP, Microsoft Power Platform, Epic/FHIR
governance    NIST CSF, ISO 42001, EU AI Act, GDPR, HIPAA, GxP, on-prem and sovereign, multi-tenancy, PII redaction, RBAC
languages     Python, C++/CUDA, SQL, TypeScript, Go, Java, C#, R
```

[mikitadaroshkin.com](https://mikitadaroshkin.com/) | mikitadaroshkin@gmail.com
