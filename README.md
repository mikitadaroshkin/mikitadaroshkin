# Mikita Daroshkin

I build LLM systems that hold up in production: the eval harness, the retrieval, the serving, and the failure modes that only show up under real traffic.

Most of my work is agentic, and the hard parts are reliability and identity, not the happy path: maker-checker verification, log-prob confidence gates with deterministic fallbacks, token-exchange auth between agents, sandboxed tool execution, bounded retries. Agents fail quietly, so evals go in first: golden datasets, LLM-as-judge and faithfulness checks in CI, OpenTelemetry tracing, deterministic replay, merge-blocking regression gates.

Retrieval is where RAG quality lives, so I spend time there instead of on model swaps: hybrid dense and BM25 with reciprocal-rank fusion, cross-encoder and ColBERT reranking, query rewriting and decomposition, semantic chunking, hard-negative mining, HNSW and IVF-PQ tradeoffs. GraphRAG and text-to-SQL when the question fits a graph or a schema better than embeddings.

On serving I work down to the kernel: continuous batching, paged KV-cache, prefix caching, speculative decoding, FlashAttention, model routing, and enough profiling to find where the time actually goes. I have quantized and distilled models to run offline on-device, and tuned serving fleets for throughput per dollar.

The base under it: CUDA and MPI, distributed training on FSDP and DeepSpeed, computer vision (ViT, 3D U-Net, Mask R-CNN, CLIP) and classical ML (gradient boosting, calibration, conformal prediction, causal inference) from before the LLM wave. Three companies co-founded as the technical half; ML shipped in games, medical imaging, and finance.

Repos below are small, public, and runnable, reference implementations of the methods rather than the proprietary work.

## Repositories

- [agentic-rag-evals](https://github.com/mikitadaroshkin/agentic-rag-evals): LangGraph agentic RAG with hybrid retrieval, an LLM-as-judge eval suite, and a CI gate that fails on eval-metric regression. OpenTelemetry, FastAPI, Docker.
- [gpu-mpi-search](https://github.com/mikitadaroshkin/gpu-mpi-search): BM25 with cross-encoder rerank, tiled CUDA kernels, data-parallel scoring over MPI and multiprocessing.
- [mri-segmentation](https://github.com/mikitadaroshkin/mri-segmentation): 3D U-Net with soft-Dice/BCE loss and volumetric inference.
- [image-captioning-vit](https://github.com/mikitadaroshkin/image-captioning-vit): ViT encoder with a Transformer decoder for image captioning.
- [nutrizy](https://github.com/mikitadaroshkin/nutrizy): cross-modal retrieval over a shared text and image embedding space with CLIP.
- [greenbox](https://github.com/mikitadaroshkin/greenbox): a CNN disease classifier, a LightGBM growth model, and an MQTT edge pipeline.
- [warehouse-patterns](https://github.com/mikitadaroshkin/warehouse-patterns): dimensional modeling, star schema, and SCD-2 with a runnable DuckDB demo.

## Stack

```
agents        LangGraph, LangChain, Google ADK, DSPy, Semantic Kernel, CrewAI, AutoGen, MCP, A2A
serving       vLLM, TensorRT-LLM, SGLang, Triton, TGI, LiteRT, ONNX Runtime
retrieval     pgvector, FAISS, Qdrant, Milvus, Vertex AI Vector Search, GraphRAG, Neo4j
training      PyTorch, JAX, FSDP, DeepSpeed, Megatron-LM, LoRA/QLoRA, DPO, GPTQ/AWQ
eval + obs    RAGAS, DeepEval, Promptfoo, LangSmith, Langfuse, OpenTelemetry, LlamaGuard
data          BigQuery, Snowflake, Databricks, Spark, dbt, DuckDB, Kafka
cloud         GCP (Vertex AI, GKE), AWS (SageMaker, Bedrock), Azure (Azure OpenAI, ML), Kubernetes, Terraform
enterprise    Palantir Foundry, Salesforce Agentforce, ServiceNow, SAP, Epic/FHIR
languages     Python, C++/CUDA, SQL, TypeScript, Go
```

[mikitadaroshkin.com](https://mikitadaroshkin.com/) | mikitadaroshkin@gmail.com
