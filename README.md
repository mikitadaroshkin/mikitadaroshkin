# Mikita Daroshkin

I build LLM systems that hold up in production: the eval harness, the retrieval, the serving, and the failure modes that only show up under real load.

Most of my work is agentic, where the hard parts are reliability and identity, not the happy path. Maker-checker and critic-verifier loops, log-prob confidence gating with deterministic fallbacks, bounded retries and circuit breakers, token-exchange identity between agents, sandboxed tool execution, egress control. Agents fail quietly, so evaluation is load-bearing: golden datasets, LLM-as-judge with calibrated rubrics, faithfulness and groundedness scoring, deterministic replay, OpenTelemetry traces, and regression gates that block the merge before behavior drifts.

Retrieval is where RAG quality actually lives, so I spend the budget there, not on model swaps. Hybrid dense and BM25 with reciprocal-rank fusion, cross-encoder and ColBERT late-interaction reranking, query decomposition and rewriting, semantic and contextual chunking, hard-negative mining, HNSW versus IVF-PQ tradeoffs on recall and memory. When a graph or a schema answers better than embeddings, GraphRAG and self-correcting text-to-SQL instead.

On serving I work down to the kernel: continuous batching, paged and radix KV-cache, chunked prefill, disaggregated prefill and decode, speculative and assisted decoding, FlashAttention, CUDA graphs, FP8, and profiling until I know where every millisecond goes. I have compressed 8B models 27x to run fully offline on a phone, and pushed multi-GPU serving fleets to 3 to 4x throughput per dollar with model routing and multi-LoRA. The training behind it runs on FSDP and DeepSpeed ZeRO, tensor and pipeline parallelism, gradient checkpointing, LoRA/QLoRA, and DPO.

Underneath is nine years of the rest: tiled CUDA and MPI, computer vision (ViT, 3D U-Net, Mask R-CNN, CLIP), classical ML (gradient boosting, calibration, conformal prediction, causal inference), streaming and distributed systems. Three companies co-founded as the technical half; ML shipped into games, medical imaging, and finance under real SLAs.

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
