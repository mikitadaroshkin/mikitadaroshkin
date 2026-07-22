# Mikita Daroshkin

I build LLM systems that hold up in production, which mostly comes down to the boring, load-bearing parts: evaluation, retrieval, inference internals, and the failure modes that only surface under real traffic.

Agentic work is most of it now, and the difficulty is never the happy path, it is reliability, identity, and control. Maker-checker and critic-verifier loops, self-consistency, log-prob confidence gating with deterministic code fallbacks, plan repair, bounded retries with jitter, circuit breakers, idempotent tool calls, token-exchange identity between agents, sandboxed execution, egress control, structured output through grammar-constrained decoding rather than hope. And because agents fail quietly, evaluation is the real system: golden datasets, LLM-as-judge with position-bias correction and calibrated rubrics, faithfulness and groundedness scoring with bootstrap confidence intervals, deterministic replay, span-level trace and token accounting, regression gates that block the merge before behavior drifts.

Retrieval is where most RAG quality is won or lost, so that is where the budget goes, not model swaps. Hybrid dense and BM25 fused with reciprocal-rank fusion, cross-encoder and ColBERT late-interaction reranking cascades, SPLADE learned-sparse, query decomposition with HyDE and step-back rewriting, contextual and semantic chunking, hard-negative mining, contrastively fine-tuned and Matryoshka embeddings, HNSW against IVF-PQ with scalar and binary vector quantization, tuned on recall and nDCG under a memory budget. GraphRAG and self-correcting text-to-SQL when a graph or a schema beats embeddings.

Serving I take down to the kernel: continuous batching, paged and radix KV-cache with cache-aware routing, chunked prefill, disaggregated prefill and decode, speculative and assisted decoding with draft-model and n-gram methods, FlashAttention, CUDA graphs, torch.compile, FP8 and INT4 with KV-cache quantization, custom Triton kernels, profiling until every millisecond is accounted for. I have compressed 8B stacks 27x to run fully offline on a phone, and driven multi-GPU fleets to 3 to 4x throughput per dollar with model routing, multi-LoRA, and SLA-aware scheduling. The training behind it is FSDP and DeepSpeed ZeRO-3, tensor, pipeline, and sequence parallelism, activation recomputation, NCCL-tuned collectives, LoRA/QLoRA, and DPO.

Under all of it is nine years of the rest: tiled CUDA with shared-memory tiling and warp-level primitives, MPI over InfiniBand, computer vision (ViT, 3D U-Net, Mask R-CNN, CLIP), classical ML with real statistical discipline (calibration, conformal prediction, causal inference), streaming and lock-free distributed systems. Three companies co-founded as the technical half; ML shipped into games, medical imaging, and finance under real SLAs.

The repos below are small, public, and runnable, reference implementations of the methods rather than the proprietary work.

## Repositories

- [agentic-rag-evals](https://github.com/mikitadaroshkin/agentic-rag-evals): LangGraph agentic RAG with hybrid retrieval, an LLM-as-judge eval suite, and a CI gate that fails on eval-metric regression. OpenTelemetry, FastAPI, Docker.
- [gpu-mpi-search](https://github.com/mikitadaroshkin/gpu-mpi-search): BM25 with cross-encoder rerank, tiled CUDA kernels with shared-memory tiling, data-parallel scoring over MPI and multiprocessing.
- [mri-segmentation](https://github.com/mikitadaroshkin/mri-segmentation): 3D U-Net with soft-Dice/BCE loss and patch-based volumetric inference.
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
