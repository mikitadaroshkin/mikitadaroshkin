# Mikita Daroshkin

I build LLM systems that hold up in production, which mostly means the parts that aren't the demo: the evaluation harness, the retrieval that has to be right, the serving that has to be fast, and the failure modes nobody notices until real traffic finds them.

Most of my work now is agentic, and the hard problems there aren't the happy path, they're reliability and identity. Maker-checker verification, log-prob confidence gates with deterministic fallbacks, token-exchange auth between agents, sandboxed tool execution. Agents fail quietly, so evaluation goes in first: golden datasets and LLM-as-judge checks wired into CI, with OpenTelemetry tracing and deterministic replay, before anything reaches a user.

A lot of RAG quality actually lives in retrieval, so I treat it as the main event rather than a vector-store call. Hybrid dense and lexical with reciprocal-rank fusion, cross-encoder or ColBERT reranking, query rewriting, and the unglamorous chunking and hard-negative mining that move the numbers more than swapping the model does. When a knowledge graph or text-to-SQL fits the question better than embeddings, I reach for that instead.

On serving I go down to the kernel: continuous batching, paged KV-cache, speculative decoding, FlashAttention, and enough profiling to know where the time actually goes. I've compressed models to run offline on phones and tuned serving fleets for throughput per dollar. I came up through CUDA and MPI, distributed training on FSDP and DeepSpeed, and years of computer vision and classical ML before any of this was called agentic, so the whole stack reads as one system to me rather than layers I bolt together.

Earlier I co-founded three companies as the technical half and shipped ML in games, medical imaging, and finance. Enough production incidents to be paranoid in the useful ways.

The repositories below are small, public, and actually run. They are reference implementations of the methods I use, not the proprietary work.

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
