# Mikita Daroshkin

I design, build, and operate production LLM systems end to end, plus the evaluation that keeps them correct: golden datasets, LLM-as-judge regression gates in CI, OpenTelemetry tracing, deterministic replay, and merge-blocking thresholds on faithfulness, hallucination, jailbreak, and toxicity.

## What I build

Multi-agent systems on Vertex AI Agent Engine and Google ADK: federated agents over A2A with OAuth token-exchange identity propagated at every hop, maker-checker verification, log-prob confidence gates with deterministic code fallbacks, MCP-wrapped tools inside egress-locked VPC-SC perimeters, OPA policy-as-code promotion gates, per-agent eval scorecards, and zero-registration agent CI/CD.

Retrieval built for correctness, not demos: hybrid dense + BM25 with reciprocal-rank fusion and cross-encoder reranking over pgvector, FAISS, and Vertex AI Vector Search; GraphRAG and text-to-SQL with schema-linking and EXPLAIN-driven self-correction; dual-encoder text-and-vision multimodal retrieval; 12 TB multimodal document lakes with vision-language agent crews.

Inference at scale on vLLM, TensorRT-LLM, SGLang, and Triton across multi-GPU GKE: continuous batching, KV-cache optimization, speculative decoding, FlashAttention, GPTQ/AWQ quantization, and Nsight kernel profiling, at 3.2 to 4.7x throughput per dollar. On-device, an 8B chat-RAG stack compressed 27x to under 600 MB with quantization, distillation, LiteRT/TFLite GPU delegates, and ONNX Runtime Mobile, running fully offline on mid-tier Android.

Post-training and reliability: LoRA/QLoRA, SFT, DPO, reward models, and RLHF on Slurm/DGX with FSDP and DeepSpeed; calibrated, bias-audited ranking with conformal prediction; document-intelligence pipelines with LLM cross-document validation at 99.2% extraction accuracy.

Foundations under all of it: 3D U-Net segmentation on volumetric MRI trained on a 10M+ scan corpus, ViT captioning, tiled CUDA and MPI kernels, gradient-boosted tabular models, distributed training and serving from the kernel up.

## Repositories

Public reference implementations, each small enough to clone and run. All build in CI; agentic-rag-evals gates its own merges on eval-metric regression.

- [agentic-rag-evals](https://github.com/mikitadaroshkin/agentic-rag-evals): LangGraph agentic RAG, hybrid retrieval, LLM-as-judge eval suite, CI regression gate, OpenTelemetry, FastAPI, Docker.
- [gpu-mpi-search](https://github.com/mikitadaroshkin/gpu-mpi-search): BM25 with cross-encoder rerank, tiled CUDA kernels, data-parallel scoring (MPI/multiprocessing). Coursework.
- [mri-segmentation](https://github.com/mikitadaroshkin/mri-segmentation): 3D U-Net, soft-Dice/BCE loss, volumetric inference. BSc thesis.
- [image-captioning-vit](https://github.com/mikitadaroshkin/image-captioning-vit): ViT encoder, Transformer decoder, image captioning.
- [nutrizy](https://github.com/mikitadaroshkin/nutrizy): cross-modal retrieval over a shared text/image embedding space (CLIP), 2019.
- [greenbox](https://github.com/mikitadaroshkin/greenbox): CNN disease classifier, LightGBM growth model, MQTT edge pipeline.
- [warehouse-patterns](https://github.com/mikitadaroshkin/warehouse-patterns): dimensional modeling, star schema, SCD-2, runnable DuckDB demo.

## Stack

```
llm / agents   LangGraph, LangChain, Google ADK, DSPy, Semantic Kernel, CrewAI, AutoGen, MCP, A2A, function calling, structured outputs
retrieval      pgvector, FAISS, Qdrant, Milvus, Vertex AI Vector Search, BM25, reciprocal-rank fusion, cross-encoder rerank, ColBERT, SPLADE, GraphRAG
serving        vLLM, TensorRT-LLM, SGLang, Triton, TGI, continuous batching, KV-cache, speculative decoding, FlashAttention, GPTQ/AWQ, distillation, LiteRT, ONNX Runtime
training       PyTorch, FSDP, DeepSpeed, Megatron-LM, NeMo, LoRA/QLoRA, SFT, DPO, RLHF, PEFT, Slurm/DGX
eval / obs     golden sets, LLM-as-judge, RAGAS, DeepEval, Promptfoo, Braintrust, LangSmith, Langfuse, Arize Phoenix, OpenTelemetry
data           BigQuery, Snowflake, Databricks, Spark, dbt, DuckDB, Kafka, pgvector
cloud          GCP (Vertex AI), AWS (SageMaker, Bedrock), Azure (Azure OpenAI), Docker, Kubernetes, Terraform, Argo, MLflow, Weights & Biases
languages      Python, C++/CUDA, SQL, TypeScript, Go
```

[mikitadaroshkin.com](https://mikitadaroshkin.com/) | mikitadaroshkin@gmail.com
