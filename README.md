# Mikita Daroshkin

AI/ML engineer building production LLM systems and the ML underneath.

Most of my work is agentic. I build multi-agent systems in LangGraph and Google ADK over A2A and MCP, with function calling and grammar-constrained decoding, ReAct and Plan-and-Execute planning, and a reliability layer of maker-checker and critic-verifier loops, self-consistency, log-prob confidence gating with deterministic fallbacks, bounded retries and circuit breakers over idempotent tools, token-exchange identity between agents, sandboxed execution, and egress control.

I gate all of it on evaluation in CI: golden datasets scored by LLM-as-judge with position-bias correction and calibrated rubrics, faithfulness and groundedness with bootstrap confidence intervals, deterministic replay and span-level token accounting under OpenTelemetry, and regression thresholds that block the merge before behavior drifts.

For retrieval I reach for hybrid dense and BM25 with reciprocal-rank fusion, cross-encoder and ColBERT late-interaction reranking, SPLADE, HyDE and step-back rewriting, query decomposition, contextual chunking, hard-negative mining, contrastive and Matryoshka embeddings, HNSW and IVF-PQ under scalar and binary quantization, and GraphRAG or self-correcting text-to-SQL when a graph or schema fits better than embeddings.

Serving is where I go down to the kernel: continuous batching, paged and radix KV-cache, chunked prefill, disaggregated prefill/decode, speculative and assisted decoding, FlashAttention, CUDA graphs, torch.compile, FP8 and INT4 with KV-cache quantization, custom Triton kernels, model routing, multi-LoRA, and SLA-aware scheduling on vLLM, TensorRT-LLM, and SGLang. I have taken an 8B stack down 27x to run offline on a phone and held multi-GPU fleets at 3 to 4x throughput per dollar.

Behind the models I run FSDP and DeepSpeed ZeRO-3 with tensor, pipeline, and sequence parallelism, activation recomputation and NCCL-tuned collectives in BF16 or FP8, and post-train with LoRA/QLoRA, SFT, DPO, reward modeling, and distillation.

Nine years back it started elsewhere: tiled CUDA with shared-memory tiling and warp-level primitives, MPI over InfiniBand, computer vision on ViT, 3D U-Net, Mask R-CNN, and CLIP, and classical ML with real statistical discipline in gradient boosting, probability calibration, conformal prediction, and causal inference, on lock-free, streaming, distributed systems.

I deliver across GCP, AWS, and Azure, on Databricks, Snowflake, and BigQuery for data and pgvector, FAISS, and Qdrant for vectors, integrate with Palantir Foundry and Salesforce Agentforce, and ship under NIST CSF, ISO 42001, the EU AI Act, HIPAA, and GxP when the domain calls for it.

## Repositories

Public reference implementations of the methods above, small enough to clone and run.

- [agentic-rag-evals](https://github.com/mikitadaroshkin/agentic-rag-evals): LangGraph agentic RAG, hybrid retrieval, LLM-as-judge eval suite, CI gate on eval-metric regression, OpenTelemetry, FastAPI, Docker.
- [gpu-mpi-search](https://github.com/mikitadaroshkin/gpu-mpi-search): BM25, cross-encoder rerank, tiled CUDA kernels, data-parallel scoring over MPI and multiprocessing.
- [mri-segmentation](https://github.com/mikitadaroshkin/mri-segmentation): 3D U-Net, soft-Dice/BCE loss, patch-based volumetric inference.
- [image-captioning-vit](https://github.com/mikitadaroshkin/image-captioning-vit): ViT encoder, Transformer decoder, image captioning.
- [nutrizy](https://github.com/mikitadaroshkin/nutrizy): cross-modal retrieval over a shared text and image embedding space, CLIP.
- [greenbox](https://github.com/mikitadaroshkin/greenbox): CNN disease classifier, LightGBM growth model, MQTT edge pipeline.
- [warehouse-patterns](https://github.com/mikitadaroshkin/warehouse-patterns): dimensional modeling, star schema, SCD-2, runnable DuckDB demo.

## Contact

I write about the engineering at [mikitadaroshkin.com](https://mikitadaroshkin.com/). Email is the fastest way to reach me: mikitadaroshkin@gmail.com.
