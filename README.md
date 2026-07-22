# Mikita Daroshkin

AI/ML engineer building production LLM systems and the ML underneath.

I build multi-agent systems orchestrated in LangGraph and Google ADK over A2A and MCP, calling tools through function calling and grammar-constrained decoding, planned with ReAct and Plan-and-Execute, and kept reliable by maker-checker and critic-verifier loops, self-consistency, log-prob confidence gating with deterministic fallbacks, bounded retries and circuit breakers over idempotent tools, token-exchange identity between agents, sandboxed execution, and egress control.

Evaluation runs in CI over golden datasets, scored by LLM-as-judge with position-bias correction and calibrated rubrics, faithfulness and groundedness measured with bootstrap confidence intervals, deterministic replay and span-level token accounting under OpenTelemetry, and regression gates that block the merge before behavior drifts.

Retrieval is hybrid dense and BM25 fused with reciprocal-rank fusion, reranked by cross-encoders and ColBERT late-interaction, extended with SPLADE, HyDE and step-back rewriting, and query decomposition, fed by contextual chunking and hard-negative mining over contrastive and Matryoshka embeddings, indexed with HNSW or IVF-PQ under scalar and binary quantization, and swapped for GraphRAG or self-correcting text-to-SQL when a graph or schema answers better than embeddings.

Serving goes down to the kernel with continuous batching, paged and radix KV-cache, chunked prefill and disaggregated prefill/decode, speculative and assisted decoding, FlashAttention, CUDA graphs and torch.compile, FP8 and INT4 with KV-cache quantization, custom Triton kernels, model routing, multi-LoRA, and SLA-aware scheduling on vLLM, TensorRT-LLM, and SGLang. I have compressed 8B stacks 27x to run offline on a phone and driven multi-GPU fleets to 3 to 4x throughput per dollar.

Training runs on FSDP and DeepSpeed ZeRO-3 with tensor, pipeline, and sequence parallelism, activation recomputation and NCCL-tuned collectives in BF16 or FP8, using LoRA/QLoRA, SFT, DPO, reward modeling, and distillation.

Underneath sits nine years of the rest: tiled CUDA with shared-memory tiling and warp-level primitives, MPI over InfiniBand, computer vision on ViT, 3D U-Net, Mask R-CNN, and CLIP, and classical ML with real statistical discipline in gradient boosting, probability calibration, conformal prediction, and causal inference, all on lock-free, streaming, distributed systems.

It ships across GCP, AWS, and Azure, with Databricks, Snowflake, and BigQuery for data and Palantir Foundry and Salesforce Agentforce on the enterprise side, under NIST CSF, ISO 42001, the EU AI Act, HIPAA, and GxP where the domain demands it.

## Repositories

Public reference implementations of the methods above, small enough to clone and run.

- [agentic-rag-evals](https://github.com/mikitadaroshkin/agentic-rag-evals): LangGraph agentic RAG, hybrid retrieval, LLM-as-judge eval suite, CI gate on eval-metric regression, OpenTelemetry, FastAPI, Docker.
- [gpu-mpi-search](https://github.com/mikitadaroshkin/gpu-mpi-search): BM25, cross-encoder rerank, tiled CUDA kernels, data-parallel scoring over MPI and multiprocessing.
- [mri-segmentation](https://github.com/mikitadaroshkin/mri-segmentation): 3D U-Net, soft-Dice/BCE loss, patch-based volumetric inference.
- [image-captioning-vit](https://github.com/mikitadaroshkin/image-captioning-vit): ViT encoder, Transformer decoder, image captioning.
- [nutrizy](https://github.com/mikitadaroshkin/nutrizy): cross-modal retrieval over a shared text and image embedding space, CLIP.
- [greenbox](https://github.com/mikitadaroshkin/greenbox): CNN disease classifier, LightGBM growth model, MQTT edge pipeline.
- [warehouse-patterns](https://github.com/mikitadaroshkin/warehouse-patterns): dimensional modeling, star schema, SCD-2, runnable DuckDB demo.

[mikitadaroshkin.com](https://mikitadaroshkin.com/) | mikitadaroshkin@gmail.com
