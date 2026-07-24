# Mikita Daroshkin

AI/ML engineer building production LLM systems and the ML underneath.

Most of the work is agentic: multi-agent meshes on LangGraph, LangChain, and Google ADK, with CrewAI and DSPy where they fit, communicating over A2A and MCP, deployed on Vertex AI Agent Engine and Amazon Bedrock. Graphs are explicit state machines with checkpointed state, human-in-the-loop interrupts, and streamed partial output. Tool calls go through function calling and grammar-constrained decoding against a JSON schema, so arguments either parse or fail loudly. Planning is ReAct or Plan-and-Execute. Context is managed rather than assumed: windowed history, episodic and semantic memory, summarisation and compaction at the boundary. The reliability layer carries the weight, with maker-checker and critic-verifier loops, self-consistency, log-prob confidence gating with deterministic fallback, bounded retries with jitter and circuit breakers over idempotent tools, and dead-letter paths for what still fails. Agents authenticate by token exchange, hold least-privilege scopes, execute sandboxed under egress control, and get tested against prompt injection.

Behavior is gated on evaluation in CI. Golden datasets scored by LLM-as-judge with position-bias correction, calibrated rubrics, and pairwise comparison where absolute scores are too noisy to trust. Faithfulness and groundedness report bootstrap confidence intervals. Retrieval is scored on recall@k, nDCG, and MRR. Runs replay deterministically from recorded fixtures, and OpenTelemetry carries span-level token accounting and latency. Regression thresholds block the merge; drift is watched in production on shadow traffic and canaries. A prompt is code and should be able to fail the build.

Retrieval starts hybrid: dense and BM25 fused by reciprocal rank, reranked by cross-encoder or ColBERT late interaction, SPLADE where the match must stay lexical but learned. Queries get HyDE, step-back rewriting, decomposition, and routing. Chunking is contextual, with parent-document and sentence-window expansion and metadata filtering under it. Embeddings come from contrastive training and hard-negative mining, Matryoshka where vectors have to shorten without retraining. Indexes are HNSW or IVF-PQ under scalar or binary quantization, built incrementally so freshness does not cost a rebuild, over pgvector, FAISS, Qdrant, or Elasticsearch at twelve-terabyte corpus scale. GraphRAG and knowledge graphs in Neo4j where the relations carry the meaning, self-correcting text-to-SQL where a schema fits better than embeddings. Operating points get chosen rather than defaulted: 94% precision at 90% recall is a decision about which error costs more.

Serving goes down to the kernel: continuous batching over paged and radix KV-cache with prefix reuse, chunked prefill, prefill/decode disaggregation, speculative and assisted decoding, FlashAttention, CUDA graphs, torch.compile, FP8 and INT4 against a quantized KV-cache, custom Triton kernels where the framework falls short, model routing, multi-LoRA, and SLA-aware scheduling on vLLM, TensorRT-LLM, and SGLang, against the Claude, Gemini, and OpenAI APIs. What gets tuned is time-to-first-token, inter-token latency, and p99 under load, not average throughput. An 8B multimodal stack came down 27x to run offline on a mid-tier Android phone; multi-GPU fleets hold 3 to 4x throughput per dollar.

Training is FSDP or DeepSpeed ZeRO-3, sharded across tensor, pipeline, and sequence parallelism, with gradient accumulation, activation recomputation, sharded checkpointing, sequence packing, and NCCL collectives tuned to the interconnect, in BF16 or FP8. Post-training is LoRA and QLoRA, SFT, DPO, RLHF, reward modeling, and distillation into the smallest model that still holds the metric.

Underneath is Python and PyTorch, with TensorFlow, Hugging Face Transformers, scikit-learn, and XGBoost where the problem is not a language model, and older layers still: tiled CUDA with shared-memory tiling and warp-level primitives, MPI over InfiniBand, computer vision on ViT, 3D U-Net, Mask R-CNN, and CLIP, recommender systems, and classical ML with statistical discipline in gradient boosting, probability calibration, conformal prediction, and causal inference over lock-free streaming and distributed systems. It is why a calibration curve still comes before a bigger model.

The delivery surface is GCP, AWS, and Azure on Vertex AI, SageMaker, and Azure OpenAI, in Docker and Kubernetes under Terraform and CI/CD, with Airflow, model monitoring, and A/B testing over Databricks, Snowflake, BigQuery, Spark and PySpark, Kafka, dbt, Postgres, and MongoDB, integrating Palantir Foundry, Salesforce Agentforce, ServiceNow, Workday, and FHIR, under NIST CSF, ISO 42001, the EU AI Act, HIPAA, and GxP.

## Repositories

The client work is closed, so these are the public equivalents: reference implementations of the methods above, small enough to clone and run. Each one builds and tests in CI.

- [agentic-rag-evals](https://github.com/mikitadaroshkin/agentic-rag-evals): LangGraph agentic RAG, hybrid retrieval, LLM-as-judge eval suite, CI gate on eval-metric regression, OpenTelemetry, FastAPI, Docker.
- [gpu-mpi-search](https://github.com/mikitadaroshkin/gpu-mpi-search): BM25, cross-encoder rerank, tiled CUDA kernels, data-parallel scoring over MPI and multiprocessing.
- [mri-segmentation](https://github.com/mikitadaroshkin/mri-segmentation): 3D U-Net, soft-Dice/BCE loss, patch-based volumetric inference.
- [image-captioning-vit](https://github.com/mikitadaroshkin/image-captioning-vit): ViT encoder, Transformer decoder, image captioning.
- [nutrizy](https://github.com/mikitadaroshkin/nutrizy): cross-modal retrieval over a shared text and image embedding space, CLIP.
- [greenbox](https://github.com/mikitadaroshkin/greenbox): CNN disease classifier, LightGBM growth model, MQTT edge pipeline.
- [warehouse-patterns](https://github.com/mikitadaroshkin/warehouse-patterns): dimensional modeling, star schema, SCD-2, runnable DuckDB demo.

## Contact

I write about the engineering at [mikitadaroshkin.com](https://mikitadaroshkin.com/). Email is the fastest way to reach me: mikitadaroshkin@gmail.com.
