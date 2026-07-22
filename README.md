# Mikita Daroshkin

Production LLM and ML systems end to end, from CUDA kernels to multi-agent orchestration, with evaluation wired into CI. What I work in, in depth:

```
agents         multi-agent orchestration (supervisor, hierarchical, blackboard), A2A, MCP, tool use and function calling, constrained and structured decoding, planning and reasoning (ReAct, Plan-and-Execute, Reflexion, tree search, test-time compute), episodic and semantic memory, maker-checker and critic-verifier loops, self-consistency, log-prob confidence gating, deterministic fallbacks, token-exchange identity, sandboxed execution, computer-use
retrieval      hybrid dense + BM25, reciprocal-rank fusion, cross-encoder and ColBERT late-interaction reranking, SPLADE, query rewriting and decomposition (HyDE, multi-query, self-query), semantic and late chunking, parent-document, matryoshka and fine-tuned embeddings, hard-negative mining, HNSW, IVF-PQ, GraphRAG, knowledge graphs, entity resolution, text-to-SQL (schema-linking, EXPLAIN self-correction), grounding and citation
fine-tuning    SFT, LoRA/QLoRA, PEFT, preference optimization (RLHF, RLAIF, DPO, ORPO), reward modeling, quantization (GPTQ, AWQ, bitsandbytes, FP8/INT4), distillation, pruning, FSDP, DeepSpeed ZeRO, tensor and pipeline parallelism, gradient checkpointing, mixed precision (BF16/FP8), synthetic data, weak supervision
serving        continuous batching, paged KV-cache, prefix caching, chunked prefill, speculative and assisted decoding, FlashAttention, kernel fusion, model routing and cascades, disaggregated prefill/decode, multi-LoRA, autoscaling; on-device quantization, LiteRT/TFLite, ONNX Runtime, NNAPI, GPU delegates, Nsight profiling
eval + safety  golden datasets, LLM-as-judge (pairwise, rubric), faithfulness and groundedness, regression gates in CI, deterministic replay, red-teaming (prompt injection, jailbreak, data poisoning, model inversion, adversarial), guardrails, content classifiers, PII redaction, OpenTelemetry tracing, token and cost accounting, drift detection
ml + vision    transformers, ViT, ResNet, Mask R-CNN, 3D U-Net, CLIP, RotationNet, segmentation, detection; LSTMs, time series; gradient boosting, probability calibration, conformal prediction, uplift modeling, causal inference, survival analysis; learning-to-rank, DSSM
systems + hpc  tiled CUDA kernels (shared-memory tiling), MPI, InfiniBand, lock-free concurrency, Kafka streaming; Kubernetes (Karpenter, Istio), Terraform, GitOps, model registry, blue-green and shadow deploys, feature stores; dimensional modeling, CDC, data lineage

frameworks     PyTorch, JAX, LangGraph, LangChain, Google ADK, DSPy, Semantic Kernel, CrewAI, AutoGen, OpenAI Agents SDK, Megatron-LM, NeMo, scikit-learn, XGBoost, LightGBM, CatBoost
inference      vLLM, TensorRT-LLM, SGLang, Triton, TGI, LiteRT, ONNX Runtime; open-weight models (Llama, Qwen, DeepSeek, Gemma, Mistral, Phi)
retrieval infra pgvector, FAISS, Qdrant, Chroma, Milvus, OpenSearch, Vertex AI Vector Search, Azure AI Search, Neo4j, Neptune
eval infra     RAGAS, DeepEval, Promptfoo, Braintrust, TruLens, Giskard, LangSmith, Langfuse, Arize Phoenix, LlamaGuard, NeMo Guardrails, Vertex AI Evaluation, Mosaic AI Eval
voice          Whisper, OpenAI Realtime API, ElevenLabs, Deepgram, LiveKit, Pipecat, Twilio; VAD, barge-in, speech-to-speech latency, IVR
data           BigQuery, Snowflake, Databricks (Unity Catalog, DBRX, Delta Live Tables), Spark, dbt, Dagster, Airflow, DuckDB, Iceberg, Hudi, Microsoft Fabric, Azure Synapse, Trino/Presto, Kafka, Debezium
cloud + infra  GCP (Vertex AI, GKE, BigQuery), AWS (SageMaker, Bedrock, Amazon Q, Trainium), Azure (Azure OpenAI, Azure ML); Docker, Kubernetes, Terraform, OpenTofu, Pulumi, Argo, MLflow, Weights and Biases, Slurm/DGX, Run:ai, H100/H200/GB200
enterprise     Palantir Foundry/AIP, Salesforce Agentforce, ServiceNow Now Assist, SAP, Microsoft Power Platform, Microsoft Graph, Epic/FHIR, HealthLake, Veeva, IQVIA
governance     Responsible AI, model risk and auditability, NIST CSF, ISO 27001, ISO 42001, EU AI Act, GDPR/CCPA, HIPAA, GxP, on-prem and sovereign, multi-tenancy, PII redaction, SSO (OIDC/SAML), RBAC
languages      Python, C++/CUDA, SQL, TypeScript, Go, Java, C#, R
```

## Repositories

Public reference implementations, each small enough to clone and run. All build in CI; agentic-rag-evals gates its own merges on eval-metric regression.

- [agentic-rag-evals](https://github.com/mikitadaroshkin/agentic-rag-evals): LangGraph agentic RAG, hybrid retrieval, LLM-as-judge eval suite, CI regression gate, OpenTelemetry, FastAPI, Docker.
- [gpu-mpi-search](https://github.com/mikitadaroshkin/gpu-mpi-search): BM25 with cross-encoder rerank, tiled CUDA kernels, data-parallel scoring (MPI/multiprocessing).
- [mri-segmentation](https://github.com/mikitadaroshkin/mri-segmentation): 3D U-Net, soft-Dice/BCE loss, volumetric inference.
- [image-captioning-vit](https://github.com/mikitadaroshkin/image-captioning-vit): ViT encoder, Transformer decoder, image captioning.
- [nutrizy](https://github.com/mikitadaroshkin/nutrizy): cross-modal retrieval over a shared text/image embedding space (CLIP).
- [greenbox](https://github.com/mikitadaroshkin/greenbox): CNN disease classifier, LightGBM growth model, MQTT edge pipeline.
- [warehouse-patterns](https://github.com/mikitadaroshkin/warehouse-patterns): dimensional modeling, star schema, SCD-2, runnable DuckDB demo.

[mikitadaroshkin.com](https://mikitadaroshkin.com/) | mikitadaroshkin@gmail.com
