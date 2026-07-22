# Mikita Daroshkin

I design, build, and operate production LLM systems end to end for regulated enterprises, plus the evaluation that keeps them correct: golden datasets, LLM-as-judge regression gates in CI, OpenTelemetry tracing, deterministic replay, and merge-blocking thresholds on faithfulness, hallucination, jailbreak, and toxicity. Nine years and thirty-plus production systems across LLMs, computer vision, recommenders, real-time systems, HPC, and data platforms.

## What I build now

Multi-agent systems on Vertex AI Agent Engine and Google ADK, on Amazon Bedrock, and on Azure AI Foundry: federated agents over A2A with OAuth token-exchange identity propagated at every hop, maker-checker verification, log-prob confidence gates with deterministic code fallbacks, MCP-wrapped tools inside egress-locked VPC-SC perimeters, OPA policy-as-code promotion gates, per-agent eval scorecards, and zero-registration agent CI/CD. Function calling and structured outputs, agent memory, computer-use and browser agents, real-time voice (streaming ASR, speech-to-speech, VAD turn-taking, barge-in) over telephony.

Retrieval built for correctness, not demos: hybrid dense + BM25 with reciprocal-rank fusion and cross-encoder reranking over pgvector, FAISS, Qdrant, and Vertex AI Vector Search; GraphRAG and knowledge graphs; text-to-SQL with schema-linking and EXPLAIN-driven self-correction; dual-encoder text-and-vision multimodal retrieval across 12 TB document lakes with vision-language agent crews.

Inference at scale on vLLM, TensorRT-LLM, SGLang, and Triton across multi-GPU GKE: continuous batching, KV-cache optimization, speculative decoding, FlashAttention, GPTQ/AWQ quantization, prompt caching, and model routing, at 3.2 to 4.7x throughput per dollar on seven-figure inference budgets. On-device, an 8B chat-RAG stack compressed 27x to under 600 MB with quantization, distillation, LiteRT/TFLite GPU delegates, ONNX Runtime Mobile, and NNAPI, running fully offline on mid-tier Android.

Post-training and distributed training: LoRA/QLoRA, SFT, DPO, reward models, and RLHF on Slurm/DGX with FSDP, DeepSpeed, and Megatron-LM across H100/H200 and InfiniBand clusters; calibrated, bias-audited ranking with conformal prediction; document intelligence with LLM cross-document validation at 99.2% extraction accuracy.

Evaluation and safety are part of the build, not an afterthought: red-teaming for data poisoning, model inversion, and adversarial attacks; faithfulness and fairness scoring; guardrails with LlamaGuard and NeMo Guardrails; shipped under NIST CSF, ISO 42001, the EU AI Act, HIPAA, and GxP, on-prem and sovereign where required.

## Also shipped

Computer vision and medical imaging: 3D U-Net segmentation on volumetric MRI, from a BSc thesis to a 10M+ scan quality-control corpus with a multimodal RotationNet trained on FSDP and DeepSpeed in BF16; ViT image captioning that took first place in a data-science competition; Mask R-CNN segmentation and CLIP-style joint text-image embeddings; image processing and computer graphics written from scratch. Two patent disclosures from the research years.

Generative NLP, before the current wave: an ML copywriting platform taken from zero to launch, retrieval-grounded long-form generation on GPT-2 and GPT-3 with per-brand style memory and a training-data flywheel of edit capture, weak supervision, and dedup, whose fine-tuned GPT-2 beat GPT-3 few-shot on brand voice at one eighteenth the cost, served multi-tenant with cost-aware routing between self-hosted and hosted models.

Recommenders and predictive ML: retrieval-cascade recommenders (multi-armed bandits, FAISS candidate generation, DSSM ranking with CNN-attention ensembles, graph-ML over a social graph); LTV forecasting with unsupervised user embeddings, LSTMs, and a ZILN-loss framework (R2 up 23%); churn modeling with gradient boosting, SHAP, and probability calibration; uplift-targeted campaigns.

Real-time and systems: game backends at over a million daily users, lock-free matchmaking and session telemetry, Kafka streaming over protobuf, statistical anti-cheat with isolation forests under a 0.1% false-positive ban rate, in-process model scoring, GC-tuned C++ and native interop.

Distributed systems and HPC: tiled GPU (CUDA) and MPI algorithms; a hierarchical search engine over PageRank, BM25, and BERT; distributed systems on Nginx and Cassandra.

Data engineering and BI: countrywide finance BI systems, star-schema warehouses and SSIS ETL cut from six hours to forty minutes, SSAS OLAP semantic layers, dimensional modeling, and bank-grade governance with row-level security, TDE, and audit.

Edge and IoT: plant-health at the edge, a CNN disease classifier at 99.85%, LightGBM growth models, and an MQTT sensor-fusion pipeline on Raspberry Pi, a hackathon win that converted to a development grant.

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
agents         LangGraph, LangChain, Google ADK, DSPy, Semantic Kernel, CrewAI, AutoGen, OpenAI Agents SDK, MCP, A2A, ReAct, Plan-and-Execute, function calling, structured outputs, computer-use, Copilot Studio, Azure AI Foundry, Bedrock AgentCore
models         GPT, Gemini, open-weight (Llama, Qwen, DeepSeek, Gemma, Phi, Mistral); LoRA/QLoRA, SFT, DPO, RLHF/RLAIF, PEFT; GPTQ/AWQ/bitsandbytes; distillation; reasoning and test-time compute
retrieval      pgvector, FAISS, Qdrant, Chroma, Milvus, OpenSearch, Vertex AI Vector Search, Azure AI Search; BM25, reciprocal-rank fusion, cross-encoder rerank, ColBERT, SPLADE; GraphRAG, Neo4j, Neptune, text-to-SQL, entity resolution
serving        vLLM, TensorRT-LLM, SGLang, Triton, TGI; continuous batching, KV-cache, speculative decoding, FlashAttention, prompt caching, model routing; LiteRT/TFLite, ONNX Runtime Mobile, NNAPI; Nsight
training       PyTorch, JAX, FSDP, DeepSpeed, Megatron-LM, NeMo, NCCL, ZeRO; Slurm/DGX, Run:ai, Anyscale; H100/H200/GB200, InfiniBand, Trainium/Inferentia
vision         PyTorch, 3D U-Net, ViT, ResNet, Mask R-CNN, CLIP, FSDP/DeepSpeed, computer graphics
voice          Whisper, OpenAI Realtime API, ElevenLabs, Deepgram, LiveKit, Pipecat, Twilio; VAD, barge-in, speech-to-speech latency, IVR
eval + safety  golden sets, LLM-as-judge, RAGAS, DeepEval, Promptfoo, Braintrust, TruLens, Giskard, Vertex AI Evaluation, Mosaic AI Eval; Guardrails AI, LlamaGuard, NeMo Guardrails; red-teaming, XAI, fairness
observability  OpenTelemetry, LangSmith, Langfuse, Arize Phoenix, Prometheus, Grafana; drift and model monitoring, AIOps
data           BigQuery, Snowflake, Databricks (Unity Catalog, DBRX, Delta Live Tables), Spark, dbt, Dagster, Airflow, Kafka, DuckDB, Iceberg, Hudi, Microsoft Fabric, Azure Synapse; CDC/Debezium, Trino/Presto; lineage and governance (Collibra, Purview)
classical ml   scikit-learn, XGBoost, LightGBM, CatBoost, H2O, SHAP, calibration, causal inference, uplift modeling, survival analysis, time series, statistics
cloud          GCP (Vertex AI, GKE, BigQuery), AWS (SageMaker, Bedrock, Amazon Q), Azure (Azure OpenAI, Azure ML); Docker, Kubernetes, Terraform, OpenTofu, Pulumi, Argo, GitOps, Karpenter, Istio, MLflow, Weights and Biases
enterprise     Palantir Foundry/AIP, Salesforce Agentforce, ServiceNow Now Assist, SAP, Microsoft Power Platform, Microsoft Graph, Epic/FHIR, HealthLake, Veeva, IQVIA
governance     Responsible AI, model risk and auditability, NIST CSF, ISO 27001, ISO 42001, EU AI Act, GDPR/CCPA, HIPAA, GxP; on-prem and sovereign AI, multi-tenancy, PII redaction, SSO (OIDC/SAML), RBAC
languages      Python, C++/CUDA, SQL, TypeScript, Go, Java, C#, R
```

[mikitadaroshkin.com](https://mikitadaroshkin.com/) | mikitadaroshkin@gmail.com
