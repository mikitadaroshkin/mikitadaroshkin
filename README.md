# Mikita Daroshkin

I build production LLM and ML systems end to end at Fortune 500 scale, plus the evaluation that keeps them correct: golden datasets, LLM-as-judge regression gates in CI, OpenTelemetry tracing, deterministic replay, merge-blocking guardrails for faithfulness, hallucination, jailbreak, and toxicity. Nine years, thirty-plus production systems, from CUDA kernels up.

## Selected systems

Anonymized; production, regulated industries.

- Multi-agent research mesh: a seven-agent Google ADK portfolio federated with an enterprise LLM on Vertex AI Agent Engine over A2A, OAuth token-exchange identity at every hop, maker-checker verification, log-prob confidence gates with deterministic fallbacks. Multi-step research from weeks to 1.4 minutes.
- On-device multimodal assistant: an 8B chat-RAG stack compressed 27x to under 600 MB (quantization, distillation, LiteRT/TFLite GPU delegates, ONNX Runtime Mobile, NNAPI), running fully offline on mid-tier Android.
- LLM serving fleet: vLLM, TensorRT-LLM, SGLang, and Triton on multi-GPU GKE with continuous batching, KV-cache optimization, speculative decoding, FlashAttention, and model routing. 3.2 to 4.7x throughput per dollar on seven-figure budgets.
- Division-wide LLM evaluation and observability: golden datasets, LLM-as-judge regression gates in CI, OpenTelemetry tracing, merge-blocking hallucination, jailbreak, and toxicity guardrails. Release regressions down 80%.
- Document intelligence: OCR and IDP with rule-plus-LLM cross-document validation and human-in-the-loop routing. 99.2% extraction accuracy, 2.8x straight-through processing.
- Calibrated triage ranking: tabular (XGBoost), imaging (ResNet-50), and LLM-enriched features fused under conformal prediction and bias audits, with dual-encoder retrieval. NDCG up 28 points.
- Agentic HR automation: function-calling agents over enterprise HCM APIs at 65k-employee scale, HITL escalation, roughly 70% self-service resolution.
- Conversational booking agent on Amazon Bedrock over legacy reservation APIs: multi-turn scenario eval suites, p95 under 2.5 seconds at national scale, surge-tested to 12x peak.
- GraphRAG and text-to-SQL: schema-linking with EXPLAIN-driven self-correction, and vision-language agent crews over a 12 TB multimodal document lake.
- Company-wide AI platform: a research-agent swarm, gold-set eval harness, and semantic dedup on Vertex AI Vector Search with manifest-generated IaC. Tabular models to production in 3 days, down from 6 weeks.
- Air-gapped research assistant: a QLoRA-tuned 7B with hybrid retrieval, reranking, and grounded citations. Knowledge-acquisition up 4.6x.
- 3D-scan quality control: a multimodal RotationNet trained on a 10M+ scan corpus with FSDP and DeepSpeed in BF16. Critical complaints down 16%.

## Also shipped

- Generative NLP, founder: an ML copywriting platform zero to launch, retrieval-grounded generation on GPT-2 and GPT-3, a training-data flywheel of weak supervision and dedup; the fine-tuned GPT-2 beat GPT-3 few-shot on brand voice at one eighteenth the cost.
- Computer vision: ViT image captioning that took first place in a data-science competition; 3D U-Net MRI segmentation (thesis); Mask R-CNN and CLIP-style joint text-image embeddings. Two patent disclosures.
- Recommenders and predictive ML: retrieval-cascade recommenders (multi-armed bandits, FAISS, DSSM ranking, CNN-attention, graph-ML over a social graph); LTV forecasting with unsupervised embeddings, LSTMs, and a ZILN-loss framework (R2 up 23%); churn with gradient boosting, SHAP, and calibration.
- Real-time systems: game backends at over a million daily users, lock-free matchmaking, Kafka telemetry over protobuf, statistical anti-cheat with isolation forests under a 0.1% false-positive ban rate.
- HPC and search: tiled GPU (CUDA) and MPI algorithms; a hierarchical search engine over PageRank, BM25, and BERT; distributed systems on Nginx and Cassandra.
- Data and BI: countrywide finance BI systems, star-schema warehouses and SSIS ETL cut from six hours to forty minutes, SSAS OLAP semantic layers, bank-grade governance (row-level security, TDE, audit).
- Edge and IoT: plant-disease detection at 99.85% with LightGBM growth models and an MQTT sensor-fusion pipeline on Raspberry Pi, a hackathon win that converted to a development grant.

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
vision         3D U-Net, ViT, ResNet, Mask R-CNN, CLIP, RotationNet, FSDP/DeepSpeed, computer graphics
voice          Whisper, OpenAI Realtime API, ElevenLabs, Deepgram, LiveKit, Pipecat, Twilio; VAD, barge-in, speech-to-speech latency, IVR
eval + safety  golden sets, LLM-as-judge, RAGAS, DeepEval, Promptfoo, Braintrust, TruLens, Giskard, Vertex AI Evaluation, Mosaic AI Eval; Guardrails AI, LlamaGuard, NeMo Guardrails; red-teaming, XAI, fairness
observability  OpenTelemetry, LangSmith, Langfuse, Arize Phoenix, Prometheus, Grafana; drift and model monitoring, AIOps
data           BigQuery, Snowflake, Databricks (Unity Catalog, DBRX, Delta Live Tables), Spark, dbt, Dagster, Airflow, Kafka, DuckDB, Iceberg, Hudi, Microsoft Fabric, Azure Synapse; CDC/Debezium, Trino/Presto; lineage and governance (Collibra, Purview)
classical ml   scikit-learn, XGBoost, LightGBM, CatBoost, H2O, SHAP, calibration, causal inference, uplift modeling, survival analysis, time series
cloud          GCP (Vertex AI, GKE, BigQuery), AWS (SageMaker, Bedrock, Amazon Q), Azure (Azure OpenAI, Azure ML); Docker, Kubernetes, Terraform, OpenTofu, Pulumi, Argo, GitOps, Karpenter, Istio, MLflow, Weights and Biases
enterprise     Palantir Foundry/AIP, Salesforce Agentforce, ServiceNow Now Assist, SAP, Microsoft Power Platform, Microsoft Graph, Epic/FHIR, HealthLake, Veeva, IQVIA
governance     Responsible AI, model risk and auditability, NIST CSF, ISO 27001, ISO 42001, EU AI Act, GDPR/CCPA, HIPAA, GxP; on-prem and sovereign AI, multi-tenancy, PII redaction, SSO (OIDC/SAML), RBAC
languages      Python, C++/CUDA, SQL, TypeScript, Go, Java, C#, R
```

[mikitadaroshkin.com](https://mikitadaroshkin.com/) | mikitadaroshkin@gmail.com
