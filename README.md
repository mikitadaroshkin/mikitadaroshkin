# Mikita Daroshkin

I build production LLM systems and the evaluation that gates them: golden datasets, LLM-as-judge scoring in CI, OpenTelemetry traces, deterministic replay, regression thresholds that block merges.

Agent orchestration on LangGraph and Google ADK over A2A and MCP; multi-agent RAG and GraphRAG; hybrid retrieval (dense + BM25, reciprocal-rank fusion, cross-encoder rerank) across pgvector, FAISS, and Vertex AI Vector Search. Reliability is part of the build: maker-checker agent verification, log-prob confidence gates with deterministic fallbacks, structured outputs and function calling, guardrails, OAuth token-exchange identity between agents, egress-controlled perimeters, and policy-as-code promotion gates.

Fine-tuning with LoRA/QLoRA and DPO; distributed training on FSDP and DeepSpeed. Inference serving on vLLM, TensorRT-LLM, and SGLang tuned for throughput per dollar: continuous batching, KV-cache reuse, speculative decoding, GPTQ/AWQ quantization.

Before LLMs: 3D U-Net segmentation on volumetric MRI, ViT captioning, tiled CUDA and MPI kernels, gradient-boosted tabular models.

## Repositories

All build in CI. agentic-rag-evals gates merges on eval-metric regression.

- [agentic-rag-evals](https://github.com/mikitadaroshkin/agentic-rag-evals): LangGraph agentic RAG, hybrid retrieval, LLM-as-judge eval suite, CI regression gate, OpenTelemetry, FastAPI, Docker.
- [gpu-mpi-search](https://github.com/mikitadaroshkin/gpu-mpi-search): BM25 with cross-encoder rerank, tiled CUDA kernels, data-parallel scoring (MPI/multiprocessing). Coursework.
- [mri-segmentation](https://github.com/mikitadaroshkin/mri-segmentation): 3D U-Net, soft-Dice/BCE loss, volumetric inference. BSc thesis.
- [image-captioning-vit](https://github.com/mikitadaroshkin/image-captioning-vit): ViT encoder, Transformer decoder, image captioning.
- [nutrizy](https://github.com/mikitadaroshkin/nutrizy): cross-modal retrieval over a shared text/image embedding space (CLIP), 2019.
- [greenbox](https://github.com/mikitadaroshkin/greenbox): CNN disease classifier, LightGBM growth model, MQTT edge pipeline.
- [warehouse-patterns](https://github.com/mikitadaroshkin/warehouse-patterns): dimensional modeling, star schema, SCD-2, runnable DuckDB demo.

## Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat-square&logo=nvidia&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![LangChain](https://img.shields.io/badge/LangGraph-1C3C3C?style=flat-square&logo=langchain&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/pgvector-4169E1?style=flat-square&logo=postgresql&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)
![Kubernetes](https://img.shields.io/badge/Kubernetes-326CE5?style=flat-square&logo=kubernetes&logoColor=white)
![Terraform](https://img.shields.io/badge/Terraform-7B42BC?style=flat-square&logo=terraform&logoColor=white)
![Google Cloud](https://img.shields.io/badge/Google%20Cloud-4285F4?style=flat-square&logo=googlecloud&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-232F3E?style=flat-square&logo=amazonwebservices&logoColor=white)
![Azure](https://img.shields.io/badge/Azure-0078D4?style=flat-square&logo=microsoftazure&logoColor=white)
![OpenTelemetry](https://img.shields.io/badge/OpenTelemetry-000000?style=flat-square&logo=opentelemetry&logoColor=white)

[mikitadaroshkin.com](https://mikitadaroshkin.com/) | mikitadaroshkin@gmail.com
