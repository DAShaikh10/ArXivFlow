# ArXivFlow

An end-to-end **research paper recommender system** for the NLP domain — scrape → quality-gate →
annotate → embed → recommend → evaluate → serve, built as a reproducible [Moon][moonrepo-url]
monorepo and deployed on [Kubernetes][k8s-url].

Given a paper _(or a free-text query)_ ArXivFlow returns the most relevant papers from a curated
corpus, combining **dense semantic embeddings**, **lexical (BM25)** retrieval, **citation-graph
overlap**, and **NER entity overlap** — fused with Reciprocal Rank Fusion and evaluated _honestly_
against two independent ground truths.

---

<div align = "center">

![Moonrepo][moonrepo-shield]
![proto][proto-shield]
![UV][uv-shield]
![Python][python-shield]
![K8S][k8s-shield]
![FastAPI][fastapi-shield]
![Next.js][nextjs-shield]
![ChromaDB][chromadb-shield]
![W&B][wandb-shield]

</div>

<div align = "center">

![Docker Image Size][arxivflow-crucible-image-shield]
![Docker Image Size][arxivflow-annotate-image-shield]
![Docker Image Size][arxivflow-lsml-image-shield]
![Docker Image Size][arxivflow-papervec-image-shield]
![Docker Image Size][arxivflow-api-image-shield]
![Docker Image Size][arxivflow-client-image-shield]

</div>

---

## Pipeline

```text
arXiv API + Semantic Scholar API        (crucible)
  → Kubernetes Jobs scrape into a shared PVC
  → Data-quality gate                    (crucible)          1000 → 992 clean papers
  → Label Studio + model-assisted NER    (annotate + lsml)
  → Vector embeddings in ChromaDB        (papervec)          4 encoder models × 992 vectors
  → Recommender: dense + BM25 + citation + entity, RRF fusion   (compass)
  → Evaluation: tag-overlap + citation-overlap proxies + LLM-judge gold   (compass)
  → Perturbation / robustness analysis   (compass)
  → FastAPI backend + Next.js frontend   (api + client)
```

- **Scrape** 1,000 NLP papers _(arXiv `cs.CL` / ACM I.2.7 / MSC 68T50)_ from the [ArXiv API][arxiv-api-url]
  and enrich them with influential-citation counts and reference graphs from the [Semantic Scholar API][semantic-scholar-api-url].
- **Quality-gate** the raw dataset _(completeness, uniqueness, consistency, validity)_ down to a clean
  **992-paper** corpus before anything is annotated or embedded.
- **Annotate** abstracts with 7-label Named Entity Recognition in [Label Studio][label-studio-url],
  assisted by a custom ML backend and cross-checked against GLiNER, phi-4, Qwen2.5-14B and Claude Opus 4.8.
- **Embed** every paper with domain-specific scientific encoders _(SPECTER2, SciNCL, BGE-large, Qwen3-8B)_
  into [ChromaDB][chromadb-url] _(HNSW, cosine)_.
- **Recommend** with a multi-signal recommender fused by Reciprocal Rank Fusion, and **evaluate** it against
  two independent proxy ground truths plus an LLM-judge gold set.
- **Serve** the corpus, similarity neighbours and a 2D embedding atlas through a FastAPI backend and a
  Next.js frontend.

---

## Monorepo

The workspace is orchestrated by [Moon][moonrepo-url] _(task runner + project graph)_, with tool
versions pinned by [proto][proto-url] and Python dependencies managed by [uv][uv-url]. Each package /
app owns its own `moon.yml`, `pyproject.toml` and `Dockerfile.tera`.

| Project                         | Kind    | Role                                                                      |
| :------------------------------ | :------ | :------------------------------------------------------------------------ |
| [`crucible`](packages/crucible) | package | Scrape arXiv + Semantic Scholar, enrich metadata, pre-annotation QC       |
| [`annotate`](packages/annotate) | package | NER annotation _(Label Studio / GLiNER / LM)_ + inter-annotator agreement |
| [`lsml`](packages/lsml)         | package | Label Studio ML backend serving GLiNER annotation suggestions             |
| [`papervec`](packages/papervec) | package | Generate SPECTER2 _(+ alt encoders)_ embeddings into ChromaDB             |
| [`compass`](packages/compass)   | package | Framework-agnostic recommender logic, evaluation & robustness harness     |
| [`api`](apps/api)               | app     | FastAPI backend — listings, cosine neighbours, embedding atlas            |
| [`client`](apps/client)         | app     | Next.js _(App Router)_ + SCSS frontend                                    |

Run any task from the workspace root:

```bash
moon run <project>:<task>     # e.g. moon run crucible:dockerize, moon run compass:evaluate
```

---

## Structure

```bash
.
├── apps/
│ ├── api/          # FastAPI backend (serves listings + SPECTER2 neighbours + atlas)
│ └── client/       # Next.js + SCSS frontend
├── packages/
│ ├── crucible/     # scraping + enrichment + pre-annotation QC
│ ├── annotate/     # NER annotation + IAA
│ ├── lsml/         # Label Studio ML backend (GLiNER)
│ ├── papervec/     # SPECTER2 / alt-encoder embeddings → ChromaDB
│ └── compass/      # recommender + evaluation + perturbation/robustness
├── data/           # shared artifacts (raw dataset, annotations, embeddings, eval reports)
├── k8s/            # cluster-wide manifests (PVC + PVC inspector)
├── scripts/        # workspace helpers
├── .moon/          # Moon workspace + toolchain config
├── .prototools     # pinned tool versions (moon, proto, uv)
├── PLAN.md         # work-package plan / lab notebook
├── SOURCES.md      # presentation fact sheet (every headline number, sourced from the repo)
└── README.md       # this file
```

Shared artifacts live in `data/` and flow **one way** between packages — `crucible` writes the cleaned
dataset, `papervec` writes the embeddings, and `compass` / `api` only ever _read_ them.

---

## Kubernetes

All heavy workloads run as Jobs / Deployments on the cluster, mounting a shared
**`arxivflow-pvc`** _(5Gi, ReadWriteMany)_ so each stage hands artifacts to the next through the PVC.
GPU workloads target **NVIDIA L4** nodes _(24GB, Ada Lovelace)_ via `nodeSelector`.

```bash
kubectl apply -f k8s/pvc.yml            # shared 5Gi RWX volume
kubectl apply -f k8s/inspect-pvc.yml    # long-lived inspector pod for kubectl cp in/out
```

Per-package cluster usage _(image build, secrets, jobs, teardown)_ is documented in each package README.
Images are published to Docker Hub under [`dashaikh10/`][dockerhub-url].

<!-- REFERENCES -->

[arxiv-api-url]: https://info.arxiv.org/help/api/basics.html
[semantic-scholar-api-url]: https://www.semanticscholar.org/product/api
[label-studio-url]: https://labelstud.io/
[chromadb-url]: https://www.trychroma.com/
[dockerhub-url]: https://hub.docker.com/repositories/dashaikh10
[moonrepo-url]: https://moonrepo.dev/
[proto-url]: https://moonrepo.dev/proto
[uv-url]: https://github.com/astral-sh/uv
[arxiv-api]: https://info.arxiv.org/help/api/index.html
[acm-categories]: https://dl.acm.org/ccs
[arxiv-categories]: https://arxiv.org/category_taxonomy
[msc-categories]: https://mathscinet.ams.org/mathscinet/msc/pdfs/classifications2020.pdf
[k8s-url]: https://kubernetes.io/
[moonrepo-shield]: https://img.shields.io/badge/Moonrepo-Informational?style=flat&logo=moonrepo&labelColor=fff&color=%236f53f3
[proto-shield]: https://img.shields.io/badge/proto-Informational?style=flat&logo=moonrepo&labelColor=fff&color=%236f53f3
[uv-shield]: https://img.shields.io/badge/UV-Informational?style=flat&logo=uv&labelColor=fff&color=%23de5fe9
[python-shield]: https://img.shields.io/badge/Python%203.14-Informational?style=flat&logo=python&logoColor=ffd43b&labelColor=306998&color=306998
[k8s-shield]: https://img.shields.io/badge/Kubernetes-Informational?style=flat&logo=kubernetes&logoColor=326ce5&labelColor=fff&color=326ce5
[fastapi-shield]: https://img.shields.io/badge/FastAPI-Informational?style=flat&logo=fastapi&logoColor=fff&labelColor=009688&color=009688
[nextjs-shield]: https://img.shields.io/badge/Next.js-Informational?style=flat&logo=nextdotjs&logoColor=fff&labelColor=000&color=000
[chromadb-shield]: https://img.shields.io/badge/ChromaDB-Informational?style=flat&logo=chromadb&logoColor=fff&labelColor=ff6d00&color=ff6d00
[wandb-shield]: https://img.shields.io/badge/Weights%20%26%20Biases-Informational?style=flat&logo=weightsandbiases&logoColor=fff&labelColor=ffbe00&color=ffbe00
[arxivflow-crucible-image-shield]: https://img.shields.io/docker/image-size/dashaikh10/arxivflow-crucible?style=flat&label=arxivflow-crucible
[arxivflow-annotate-image-shield]: https://img.shields.io/docker/image-size/dashaikh10/arxivflow-annotate?style=flat&label=arxivflow-annotate
[arxivflow-lsml-image-shield]: https://img.shields.io/docker/image-size/dashaikh10/arxivflow-lsml?style=flat&label=arxivflow-lsml
[arxivflow-papervec-image-shield]: https://img.shields.io/docker/image-size/dashaikh10/arxivflow-papervec?style=flat&label=arxivflow-papervec
[arxivflow-api-image-shield]: https://img.shields.io/docker/image-size/dashaikh10/arxivflow-api?style=flat&label=arxivflow-api
[arxivflow-client-image-shield]: https://img.shields.io/docker/image-size/dashaikh10/arxivflow-client?style=flat&label=arxivflow-client
