# Agentic System for Medicare Coverage Eligibility Review

Healthcare reviewers spend significant time manually determining whether a patient's clinical documentation satisfies Medicare coverage requirements for a requested medical service. Although CMS publishes Local Coverage Determinations (LCDs) through the Medicare Coverage Database, reviewers must read lengthy clinical notes, extract the relevant clinical information, evaluate it against Medicare coverage criteria, determine whether the requested service meets those criteria, and document the rationale for their decision.

This project automates that first-pass coverage eligibility review using an agentic Retrieval-Augmented Generation (RAG) workflow. Given a clinical note and a requested procedure (starting with Lumbar MRI), the system:

* Extracts structured clinical facts from unstructured documentation,
* Retrieves the relevant CMS Local Coverage Determination (LCD),
* Evaluates the extracted facts against documented coverage criteria using deterministic rules,
* Invokes an LLM only when additional clinical reasoning is required,
* Returns an Approve, Deny, or Insufficient Information recommendation,
* Provides policy citations and an auditable reasoning trail for human review.

## Business goal
The system serves as a clinical decision-support assistant by Reducing the manual effort required to evaluate Medicare coverage eligibility by automating the first-pass comparison between clinical documentation and CMS coverage criteria while preserving human oversight and full decision traceability.


## Scope for this build: 

**lumbar spine MRI requests**, evaluated against the real CMS Local Coverage Determination for lumbar MRI (L34220).

## Architecture

![Architecture diagram](docs/architecture.png)


## Technologies

| Layer | Tool | Purpose |
|---|---|---|
| Orchestration | LangGraph | Multi-step agent workflow across the nodes above |
| Retrieval | LangChain + Pinecone | Chunk, embed, and retrieve policy documents |
| LLM | OpenAI | Structured fact extraction and clinical judgment |
| Evaluation | RAGAS | Retrieval quality and decision-accuracy scoring against labeled scenarios |
| Experiment tracking | MLflow | Logging retrieval experiments and prompt variants (not model training) |
| API | FastAPI | Serves the pipeline as a web service |
| Containerization | Docker | Packages the FastAPI service |
| Deployment | AWS ECS | Hosts the containerized app |
| CI/CD | GitHub Actions | Lint/test on PR, build + push image, deploy on merge |

## Data sources

- **Clinical notes (real):** `AGBonnet/augmented-clinical-notes` (Hugging
  Face) — case narratives sourced from PMC-Patients, real doctor-written
  case presentations from open-access PubMed Central case studies.
- **Coverage policy (real):** CMS Local Coverage Determination L34220
  (Lumbar MRI) — a U.S. government work, public domain.
- **Ground-truth labels:** hand-labeled prior-auth scenarios, built by
  pairing a note + requested service + decision + rationale against the
  actual criteria in L34220.

## Evaluation

The labeled scenario set serves as ground truth for:
- Retrieval quality (did the system pull the correct policy clause)
- Decision accuracy (does the output match the expected Approve/Deny/
  Insufficient Information label)
- Citation correctness (is the cited clause the one that actually
  justifies the decision)
- Missing-information detection (does the system correctly flag
  Insufficient Information rather than guessing)

## Status

This README describes the intended end-to-end architecture. Build status
and progress are tracked separately as the project develops.
