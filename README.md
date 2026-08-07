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

  # Build Log: Step-by-Step Development Process

This document records exactly how this project was built, in the order
it was built, so the process can be reproduced from scratch. Each phase
includes the goal, the concrete steps taken (accounts, installs,
commands), what was produced, and why each decision was made.

---

## Phase 0: Environment Setup

**Goal:** A working environment with access to an LLM, an embedding
model, and a vector database.

**Steps:**
1. Chose Google Colab as the development environment (free GPU/runtime
   access, no local setup required).
2. Created accounts and generated API keys for three services:
   - **Pinecone** (pinecone.io) - sign up, navigate to API Keys, copy the
     default key.
   - **Voyage AI** (voyageai.com) - sign up, navigate to API Keys,
     generate a new key.
   - **Google AI Studio** (aistudio.google.com) - sign up, click "Get API
     key," create a key (used for Gemini).
3. Stored all three keys in Colab's built-in Secrets manager (key icon
   in the left sidebar) rather than hardcoding them in notebook cells:
   `PINECONE_API_KEY`, `VOYAGE_API_KEY`, `GOOGLE_API_KEY`.
4. Installed required packages:
   ```
   pip install google-genai voyageai pinecone
   ```
   (Note: the Pinecone Python package was renamed from `pinecone-client`
   to `pinecone` during development - the install command above reflects
   the current correct name.)

**Why Colab Secrets over hardcoding:** keeps credentials out of notebook
files that might be shared or committed to version control.

---

## Phase 1: Data Acquisition

**Goal:** Real clinical notes and a real coverage policy document to
build and test against - no synthetic/fabricated data.

**Steps:**
1. **Clinical notes:** Identified `AGBonnet/augmented-clinical-notes` on
   Hugging Face - real case narratives derived from PMC-Patients
   (published, open-access PubMed Central case studies).
2. Downloaded the dataset (this step requires an environment with
   internet access to huggingface.co):
   ```python
   from datasets import load_dataset
   ds = load_dataset("AGBonnet/augmented-clinical-notes", split="train")
   df = ds.to_pandas()
   ```
3. Filtered the ~30,000-row dataset down to a back-pain/musculoskeletal
   subset (~3,600 rows) using a keyword filter on the note text, since
   the project's initial policy scope is lumbar MRI.
4. Saved both the full dataset and the filtered subset to Google Drive
   (`full_notes.jsonl`, `back_pain_subset.jsonl`) so they persist across
   Colab sessions.
5. **Coverage policy:** Retrieved the real, current text of CMS Local Coverage Determination L34220 (Lumbar MRI) from the Medicare Coverage Database (cms.gov). This document defines the specific criteria a lumbar MRI request must meet to be covered — including a list of "red-flag" conditions (e.g., major trauma, cancer history, motor weakness) that qualify a request immediately, and a separate rule for cases without a red flag requiring at least 4 weeks of documented, failed conservative treatment. Copied the relevant sections into a plain text file, preserving the original section structure (Red Flag Conditions, Non-Red-Flag Criteria, Documentation Requirements, etc.), since that structure was later used directly for chunking in Phase 3. Saved as a clean text file in `data/policy_docs/L34220_lumbar_mri.txt`.

---

## Phase 2: Ground-Truth Scenario Set

**Goal:** A small set of real patient cases with a manually-determined
correct answer, to evaluate the system against later.

**Steps:**
1. Sampled batches of notes from the back-pain subset and manually
   reviewed them against L34220's actual criteria.
2. Discarded false-positive keyword matches (e.g., notes mentioning
   "lumbar" as an anatomical landmark rather than as a back-pain
   complaint).
3. Hand-labeled **5 scenarios**, each pairing: a real note, a requested
   service, an expected decision, and the specific policy clause that
   justifies it.

**Why do this before building the pipeline:** having ground truth
defined up front means every later component (retrieval, rule logic,
LLM judgment) can be checked against a known-correct answer, rather than
just "eyeballing" whether output looks reasonable.

**Reference:** `data/processed/prior_auth_scenarios.json`

---

## Phase 3: Node 2 - Policy Retrieval

**Goal:** Given a natural-language question, retrieve the correct
section of the L34220 policy text.

**Steps:**
1. Split the policy text into 8 sections matching its actual structure
   (Red Flag Conditions, Non-Red-Flag Criteria, Documentation
   Requirements, etc.).
   - **First attempt** used a regex to detect section headers written in
     ALL CAPS. This produced only 5 chunks - the regex missed "RED FLAG
     CONDITIONS" and "NON-RED-FLAG CRITERIA" because their titles include
     a lowercase parenthetical explanation immediately after the caps.
     This silently merged the Red Flag Conditions text into an adjacent
     section.
   - **Fix:** manually defined each section's exact boundaries in code
     instead of relying on pattern detection, since the source document
     is small, fixed, and fully known. This produced the correct 8
     chunks.
2. Embedded each chunk using Voyage AI:
   ```python
   result = vo.embed(chunk_texts, model="voyage-3-large", input_type="document")
   ```
3. Created a Pinecone index (`prior-auth-policy`) and upserted the 8
   embedded chunks, each tagged with its section name and full text as
   metadata.
4. Wrote a `retrieve_policy(query, top_k)` function: embeds the query
   (using `input_type="query"`, Voyage's separate mode for search
   queries vs. stored documents) and returns Pinecone's closest matches.
5. **Validated** by querying with a red-flag-style question and
   confirming the "Red Flag Conditions" chunk was returned - this
   surfaced the chunking bug above on the first attempt (the correct
   chunk didn't even appear in the top 2 results), and confirmed the fix
   after rebuilding the chunks.

**Why manual chunking instead of an automated splitter:** for a small,
fixed source document, manually verifying chunk boundaries is more
reliable than a general-purpose splitting heuristic, and the earlier bug
demonstrated exactly how a heuristic can silently fail in a way that's
hard to notice until retrieval quality is tested.

**Reference:** `src/node2_retrieval.py`

---

## Phase 4: Node 1 - Structured Fact Extraction

**Goal:** Convert a raw clinical note into a fixed set of structured
facts needed to evaluate coverage criteria.

**Steps:**
1. Defined a JSON schema listing exactly which fields to extract: age,
   symptom duration, conservative treatment (bool/null), neurological
   deficit, red flags, requested service.
2. Wrote an extraction function using Gemini's schema-constrained
   structured output, so the model's response is always valid JSON in
   this exact shape:
   ```python
   response = client.models.generate_content(
       model="gemini-3.6-flash",
       contents=prompt,
       config=types.GenerateContentConfig(
           response_mime_type="application/json",
           response_schema=extraction_schema
       )
   )
   ```
3. The prompt explicitly instructs the model not to guess or infer
   information beyond what the note states.
4. **Validated** against all 5 labeled scenarios - extraction correctly
   pulled out age, duration, and red-flag details in every case, and
   correctly returned `null` for fields the note didn't mention (e.g.,
   conservative treatment history), rather than fabricating an answer.

**Model changes during development:** the reasoning/extraction model was
originally planned as OpenAI, briefly switched to Claude Sonnet, and
finally set on **Gemini** (`gemini-3.6-flash`) to keep a two-vendor stack
(Gemini + Voyage AI) instead of three separate providers.

**Reference:** `src/node1_extraction.py`

---

## Phase 5: Bridge (Node 1 to Node 2)

**Goal:** Automatically generate Node 2's search query from Node 1's
extracted facts, so no one has to hand-write a query per patient.

**Steps:**
1. Wrote a plain-code function (no LLM call) that assembles a sentence
   from the structured facts - e.g., "Patient has a red-flag condition:
   motor weakness. Symptom duration: 6 months."
2. Chained it directly into `retrieve_policy()` via an
   `extract_and_retrieve()` wrapper function.
3. **Validated:** the auto-generated query for a red-flag case correctly
   retrieved the Red Flag Conditions section as the top result - and
   scored it *higher* than an earlier hand-written test query had,
   likely because the generated query states the red flag explicitly
   rather than more loosely.

**Why plain code, not another LLM call:** this step has a clear,
deterministic transformation (facts -> sentence) with no ambiguity to
resolve - an LLM call here would only add latency and cost.

**Reference:** included in `src/node2_retrieval.py`
(`build_retrieval_query`, `extract_and_retrieve`)

---

## Phase 6: Node 3 - Rule-Based Criteria Match

**Goal:** Check the extracted facts directly against L34220's two
coverage pathways, resolving clear cases without any LLM call.

**Steps:**
1. Implemented the **red-flag pathway** check: a direct read of
   `red_flags_present` from Node 1's output.
2. Implemented the **non-red-flag pathway** check: requires both (a)
   symptom duration >= 4 weeks and (b) documented conservative treatment.
3. Wrote a duration parser converting free-text durations ("6 months",
   "2 weeks") into a comparable number of weeks.
4. Designed the function so that if either required value can't be
   determined (unparseable duration, or conservative treatment not
   documented either way), the case is flagged `needs_llm_judgment: True`
   with a specific reason - rather than defaulting to an assumption.
5. **Validated** against two known cases: a red-flag case resolved
   entirely by this step alone (no judgment needed), and a sparse-info
   case correctly flagged for further judgment.

**Reference:** `src/node3_rules.py`

---

## Phase 7: Node 4 - LLM Judgment (conditional)

**Goal:** Resolve the specific ambiguous point Node 3 could not, using
only the original note and the retrieved policy text - not general
knowledge.

**Steps:**
1. Defined a second structured-output schema (`resolution`, `reasoning`,
   `policy_citation`).
2. Wrote a prompt providing the original note, Node 1's facts, Node 3's
   specific unresolved question, and the retrieved policy text -
   instructing the model to answer only from that material.
3. Added a safety check after observing a failure mode where the API
   occasionally returned valid-but-empty JSON (`null`) rather than a
   real answer - likely tied to transient server load. The check raises
   a clear error prompting a retry instead of silently passing along an
   empty result.
4. **Validated** on the sparse-info case: correctly returned
   `still_insufficient`, with reasoning citing the specific missing
   duration and conservative-treatment documentation.

**Reference:** `src/node4_judgment.py`

---

## Phase 8: Node 5 - Final Decision

**Goal:** Combine Node 3 and Node 4's outputs into one final decision.

**Steps:**
1. Wrote plain combination logic (no LLM call): red-flag pathway met ->
   Approve; non-red-flag pathway clearly met/not met -> Approve/Deny;
   Node 4's judgment (if it ran) maps to Approve/Deny.
2. **Decision model changed during development:** initially a three-way
   outcome (Approve / Deny / Insufficient Information), matching the
   original project scope. Later collapsed to **two-way (Approve /
   Deny)** - cases that would have been "Insufficient Information" now
   resolve to Deny, with the missing documentation surfaced as a
   separate `missing_information` field.

**Reference:** `src/node5_decision.py`

---

## Phase 9: Node 6 - Audit Trail and Human Review

**Goal:** Present the full reasoning chain in a readable format, and
require a human reviewer to independently confirm the final decision.

**Steps:**
1. Wrote a formatting function assembling every prior node's output into
   one top-to-bottom summary: facts extracted, policy retrieved, rules
   matched, judgment (if used), decision.
2. Built a reviewer step using `input()`: shows the AI's recommended
   decision and reasoning as reference, then requires the reviewer to
   type their own decision and reason - the human's input is the final
   record, and the system separately tracks whether it matched the AI's
   recommendation.
3. Added a color-coded (green/red) HTML decision display for fast visual
   confirmation of the outcome, using `IPython.display.HTML`.

**Reference:** `src/node6_audit_trail.py`, `src/reviewer_action.py`,
`src/decision_display.py`

---

## Phase 10: End-to-End Validation

**Goal:** Confirm the full chain (Node 1 through Node 6) works correctly
across multiple real patient cases, not just in isolation.

**Steps:**
1. Wrote a script running all 5 labeled scenarios through the complete
   pipeline in sequence, printing each result against its expected
   outcome.
2. Validated individual end-to-end runs, including both Approve and Deny
   outcomes and a case requiring Node 4's judgment step.
3. A full batch run of all 5 in one session has not yet completed
   successfully due to hitting the Gemini API free-tier daily request
   quota partway through - a usage-limit issue, not a defect in the
   pipeline logic.

**Reference:** `src/run_all_5_scenarios.py`

## Status

This README describes the intended end-to-end architecture. Build status
and progress are tracked separately as the project develops.
