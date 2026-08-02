# Agentic-system-for-prior-authorization

An agentic system that automates the first-pass review a payer's clinical
staff performs manually: given a clinical note and a requested medical
service, it extracts the relevant clinical facts, retrieves the applicable
coverage policy, checks documented criteria against that policy, and returns
a decision recommendation — Approve, Deny, or Insufficient Information —
with a citation back to the specific policy clause that justifies it.

## Why this project

Prior-authorization review currently requires a human reviewer to read a
full clinical note, look up the relevant coverage policy, and manually
cross-reference documented criteria before making a decision. This project
automates that first pass so a reviewer can confirm the agent's reasoning
in minutes instead of doing the full manual review — while keeping a human
in the loop for the final accept/override decision.

Scope for this build: **lumbar spine MRI requests**, evaluated against the
real CMS Local Coverage Determination for lumbar MRI (L34220).

## Architecture

```
Requested service + clinical note
        |
        v
[1] Structured Fact Extraction (OpenAI)
    -> age, symptom duration, conservative treatment (bool),
       neurological deficit (bool), red flags (bool), etc.
        |
        v
[2] Policy Retrieval (LangChain + Pinecone)
    -> relevant LCD clause(s), with section/paragraph reference
        |
        v
[3] Rule-Based Criteria Match
    -> checklist of which required criteria are met / unmet / undetermined
        |
        v
[4] LLM Judgment (only for cases the rule engine can't resolve cleanly)
        |
        v
[5] Decision
    -> Approve / Deny / Insufficient Information
    -> rule-match ratio as the primary confidence signal (e.g. "6/6 matched")
    -> counterfactual for Deny/Insufficient cases, grounded only in the
       retrieved policy text
        |
        v
[6] Audit Trail Output
    -> full reasoning chain rendered top to bottom: facts extracted ->
       policy retrieved -> rules matched/unmatched -> decision ->
       evidence citation -> reviewer accept/override
```

graph TD
    %% [START]
    StartNode(START)
    StartNode --> InputNode

    %% INPUT: Requested Service + Clinical Note
    InputNode[INPUT: Requested Service + Clinical Note]

    %% Splitting into concurrent paths
    InputNode --> PathAPolicies
    InputNode --> PathBMatching

    %% Path A: Policies
    subgraph PathA [Path A: Policy Context Management]
        PathAPolicies[Concurrent Path A: Policies]
        PathAPolicies --> FolderNode

        FolderNode[Folder / Central Context]
        FolderNode --> Step3Retrieval

        %% [STEP 3]: Policy Retrieval (LangChain + Pinecone)
        Step3Retrieval[STEP 3: Policy Retrieval]
        Step3Retrieval --> |LangChain + Pinecone Agent| ActionStep3
        ActionStep3(Action: Retrieve relevant LCD Clauses & section/paragraph references)
        ActionStep3 --> RetrievedPoliciesData

        RetrievedPoliciesData[Retrieved Policy Data]
    end

    %% Path B: Matching
    subgraph PathB [Path B: Rule-Based Fact Matching]
        PathBMatching[Concurrent Path B: Matching]
        PathBMatching --> Step1Extraction

        %% [STEP 1]: Structured Fact Extraction (OpenAI Agent)
        Step1Extraction[STEP 1: Structured Fact Extraction]
        Step1Extraction --> |OpenAI Agent| ActionStep1
        ActionStep1(Action: Extract detailed facts like Age, Symptom Duration, Red Flags, Deficits, and Treatments)
        ActionStep1 --> FactsData

        FactsData[Facts Data]
        FactsData --> Step2Matching

        %% [STEP 2]: Rule-Based Criteria Match
        Step2Matching[STEP 2: Rule-Based Criteria Match]
        Step2Matching --> |Evaluate facts against rules| Step2Evaluate

        %% Statuses from Step 2
        Step2Evaluate(Output: MET / UNMET / UNDETERMINED statuses)
        Step2Evaluate --> Step2Logic

        %% Decision branch for Undetermined status
        Step2Logic{Is Status UNDETERMINED?}
        Step2Logic -- Yes --> Step5Escalation
        Step2Logic -- No (MET/UNMET) --> ResolvedMatchingDataStart with the Input and split into concurrent processing paths.

### Input
- **Requested Service + Clinical Note**

### Processing Path A: Policy Context Management
1.  **Folder / Central Context**: Holds the original request and context.
2.  **[STEP 3] Policy Retrieval**:
    - **Agent**: LangChain + Pinecone
    - **Action**: Retrieve relevant LCD Clauses and section/paragraph references.
    - **Output**: **Retrieved Policy Data** contributes to the Final Decision.

### Processing Path B: Rule-Based Fact Matching
1.  **[STEP 1] Structured Fact Extraction**:
    - **Agent**: OpenAI Agent (implied lightbulb)
    - **Action**: Extract detailed, structured facts such as: Age, Symptom Duration, Red Flags (bool), Neurological Deficit (bool), Conservative Treatment (bool), etc.
2.  **[STEP 2] Rule-Based Criteria Match**:
    - **Action**: Evaluate the extracted facts against pre-defined rules.
    - **Outcome Statuses**: MET, UNMET, UNDETERMINED.
3.  **Core Agentic Escalation Logic (Nested)**:
    - **Condition**: IF Rule-Based Match status is **UNDETERMINED**.
    - **[STEP 5] LLM Judgment (Escalation)**:
        - **Agent**: Escalation Agent (implied second lightbulb)
        - **Action**: Agent uses the clinical note and existing facts to re-evaluate undetermined criteria.
        - **Feedback**: Agent re-evaluation feeds back into **[STEP 2]** to refine the matching status.
4.  **Refined Matching Data**: Once resolved or determined (MET/UNMET/LLM Judgment completed), the data contributes to the Final Decision.

### Final Decision & Synthesis
1.  **[Decision Data Convergence]**: Combine output from retrieved policies and refined matching data.
2.  **[STEP 4] Final Decision**:
    - **Action**: Agent synthesizes all data to make a formal judgment.
    - **Outputs**:
        - **Approval Status**: Approve, Deny, or Insufficient Info.
        - **Match Ratio**: Primary confidence signal (e.g., 6/6 matched).
        - **Grounded Counterfactual**: Explain 'Why denied' using grounded text.

### Output & Review
1.  **[STEP 6] Audit Trail Output**:
    - **Action**: Generate a comprehensive log.
    - **Log Contents**: Full Reasoning Chain, Evidence Citations, Facts Extracted, Rule Match Statuses (Matched/Unmatched), Decision.
2.  **[STEP 7] Reviewer Accept/Override**:
    - **Final Step**: A human reviewer checks the audit trail to confirm or override the final decision.
### **[END]**

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
