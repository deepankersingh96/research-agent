# Research Agent

An AI-powered academic research assistant for structured literature discovery and evidence synthesis. The project targets the **research analyst** level of capability: it should synthesize literature, identify trends and contradictions, evaluate evidence, and surface research gaps.

## Research intents

The agent is being designed to support several ways of researching a topic:

- **Survey:** understand the current status of a field.
- **Decision:** compare approaches and determine which is most suitable.
- **Gap discovery:** identify unresolved questions and research opportunities.
- **Replication:** understand how a study or approach was carried out.
- **Novelty validation:** assess whether an idea has already been explored.

## Architecture

The top-level workflow is sequential. Each research stage is encapsulated in its own subgraph, allowing the stages to evolve independently.

```text
Question formulation
        |
        v
Query expansion
        |
        v
Retrieval
        |
        v
Filtering and re-ranking
        |
        v
Evidence synthesis
        |
        v
Report generation
```

## Current implementation

| Stage | Status | Current capability |
| --- | --- | --- |
| Question formulation | Implemented | Formulates research questions using the PICOC framework, with a clarification loop when additional information is needed. |
| Query expansion | Implemented | Extracts concepts and synonyms from a PICOC question and generates arXiv-formatted search queries. |
| Retrieval | In progress | An arXiv retrieval tool is being created. Retrieval is not yet an end-to-end supported capability. |
| Filtering and re-ranking | Planned | Filter retrieved works and rank the most relevant evidence. |
| Evidence synthesis | Planned | Analyze trends, contradictions, evidence quality, and research gaps. |
| Report generation | Planned | Produce a structured research report from synthesized evidence. |

## Current limitations

- Question formulation currently supports **PICOC** only. Framework selection is not yet driven by research domain.
- Future work will add frameworks such as **PCC**, **PICO**, and **SPYDER**, selected according to domains such as medicine, social science, and technology.
- Query expansion currently supports only **arXiv** query syntax.
- Core-concept extraction, synonym generation, and database-specific query translation are currently combined in the query-expansion stage.
- Additional databases and their dedicated query translators are planned.

## Roadmap

- Complete arXiv retrieval, then add support for further scholarly databases.
- Implement retrieval filtering and re-ranking.
- Separate core-concept extraction and synonym generation from database-specific query translation.
- Add domain-aware research-question framework selection, including PCC, PICO, SPYDER, and additional frameworks where appropriate.
- Support multiple evidence-synthesis types: systematic reviews, meta-analyses, scoping reviews, and rapid reviews.
- Build evidence synthesis for trend analysis, contradictions, evidence gaps, and evidence quality.
- Implement report generation.

## Responsible AI goal

The long-term goal is for the Research Agent to become compliant with **RAISE** principles for Responsible AI in Evidence Synthesis. This is a future target, not a current compliance or certification claim.

> TODO: Define and maintain a RAISE compliance checklist to guide implementation and evaluate the agent as its capabilities expand.
