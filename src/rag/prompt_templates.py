"""
Prompt templates for the ResearchPilot RAG system.

PROMPT ENGINEERING IS NOT OPTIONAL.
The difference between a good RAG system and a bad one
is often entirely in the prompt design.

Key principles we apply:
    1. EXPLICIT GROUNDING: Tell the LLM to ONLY use provided context
    2. CITATION REQUIREMENT: Force the LLM to cite which paper it used
    3. UNCERTAINTY ACKNOWLEDGMENT: If context doesn't answer, say so
    4. STRUCTURED OUTPUT: Consistent format makes parsing reliable
"""

SYSTEM_PROMPT = """You are ResearchPilot, an expert AI research assistant 
specialized in machine learning and AI research papers.

Your job is to answer questions based EXCLUSIVELY on the research paper 
excerpts provided in the context below. 

STRICT RULES:
    1. Only use information from the provided context excerpts
    2. Always cite the paper title and ID when using information from it
    3. If the context does not contain enough information to answer, 
    say "The provided papers do not contain sufficient information 
    to answer this question" - do NOT make up information
    4. Be precise and technical - your users are ML researchers and engineers
    5. When multiple papers discuss the same topic, synthesize their findings
    6. Keep answers focused and well-structured
"""


def build_rag_prompt(query: str, context_chunks: list[dict]) -> str:
    """
    Build the full prompt for the LLM with retrieved context.

    Args:
        query:          User's question
        context_chunks: List of retrieved chunk dicts from RetrievalPipeline

    Returns:
        Formatted prompt string ready to send to the LLM

    PROMPT STRUCTURE:
        [System prompt]
        [Context block - all retrieved chunks with citations]
        [User question]

    WHY WE FORMAT CONTEXT THIS WAY:
        Each chunk is labeled with its paper title and ID.
        This enables the LLM to produce citations like:
        "According to [2603.12248], LoRA constrains..."
        
        Without this labeling, the LLM cannot cite sources
        even if it wanted to.
    """

    # Build context block from retrieved chunks
    context_parts = []


    for i, chunk in enumerate(context_chunks, 1):
        paper_id = chunk.get("paper_id", "unknown")
        title    = chunk.get("title", "Unknown Paper")
        date     = chunk.get("published_date", "")
        text     = chunk.get("text", "")

        context_parts.append(
            f"[SOURCE {i}]\n"
            f"Paper ID: {paper_id}\n"
            f"Title: {title}\n"
            f"Published: {date}\n"
            f"Excerpt:\n{text}\n"
        ) 


    context_block = "\n---\n".join(context_parts)


    prompt = f"""
    CONTEXT - Research Paper Excerpts:
    {context_block}

    ---

    QUESTION: {query}

    INSTRUCTIONS: Answer the question using ONLY the context above. 
    Cite sources using their Paper ID in brackets, e.g. [2603.12248].
    If the context is insufficient, say so clearly.
    """

    return prompt



def build_citation_list(context_chunks: list[dict]) -> list[dict]:
    """
    Build a structured list of cited sources from retrieved chunks.

    Returns deduplicated list of papers used as sources. 
    """
    seen_papers = set()
    citations = []


    for chunk in context_chunks:
        paper_id = chunk.get("paper_id", "")
        if paper_id and paper_id not in seen_papers:
            seen_papers.add(paper_id)
            citations.append(
                {
                    "paper_id":         paper_id,
                    "title":            chunk.get("title", ""),
                    "authors":          chunk.get("authors", []),
                    "published_date":   chunk.get("published_date", ""),
                    "arxiv_url":        chunk.get("arxiv_url", ""),
                }
            )

    return citations