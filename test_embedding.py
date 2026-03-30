"""Verify embedding model works correctly before full pipeline run."""

import numpy as np
from src.utils.logger import setup_logger, get_logger
from src.embeddings.embedding_model import EmbeddingModel


setup_logger()
logger = get_logger(__name__)



def main():
    model = EmbeddingModel()


    # Test 1: Document embedding shape
    docs = [
        "The transformer model uses self-attention mechanisms.",
        "UAV delivery systems require multi-agent coordination.",
        "Gradient descent optimizes neural network parameters.",
    ]

    doc_embeddings = model.embed_documents(docs, show_progress = False)
    assert doc_embeddings.shape == (3, 768), f"Wrong shape: {doc_embeddings.shape}"
    logger.info(f"✅ Document embedding shape: {doc_embeddings.shape}")

    # Test 2: Query embedding shape
    query_emb = model.embed_query("what is attention mechanism?")
    assert query_emb.shape == (768,), f"Wrong shape: {query_emb.shape}"
    logger.info(f"✅ Query embedding shape: {query_emb.shape}")


    # Test 3: Semantic similarity ordering
    # The first two docs are about ML models - should be more similar
    # to each other than to the UAV doc
    sim_01 = float(np.dot(doc_embeddings[0], doc_embeddings[1]))
    sim_02 = float(np.dot(doc_embeddings[0], doc_embeddings[2]))
    sim_12 = float(np.dot(doc_embeddings[1], doc_embeddings[2])) 


    logger.info(f"Similarity (transformer ↔ gradient descent): {sim_02:.3f}")
    logger.info(f"Similarity (transformer ↔ UAV):              {sim_01:.3f}")
    logger.info(f"Similarity (UAV ↔ gradient descent):         {sim_12:.3f}")


    # Test 4: Query-document similarity direction
    # Query about attention should be closest to doc[0]
    query_emb_2d = query_emb.reshape(1, -1)
    sims = doc_embeddings @ query_emb_2d.T
    best_match = int(np.argmax(sims))
    logger.info(f"✅ Query 'attention mechanism' matched doc[{best_match}]: '{docs[best_match][:50]}'")
    assert best_match == 0, f"Expected doc[0] but got doc[{best_match}]"

    # Test 5: Verify normalization (all vectors should have magnitude ≈ 1.0)
    norms = np.linalg.norm(doc_embeddings, axis = 1)
    assert np.allclose(norms, 1.0, atol = 1e-5), f"Not normalized: {norms}"

    logger.info(f"✅ All embeddings L2-normalized (norms: {norms})")

    logger.info(f"\n✅ All embedding tests passed. Ready for full pipeline.")



if __name__ == "__main__":
    main()  