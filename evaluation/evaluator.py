from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from loguru import logger


def run_ragas_evaluation(rag_pipeline) -> dict:
    test_cases = [
        {
            "question": "What are the main risk factors for Apple?",
            "ground_truth": "Apple faces risks including macroeconomic conditions, international operations, foreign exchange exposure, retail store risks, and factors affecting business reputation and financial performance."
        },
        {
            "question": "What is Apple's exposure to foreign exchange risk?",
            "ground_truth": "Apple is exposed to movements in foreign exchange rates. The company's financial performance is subject to risks associated with changes in the value of the U.S. dollar relative to local currencies."
        },
        {
            "question": "What does Item 1A cover in this document?",
            "ground_truth": "Item 1A covers Risk Factors - the various risks that could affect Apple's business, reputation, results of operations, financial condition and stock price."
        },
    ]

    logger.info(f"Running RAGAS evaluation on {len(test_cases)} test cases...")

    rows = []
    for case in test_cases:
        logger.info(f"Testing: {case['question'][:60]}")
        result = rag_pipeline.query(
            question=case["question"],
            session_id="ragas_eval"
        )
        rows.append({
            "question": case["question"],
            "answer": result["answer"],
            "contexts": [s["preview"] for s in result["sources"]],
            "ground_truth": case["ground_truth"],
        })

    ollama_llm = LangchainLLMWrapper(
        OllamaLLM(model="mistral", temperature=0)
    )
    ollama_embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model="nomic-embed-text")
    )

    dataset = Dataset.from_list(rows)

    scores = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        ],
        llm=ollama_llm,
        embeddings=ollama_embeddings,
    )

    score_df = scores.to_pandas()
    logger.info("\n" + str(score_df[["faithfulness", "answer_relevancy", "context_recall", "context_precision"]]))

    return scores
