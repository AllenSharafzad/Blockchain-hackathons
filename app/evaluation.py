import json
from pathlib import Path
import asyncio
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from bert_score import score as bert_score
from app.chatbot import generate_chat_response

# Configuration
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
QA_PATH = DATA_DIR / "auto_questions.json"
RESULTS_DIR.mkdir(exist_ok=True)

REQUIRED_COLUMNS = {'user_input', 'retrieved_contexts', 'response', 'reference'}

async def run_evaluation():
    """Main evaluation pipeline handling both RAGAS and BERTScore"""
    questions = load_questions()
    if not questions:
        print("⚠️ No questions loaded. Exiting evaluation.")
        return

    # Phase 1: Generate responses and contexts
    predictions = await generate_responses(questions)
    
    # Phase 2: RAGAS Evaluation
    rag_results = await run_ragas(predictions)
    if rag_results is not None:
        analyze_rag_results(rag_results)
    
    # Phase 3: BERTScore Evaluation
    await run_bertscore(predictions)

def load_questions():
    """Load evaluation questions from JSON file"""
    try:
        with open(QA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading questions: {str(e)}")
        return []

async def generate_responses(questions):
    """Generate chatbot responses for all questions"""
    predictions = []
    for q in questions:
        try:
            response = await generate_chat_response({
                "query": q["Question"],
                "user_id": "eval_bot",
                "history": []
            })
            
            retrieved_contexts = response.get("chunks", [])
            if isinstance(retrieved_contexts, str):
                retrieved_contexts = retrieved_contexts.split("\n")
            if not retrieved_contexts:
                print(f"⚠️ Warning: No contexts retrieved for question: {q['Question']}")
            else:
                retrieval_score = response.get("retrieval_score", 0.0)
                print(f"📏 Retrieval score for '{q['Question']}': {retrieval_score:.4f}")

            predictions.append({
                "user_input": q["Question"],
                "reference": q["Answer"],  # Ground truth
                "response": response.get("answer", ""),
                "retrieved_contexts": retrieved_contexts
            })
            
        except Exception as e:
            print(f"❌ Error processing '{q['Question']}': {str(e)}")
            predictions.append({
                "user_input": q["Question"],
                "reference": q["Answer"],
                "response": f"Error: {str(e)}",
                "retrieved_contexts": []
            })
    return predictions

async def run_ragas(predictions):
    """Run RAGAS evaluation metrics"""
    try:
        dataset = Dataset.from_list([{
            "user_input": p["user_input"],
            "retrieved_contexts": p["retrieved_contexts"],
            "response": p["response"],
            "reference": p["reference"]
        } for p in predictions])
        
        if not REQUIRED_COLUMNS.issubset(dataset.features):
            missing = REQUIRED_COLUMNS - set(dataset.features)
            raise ValueError(f"Missing columns in dataset: {missing}")

        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_recall],
        )
        
        # Save results
        df = result.to_pandas()
        df.to_json(RESULTS_DIR / "ragas_results.json", orient="records", indent=2)
        df.to_csv(RESULTS_DIR / "ragas_results.csv", index=False)
        
        return df
        
    except Exception as e:
        print(f"❌ RAGAS evaluation failed: {str(e)}")
        return None

def analyze_rag_results(df):
    """Analyze and report RAG evaluation results"""
    print("\n🔍 RAGAS Results Summary:")
    print(df[['faithfulness', 'answer_relevancy', 'context_recall']].describe())
    
    # Identify low context recall cases
    low_recall = df[df['context_recall'] < 0.5]
    if not low_recall.empty:
        print("\n⚠️ Low Context Recall Cases:")
        for _, row in low_recall.iterrows():
            print(f"\nQuestion: {row['user_input']}")
            print(f"Context Recall: {row['context_recall']:.2f}")
            print(f"Retrieved Contexts: {str(row['retrieved_contexts'])[:100]}...")
            print(f"Reference Answer: {str(row['reference'])[:100]}...")

async def run_bertscore(predictions):
    """Calculate BERTScore for all responses"""
    try:
        references = [p["reference"] for p in predictions]
        candidates = [p["response"] for p in predictions]
        
        P, R, F1 = bert_score(
            candidates,
            references,
            lang="en",
            verbose=True
        )
        
        print(f"\n✅ BERTScore Metrics:")
        print(f"Precision: {P.mean().item():.4f}")
        print(f"Recall: {R.mean().item():.4f}")
        print(f"F1: {F1.mean().item():.4f}")
        
    except Exception as e:
        print(f"❌ BERTScore failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(run_evaluation())