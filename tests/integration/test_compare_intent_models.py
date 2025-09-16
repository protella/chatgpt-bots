#!/usr/bin/env python3
"""
Compare intent classification performance between GPT-5-mini and GPT-4.1-mini
"""

import os
import sys
import time
from typing import Dict, List, Tuple
from dotenv import load_dotenv
from collections import defaultdict
from tabulate import tabulate
import json

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
# Also add the project root
project_root = os.path.dirname(parent_dir)
sys.path.insert(0, project_root)

from openai_client import OpenAIClient
from config import config
from logger import setup_logger

# Load environment variables
load_dotenv()

# Setup logging
setup_logger("model_comparison", level="INFO")

# Models to compare
MODELS_TO_TEST = [
    "gpt-5-mini",
    "gpt-4.1-mini-2025-04-14"
]

# Test cases from the original test (subset for faster comparison)
TEST_CASES = {
    "new_image": [
        "draw me a picture of a cat",
        "generate an image of a sunset over mountains",
        "create a logo for my coffee shop",
        "I need a picture of a beach scene",
        "make me a picture of a cityscape at night",
    ],
    "edit_image": [
        "make the sky blue in this image",
        "remove the background from this photo",
        "change the color of the car to red",
        "add snow to this scene",
        "make the colors more vibrant"
    ],
    "vision": [
        "what do you see in this image?",
        "describe what's in this picture",
        "analyze this document for me",
        "can you read the text in this image?",
        "what information is in this PDF?"
    ],
    "text_only": [
        "what's the weather like today?",
        "tell me a joke",
        "how do I cook pasta?",
        "explain quantum computing",
        "what are the benefits of exercise?"
    ],
    "ambiguous_image": [
        "show me",
        "make it better",
        "fix it",
        "create something",
        "I want to see it"
    ]
}

def create_conversation_context(has_recent_image: bool = False, has_document: bool = False) -> List[Dict]:
    """Create a realistic conversation context"""
    context = []

    if has_recent_image:
        context.append({
            "role": "user",
            "content": "draw a picture of a house"
        })
        context.append({
            "role": "assistant",
            "content": "I've generated an image of a house with a cozy design featuring a red brick exterior and white trim."
        })
    elif has_document:
        context.append({
            "role": "user",
            "content": "Here's a document to review\n[Document content truncated for classification]"
        })
        context.append({
            "role": "assistant",
            "content": "I've reviewed the document. It contains financial data and order information."
        })
    else:
        context.append({
            "role": "user",
            "content": "Hello, how are you?"
        })
        context.append({
            "role": "assistant",
            "content": "Hello! I'm doing well, thank you. How can I help you today?"
        })

    return context

def test_single_classification(client: OpenAIClient, query: str, expected_intent: str,
                              context_type: str, model: str) -> Tuple[str, bool, float]:
    """Test a single intent classification with specified model"""

    # Temporarily override the utility model
    original_model = config.utility_model
    config.utility_model = model

    # Create appropriate context
    if context_type == "with_image":
        context = create_conversation_context(has_recent_image=True)
    elif context_type == "with_document":
        context = create_conversation_context(has_document=True)
    else:
        context = create_conversation_context()

    has_attached = False
    if expected_intent == "vision":
        has_attached = True
        query = query + "\n[Note: User has attached images with this message]"

    start_time = time.time()
    try:
        result = client.classify_intent(
            messages=context,
            last_user_message=query,
            has_attached_images=has_attached
        )
        elapsed = time.time() - start_time
        is_correct = result == expected_intent

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"Error with {model} on '{query[:30]}...': {e}")
        result = "error"
        is_correct = False
    finally:
        # Restore original model
        config.utility_model = original_model

    return result, is_correct, elapsed

def compare_models():
    """Compare intent classification between models"""

    print("="*100)
    print("MODEL COMPARISON: INTENT CLASSIFICATION")
    print("="*100)
    print(f"Comparing models: {', '.join(MODELS_TO_TEST)}")
    print(f"Current configured model: {config.utility_model}")
    print("="*100)

    # Initialize client
    client = OpenAIClient()

    # Store results for each model
    model_results = {model: defaultdict(list) for model in MODELS_TO_TEST}
    model_times = {model: [] for model in MODELS_TO_TEST}
    detailed_comparisons = []

    # Test each case with each model
    total_tests = sum(len(cases) for cases in TEST_CASES.values())
    test_num = 0

    for expected_intent, queries in TEST_CASES.items():
        print(f"\n{'='*50}")
        print(f"Testing {expected_intent.upper()} intent")
        print(f"{'='*50}")

        for i, query in enumerate(queries, 1):
            test_num += 1
            print(f"\n[{test_num}/{total_tests}] Query: {query[:60]}...")

            # Determine context type
            if expected_intent == "edit_image" or expected_intent == "ambiguous_image":
                context_type = "with_image" if i % 2 == 0 else "normal"
            elif expected_intent == "vision":
                context_type = "with_document"
            else:
                context_type = "normal"

            comparison_row = {
                "Intent": expected_intent,
                "Query": query[:40] + "..." if len(query) > 40 else query
            }

            # Test each model
            for model in MODELS_TO_TEST:
                print(f"  Testing {model}...", end=" ")
                result, is_correct, elapsed = test_single_classification(
                    client, query, expected_intent, context_type, model
                )

                model_results[model][expected_intent].append(is_correct)
                model_times[model].append(elapsed)

                status = "✓" if is_correct else f"✗ ({result})"
                print(f"{status} [{elapsed:.2f}s]")

                comparison_row[f"{model}_result"] = result
                comparison_row[f"{model}_correct"] = "✓" if is_correct else "✗"
                comparison_row[f"{model}_time"] = f"{elapsed:.2f}s"

                # Small delay between API calls
                time.sleep(0.3)

            detailed_comparisons.append(comparison_row)

    # Print comparison summary
    print("\n" + "="*100)
    print("SUMMARY COMPARISON")
    print("="*100)

    summary_data = []

    # Calculate per-intent accuracy for each model
    for intent in TEST_CASES.keys():
        row = {"Intent": intent}
        for model in MODELS_TO_TEST:
            correct = sum(model_results[model][intent])
            total = len(model_results[model][intent])
            accuracy = (correct / total * 100) if total > 0 else 0
            row[f"{model[:15]}"] = f"{accuracy:.0f}% ({correct}/{total})"
        summary_data.append(row)

    # Add overall accuracy
    row = {"Intent": "OVERALL"}
    for model in MODELS_TO_TEST:
        all_results = []
        for intent_results in model_results[model].values():
            all_results.extend(intent_results)
        correct = sum(all_results)
        total = len(all_results)
        accuracy = (correct / total * 100) if total > 0 else 0
        row[f"{model[:15]}"] = f"{accuracy:.0f}% ({correct}/{total})"
    summary_data.append(row)

    print(tabulate(summary_data, headers="keys", tablefmt="grid"))

    # Performance comparison
    print("\n" + "="*100)
    print("PERFORMANCE COMPARISON")
    print("="*100)

    perf_data = []
    for model in MODELS_TO_TEST:
        times = model_times[model]
        if times:
            perf_data.append({
                "Model": model,
                "Avg Time": f"{sum(times)/len(times):.3f}s",
                "Min Time": f"{min(times):.3f}s",
                "Max Time": f"{max(times):.3f}s",
                "Total Time": f"{sum(times):.1f}s"
            })

    print(tabulate(perf_data, headers="keys", tablefmt="grid"))

    # Show disagreements
    print("\n" + "="*100)
    print("CLASSIFICATION DISAGREEMENTS")
    print("="*100)

    disagreements = []
    for row in detailed_comparisons:
        results = [row.get(f"{model}_result", "") for model in MODELS_TO_TEST]
        if len(set(results)) > 1:  # Models disagreed
            disagreement = {
                "Expected": row["Intent"],
                "Query": row["Query"]
            }
            for model in MODELS_TO_TEST:
                disagreement[model[:15]] = row.get(f"{model}_result", "?")
            disagreements.append(disagreement)

    if disagreements:
        print(tabulate(disagreements, headers="keys", tablefmt="grid"))
    else:
        print("No disagreements - all models classified identically!")

    # Save detailed results to file
    output_file = "intent_model_comparison.json"
    with open(output_file, "w") as f:
        json.dump({
            "models": MODELS_TO_TEST,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": summary_data,
            "performance": perf_data,
            "detailed": detailed_comparisons
        }, f, indent=2)
    print(f"\n✓ Detailed results saved to {output_file}")

if __name__ == "__main__":
    try:
        compare_models()
    except KeyboardInterrupt:
        print("\n\nComparison interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nComparison failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)