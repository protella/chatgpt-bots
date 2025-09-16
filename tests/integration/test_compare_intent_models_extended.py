#!/usr/bin/env python3
"""
Extended comparison of intent classification between GPT-5-mini and GPT-4.1-mini
with more comprehensive test cases and context scenarios
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
setup_logger("model_comparison_extended", level="INFO")

# Models to compare
MODELS_TO_TEST = [
    "gpt-5-mini",
    "gpt-4.1-mini-2025-04-14"
]

# Extended test cases with more variety and edge cases
TEST_CASES = {
    "new_image": [
        # Clear image generation requests
        "draw me a picture of a cat",
        "generate an image of a sunset over mountains",
        "create a logo for my coffee shop",
        "I need a picture of a beach scene",
        "make me a picture of a cityscape at night",

        # More complex generation requests
        "can you create an illustration of a robot playing chess with a dinosaur",
        "design a poster for a jazz concert",
        "draw a fantasy landscape with dragons",
        "create an infographic about climate change",
        "generate a cyberpunk street scene",

        # Subtle generation requests
        "I want to see what a fusion of art deco and modern architecture looks like",
        "show me your interpretation of happiness",
        "visualize the concept of time",
        "paint me a dream",
        "illustrate a haiku about spring",

        # Commands that might be confused with edits
        "produce a new image of a car",
        "make a fresh illustration of a tree",
        "start from scratch and draw a house",
        "create something completely new with flowers",
        "generate original artwork of a sunset"
    ],

    "edit_image": [
        # Clear edit requests
        "make the sky blue in this image",
        "remove the background from this photo",
        "change the color of the car to red",
        "add snow to this scene",
        "make the colors more vibrant",

        # Complex edits
        "replace the person in the foreground with a robot",
        "transform this photo into a watercolor painting style",
        "make this daytime scene look like nighttime",
        "add dramatic lighting to this portrait",
        "remove all the text from this image",

        # Style transfers
        "make this look like van gogh painted it",
        "apply a vintage filter to this",
        "turn this into pixel art",
        "make it look more professional",
        "give this a cinematic look",

        # Specific modifications
        "crop this to focus on the center",
        "blur the background but keep the subject sharp",
        "enhance the details in the shadows",
        "fix the exposure in this overexposed photo",
        "correct the white balance"
    ],

    "vision": [
        # Analysis requests
        "what do you see in this image?",
        "describe what's in this picture",
        "analyze this document for me",
        "can you read the text in this image?",
        "what information is in this PDF?",

        # Specific vision tasks
        "identify all the objects in this photo",
        "transcribe the handwritten text",
        "what's the dominant color in this image?",
        "count how many people are in this picture",
        "what emotions do you detect in these faces?",

        # Document understanding
        "summarize this screenshot",
        "extract the key points from this slide",
        "what does this graph show?",
        "interpret this diagram for me",
        "explain what this flowchart represents",

        # Complex analysis
        "compare these two images",
        "what's unusual about this photo?",
        "identify the architectural style",
        "what time period does this appear to be from?",
        "analyze the composition of this artwork"
    ],

    "text_only": [
        # Clear text requests
        "what's the weather like today?",
        "tell me a joke",
        "how do I cook pasta?",
        "explain quantum computing",
        "what are the benefits of exercise?",

        # Knowledge questions
        "who was the first person on the moon?",
        "explain the theory of relativity",
        "what's the difference between RAM and ROM?",
        "how does photosynthesis work?",
        "what caused the 2008 financial crisis?",

        # Creative writing
        "write a short story about time travel",
        "compose a limerick about programming",
        "create a recipe for chocolate cake",
        "draft an email declining a meeting",
        "write a product description for a smartwatch",

        # Problem solving
        "help me debug this Python code",
        "solve this math equation: 2x + 5 = 15",
        "suggest improvements for my resume",
        "plan a 3-day trip to Tokyo",
        "how can I improve my public speaking?"
    ],

    "ambiguous_image": [
        # Very ambiguous
        "show me",
        "make it better",
        "fix it",
        "create something",
        "I want to see it",

        # Could be edit or new
        "change this",
        "do something cool",
        "make it different",
        "improve this",
        "transform it",

        # References without clear intent
        "do that again",
        "another one",
        "more of this",
        "something similar",
        "like before",

        # Vague requests
        "surprise me",
        "be creative",
        "use your imagination",
        "do your thing",
        "show me what you can do"
    ]
}

def create_rich_conversation_context(context_type: str) -> List[Dict]:
    """Create richer, more realistic conversation contexts"""

    if context_type == "with_recent_image_generation":
        # Context where user just generated an image
        return [
            {"role": "user", "content": "Create an image of a futuristic city"},
            {"role": "assistant", "content": "I've generated an image of a futuristic city with towering skyscrapers, flying vehicles, and neon lights illuminating the night sky."},
            {"role": "user", "content": "That's amazing! The colors are perfect"},
            {"role": "assistant", "content": "Thank you! I'm glad you like the vibrant cyberpunk aesthetic with the purple and blue neon colors contrasting against the dark skyline."}
        ]

    elif context_type == "with_recent_image_edit":
        # Context where user just edited an image
        return [
            {"role": "user", "content": "I have a photo that needs editing"},
            {"role": "assistant", "content": "I'd be happy to help edit your photo. What changes would you like to make?"},
            {"role": "user", "content": "Remove the people in the background"},
            {"role": "assistant", "content": "I've successfully removed the people from the background while preserving the main subject and the natural scenery."}
        ]

    elif context_type == "with_document_analysis":
        # Context with recent document analysis
        return [
            {"role": "user", "content": "Here's my quarterly report\n[Document content truncated for classification]"},
            {"role": "assistant", "content": "I've analyzed your quarterly report. The document shows a 15% increase in revenue compared to last quarter, with particularly strong performance in the digital services division."},
            {"role": "user", "content": "What were the main cost drivers?"},
            {"role": "assistant", "content": "The main cost drivers were: 1) Marketing expenses (up 22%), 2) R&D investments (up 18%), and 3) New hire onboarding costs (up 12%)."}
        ]

    elif context_type == "technical_discussion":
        # Technical conversation context
        return [
            {"role": "user", "content": "I'm working on a machine learning model for image classification"},
            {"role": "assistant", "content": "That's interesting! What type of architecture are you using? CNNs are typically effective for image classification tasks."},
            {"role": "user", "content": "Yes, I'm using a ResNet50 but having issues with overfitting"},
            {"role": "assistant", "content": "For overfitting in ResNet50, you might want to try: data augmentation, dropout layers, L2 regularization, or reducing model complexity. What's your current dataset size?"}
        ]

    elif context_type == "mixed_media":
        # Context with both text and image discussions
        return [
            {"role": "user", "content": "I'm writing an article about architecture"},
            {"role": "assistant", "content": "I'd be happy to help with your architecture article. What aspect are you focusing on?"},
            {"role": "user", "content": "Modern sustainable design. I need both content and visuals"},
            {"role": "assistant", "content": "For sustainable architecture, key topics include passive solar design, green roofs, recycled materials, and LEED certification. Would you like me to elaborate on any of these?"}
        ]

    else:  # normal conversation
        return [
            {"role": "user", "content": "Hello, how are you today?"},
            {"role": "assistant", "content": "Hello! I'm doing well, thank you. How can I help you today?"},
            {"role": "user", "content": "I have various tasks I need help with"},
            {"role": "assistant", "content": "I'm here to help! I can assist with writing, analysis, image generation, editing, answering questions, and many other tasks. What would you like to start with?"}
        ]

def test_single_classification(client: OpenAIClient, query: str, expected_intent: str,
                              context_type: str, model: str) -> Tuple[str, bool, float]:
    """Test a single intent classification with specified model"""

    # Temporarily override the utility model
    original_model = config.utility_model
    config.utility_model = model

    # Create appropriate context
    context = create_rich_conversation_context(context_type)

    # Handle special cases for vision intent
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

def compare_models_extended():
    """Run extended comparison between models"""

    print("="*100)
    print("EXTENDED MODEL COMPARISON: INTENT CLASSIFICATION")
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

    # Context rotation for variety
    context_types = [
        "normal",
        "with_recent_image_generation",
        "with_recent_image_edit",
        "with_document_analysis",
        "technical_discussion",
        "mixed_media"
    ]

    # Test each case with each model
    total_tests = sum(len(cases) for cases in TEST_CASES.values())
    test_num = 0

    for expected_intent, queries in TEST_CASES.items():
        print(f"\n{'='*50}")
        print(f"Testing {expected_intent.upper()} intent ({len(queries)} cases)")
        print(f"{'='*50}")

        for i, query in enumerate(queries, 1):
            test_num += 1

            # Determine context type based on intent and rotation
            if expected_intent == "edit_image":
                # Mix of contexts, some with recent images
                if i % 3 == 0:
                    context_type = "with_recent_image_generation"
                elif i % 3 == 1:
                    context_type = "with_recent_image_edit"
                else:
                    context_type = "mixed_media"
            elif expected_intent == "vision":
                # Always needs document/analysis context
                context_type = "with_document_analysis" if i % 2 == 0 else "mixed_media"
            elif expected_intent == "ambiguous_image":
                # Rotate through all contexts for ambiguous
                context_type = context_types[i % len(context_types)]
            elif expected_intent == "new_image":
                # Mix of normal and technical contexts
                context_type = "technical_discussion" if i % 3 == 0 else "normal"
            else:
                # Text only - normal or technical
                context_type = "technical_discussion" if i % 2 == 0 else "normal"

            print(f"\n[{test_num}/{total_tests}] Query: {query[:50]}...")
            print(f"  Context: {context_type}")

            comparison_row = {
                "Intent": expected_intent,
                "Query": query[:35] + "..." if len(query) > 35 else query,
                "Context": context_type[:20]
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
                time.sleep(0.2)

            detailed_comparisons.append(comparison_row)

            # Progress indicator every 10 tests
            if test_num % 10 == 0:
                print(f"\n>>> Progress: {test_num}/{total_tests} tests completed ({test_num*100//total_tests}%)")

    # Print comparison summary
    print("\n" + "="*100)
    print("SUMMARY COMPARISON")
    print("="*100)

    summary_data = []

    # Calculate per-intent accuracy for each model
    for intent in TEST_CASES.keys():
        row = {"Intent": intent, "Tests": len(TEST_CASES[intent])}
        for model in MODELS_TO_TEST:
            correct = sum(model_results[model][intent])
            total = len(model_results[model][intent])
            accuracy = (correct / total * 100) if total > 0 else 0
            row[f"{model[:15]}"] = f"{accuracy:.1f}% ({correct}/{total})"
        summary_data.append(row)

    # Add overall accuracy
    row = {"Intent": "OVERALL", "Tests": total_tests}
    for model in MODELS_TO_TEST:
        all_results = []
        for intent_results in model_results[model].values():
            all_results.extend(intent_results)
        correct = sum(all_results)
        total = len(all_results)
        accuracy = (correct / total * 100) if total > 0 else 0
        row[f"{model[:15]}"] = f"{accuracy:.1f}% ({correct}/{total})"
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
                "Median": f"{sorted(times)[len(times)//2]:.3f}s",
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
                "Query": row["Query"],
                "Context": row["Context"]
            }
            for model in MODELS_TO_TEST:
                disagreement[model[:15]] = row.get(f"{model}_result", "?")
            disagreements.append(disagreement)

    if disagreements:
        print(f"Found {len(disagreements)} disagreements:")
        print(tabulate(disagreements[:20], headers="keys", tablefmt="grid"))  # Show first 20
        if len(disagreements) > 20:
            print(f"... and {len(disagreements) - 20} more disagreements")
    else:
        print("No disagreements - all models classified identically!")

    # Analyze failure patterns
    print("\n" + "="*100)
    print("FAILURE ANALYSIS")
    print("="*100)

    for model in MODELS_TO_TEST:
        print(f"\n{model} Failures:")
        failures = []
        for row in detailed_comparisons:
            if row.get(f"{model}_correct") == "✗":
                failures.append({
                    "Expected": row["Intent"],
                    "Got": row.get(f"{model}_result"),
                    "Query": row["Query"],
                    "Context": row["Context"]
                })

        if failures:
            print(tabulate(failures[:10], headers="keys", tablefmt="grid"))
            if len(failures) > 10:
                print(f"... and {len(failures) - 10} more failures")
        else:
            print("  No failures!")

    # Save detailed results to file
    output_file = "intent_model_comparison_extended.json"
    with open(output_file, "w") as f:
        json.dump({
            "models": MODELS_TO_TEST,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_tests": total_tests,
            "summary": summary_data,
            "performance": perf_data,
            "disagreements": disagreements,
            "detailed": detailed_comparisons
        }, f, indent=2)
    print(f"\n✓ Detailed results saved to {output_file}")

if __name__ == "__main__":
    try:
        compare_models_extended()
    except KeyboardInterrupt:
        print("\n\nComparison interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nComparison failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)