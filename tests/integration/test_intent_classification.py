#!/usr/bin/env python3
"""
Comprehensive test for intent classification with real API calls
Tests all 5 intent types: new_image, edit_image, vision, text_only, ambiguous_image
"""

import os
import sys
import time
from typing import Dict, List, Tuple
from dotenv import load_dotenv
from collections import defaultdict
from tabulate import tabulate

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai_client import OpenAIClient
from config import config
from logger import setup_logger

# Load environment variables
load_dotenv()

# Setup logging
setup_logger("test_intent", level="INFO")

# Test cases for each intent type
TEST_CASES = {
    "new_image": [
        "draw me a picture of a cat",
        "generate an image of a sunset over mountains",
        "create a logo for my coffee shop",
        "make an illustration of a robot dancing",
        "draw an image of a bird",
        "I need a picture of a beach scene",
        "can you create an image of a dragon",
        "generate a portrait of a wizard",
        "make me a picture of a cityscape at night",
        "draw a cute puppy playing in the snow"
    ],
    "edit_image": [
        "make the sky blue in this image",
        "remove the background from this photo",
        "change the color of the car to red",
        "make it look more vintage",
        "add snow to this scene",
        "make the image brighter",
        "crop out the person on the left",
        "turn this into a cartoon style",
        "fix the lighting in this photo",
        "make the colors more vibrant"
    ],
    "vision": [
        "what do you see in this image?",
        "describe what's in this picture",
        "analyze this document for me",
        "what's in this photo?",
        "can you read the text in this image?",
        "tell me about this chart",
        "what does this diagram show?",
        "identify the objects in this picture",
        "summarize this document",
        "what information is in this PDF?"
    ],
    "text_only": [
        "what's the weather like today?",
        "tell me a joke",
        "how do I cook pasta?",
        "what's the capital of France?",
        "explain quantum computing",
        "write a haiku about spring",
        "what's 2+2?",
        "tell me about the history of computers",
        "how do I tie a tie?",
        "what are the benefits of exercise?"
    ],
    "ambiguous_image": [
        "show me",
        "can you do that again?",
        "make it better",
        "fix it",
        "change it",
        "show me something cool",
        "do something with this",
        "make one",
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
        # Add truncated document context
        context.append({
            "role": "user",
            "content": "Here's a document to review\n[Document content truncated for classification]"
        })
        context.append({
            "role": "assistant",
            "content": "I've reviewed the document. It contains financial data and order information."
        })
    else:
        # Regular conversation
        context.append({
            "role": "user",
            "content": "Hello, how are you?"
        })
        context.append({
            "role": "assistant",
            "content": "Hello! I'm doing well, thank you. How can I help you today?"
        })
    
    return context

def test_intent_classification(client: OpenAIClient, query: str, expected_intent: str, 
                              context_type: str = "normal") -> Tuple[str, bool, float]:
    """Test a single intent classification"""
    
    # Create appropriate context
    if context_type == "with_image":
        context = create_conversation_context(has_recent_image=True)
    elif context_type == "with_document":
        context = create_conversation_context(has_document=True)
    else:
        context = create_conversation_context()
    
    # For edit_image and ambiguous cases, having a recent image affects classification
    has_attached = False
    if expected_intent == "vision":
        # Simulate having an attached image/document
        has_attached = True
        query = query + "\n[Note: User has attached images with this message]"
    
    start_time = time.time()
    try:
        # Call the actual classify_intent method
        result = client.classify_intent(
            messages=context,
            last_user_message=query,
            has_attached_images=has_attached
        )
        elapsed = time.time() - start_time
        
        # Check if it matches expected
        is_correct = result == expected_intent
        
        return result, is_correct, elapsed
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"Error classifying '{query}': {e}")
        return "error", False, elapsed

def run_comprehensive_test():
    """Run comprehensive intent classification tests"""
    
    print("="*80)
    print("INTENT CLASSIFICATION COMPREHENSIVE TEST")
    print("="*80)
    print(f"Using model: {config.utility_model}")
    print(f"API Key: {config.openai_api_key[:20]}...")
    print("="*80)
    
    # Initialize client
    client = OpenAIClient()
    
    # Store results
    results = defaultdict(list)
    detailed_results = []
    
    # Test each intent type
    for expected_intent, queries in TEST_CASES.items():
        print(f"\nTesting {expected_intent} intent...")
        print("-" * 40)
        
        for i, query in enumerate(queries, 1):
            # Determine context type based on intent
            if expected_intent == "edit_image" or expected_intent == "ambiguous_image":
                # These often need image context
                context_type = "with_image" if i % 2 == 0 else "normal"
            elif expected_intent == "vision":
                # Vision always needs attachment or document context
                context_type = "with_document"
            else:
                context_type = "normal"
            
            # Test the query
            result, is_correct, elapsed = test_intent_classification(
                client, query, expected_intent, context_type
            )
            
            # Store results
            results[expected_intent].append(is_correct)
            detailed_results.append({
                "Expected": expected_intent,
                "Query": query[:50] + "..." if len(query) > 50 else query,
                "Result": result,
                "Correct": "✓" if is_correct else "✗",
                "Time": f"{elapsed:.2f}s"
            })
            
            # Print progress
            status = "✓" if is_correct else f"✗ (got: {result})"
            print(f"  {i:2d}. {status} {query[:60]}... ({elapsed:.2f}s)")
            
            # Small delay to avoid rate limiting
            time.sleep(0.5)
    
    # Print summary results
    print("\n" + "="*80)
    print("SUMMARY RESULTS")
    print("="*80)
    
    # Calculate statistics for each intent
    summary_data = []
    total_correct = 0
    total_tests = 0
    
    for intent in TEST_CASES.keys():
        correct = sum(results[intent])
        total = len(results[intent])
        percentage = (correct / total * 100) if total > 0 else 0
        
        summary_data.append({
            "Intent Type": intent,
            "Correct": correct,
            "Total": total,
            "Accuracy": f"{percentage:.1f}%",
            "Failures": total - correct
        })
        
        total_correct += correct
        total_tests += total
    
    # Add overall summary
    overall_accuracy = (total_correct / total_tests * 100) if total_tests > 0 else 0
    summary_data.append({
        "Intent Type": "OVERALL",
        "Correct": total_correct,
        "Total": total_tests,
        "Accuracy": f"{overall_accuracy:.1f}%",
        "Failures": total_tests - total_correct
    })
    
    # Print summary table
    print(tabulate(summary_data, headers="keys", tablefmt="grid"))
    
    # Print failed cases for debugging
    print("\n" + "="*80)
    print("FAILED CLASSIFICATIONS (for debugging)")
    print("="*80)
    
    failed_cases = [r for r in detailed_results if r["Correct"] == "✗"]
    if failed_cases:
        print(tabulate(failed_cases, headers="keys", tablefmt="grid"))
    else:
        print("No failures! All classifications were correct.")
    
    # Print timing statistics
    print("\n" + "="*80)
    print("PERFORMANCE STATISTICS")
    print("="*80)
    
    all_times = [float(r["Time"][:-1]) for r in detailed_results]
    avg_time = sum(all_times) / len(all_times)
    min_time = min(all_times)
    max_time = max(all_times)
    
    print(f"Average classification time: {avg_time:.2f}s")
    print(f"Minimum classification time: {min_time:.2f}s")
    print(f"Maximum classification time: {max_time:.2f}s")
    print(f"Total test duration: {sum(all_times):.2f}s")
    
    # Return overall success
    return overall_accuracy >= 80  # Consider test passed if 80% or more accurate

if __name__ == "__main__":
    try:
        success = run_comprehensive_test()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)