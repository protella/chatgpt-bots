#!/bin/bash
# Run all tests and show summary

echo "Running Unit Tests..."
python3 -m pytest tests/unit/ -q --tb=no 2>&1 | tail -3
UNIT_EXIT=$?

echo ""
echo "Running Integration Tests - Message Flow..."
python3 -m pytest tests/integration/test_message_flow.py -q --tb=no 2>&1 | tail -3
FLOW_EXIT=$?

echo ""
echo "Running Integration Tests - OpenAI..."
python3 -m pytest tests/integration/test_openai_integration.py -q --tb=no 2>&1 | tail -3
OPENAI_EXIT=$?

echo ""
echo "Running Integration Tests - Streaming..."
python3 -m pytest tests/integration/test_streaming_integration.py -q --tb=no 2>&1 | tail -3
STREAM_EXIT=$?

echo ""
echo "================================"
echo "Test Suite Summary:"
echo "================================"
echo "Unit Tests: $([ $UNIT_EXIT -eq 0 ] && echo 'PASSED ✓' || echo 'FAILED ✗')"
echo "Integration - Message Flow: $([ $FLOW_EXIT -eq 0 ] && echo 'PASSED ✓' || echo 'FAILED ✗')"
echo "Integration - OpenAI: $([ $OPENAI_EXIT -eq 0 ] && echo 'PASSED ✓' || echo 'FAILED ✗')"
echo "Integration - Streaming: $([ $STREAM_EXIT -eq 0 ] && echo 'PASSED ✓' || echo 'FAILED ✗')"
echo "================================"

# Exit with failure if any test suite failed
if [ $UNIT_EXIT -ne 0 ] || [ $FLOW_EXIT -ne 0 ] || [ $OPENAI_EXIT -ne 0 ] || [ $STREAM_EXIT -ne 0 ]; then
    echo "OVERALL: FAILED ✗"
    exit 1
else
    echo "OVERALL: PASSED ✓"
    echo ""
    echo "All tests pass successfully!"
    exit 0
fi