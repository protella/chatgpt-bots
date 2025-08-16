# Testing Strategy

## Test Types

### Unit Tests (`tests/unit/`)
Tests individual components in isolation using mocks.

**Coverage Focus**: Individual functions and classes
- ✅ Fast execution (< 1 second per test)
- ✅ Easy to debug failures
- ✅ High test isolation
- ❌ Doesn't test component interactions
- ❌ May miss integration issues

**Example Components**:
- `config.py` - Configuration loading and validation
- `markdown_converter.py` - Text format conversion
- `thread_manager.py` - State management logic

### Integration Tests (`tests/integration/`)
Tests multiple components working together.

**Coverage Focus**: Component interactions and workflows
- ✅ Tests realistic scenarios
- ✅ Catches integration bugs
- ✅ Better coverage per test
- ❌ Slower execution (1-10 seconds per test)
- ❌ Harder to debug failures

**Example Scenarios**:
- Slack message → Process → OpenAI → Response
- Database persistence and recovery
- Multi-turn conversations with context
- Concurrent thread handling

## Coverage Measurement

**Both test types contribute to coverage metrics!**

```bash
# Run only unit tests with coverage
make test-unit

# Run only integration tests with coverage  
make test-integration

# Run all tests (unit + integration)
make test

# Coverage includes BOTH:
# - Lines hit by unit tests
# - Lines hit by integration tests
```

## Test Pyramid for This Project

```
        /\
       /  \        E2E Tests (Optional)
      /    \       - Full bot running
     /------\      - Real Slack/Discord
    /        \     
   /----------\    Integration Tests (20-30%)
  /            \   - Component interactions
 /              \  - Database + Logic
/--------------  \ - Message flow pipelines
                   
Unit Tests (70-80%)
- Individual functions
- Class methods
- Utility functions
```

## Writing Effective Tests

### Unit Test Best Practices
```python
def test_specific_behavior(self):
    # Arrange - Setup test data
    config = BotConfig()
    
    # Act - Execute the function
    result = config.get_thread_config()
    
    # Assert - Verify the result
    assert result["model"] == "gpt-5"
```

### Integration Test Best Practices
```python
@pytest.mark.integration
async def test_complete_workflow(self):
    # Setup multiple components
    thread_mgr = ThreadStateManager(db)
    processor = MessageProcessor(openai, thread_mgr)
    
    # Execute workflow
    await processor.handle_message(test_message)
    
    # Verify system state changes
    assert thread_mgr.get_stats()["active_threads"] == 1
    assert db.get_messages(thread_id)  # Check persistence
```

## Coverage Goals

### Current Coverage
- **Overall**: 13%
- **Unit Tests**: Cover isolated logic
- **Integration Tests**: Not yet implemented

### Target Coverage
- **Overall**: 80%+
- **Critical Paths**: 95%+ (message processing, OpenAI calls)
- **Utilities**: 70%+ (logging, helpers)
- **Entry Points**: 50%+ (main.py, CLI arguments)

## Test Execution Speed

```bash
# Fast feedback loop (unit tests only)
make test-fast        # ~1 second

# Standard test run
make test            # ~5 seconds

# Full validation
make test-verbose    # ~5 seconds with details

# Continuous testing during development
make test-watch      # Auto-rerun on file changes
```

## Mocking Strategy

### Unit Tests
- Mock ALL external dependencies
- Use `unittest.mock` for Python objects
- Use `pytest-mock` for fixtures

### Integration Tests  
- Mock external services (Slack API, OpenAI API)
- Use real internal components
- Consider test databases for persistence tests

### End-to-End Tests (Optional)
- Use test Slack workspace
- Use OpenAI test API keys
- Separate test environment

## Running Specific Test Types

```bash
# Unit tests only
pytest tests/unit -v

# Integration tests only  
pytest tests/integration -v

# Tests for specific module
pytest tests/unit/test_config.py -v

# Tests matching pattern
pytest -k "test_thread" -v

# Run slow tests too
pytest -m "slow" -v

# Skip slow tests
pytest -m "not slow" -v
```