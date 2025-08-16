# Test Request Templates for Reliability

## 1. 🔒 **Regression Tests** (Prevent Breaking Changes)
*"Write tests that ensure existing functionality doesn't break when we modify the code"*

### Good Request:
"Write regression tests for the message processing pipeline that verify:
- Messages still get sent to OpenAI with correct format
- Thread context is maintained between messages
- Config overrides are still applied
These should catch if someone accidentally breaks the flow"

### What You Get:
```python
def test_message_pipeline_regression():
    """Ensure core message flow doesn't break"""
    # Tests the critical path that MUST keep working
```

## 2. 🔗 **Integration Tests** (Verify Components Work Together)
*"Write tests that verify these components still work together correctly"*

### Good Request:
"Write integration tests that verify:
- ThreadStateManager correctly stores messages that MessageProcessor sends
- Config changes in BotConfig are actually used by OpenAIClient
- When SlackClient receives a message, it ends up in the database"

### What You Get:
```python
def test_component_integration():
    """Verify components communicate correctly"""
    # Tests the handoffs between modules
```

## 3. 📝 **Contract Tests** (Ensure APIs/Interfaces Don't Change)
*"Write tests that verify the interface between X and Y hasn't changed"*

### Good Request:
"Write contract tests that ensure:
- OpenAIClient.create_response() still accepts the parameters we use
- SlackClient.post_message() return format hasn't changed
- Database schema matches what ThreadStateManager expects"

### What You Get:
```python
def test_api_contract():
    """Ensure external interfaces haven't changed"""
    # Catches when APIs or data formats change
```

## 4. 🎭 **Scenario Tests** (Real User Workflows)
*"Write tests for this user scenario that should always work"*

### Good Request:
"Write scenario tests for:
- User starts conversation → bot responds → user continues → context maintained
- User requests image → image generated → URL saved → user can reference it later
- Bot timeout → user sends message → bot recovers gracefully"

### What You Get:
```python
def test_user_scenario_conversation_flow():
    """Test complete user workflow works end-to-end"""
    # Ensures user experience doesn't break
```

## 5. 🛡️ **Smoke Tests** (Basic Functionality)
*"Write smoke tests that verify the system basically works"*

### Good Request:
"Write smoke tests that verify:
- Bot can start up with current config
- Can connect to Slack
- Can process at least one message
- Database is accessible
Run these before each coding session"

### What You Get:
```python
def test_smoke_basic_functionality():
    """Quick tests to verify system is operational"""
    # Run these first to catch major breaks
```

## 6. 🔄 **State Tests** (Data Persistence)
*"Write tests that verify state/data persists correctly"*

### Good Request:
"Write state tests that verify:
- Thread messages persist across ThreadStateManager restarts
- Config overrides survive bot restart
- Image metadata is retrievable after database reconnection"

### What You Get:
```python
def test_state_persistence():
    """Ensure data survives restarts"""
    # Catches data loss issues
```

## 7. ⚠️ **Critical Path Tests** (Must Never Break)
*"Write tests for the critical paths that absolutely must work"*

### Good Request:
"Write critical path tests for:
- CRITICAL: User message MUST reach OpenAI
- CRITICAL: OpenAI response MUST return to user
- CRITICAL: Errors MUST NOT crash the bot
Mark these as @pytest.mark.critical"

### What You Get:
```python
@pytest.mark.critical
def test_critical_message_flow():
    """THIS MUST NEVER FAIL"""
    # The absolute minimum functionality
```

## 8. 🔍 **Diagnostic Tests** (Help Debug Issues)
*"Write tests that help diagnose what broke"*

### Good Request:
"Write diagnostic tests that:
- Log the exact state when something fails
- Capture intermediate values in the pipeline
- Show exactly where the flow breaks
- Include helpful error messages"

### What You Get:
```python
def test_with_diagnostics():
    """Provides detailed info when things break"""
    # Makes debugging easier between sessions
```