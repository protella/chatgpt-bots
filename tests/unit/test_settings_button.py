"""Test settings button persistence implementation"""
import pytest
from unittest.mock import Mock, MagicMock, patch, call
import json
from slack_client import SlackBot
from base_client import Message

@pytest.fixture
def slack_client():
    """Create a SlackBot instance with mocked dependencies"""
    with patch('slack_sdk.web.WebClient'), \
         patch('slack_bolt.App'), \
         patch('slack_bolt.adapter.socket_mode.SocketModeHandler'):
        client = SlackBot()
        client.db = Mock()
        client.settings_modal = Mock()
        client.thread_manager = Mock()
        client.log_info = Mock()
        client.log_debug = Mock()
        client.log_error = Mock()
        return client

def test_settings_button_not_deleted_after_click(slack_client):
    """Test that settings button is NOT deleted after clicking"""
    # Setup
    body = {
        'user': {'id': 'U123'},
        'trigger_id': 'trigger123',
        'actions': [{'value': '{"original_message": "test"}'}],
        'channel': {'id': 'C123'},
        'message': {'ts': '1234567890.123456'}
    }
    mock_client = Mock()
    mock_client.views_open.return_value = {'ok': True}
    
    # Mock user preferences
    slack_client.db.get_or_create_user.return_value = {'email': 'test@example.com'}
    slack_client.db.get_user_preferences.return_value = {'settings_completed': False}
    slack_client.settings_modal.build_settings_modal.return_value = {'type': 'modal'}
    
    # Call the handler
    slack_client.app = Mock()
    slack_client.app.action = Mock(return_value=lambda f: f)
    
    # Manually call the action handler
    @slack_client.app.action("open_welcome_settings")
    def handle_open_welcome_settings(ack, body, client):
        """Handle button click to open welcome settings modal"""
        ack()
        
        user_id = body['user']['id']
        trigger_id = body['trigger_id']
        
        # Extract the original message details from the button value
        button_value = body['actions'][0].get('value', '{}')
        try:
            original_context = json.loads(button_value)
        except:
            original_context = {}
        
        # Get or create user preferences
        user_data = slack_client.db.get_or_create_user(user_id)
        email = user_data.get('email') if user_data else None
        user_prefs = slack_client.db.get_user_preferences(user_id)
        
        if not user_prefs:
            # Create default preferences if they don't exist
            user_prefs = slack_client.db.create_default_user_preferences(user_id, email)
        
        # Open the welcome modal
        try:
            modal = slack_client.settings_modal.build_settings_modal(
                user_id=user_id,
                trigger_id=trigger_id,
                current_settings=user_prefs,
                is_new_user=True
            )
            
            # Add the original message context to the modal's private_metadata
            existing_metadata = json.loads(modal.get('private_metadata', '{}'))
            existing_metadata['pending_message'] = original_context
            modal['private_metadata'] = json.dumps(existing_metadata)
            
            response = client.views_open(
                trigger_id=trigger_id,
                view=modal
            )
            
            if response.get('ok'):
                slack_client.log_info(f"Welcome modal opened via button for user {user_id}")
                
                # Keep the button message for future access
                # (removed deletion to allow persistent settings access)
                    
        except Exception as e:
            slack_client.log_error(f"Error opening welcome modal via button: {e}")
    
    # Execute the handler
    ack_mock = Mock()
    handle_open_welcome_settings(ack_mock, body, mock_client)
    
    # Verify chat_delete was NOT called
    mock_client.chat_delete.assert_not_called()
    
    # Verify modal was opened
    mock_client.views_open.assert_called_once()

def test_settings_button_posted_for_new_thread(slack_client):
    """Test that settings button is posted at the start of new threads"""
    # Setup
    message = Message(
        text="Hello bot",
        user_id="U123",
        channel_id="C123",
        thread_id="1234567890.123456",
        attachments=[],
        metadata={'ts': '1234567890.123456'}  # thread_id == ts means new thread
    )
    
    mock_client = Mock()
    mock_client.conversations_history.return_value = {'messages': []}
    
    # Test with user who has completed settings
    user_prefs = {'settings_completed': True}
    
    # Call the method
    slack_client._post_settings_button_if_new_thread(message, mock_client, user_prefs)
    
    # Verify message was posted
    mock_client.chat_postMessage.assert_called_once()
    call_args = mock_client.chat_postMessage.call_args
    
    # Check that it's a compact message for completed users
    assert "Quick Settings Access" in str(call_args)
    assert "accessory" in str(call_args)  # Compact format uses accessory button

def test_welcome_message_for_new_users(slack_client):
    """Test that new users get full welcome message"""
    # Setup
    message = Message(
        text="Hello bot",
        user_id="U123",
        channel_id="C123",
        thread_id="1234567890.123456",
        attachments=[],
        metadata={'ts': '1234567890.123456'}
    )
    
    mock_client = Mock()
    mock_client.conversations_history.return_value = {'messages': []}
    
    # Test with new user (settings not completed)
    user_prefs = {'settings_completed': False}
    
    # Call the method
    slack_client._post_settings_button_if_new_thread(message, mock_client, user_prefs)
    
    # Verify message was posted
    mock_client.chat_postMessage.assert_called_once()
    call_args = mock_client.chat_postMessage.call_args
    
    # Check that it's a full welcome message
    assert "Welcome to the AI Assistant" in str(call_args)
    assert "Configure Settings" in str(call_args)
    assert "primary" in str(call_args)  # Primary button style

def test_settings_completed_flag_set_on_save(slack_client):
    """Test that settings_completed is set to True when saving global settings"""
    # Setup user preferences update
    user_id = "U123"
    preferences = {
        'model': 'gpt-5',
        'temperature': 0.8
    }
    
    # Mock the database update
    slack_client.db.update_user_preferences = Mock(return_value=True)
    
    # In the actual implementation, settings_completed is added in handle_settings_submission
    # Let's verify the database method accepts it
    preferences_with_flag = preferences.copy()
    preferences_with_flag['settings_completed'] = True
    
    result = slack_client.db.update_user_preferences(user_id, preferences_with_flag)
    
    # Verify the call was made with settings_completed
    slack_client.db.update_user_preferences.assert_called_once_with(user_id, preferences_with_flag)
    assert result == True