"""Unit tests for slackbot.py - Slack bot launcher module"""

import pytest
import sys
from unittest.mock import Mock, patch, MagicMock


class TestSlackBotLauncher:
    """Test the Slack bot launcher script"""
    
    @patch('slackbot.ChatBotV2')
    def test_main_creates_slack_bot(self, mock_chatbot_class):
        """Test that main() creates a ChatBotV2 with slack platform"""
        from slackbot import main
        
        # Setup mock
        mock_instance = Mock()
        mock_chatbot_class.return_value = mock_instance
        
        # Execute
        main()
        
        # Verify
        mock_chatbot_class.assert_called_once_with(platform="slack")
        mock_instance.run.assert_called_once()
    
    @patch('slackbot.ChatBotV2')
    def test_main_handles_run_exception(self, mock_chatbot_class):
        """Test that main() handles exceptions from bot.run()"""
        from slackbot import main
        
        # Setup mock to raise exception
        mock_instance = Mock()
        mock_instance.run.side_effect = Exception("Test error")
        mock_chatbot_class.return_value = mock_instance
        
        # Execute - should not crash
        with pytest.raises(Exception, match="Test error"):
            main()
        
        # Verify bot was created
        mock_chatbot_class.assert_called_once_with(platform="slack")
    
    @patch('slackbot.ChatBotV2')
    def test_script_execution(self, mock_chatbot_class):
        """Test script execution via __main__"""
        # Setup mock bot
        mock_instance = Mock()
        mock_chatbot_class.return_value = mock_instance
        
        # Import the module
        import slackbot
        
        # Call main directly to test it
        slackbot.main()
        
        # Verify bot was created and run
        mock_chatbot_class.assert_called_with(platform="slack")
        mock_instance.run.assert_called_once()
    
    def test_module_imports(self):
        """Test that slackbot module imports correctly"""
        # Should be able to import without errors
        import slackbot
        
        # Should have main function
        assert hasattr(slackbot, 'main')
        assert callable(slackbot.main)
        
        # Should have ChatBotV2 imported
        assert hasattr(slackbot, 'ChatBotV2')
    
    @patch('slackbot.ChatBotV2')
    def test_bot_initialization_flow(self, mock_chatbot_class):
        """Test the complete initialization flow"""
        from slackbot import main
        
        # Create a more detailed mock
        mock_bot = MagicMock()
        mock_chatbot_class.return_value = mock_bot
        
        # Set up run method to track calls
        run_called = []
        def track_run():
            run_called.append(True)
        mock_bot.run = track_run
        
        # Execute
        main()
        
        # Verify initialization sequence
        assert mock_chatbot_class.called
        assert mock_chatbot_class.call_args[1]['platform'] == 'slack'
        assert len(run_called) == 1
    
    @patch('slackbot.ChatBotV2')
    def test_keyboard_interrupt_handling(self, mock_chatbot_class):
        """Test handling of keyboard interrupt"""
        from slackbot import main
        
        # Setup mock to raise KeyboardInterrupt
        mock_instance = Mock()
        mock_instance.run.side_effect = KeyboardInterrupt()
        mock_chatbot_class.return_value = mock_instance
        
        # Execute - should handle gracefully
        with pytest.raises(KeyboardInterrupt):
            main()
        
        # Bot should still be created
        mock_chatbot_class.assert_called_once_with(platform="slack")
    
    @pytest.mark.critical
    def test_critical_slack_platform_selection(self):
        """Critical test that slack platform is correctly selected"""
        with patch('slackbot.ChatBotV2') as mock_chatbot_class:
            from slackbot import main
            
            mock_instance = Mock()
            mock_chatbot_class.return_value = mock_instance
            
            # Execute
            main()
            
            # Must use slack platform
            args, kwargs = mock_chatbot_class.call_args
            assert kwargs.get('platform') == 'slack'
            assert mock_instance.run.called


class TestSlackBotIntegration:
    """Integration tests for slackbot launcher"""
    
    @pytest.mark.integration
    @patch('slackbot.ChatBotV2')
    def test_integration_with_main_module(self, mock_chatbot_class):
        """Test integration between slackbot and main module"""
        mock_bot = Mock()
        mock_chatbot_class.return_value = mock_bot
        
        from slackbot import main
        
        # Execute
        main()
        
        # Should have created a bot with slack platform
        mock_chatbot_class.assert_called_once_with(platform="slack")
        # Should have called run on the bot instance
        mock_bot.run.assert_called_once()
    
    @pytest.mark.smoke
    def test_smoke_import_chain(self):
        """Smoke test for import chain"""
        # Should be able to import all the way down
        import slackbot
        from main import ChatBotV2
        
        # Verify types
        assert isinstance(ChatBotV2, type)
        assert hasattr(ChatBotV2, 'run')
        assert hasattr(ChatBotV2, '__init__')