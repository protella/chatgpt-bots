"""Unit tests for slackbot.py - Slack bot launcher module (async wrapper)"""

import pytest
from unittest.mock import Mock, patch, AsyncMock


class TestSlackBotLauncher:
    """Test the async Slack bot launcher script"""

    @patch('slackbot.ChatBotV2')
    @pytest.mark.asyncio
    async def test_main_creates_slack_bot(self, mock_chatbot_class):
        """Test that main() creates a ChatBotV2 with slack platform and runs it"""
        from slackbot import main

        mock_instance = Mock()
        mock_instance.run = AsyncMock()
        mock_chatbot_class.return_value = mock_instance

        await main()

        mock_chatbot_class.assert_called_once_with(platform="slack")
        mock_instance.run.assert_awaited_once()

    @patch('slackbot.ChatBotV2')
    @pytest.mark.asyncio
    async def test_main_propagates_run_exception(self, mock_chatbot_class):
        """Test that main() propagates exceptions from bot.run()"""
        from slackbot import main

        mock_instance = Mock()
        mock_instance.run = AsyncMock(side_effect=Exception("Test error"))
        mock_chatbot_class.return_value = mock_instance

        with pytest.raises(Exception, match="Test error"):
            await main()

        mock_chatbot_class.assert_called_once_with(platform="slack")

    @patch('slackbot.ChatBotV2')
    @pytest.mark.asyncio
    async def test_keyboard_interrupt_handling(self, mock_chatbot_class):
        """Test KeyboardInterrupt propagates from run()"""
        from slackbot import main

        mock_instance = Mock()
        mock_instance.run = AsyncMock(side_effect=KeyboardInterrupt())
        mock_chatbot_class.return_value = mock_instance

        with pytest.raises(KeyboardInterrupt):
            await main()

        mock_chatbot_class.assert_called_once_with(platform="slack")

    def test_module_imports(self):
        """Test that slackbot module imports correctly"""
        import slackbot

        assert hasattr(slackbot, 'main')
        assert callable(slackbot.main)
        assert hasattr(slackbot, 'ChatBotV2')

    @pytest.mark.critical
    @pytest.mark.asyncio
    async def test_critical_slack_platform_selection(self):
        """Critical test that slack platform is correctly selected"""
        with patch('slackbot.ChatBotV2') as mock_chatbot_class:
            from slackbot import main

            mock_instance = Mock()
            mock_instance.run = AsyncMock()
            mock_chatbot_class.return_value = mock_instance

            await main()

            args, kwargs = mock_chatbot_class.call_args
            assert kwargs.get('platform') == 'slack'
            mock_instance.run.assert_awaited_once()


class TestSlackBotIntegration:
    """Integration tests for slackbot launcher"""

    @pytest.mark.smoke
    def test_smoke_import_chain(self):
        """Smoke test for import chain"""
        from main import ChatBotV2

        assert isinstance(ChatBotV2, type)
        assert hasattr(ChatBotV2, 'run')
        assert hasattr(ChatBotV2, '__init__')
