from __future__ import annotations

import re


class SlackFormattingMixin:
    def _clean_mentions(self, text: str) -> str:
        """Remove Slack user mentions from text"""
        return re.sub(r'<@[A-Z0-9]+>', '', text).strip()

    def format_text(self, text: str) -> str:
        """Format text for Slack using mrkdwn"""
        return self.markdown_converter.convert(text)

    def format_error_message(self, error: str) -> str:
        """Format error messages for Slack with emojis and code blocks"""
        import re
        
        # Extract error code if present
        error_code_match = re.search(r'Error code: (\d+)', error)
        error_code = error_code_match.group(1) if error_code_match else "Unknown"
        
        # Try to extract the actual error message
        if "{'error':" in error:
            # Parse OpenAI API error format
            try:
                import json
                error_dict_str = error[error.find("{'error':"):].replace("'", '"')
                error_dict = json.loads(error_dict_str)
                error_message = error_dict.get('error', {}).get('message', error)
                error_type = error_dict.get('error', {}).get('type', 'unknown_error')
            except Exception:
                # Fallback to simpler extraction
                if "'message':" in error:
                    msg_start = error.find("'message': '") + len("'message': '")
                    msg_end = error.find("',", msg_start)
                    if msg_end > msg_start:
                        error_message = error[msg_start:msg_end]
                    else:
                        error_message = error
                else:
                    error_message = error
                error_type = "api_error"
        else:
            error_message = error
            error_type = "general_error"
        
        # Format the error message for Slack
        formatted = ":warning: *Oops! Something went wrong*\n\n"
        formatted += f"*Error Code:* `{error_code}`\n"
        formatted += f"*Type:* `{error_type}`\n\n"
        formatted += f"*Details:*\n```{error_message}```\n\n"
        formatted += ":bulb: *What you can do:*\n"
        
        # Add helpful suggestions based on error type
        if "rate_limit" in error_type.lower():
            formatted += "• Wait a moment and try again\n"
            formatted += "• The API rate limit has been reached"
        elif "invalid_request" in error_type.lower():
            formatted += "• Try rephrasing your request\n"
            formatted += "• The request format may be invalid"
        elif "context_length" in error_message.lower():
            formatted += "• Start a new thread\n"
            formatted += "• The conversation has become too long"
        else:
            formatted += "• Try again in a moment\n"
            formatted += "• If the problem persists, contact support"
        
        return formatted
