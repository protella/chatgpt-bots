"""
MCP Tool Handler Mixin
Handles invocation of MCP tools when intent is "none" or text-only
"""
import json
from typing import Optional, Dict, Any
from bs4 import BeautifulSoup
from base_client import Response
from config import config
from logger import LoggerMixin


class MCPHandlerMixin(LoggerMixin):
    """Handles MCP tool invocation and responses"""

    async def _handle_mcp_tool(
        self,
        user_content: str,
        thread_state,
        client,
        message,
        thinking_id: Optional[str] = None
    ) -> Optional[Response]:
        """
        Check if message should invoke an MCP tool and handle it

        Args:
            user_content: The user's message content
            thread_state: Current thread state
            client: Platform client for updates
            message: Original message object
            thinking_id: ID of thinking indicator to update

        Returns:
            Response if MCP tool was invoked, None otherwise
        """
        if not self.mcp_manager or not self.mcp_manager.initialized:
            return None

        try:
            # Use LLM to select appropriate tool
            tool_selection = await self.mcp_manager.select_tool_for_message(
                user_content,
                self.openai_client
            )

            if not tool_selection:
                # No MCP tool matches, return None to fall through to text response
                return None

            tool_info = tool_selection.get("tool", {})
            parameters = tool_selection.get("parameters", {})

            # Ensure tool_info is a dict
            if not isinstance(tool_info, dict):
                tool_info = {"name": str(tool_info)} if tool_info else {}

            # Update thinking message with tool being called only if we have a valid title
            if thinking_id and tool_info.get('title'):
                self._update_status(
                    client,
                    message.channel_id,
                    thinking_id,
                    f"Searching {tool_info['title']}...",
                    emoji=config.thinking_emoji
                )

            self.log_info(f"Invoking MCP tool: {tool_info.get('name', 'unknown')} with parameters: {parameters}")

            # Call the MCP tool
            result = await self.mcp_manager.call_tool(
                tool_info.get("name", "unknown"),
                parameters,
                server_name=tool_info.get("server")
            )

            # Check if the tool returned an error
            if isinstance(result, dict) and result.get("status") == "error":
                # Tool failed - let the LLM handle the error gracefully
                tool_title = tool_info.get('title', tool_info.get('name', 'external'))
                error_context = f"The {tool_title} tool encountered an issue: {result.get('error', 'Unknown error')}"

                # Add the user's question to thread history
                formatted_user = self._format_user_content_with_username(user_content, message)
                thread_state.messages.append({"role": "user", "content": formatted_user})

                # Add tool error as system context
                thread_state.messages.append({
                    "role": "assistant",
                    "content": f"I attempted to search for that information but encountered a technical issue. Let me help you based on general knowledge instead."
                })

                # Cache messages if database available
                if self.db:
                    thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
                    message_ts = message.metadata.get("ts") if message.metadata else None
                    await self.db.cache_message_async(thread_key, "user", formatted_user, message_ts=message_ts)

                # Pass through text handler to generate a proper response
                # The LLM will see the error context and provide a helpful fallback response
                return await self._handle_text_response(
                    f"{user_content}\n\n[Note: {error_context}]",
                    thread_state,
                    client,
                    message,
                    thinking_id,
                    retry_count=0
                )

            # Tool succeeded - format and return the response directly
            self.log_debug(f"Tool result type: {type(result)}, tool_info: {tool_info}")

            # Extract the actual response content
            if hasattr(result, 'content'):
                # FastMCP CallToolResult format
                if isinstance(result.content, list) and len(result.content) > 0:
                    # Get the first content item (usually TextContent)
                    content_item = result.content[0]
                    if hasattr(content_item, 'text'):
                        response_content = content_item.text
                    else:
                        response_content = str(content_item)
                else:
                    response_content = str(result.content)
            elif isinstance(result, dict):
                # Direct dict response
                if "content" in result:
                    response_content = result["content"]
                elif "data" in result:
                    response_content = result["data"]
                else:
                    response_content = json.dumps(result)
            else:
                response_content = str(result)

            # Parse the response content if it's JSON
            try:
                if isinstance(response_content, str) and response_content.strip().startswith('{'):
                    parsed = json.loads(response_content)
                    if parsed.get("status") == "success" and "content" in parsed:
                        # Extract HTML content from successful response
                        html_content = parsed["content"]
                        # Convert HTML to platform-appropriate markdown
                        response_text = self._convert_html_to_markdown(html_content, client)
                    elif parsed.get("status") == "error":
                        # Handle error response gracefully without exposing technical details
                        error_msg = parsed.get("error", "")

                        # Get a friendly tool name
                        tool_display_name = tool_info.get('title') or tool_info.get('name', 'the requested service')
                        # Clean up technical names (e.g., "get_market_intelligence" -> "Market Intelligence")
                        if '_' in tool_display_name:
                            # Extract meaningful parts from snake_case names
                            parts = tool_display_name.split('_')
                            # Look for key words like "market", "intelligence", "report", etc.
                            meaningful_parts = [p for p in parts if len(p) > 3 and p.lower() not in ['get', 'fetch', 'query', 'call']]
                            if meaningful_parts:
                                tool_display_name = ' '.join(meaningful_parts).title()
                            else:
                                tool_display_name = 'the data service'

                        # Map technical errors to user-friendly messages
                        if "auth" in error_msg.lower() or "401" in str(parsed.get("status_code", "")):
                            response_text = f"⚠️ Unable to access {tool_display_name}. Please try again later or contact support if the issue persists."
                        elif "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                            response_text = f"⚠️ {tool_display_name} took too long to respond. Please try again with a more specific query."
                        elif "not found" in error_msg.lower() or "404" in str(parsed.get("status_code", "")):
                            response_text = f"⚠️ No data found in {tool_display_name} for your query. Please try rephrasing or searching for different terms."
                        elif "rate limit" in error_msg.lower() or "429" in str(parsed.get("status_code", "")):
                            response_text = f"⚠️ {tool_display_name} is currently busy. Please wait a moment and try again."
                        else:
                            # Generic error message
                            response_text = f"⚠️ Unable to retrieve information from {tool_display_name} at this time. Please try again later."

                        self.log_error(f"MCP tool error: {error_msg} (status: {parsed.get('status_code', 'unknown')})")
                    else:
                        # Other structured response
                        response_text = self._format_tool_result(tool_info, parsed)
                else:
                    # Plain text or HTML response
                    if '<' in response_content and '>' in response_content:
                        # Looks like HTML
                        response_text = self._convert_html_to_markdown(response_content, client)
                    else:
                        # Plain text
                        response_text = response_content
            except json.JSONDecodeError:
                # Not JSON, treat as plain text/HTML
                if '<' in response_content and '>' in response_content:
                    response_text = self._convert_html_to_markdown(response_content, client)
                else:
                    response_text = response_content

            # Add the user's original question to thread history
            formatted_user = self._format_user_content_with_username(user_content, message)
            thread_state.messages.append({"role": "user", "content": formatted_user})

            # Add the MCP response as the assistant's response
            thread_state.messages.append({"role": "assistant", "content": response_text})

            # Cache messages if database available
            if self.db:
                thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
                message_ts = message.metadata.get("ts") if message.metadata else None
                await self.db.cache_message_async(thread_key, "user", formatted_user, message_ts=message_ts)
                await self.db.cache_message_async(thread_key, "assistant", response_text)

            # Log token usage
            if hasattr(thread_state, "messages"):
                tokens = self.thread_manager._token_counter.count_thread_tokens(thread_state.messages)
                model = thread_state.current_model or config.gpt_model
                max_tokens = config.get_model_token_limit(model)
                self.log_debug(f"MESSAGE ADDED | Role: assistant | Total: {tokens}/{max_tokens}")

            return Response(
                type="text",  # Return as text, not mcp_tool
                content=response_text,
                metadata={
                    "tool": tool_info.get("name", "unknown"),
                    "server": tool_info.get("server"),
                    "parameters": parameters
                }
            )

        except Exception as e:
            self.log_error(f"Error invoking MCP tool: {e}")
            # Return error response
            error_message = f"❌ **Tool Error**\n\nFailed to execute tool: {str(e)}"
            return Response(
                type="error",
                content=error_message
            )

    def _format_tool_result_as_context(self, tool_info: Dict[str, Any], result: Any) -> str:
        """
        Format tool result as context for the LLM to process

        Args:
            tool_info: Tool metadata
            result: Tool execution result

        Returns:
            Formatted context string
        """
        # Check if result is already a string (from CallToolResult)
        if not isinstance(result, dict):
            return str(result)

        # For dict results, convert to readable format
        if "data" in result:
            data = result["data"]
            if isinstance(data, list):
                # Format list of items
                formatted = ""
                for i, item in enumerate(data, 1):
                    if isinstance(item, dict):
                        formatted += f"Item {i}:\n"
                        for key, value in item.items():
                            formatted += f"  - {key}: {value}\n"
                    else:
                        formatted += f"{i}. {item}\n"
                return formatted
            elif isinstance(data, dict):
                # Format dict data
                formatted = ""
                for key, value in data.items():
                    formatted += f"{key}: {value}\n"
                return formatted
            else:
                return str(data)

        # For other dict formats, just convert to readable format
        formatted = ""
        for key, value in result.items():
            if key not in ["status", "status_code"]:  # Skip status fields
                formatted += f"{key}: {value}\n"

        return formatted.strip() if formatted else str(result)

    def _convert_html_to_markdown(self, html_content: str, client) -> str:
        """
        Convert HTML response to platform-appropriate markdown

        Args:
            html_content: HTML string from MCP tool
            client: Platform client (for platform-specific formatting)

        Returns:
            Markdown-formatted string
        """
        if not html_content:
            return "No content returned"

        # Parse HTML
        soup = BeautifulSoup(html_content, 'html.parser')

        # Process bold/strong tags first (before other conversions)
        for tag in soup.find_all(['strong', 'b']):
            tag_text = tag.get_text()
            tag.replace_with(f"*{tag_text}*")

        # Process italic/em tags
        for tag in soup.find_all(['em', 'i']):
            tag_text = tag.get_text()
            tag.replace_with(f"_{tag_text}_")

        # Process citations
        citations_found = []
        for cite in soup.find_all('cite'):
            citation_id = cite.get('data-citation-id')
            citation_text = cite.get_text().strip().strip('[]')
            if citation_id:
                citations_found.append({
                    'id': citation_id,
                    'text': citation_text
                })
                # Replace with numbered reference
                cite.replace_with(f"[{len(citations_found)}]")

        # Convert lists
        for ul in soup.find_all('ul'):
            items = []
            for li in ul.find_all('li'):
                items.append(f"• {li.get_text().strip()}")
            ul.replace_with('\n'.join(items))

        for ol in soup.find_all('ol'):
            items = []
            for i, li in enumerate(ol.find_all('li'), 1):
                items.append(f"{i}. {li.get_text().strip()}")
            ol.replace_with('\n'.join(items))

        # Convert paragraphs
        for p in soup.find_all('p'):
            p.replace_with(f"\n{p.get_text().strip()}\n")

        # Get text and clean up
        text = soup.get_text()

        # Clean up excessive whitespace
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(line for line in lines if line)

        # Add citations section if found
        if citations_found:
            text += "\n\n📚 *Sources:*\n"
            for i, citation in enumerate(citations_found, 1):
                text += f"{i}. {citation['text']}\n"

        return text

    def _format_tool_result(self, tool_info: Dict[str, Any], result: Dict[str, Any]) -> str:
        """
        Format structured tool result for display

        Args:
            tool_info: Tool metadata
            result: Tool execution result

        Returns:
            Formatted text response
        """
        # Start with tool title
        formatted = f"**{tool_info['title']} Results**\n\n"

        # Format the result based on common patterns
        if "data" in result:
            # Tool returned data field
            data = result["data"]
            if isinstance(data, list):
                # List of items
                for item in data[:10]:  # Limit to first 10 items
                    if isinstance(item, dict):
                        # Format dict items
                        formatted += self._format_dict_item(item) + "\n"
                    else:
                        formatted += f"• {item}\n"
                if len(data) > 10:
                    formatted += f"\n_...and {len(data) - 10} more items_"
            elif isinstance(data, dict):
                # Single dict result
                formatted += self._format_dict_item(data)
            else:
                # Simple data
                formatted += str(data)
        elif "error" in result:
            # Tool returned an error
            formatted += f"⚠️ {result['error']}"
        else:
            # Generic result formatting
            for key, value in result.items():
                formatted += f"**{key.title()}**: {value}\n"

        return formatted.strip()

    def _format_dict_item(self, item: Dict[str, Any]) -> str:
        """Format a dictionary item for display"""
        # Common fields to prioritize
        priority_fields = ["title", "name", "description", "value", "summary"]

        parts = []
        # Add priority fields first
        for field in priority_fields:
            if field in item:
                parts.append(f"**{field.title()}**: {item[field]}")

        # Add remaining fields
        for key, value in item.items():
            if key not in priority_fields:
                parts.append(f"{key}: {value}")

        return " | ".join(parts[:3])  # Limit to 3 fields for brevity