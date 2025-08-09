
<!-- Page 1 -->
This API reference describes the RESTful, streaming, and realtime APIs you can use to interact with the
OpenAI platform. REST APIs are usable via HTTP in any environment that supports HTTP requests.
Language-specific SDKs are listed on the libraries page.
The OpenAI API uses API keys for authentication. Create, manage, and learn more about API keys in your
organization settings.
Remember that your API key is a secret! Do not share it with others or expose it in any client-side code
(browsers, apps). API keys should be securely loaded from an environment variable or key management
service on the server.
API keys should be provided via HTTP Bearer authentication.
If you belong to multiple organizations or access projects through a legacy user API key, pass a header to
### specify which organization and project to use for an API request:
## Introduction
Authentication
Authorization: Bearer OPENAI_API_KEY
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "OpenAI-Organization: YOUR_ORG_ID" \
  -H "OpenAI-Project: $PROJECT_ID"


<!-- Page 2 -->
Usage from these API requests counts as usage for the specified organization and project.Organization IDs
can be found on your organization settings page. Project IDs can be found on your general settings page
by selecting the specific project.
In addition to error codes returned from API responses, you can inspect HTTP response headers
containing the unique ID of a particular API request or information about rate limiting applied to your
requests. Below is an incomplete list of HTTP headers returned with API responses:
## API meta information
Rate limiting information
Debugging requests
openai-organization : The organization associated with the request
openai-processing-ms : Time taken processing your API request
openai-version : REST API version used for this request (currently 2020-10-01 )
x-request-id : Unique identifier for this API request (used in troubleshooting)
x-ratelimit-limit-requests
x-ratelimit-limit-tokens
x-ratelimit-remaining-requests
x-ratelimit-remaining-tokens
x-ratelimit-reset-requests
x-ratelimit-reset-tokens

<!-- Page 3 -->
## OpenAI recommends logging request IDs in production deployments for more efficient troubleshooting
with our support team, should the need arise. Our official SDKs provide a property on top-level response objects containing the value of the x-request-id  header.
## OpenAI is committed to providing stability to API users by avoiding breaking changes in major API versions
whenever reasonably possible. This includes:
Model prompting behavior between snapshots is subject to change. Model outputs are by their nature
variable, so expect changes in prompting and model behavior between snapshots. For example, if you
moved from gpt-4o-2024-05-13  to gpt-4o-2024-08-06 , the same system  or user  messages could
function differently between versions. The best way to ensure consistent prompting behavior and model
output is to use pinned model versions, and to implement evals for your applications.
### Backwards-compatible API changes:
## Backward compatibility
The REST API (currently v1 )
Our first-party SDKs (released SDKs adhere to semantic versioning)
Model families (like gpt-4o  or o4-mini )
Adding new resources (URLs) to the REST API and SDKs
## Adding new optional API parameters
Adding new properties to JSON response objects or event data
Changing the order of properties in a JSON response object
Changing the length or format of opaque strings, like resource identifiers and UUIDs
Adding new event types (in either streaming or the Realtime API)

<!-- Page 4 -->
See the changelog for a list of backwards-compatible changes and rare breaking changes.
OpenAI's most advanced interface for generating model responses. Supports text and image inputs, and
text outputs. Create stateful interactions with the model, using the output of previous responses as input.
Extend the model's capabilities with built-in tools for file search, web search, computer use, and more.
Allow the model access to external systems and data using function calling.
### Related guides:
POST https://api.openai.com/v1/responses
Creates a model response. Provide text or image inputs to generate text or
JSON outputs. Have the model call your own custom code or use built-in
## Responses
Quickstart
Text inputs and outputs
Image inputs
Structured Outputs
Function calling
Conversation state
Extend the models with tools
Create a model response
Text input
Image input
File input
Web search
Fil
Example request
python

<!-- Page 5 -->
tools like web search or file search to use your own data as input for the
model's response.
## Request body
Whether to run the model response in the background. Learn more.
background boolean or null
## Optional
Defaults to false
Specify additional output data to include in the model response. Currently supported
### values are:
include array or null
## Optional
code_interpreter_call.outputs : Includes the outputs of python code
execution in code interpreter tool call items.
computer_call_output.output.image_url : Include image urls from the
computer call output.
file_search_call.results : Include the search results of the file search tool
call.
message.input_image.image_url : Include image urls from the input
message.
message.output_text.logprobs : Include logprobs with assistant messages.
reasoning.encrypted_content : Includes an encrypted version of reasoning
tokens in reasoning item outputs. This enables reasoning items to be used in
multi-turn conversations when using the Responses API statelessly (like when the
store  parameter is set to false , or when an organization is enrolled in the
zero data retention program).
Text, image, or file inputs to the model, used to generate a response.
### Learn more:
input string or array
## Optional
Text inputs and outputs
Image inputs
from openai import OpenAI
client = OpenAI()
response = client.responses.create(
  model="gpt-4.1",
  input="Tell me a three sentence bedtime story 
)

## Response
 
{
  "id": "resp_67ccd2bed1ec8190b14f964abc0542670
  "object": "response",
  "created_at": 1741476542,
  "status": "completed",
  "error": null,
  "incomplete_details": null,
  "instructions": null,
  "max_output_tokens": null,
  "model": "gpt-4.1-2025-04-14",
  "output": [
    {
      "type": "message",
      "id": "msg_67ccd2bf17f0819081ff3bb2cf6508
      "status": "completed",
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "text": "In a peaceful grove beneath 
          "annotations": []
        }
      ]
    }
  ],
  "parallel_tool_calls": true,
  "previous_response_id": null,
  "reasoning": {


<!-- Page 6 -->
## Show possible types
File inputs
Conversation state
Function calling
A system (or developer) message inserted into the model's context.
When using along with previous_response_id , the instructions from a previous
response will not be carried over to the next response. This makes it simple to swap out
system (or developer) messages in new responses.
instructions string or null
## Optional
An upper bound for the number of tokens that can be generated for a response,
including visible output tokens and reasoning tokens.
max_output_tokens integer or null
## Optional
The maximum number of total calls to built-in tools that can be processed in a
response. This maximum number applies across all built-in tool calls, not per individual
tool. Any further attempts to call a tool by the model will be ignored.
max_tool_calls integer or null
## Optional
Set of 16 key-value pairs that can be attached to an object. This can be useful for
storing additional information about the object in a structured format, and querying for objects via API or the dashboard.
Keys are strings with a maximum length of 64 characters. Values are strings with a
maximum length of 512 characters.
metadata
map
## Optional
Model ID used to generate the response, like gpt-4o  or o3 . OpenAI offers a wide
range of models with different capabilities, performance characteristics, and price
points. Refer to the model guide to browse and compare available models.
model string
## Optional
parallel_tool_calls boolean or null
## Optional
Defaults to true
 
    "effort": null,
    "summary": null
  },
  "store": true,
  "temperature": 1.0,
  "text": {
    "format": {
      "type": "text"
    }
  },
  "tool_choice": "auto",
  "tools": [],
  "top_p": 1.0,
  "truncation": "disabled",
  "usage": {
    "input_tokens": 36,
    "input_tokens_details": {
      "cached_tokens": 0
    },
    "output_tokens": 87,
    "output_tokens_details": {
      "reasoning_tokens": 0
    },
    "total_tokens": 123
  },
  "user": null,
  "metadata": {}
}


<!-- Page 7 -->
Whether to allow the model to run tool calls in parallel.
The unique ID of the previous response to the model. Use this to create multi-turn
conversations. Learn more about conversation state.
previous_response_id string or null
## Optional
Reference to a prompt template and its variables. Learn more.
## Show properties
prompt object or null
Optional
Used by OpenAI to cache responses for similar requests to optimize your cache hit
rates. Replaces the user  field. Learn more.
prompt_cache_key string
## Optional
o-series models only
Configuration options for reasoning models.
## Show properties
reasoning object or null
Optional
A stable identifier used to help detect users of your application that may be violating
OpenAI's usage policies. The IDs should be a string that uniquely identifies each user.
We recommend hashing their username or email address, in order to avoid sending us
any identifying information. Learn more.
safety_identifier string
## Optional
Specifies the processing type used for serving the request.
service_tier string or null
## Optional
Defaults to auto
If set to 'auto', then the request will be processed with the service tier configured
in the Project settings. Unless otherwise configured, the Project will use 'default'.
If set to 'default', then the request will be processed with the standard pricing and
performance for the selected model.
If set to 'flex' or 'priority', then the request will be processed with the
corresponding service tier. Contact sales to learn more about Priority processing.

<!-- Page 8 -->
When the service_tier  parameter is set, the response body will include the
service_tier  value based on the processing mode actually used to serve the
request. This response value may be different from the value set in the parameter.
When not set, the default behavior is 'auto'.
Whether to store the generated model response for later retrieval via API.
store boolean or null
## Optional
Defaults to true
If set to true, the model response data will be streamed to the client as it is generated
using server-sent events. See the Streaming section below for more information.
stream boolean or null
## Optional
Defaults to false
Options for streaming responses. Only set this when you set stream: true .
## Show properties
stream_options object or null
## Optional
Defaults to null
What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make
the output more random, while lower values like 0.2 will make it more focused and
deterministic. We generally recommend altering this or top_p  but not both.
temperature number or null
## Optional
Defaults to 1
Configuration options for a text response from the model. Can be plain text or
structured JSON data. Learn more:
## Show properties
text object
Optional
Text inputs and outputs
Structured Outputs
How the model should select which tool (or tools) to use when generating a response.
See the tools  parameter to see how to specify which tools the model can call.
tool_choice string or object
## Optional

<!-- Page 9 -->
## Show possible types
An array of tools the model may call while generating a response. You can specify
which tool to use by setting the tool_choice  parameter.
### The two categories of tools you can provide the model are:
## Show possible types
tools array
Optional
Built-in tools: Tools that are provided by OpenAI that extend the model's
capabilities, like web search or file search. Learn more about built-in tools.
Function calls (custom tools): Functions that are defined by you, enabling the
model to call your own code with strongly typed arguments and outputs. Learn
more about function calling. You can also use custom tools to call your own code.
An integer between 0 and 20 specifying the number of most likely tokens to return at
each token position, each with an associated log probability.
top_logprobs integer or null
## Optional
An alternative to sampling with temperature, called nucleus sampling, where the model
considers the results of the tokens with top_p probability mass. So 0.1 means only the
tokens comprising the top 10% probability mass are considered.
We generally recommend altering this or temperature  but not both.
top_p number or null
## Optional
Defaults to 1
The truncation strategy to use for the model response.
truncation string or null
## Optional
Defaults to disabled
auto : If the context of this response and previous ones exceeds the model's
context window size, the model will truncate the response to fit the context
window by dropping input items in the middle of the conversation.
disabled  (default): If a model response will exceed the context window size for
a model, the request will fail with a 400 error.
user
## Deprecated string
Optional

<!-- Page 10 -->
## Returns
GET https://api.openai.com/v1/responses/{response_id}
Retrieves a model response with the given ID.
## Path parameters
Query parameters
This field is being replaced by safety_identifier  and prompt_cache_key . Use
prompt_cache_key  instead to maintain caching optimizations. A stable identifier for
your end-users. Used to boost cache hit rates by better bucketing similar requests and
to help OpenAI detect and prevent abuse. Learn more.
Constrains the verbosity of the model's response. Lower values will result in more
concise responses, while higher values will result in more verbose responses. Currently
supported values are low , medium , and high .
verbosity string or null
## Optional
Defaults to medium
Returns a Response object.
## Get a model response
The ID of the response to retrieve.
response_id string
## Required
include array
Optional
Example request
python
 
 
from openai import OpenAI
client = OpenAI()
response = client.responses.retrieve("resp_123")
print(response)

## Response
 
{
  "id": "resp_67cb71b351908190a308f3859487620d0
  "object": "response",
  "created_at": 1741386163,


<!-- Page 11 -->
## Returns
DELETE https://api.openai.com/v1/responses/{response_id}
Deletes a model response with the given ID.
## Path parameters
Additional fields to include in the response. See the include  parameter for Response
creation above for more information.
When true, stream obfuscation will be enabled. Stream obfuscation adds random
characters to an obfuscation  field on streaming delta events to normalize payload
sizes as a mitigation to certain side-channel attacks. These obfuscation fields are
included by default, but add a small amount of overhead to the data stream. You can
set include_obfuscation  to false to optimize for bandwidth if you trust the network
links between your application and the OpenAI API.
include_obfuscation boolean
## Optional
The sequence number of the event after which to start streaming.
starting_after integer
## Optional
If set to true, the model response data will be streamed to the client as it is generated
using server-sent events. See the Streaming section below for more information.
stream boolean
## Optional
The Response object matching the specified ID.
 
  "status": "completed",
  "error": null,
  "incomplete_details": null,
  "instructions": null,
  "max_output_tokens": null,
  "model": "gpt-4o-2024-08-06",
  "output": [
    {
      "type": "message",
      "id": "msg_67cb71b3c2b0819084d481baaaf148
      "status": "completed",
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "text": "Silent circuits hum,  \nThou
          "annotations": []
        }
      ]
    }
  ],
  "parallel_tool_calls": true,
  "previous_response_id": null,
  "reasoning": {
    "effort": null,
    "summary": null
  },
  "store": true,
  "temperature": 1.0,
  "text": {
    "format": {
      "type": "text"
    }
  },
  "tool_choice": "auto",
  "tools": [],
  "top_p": 1.0,
  "truncation": "disabled",
  "usage": {
    "input_tokens": 32,

## Delete a model response
Example request
python
 
from openai import OpenAI
client = OpenAI()


<!-- Page 12 -->
## Returns
POST https://api.openai.com/v1/responses/{response_id}/cancel
Cancels a model response with the given ID. Only responses created with
the background  parameter set to true  can be cancelled. Learn more.
## Path parameters
Returns
 
    "input_tokens_details": {
      "cached_tokens": 0
    },
    "output_tokens": 18,
    "output_tokens_details": {
      "reasoning_tokens": 0
    },
    "total_tokens": 50
  },
  "user": null,
  "metadata": {}
}

The ID of the response to delete.
response_id string
## Required
A success message.
 
response = client.responses.delete("resp_123")
print(response)

## Response
 
 
{
  "id": "resp_6786a1bec27481909a17d673315b29f6",
  "object": "response",
  "deleted": true
}

## Cancel a response
The ID of the response to cancel.
response_id string
## Required
A Response object.
## Example request
python
from openai import OpenAI
client = OpenAI()
response = client.responses.cancel("resp_123")
print(response)

## Response
 
{
  "id": "resp_67cb71b351908190a308f3859487620d0
  "object": "response",
  "created_at": 1741386163,
  "status": "completed",
  "error": null,
  "incomplete_details": null,
  "instructions": null,
  "max_output_tokens": null,
  "model": "gpt-4o-2024-08-06",
  "output": [


<!-- Page 13 -->
GET https://api.openai.com/v1/responses/{response_id}/input_items
Returns a list of input items for a given response.
## Path parameters
Query parameters
 
    {
      "type": "message",
      "id": "msg_67cb71b3c2b0819084d481baaaf148
      "status": "completed",
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "text": "Silent circuits hum,  \nThou
          "annotations": []
        }
      ]
    }
  ],
  "parallel_tool_calls": true,
  "previous_response_id": null,
  "reasoning": {
    "effort": null,
    "summary": null
  },
  "store": true,
  "temperature": 1.0,
  "text": {
    "format": {
      "type": "text"
    }
  },
  "tool_choice": "auto",
  "tools": [],
  "top_p": 1.0,
  "truncation": "disabled",
  "usage": {
    "input_tokens": 32,
    "input_tokens_details": {
      "cached_tokens": 0
    },
    "output_tokens": 18,
    "output_tokens_details": {
      "reasoning_tokens": 0
    },

## List input items
The ID of the response to retrieve input items for.
response_id string
## Required
An item ID to list items after, used in pagination.
after string
## Optional
before string
Optional
Example request
python
 
 
from openai import OpenAI
client = OpenAI()
response = client.responses.input_items.list("res
print(response.data)

## Response
 
{
  "object": "list",
  "data": [
    {
      "id": "msg_abc123",
      "type": "message",
      "role": "user",
      "content": [


<!-- Page 14 -->
## Returns
 
    "total_tokens": 50
  },
  "user": null,
  "metadata": {}

An item ID to list items before, used in pagination.
Additional fields to include in the response. See the include  parameter for Response
creation above for more information.
include array
## Optional
A limit on the number of objects to be returned. Limit can range between 1 and 100,
and the default is 20.
limit integer
## Optional
Defaults to 20
The order to return the input items in. Default is desc .
order string
## Optional
asc : Return the input items in ascending order.
desc : Return the input items in descending order.
A list of input item objects.
 
        {
          "type": "input_text",
          "text": "Tell me a three sentence bed
        }
      ]
    }
  ],
  "first_id": "msg_abc123",
  "last_id": "msg_abc123",
  "has_more": false
}

## The response object
Whether to run the model response in the background. Learn more.
background boolean or null
Unix timestamp (in seconds) of when this Response was created.
created_at number
## OBJECT The response object
 
{
  "id": "resp_67ccd3a9da748190baa7f1570fe91ac60
  "object": "response",
  "created_at": 1741476777,
  "status": "completed",


<!-- Page 15 -->
An error object returned when the model fails to generate a Response.
## Show properties
error object or null
Unique identifier for this Response.
id string
Details about why the response is incomplete.
## Show properties
incomplete_details object or null
A system (or developer) message inserted into the model's context.
When using along with previous_response_id , the instructions from a previous
response will not be carried over to the next response. This makes it simple to swap out
system (or developer) messages in new responses.
## Show possible types
instructions string or array
An upper bound for the number of tokens that can be generated for a response,
including visible output tokens and reasoning tokens.
max_output_tokens integer or null
## The maximum number of total calls to built-in tools that can be processed in a
response. This maximum number applies across all built-in tool calls, not per individual
tool. Any further attempts to call a tool by the model will be ignored.
max_tool_calls integer or null
Set of 16 key-value pairs that can be attached to an object. This can be useful for
storing additional information about the object in a structured format, and querying for objects via API or the dashboard.
metadata
map
 
  "error": null,
  "incomplete_details": null,
  "instructions": null,
  "max_output_tokens": null,
  "model": "gpt-4o-2024-08-06",
  "output": [
    {
      "type": "message",
      "id": "msg_67ccd3acc8d48190a77525dc6de64b
      "status": "completed",
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "text": "The image depicts a scenic l
          "annotations": []
        }
      ]
    }
  ],
  "parallel_tool_calls": true,
  "previous_response_id": null,
  "reasoning": {
    "effort": null,
    "summary": null
  },
  "store": true,
  "temperature": 1,
  "text": {
    "format": {
      "type": "text"
    }
  },
  "tool_choice": "auto",
  "tools": [],
  "top_p": 1,
  "truncation": "disabled",
  "usage": {
    "input_tokens": 328,
    "input_tokens_details": {


<!-- Page 16 -->
Keys are strings with a maximum length of 64 characters. Values are strings with a
maximum length of 512 characters.
Model ID used to generate the response, like gpt-4o  or o3 . OpenAI offers a wide
range of models with different capabilities, performance characteristics, and price
points. Refer to the model guide to browse and compare available models.
model string
The object type of this resource - always set to response .
object string
An array of content items generated by the model.
## Show possible types
output array
The length and order of items in the output  array is dependent on the model's
response.
Rather than accessing the first item in the output  array and assuming it's an
assistant  message with the content generated by the model, you might
consider using the output_text  property where supported in SDKs.
## SDK-only convenience property that contains the aggregated text output from all
output_text  items in the output  array, if any are present. Supported in the Python
and JavaScript SDKs.
output_text string or null
## SDK Only
Whether to allow the model to run tool calls in parallel.
parallel_tool_calls boolean
The unique ID of the previous response to the model. Use this to create multi-turn
conversations. Learn more about conversation state.
previous_response_id string or null
prompt object or null
 
      "cached_tokens": 0
    },
    "output_tokens": 52,
    "output_tokens_details": {
      "reasoning_tokens": 0
    },
    "total_tokens": 380
  },
  "user": null,
  "metadata": {}


<!-- Page 17 -->
Reference to a prompt template and its variables. Learn more.
## Show properties
Used by OpenAI to cache responses for similar requests to optimize your cache hit
rates. Replaces the user  field. Learn more.
prompt_cache_key string
o-series models only
Configuration options for reasoning models.
## Show properties
reasoning object or null
A stable identifier used to help detect users of your application that may be violating
OpenAI's usage policies. The IDs should be a string that uniquely identifies each user.
We recommend hashing their username or email address, in order to avoid sending us
any identifying information. Learn more.
safety_identifier string
Specifies the processing type used for serving the request.
When the service_tier  parameter is set, the response body will include the
service_tier  value based on the processing mode actually used to serve the
request. This response value may be different from the value set in the parameter.
service_tier string or null
If set to 'auto', then the request will be processed with the service tier configured
in the Project settings. Unless otherwise configured, the Project will use 'default'.
If set to 'default', then the request will be processed with the standard pricing and
performance for the selected model.
If set to 'flex' or 'priority', then the request will be processed with the
corresponding service tier. Contact sales to learn more about Priority processing.
When not set, the default behavior is 'auto'.
status string

<!-- Page 18 -->
The status of the response generation. One of completed , failed , in_progress ,
cancelled , queued , or incomplete .
What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make
the output more random, while lower values like 0.2 will make it more focused and
deterministic. We generally recommend altering this or top_p  but not both.
temperature number or null
Configuration options for a text response from the model. Can be plain text or
structured JSON data. Learn more:
## Show properties
text object
Text inputs and outputs
Structured Outputs
How the model should select which tool (or tools) to use when generating a response.
See the tools  parameter to see how to specify which tools the model can call.
## Show possible types
tool_choice string or object
An array of tools the model may call while generating a response. You can specify
which tool to use by setting the tool_choice  parameter.
### The two categories of tools you can provide the model are:
## Show possible types
tools array
Built-in tools: Tools that are provided by OpenAI that extend the model's
capabilities, like web search or file search. Learn more about built-in tools.
Function calls (custom tools): Functions that are defined by you, enabling the
model to call your own code with strongly typed arguments and outputs. Learn
more about function calling. You can also use custom tools to call your own code.

<!-- Page 19 -->
An integer between 0 and 20 specifying the number of most likely tokens to return at
each token position, each with an associated log probability.
top_logprobs integer or null
An alternative to sampling with temperature, called nucleus sampling, where the model
considers the results of the tokens with top_p probability mass. So 0.1 means only the
tokens comprising the top 10% probability mass are considered.
We generally recommend altering this or temperature  but not both.
top_p number or null
The truncation strategy to use for the model response.
truncation string or null
auto : If the context of this response and previous ones exceeds the model's
context window size, the model will truncate the response to fit the context
window by dropping input items in the middle of the conversation.
disabled  (default): If a model response will exceed the context window size for
a model, the request will fail with a 400 error.
Represents token usage details including input tokens, output tokens, a breakdown of
output tokens, and the total tokens used.
## Show properties
usage object
This field is being replaced by safety_identifier  and prompt_cache_key . Use
prompt_cache_key  instead to maintain caching optimizations. A stable identifier for
your end-users. Used to boost cache hit rates by better bucketing similar requests and
to help OpenAI detect and prevent abuse. Learn more.
user
## Deprecated string
Constrains the verbosity of the model's response. Lower values will result in more
concise responses, while higher values will result in more verbose responses. Currently
verbosity string or null

<!-- Page 20 -->
A list of Response items.
supported values are low , medium , and high .
## The input item list
A list of items used to generate this response.
## Show possible types
data array
The ID of the first item in the list.
first_id string
Whether there are more items available.
has_more boolean
The ID of the last item in the list.
last_id string
The type of object returned, must be list .
object string
## OBJECT The input item list
 
 
{
  "object": "list",
  "data": [
    {
      "id": "msg_abc123",
      "type": "message",
      "role": "user",
      "content": [
        {
          "type": "input_text",
          "text": "Tell me a three sentence bedt
        }
      ]
    }
  ],
  "first_id": "msg_abc123",
  "last_id": "msg_abc123",
  "has_more": false
}

## Streaming

<!-- Page 21 -->
When you create a Response with stream  set to true , the server will emit server-sent events to the client
as the Response is generated. This section contains the events that are emitted by the server.
Learn more about streaming responses.
An event that is emitted when a response is created.
response.created
The response that was created.
## Show properties
response object
The sequence number for this event.
sequence_number integer
The type of the event. Always response.created .
type string
OBJECT response.created
 
{
  "type": "response.created",
  "response": {
    "id": "resp_67ccfcdd16748190a91872c75d38539
    "object": "response",
    "created_at": 1741487325,
    "status": "in_progress",
    "error": null,
    "incomplete_details": null,
    "instructions": null,
    "max_output_tokens": null,
    "model": "gpt-4o-2024-08-06",
    "output": [],
    "parallel_tool_calls": true,
    "previous_response_id": null,
    "reasoning": {
      "effort": null,
      "summary": null
    },
    "store": true,
    "temperature": 1,
    "text": {
      "format": {


<!-- Page 22 -->
Emitted when the response is in progress.
 
        "type": "text"
      }
    },
    "tool_choice": "auto",
    "tools": [],
    "top_p": 1,
    "truncation": "disabled",
    "usage": null,
    "user": null,
    "metadata": {}
  },
  "sequence_number": 1
}

response.in_progress
The response that is in progress.
## Show properties
response object
The sequence number of this event.
sequence_number integer
The type of the event. Always response.in_progress .
type string
OBJECT response.in_progress
 
{
  "type": "response.in_progress",
  "response": {
    "id": "resp_67ccfcdd16748190a91872c75d38539
    "object": "response",
    "created_at": 1741487325,
    "status": "in_progress",
    "error": null,
    "incomplete_details": null,
    "instructions": null,
    "max_output_tokens": null,
    "model": "gpt-4o-2024-08-06",
    "output": [],
    "parallel_tool_calls": true,
    "previous_response_id": null,
    "reasoning": {
      "effort": null,
      "summary": null
    },
    "store": true,
    "temperature": 1,


<!-- Page 23 -->
Emitted when the model response is complete.
 
    "text": {
      "format": {
        "type": "text"
      }
    },
    "tool_choice": "auto",
    "tools": [],
    "top_p": 1,
    "truncation": "disabled",
    "usage": null,
    "user": null,
    "metadata": {}
  },
  "sequence_number": 1
}

response.completed
Properties of the completed response.
## Show properties
response object
The sequence number for this event.
sequence_number integer
The type of the event. Always response.completed .
type string
OBJECT response.completed
 
{
  "type": "response.completed",
  "response": {
    "id": "resp_123",
    "object": "response",
    "created_at": 1740855869,
    "status": "completed",
    "error": null,
    "incomplete_details": null,
    "input": [],
    "instructions": null,
    "max_output_tokens": null,
    "model": "gpt-4o-mini-2024-07-18",
    "output": [
      {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",


<!-- Page 24 -->
An event that is emitted when a response fails.
 
        "content": [
          {
            "type": "output_text",
            "text": "In a shimmering forest und
            "annotations": []
          }
        ]
      }
    ],
    "previous_response_id": null,
    "reasoning_effort": null,
    "store": false,
    "temperature": 1,
    "text": {
      "format": {
        "type": "text"
      }
    },
    "tool_choice": "auto",
    "tools": [],
    "top_p": 1,
    "truncation": "disabled",
    "usage": {
      "input_tokens": 0,
      "output_tokens": 0,
      "output_tokens_details": {
        "reasoning_tokens": 0
      },
      "total_tokens": 0
    },
    "user": null,
    "metadata": {}
  },
  "sequence_number": 1
}

response.failed
The response that failed.
## Show properties
response object
The sequence number of this event.
sequence_number integer
The type of the event. Always response.failed .
type string
OBJECT response.failed
 
{
  "type": "response.failed",
  "response": {
    "id": "resp_123",
    "object": "response",
    "created_at": 1740855869,
    "status": "failed",
    "error": {
      "code": "server_error",
      "message": "The model failed to generate 
    },
    "incomplete_details": null,
    "instructions": null,
    "max_output_tokens": null,
    "model": "gpt-4o-mini-2024-07-18",
    "output": [],


<!-- Page 25 -->
An event that is emitted when a response finishes as incomplete.
 
    "previous_response_id": null,
    "reasoning_effort": null,
    "store": false,
    "temperature": 1,
    "text": {
      "format": {
        "type": "text"
      }
    },
    "tool_choice": "auto",
    "tools": [],
    "top_p": 1,
    "truncation": "disabled",
    "usage": null,
    "user": null,
    "metadata": {}
  }
}

response.incomplete
The response that was incomplete.
## Show properties
response object
The sequence number of this event.
sequence_number integer
The type of the event. Always response.incomplete .
type string
OBJECT response.incomplete
 
{
  "type": "response.incomplete",
  "response": {
    "id": "resp_123",
    "object": "response",
    "created_at": 1740855869,
    "status": "incomplete",
    "error": null, 
    "incomplete_details": {
      "reason": "max_tokens"
    },
    "instructions": null,
    "max_output_tokens": null,


<!-- Page 26 -->
Emitted when a new output item is added.
 
    "model": "gpt-4o-mini-2024-07-18",
    "output": [],
    "previous_response_id": null,
    "reasoning_effort": null,
    "store": false,
    "temperature": 1,
    "text": {
      "format": {
        "type": "text"
      }
    },
    "tool_choice": "auto",
    "tools": [],
    "top_p": 1,
    "truncation": "disabled",
    "usage": null,
    "user": null,
    "metadata": {}
  },
  "sequence_number": 1
}

response.output_item.added
The output item that was added.
## Show possible types
item object
output_index integer
OBJECT response.output_item.added
{
  "type": "response.output_item.added",
  "output_index": 0,
  "item": {
    "id": "msg_123",
    "status": "in_progress",
    "type": "message",
    "role": "assistant",


<!-- Page 27 -->
Emitted when an output item is marked done.
The index of the output item that was added.
The sequence number of this event.
sequence_number integer
The type of the event. Always response.output_item.added .
type string
    "content": []
  },
  "sequence_number": 1
}

response.output_item.done
The output item that was marked done.
## Show possible types
item object
The index of the output item that was marked done.
output_index integer
The sequence number of this event.
sequence_number integer
The type of the event. Always response.output_item.done .
type string
OBJECT response.output_item.done
 
 
{
  "type": "response.output_item.done",
  "output_index": 0,
  "item": {
    "id": "msg_123",
    "status": "completed",
    "type": "message",
    "role": "assistant",
    "content": [
      {
        "type": "output_text",
        "text": "In a shimmering forest under a 
        "annotations": []
      }
    ]
  },
  "sequence_number": 1
}


<!-- Page 28 -->
Emitted when a new content part is added.
Emitted when a content part is done.
response.content_part.added
The index of the content part that was added.
content_index integer
The ID of the output item that the content part was added to.
item_id string
The index of the output item that the content part was added to.
output_index integer
The content part that was added.
## Show possible types
part object
The sequence number of this event.
sequence_number integer
The type of the event. Always response.content_part.added .
type string
OBJECT response.content_part.added
{
  "type": "response.content_part.added",
  "item_id": "msg_123",
  "output_index": 0,
  "content_index": 0,
  "part": {
    "type": "output_text",
    "text": "",
    "annotations": []
  },
  "sequence_number": 1
}

response.content_part.done
OBJECT response.content_part.done

<!-- Page 29 -->
Emitted when there is an additional text delta.
The index of the content part that is done.
content_index integer
The ID of the output item that the content part was added to.
item_id string
The index of the output item that the content part was added to.
output_index integer
The content part that is done.
## Show possible types
part object
The sequence number of this event.
sequence_number integer
The type of the event. Always response.content_part.done .
type string
{
  "type": "response.content_part.done",
  "item_id": "msg_123",
  "output_index": 0,
  "content_index": 0,
  "sequence_number": 1,
  "part": {
    "type": "output_text",
    "text": "In a shimmering forest under a sky 
    "annotations": []
  }
}

response.output_text.delta
The index of the content part that the text delta was added to.
content_index integer
delta string
OBJECT response.output_text.delta
 
{
  "type": "response.output_text.delta",
  "item_id": "msg_123",
  "output_index": 0,


<!-- Page 30 -->
Emitted when text content is finalized.
The text delta that was added.
The ID of the output item that the text delta was added to.
item_id string
The log probabilities of the tokens in the delta.
## Show properties
logprobs array
The index of the output item that the text delta was added to.
output_index integer
The sequence number for this event.
sequence_number integer
The type of the event. Always response.output_text.delta .
type string
 
  "content_index": 0,
  "delta": "In",
  "sequence_number": 1
}

response.output_text.done
The index of the content part that the text content is finalized.
content_index integer
The ID of the output item that the text content is finalized.
item_id string
The log probabilities of the tokens in the delta.
logprobs array
OBJECT response.output_text.done
 
 
{
  "type": "response.output_text.done",
  "item_id": "msg_123",
  "output_index": 0,
  "content_index": 0,
  "text": "In a shimmering forest under a sky ful
  "sequence_number": 1
}


<!-- Page 31 -->
Emitted when there is a partial refusal text.
## Show properties
The index of the output item that the text content is finalized.
output_index integer
The sequence number for this event.
sequence_number integer
The text content that is finalized.
text string
The type of the event. Always response.output_text.done .
type string
response.refusal.delta
The index of the content part that the refusal text is added to.
content_index integer
The refusal text that is added.
delta string
The ID of the output item that the refusal text is added to.
item_id string
OBJECT response.refusal.delta
{
  "type": "response.refusal.delta",
  "item_id": "msg_123",
  "output_index": 0,
  "content_index": 0,
  "delta": "refusal text so far",
  "sequence_number": 1
}


<!-- Page 32 -->
Emitted when refusal text is finalized.
The index of the output item that the refusal text is added to.
output_index integer
The sequence number of this event.
sequence_number integer
The type of the event. Always response.refusal.delta .
type string
response.refusal.done
The index of the content part that the refusal text is finalized.
content_index integer
The ID of the output item that the refusal text is finalized.
item_id string
The index of the output item that the refusal text is finalized.
output_index integer
The refusal text that is finalized.
refusal string
The sequence number of this event.
sequence_number integer
type string
OBJECT response.refusal.done
{
  "type": "response.refusal.done",
  "item_id": "item-abc",
  "output_index": 1,
  "content_index": 2,
  "refusal": "final refusal text",
  "sequence_number": 1
}


<!-- Page 33 -->
Emitted when there is a partial function-call arguments delta.
Emitted when function-call arguments are finalized.
The type of the event. Always response.refusal.done .
response.function_call_arguments.delta
The function-call arguments delta that is added.
delta string
The ID of the output item that the function-call arguments delta is added to.
item_id string
The index of the output item that the function-call arguments delta is added to.
output_index integer
The sequence number of this event.
sequence_number integer
The type of the event. Always response.function_call_arguments.delta .
type string
OBJECT response.function_call_arguments.delta
 
 
{
  "type": "response.function_call_arguments.delta
  "item_id": "item-abc",
  "output_index": 0,
  "delta": "{ \"arg\":"
  "sequence_number": 1
}

response.function_call_arguments.done

<!-- Page 34 -->
Emitted when a file search call is initiated.
The function-call arguments.
arguments string
The ID of the item.
item_id string
The index of the output item.
output_index integer
The sequence number of this event.
sequence_number integer
type string
OBJECT response.function_call_arguments.done
{
  "type": "response.function_call_arguments.done"
  "item_id": "item-abc",
  "output_index": 1,
  "arguments": "{ \"arg\": 123 }",
  "sequence_number": 1
}

response.file_search_call.in_progress
The ID of the output item that the file search call is initiated.
item_id string
The index of the output item that the file search call is initiated.
output_index integer
The sequence number of this event.
sequence_number integer
OBJECT response.file_search_call.in_progress
 
 
{
  "type": "response.file_search_call.in_progress"
  "output_index": 0,
  "item_id": "fs_123",
  "sequence_number": 1
}


<!-- Page 35 -->
Emitted when a file search is currently searching.
Emitted when a file search call is completed (results found).
The type of the event. Always response.file_search_call.in_progress .
type string
response.file_search_call.searching
The ID of the output item that the file search call is initiated.
item_id string
The index of the output item that the file search call is searching.
output_index integer
The sequence number of this event.
sequence_number integer
The type of the event. Always response.file_search_call.searching .
type string
OBJECT response.file_search_call.searching
 
 
{
  "type": "response.file_search_call.searching",
  "output_index": 0,
  "item_id": "fs_123",
  "sequence_number": 1
}

response.file_search_call.completed
The ID of the output item that the file search call is initiated.
item_id string
OBJECT response.file_search_call.completed
 
{
  "type": "response.file_search_call.completed",
  "output_index": 0,


<!-- Page 36 -->
Emitted when a web search call is initiated.
The index of the output item that the file search call is initiated.
output_index integer
The sequence number of this event.
sequence_number integer
The type of the event. Always response.file_search_call.completed .
type string
 
  "item_id": "fs_123",
  "sequence_number": 1
}

response.web_search_call.in_progress
Unique ID for the output item associated with the web search call.
item_id string
The index of the output item that the web search call is associated with.
output_index integer
The sequence number of the web search call being processed.
sequence_number integer
The type of the event. Always response.web_search_call.in_progress .
type string
OBJECT response.web_search_call.in_progress
 
 
{
  "type": "response.web_search_call.in_progress",
  "output_index": 0,
  "item_id": "ws_123",
  "sequence_number": 0
}


<!-- Page 37 -->
Emitted when a web search call is executing.
Emitted when a web search call is completed.
response.web_search_call.searching
Unique ID for the output item associated with the web search call.
item_id string
The index of the output item that the web search call is associated with.
output_index integer
The sequence number of the web search call being processed.
sequence_number integer
The type of the event. Always response.web_search_call.searching .
type string
OBJECT response.web_search_call.searching
 
 
{
  "type": "response.web_search_call.searching",
  "output_index": 0,
  "item_id": "ws_123",
  "sequence_number": 0
}

response.web_search_call.completed
Unique ID for the output item associated with the web search call.
item_id string
The index of the output item that the web search call is associated with.
output_index integer
The sequence number of the web search call being processed.
sequence_number integer
OBJECT response.web_search_call.completed
 
 
{
  "type": "response.web_search_call.completed",
  "output_index": 0,
  "item_id": "ws_123",
  "sequence_number": 0
}


<!-- Page 38 -->
The type of the event. Always response.web_search_call.completed .
type string
response.reasoning_summary_part.added

<!-- Page 39 -->
Emitted when a new reasoning summary part is added.
Emitted when a reasoning summary part is completed.
The ID of the item this summary part is associated with.
item_id string
The index of the output item this summary part is associated with.
output_index integer
The summary part that was added.
## Show properties
part object
The sequence number of this event.
sequence_number integer
The index of the summary part within the reasoning summary.
summary_index integer
The type of the event. Always response.reasoning_summary_part.added .
type string
OBJECT response.reasoning_summary_part.added
 
 
{
  "type": "response.reasoning_summary_part.added
  "item_id": "rs_6806bfca0b2481918a5748308061a26
  "output_index": 0,
  "summary_index": 0,
  "part": {
    "type": "summary_text",
    "text": ""
  },
  "sequence_number": 1
}

response.reasoning_summary_part.done
The ID of the item this summary part is associated with.
item_id string
output_index integer
OBJECT response.reasoning_summary_part.done
 
{
  "type": "response.reasoning_summary_part.done
  "item_id": "rs_6806bfca0b2481918a5748308061a2
  "output_index": 0,
  "summary_index": 0,
  "part": {


<!-- Page 40 -->
Emitted when a delta is added to a reasoning summary text.
The index of the output item this summary part is associated with.
The completed summary part.
## Show properties
part object
The sequence number of this event.
sequence_number integer
The index of the summary part within the reasoning summary.
summary_index integer
The type of the event. Always response.reasoning_summary_part.done .
type string
 
    "type": "summary_text",
    "text": "**Responding to a greeting**\n\nTh
  },
  "sequence_number": 1
}

response.reasoning_summary_text.delta
The text delta that was added to the summary.
delta string
The ID of the item this summary text delta is associated with.
item_id string
The index of the output item this summary text delta is associated with.
output_index integer
OBJECT response.reasoning_summary_text.delta
 
 
{
  "type": "response.reasoning_summary_text.delta"
  "item_id": "rs_6806bfca0b2481918a5748308061a260
  "output_index": 0,
  "summary_index": 0,
  "delta": "**Responding to a greeting**\n\nThe u
  "sequence_number": 1
}


<!-- Page 41 -->
Emitted when a reasoning summary text is completed.
The sequence number of this event.
sequence_number integer
The index of the summary part within the reasoning summary.
summary_index integer
The type of the event. Always response.reasoning_summary_text.delta .
type string
response.reasoning_summary_text.done
The ID of the item this summary text is associated with.
item_id string
The index of the output item this summary text is associated with.
output_index integer
The sequence number of this event.
sequence_number integer
The index of the summary part within the reasoning summary.
summary_index integer
The full text of the completed reasoning summary.
text string
type string
OBJECT response.reasoning_summary_text.done
 
 
{
  "type": "response.reasoning_summary_text.done",
  "item_id": "rs_6806bfca0b2481918a5748308061a260
  "output_index": 0,
  "summary_index": 0,
  "text": "**Responding to a greeting**\n\nThe us
  "sequence_number": 1
}


<!-- Page 42 -->
Emitted when a delta is added to a reasoning text.
The type of the event. Always response.reasoning_summary_text.done .
response.reasoning_text.delta
The index of the reasoning content part this delta is associated with.
content_index integer
The text delta that was added to the reasoning content.
delta string
The ID of the item this reasoning text delta is associated with.
item_id string
The index of the output item this reasoning text delta is associated with.
output_index integer
The sequence number of this event.
sequence_number integer
The type of the event. Always response.reasoning_text.delta .
type string
OBJECT response.reasoning_text.delta
{
  "type": "response.reasoning_text.delta",
  "item_id": "rs_123",
  "output_index": 0,
  "content_index": 0,
  "delta": "The",
  "sequence_number": 1
}


<!-- Page 43 -->
Emitted when a reasoning text is completed.
## Emitted when an image generation tool call has completed and the final
image is available.
response.reasoning_text.done
The index of the reasoning content part.
content_index integer
The ID of the item this reasoning text is associated with.
item_id string
The index of the output item this reasoning text is associated with.
output_index integer
The sequence number of this event.
sequence_number integer
The full text of the completed reasoning content.
text string
The type of the event. Always response.reasoning_text.done .
type string
OBJECT response.reasoning_text.done
{
  "type": "response.reasoning_text.done",
  "item_id": "rs_123",
  "output_index": 0,
  "content_index": 0,
  "text": "The user is asking...",
  "sequence_number": 4
}

response.image_generation_call.completed
OBJECT response.image_generation_call.completed

<!-- Page 44 -->
The unique identifier of the image generation item being processed.
item_id string
The index of the output item in the response's output array.
output_index integer
The sequence number of this event.
sequence_number integer
The type of the event. Always 'response.image_generation_call.completed'.
type string
{
  "type": "response.image_generation_call.complet
  "output_index": 0,
  "item_id": "item-123",
  "sequence_number": 1

response.image_generation_call.generating

<!-- Page 45 -->
## Emitted when an image generation tool call is actively generating an image
(intermediate state).
Emitted when an image generation tool call is in progress.
The unique identifier of the image generation item being processed.
item_id string
The index of the output item in the response's output array.
output_index integer
The sequence number of the image generation item being processed.
sequence_number integer
The type of the event. Always 'response.image_generation_call.generating'.
type string
OBJECT response.image_generation_call.generating
 
 
{
  "type": "response.image_generation_call.generat
  "output_index": 0,
  "item_id": "item-123",
  "sequence_number": 0
}

response.image_generation_call.in_progress
The unique identifier of the image generation item being processed.
item_id string
The index of the output item in the response's output array.
output_index integer
The sequence number of the image generation item being processed.
sequence_number integer
type string
OBJECT response.image_generation_call.in_progress
 
 
{
  "type": "response.image_generation_call.in_prog
  "output_index": 0,
  "item_id": "item-123",
  "sequence_number": 0
}


<!-- Page 46 -->
## Emitted when a partial image is available during image generation
streaming.
The type of the event. Always 'response.image_generation_call.in_progress'.
response.image_generation_call.partial_image
The unique identifier of the image generation item being processed.
item_id string
The index of the output item in the response's output array.
output_index integer
Base64-encoded partial image data, suitable for rendering as an image.
partial_image_b64 string
0-based index for the partial image (backend is 1-based, but this is 0-based for the
user).
partial_image_index integer
The sequence number of the image generation item being processed.
sequence_number integer
The type of the event. Always 'response.image_generation_call.partial_image'.
type string
OBJECT response.image_generation_call.partial_ima...
 
 
{
  "type": "response.image_generation_call.partial
  "output_index": 0,
  "item_id": "item-123",
  "sequence_number": 0,
  "partial_image_index": 0,
  "partial_image_b64": "..."
}


<!-- Page 47 -->
Emitted when there is a delta (partial update) to the arguments of an MCP
tool call.
Emitted when the arguments for an MCP tool call are finalized.
response.mcp_call_arguments.delta
A JSON string containing the partial update to the arguments for the MCP tool call.
delta string
The unique identifier of the MCP tool call item being processed.
item_id string
The index of the output item in the response's output array.
output_index integer
The sequence number of this event.
sequence_number integer
The type of the event. Always 'response.mcp_call_arguments.delta'.
type string
OBJECT response.mcp_call_arguments.delta
{
  "type": "response.mcp_call_arguments.delta",
  "output_index": 0,
  "item_id": "item-abc",
  "delta": "{",
  "sequence_number": 1
}

response.mcp_call_arguments.done
A JSON string containing the finalized arguments for the MCP tool call.
arguments string
The unique identifier of the MCP tool call item being processed.
item_id string
OBJECT response.mcp_call_arguments.done
 
{
  "type": "response.mcp_call_arguments.done",
  "output_index": 0,
  "item_id": "item-abc",
  "arguments": "{\"arg1\": \"value1\", \"arg2\":


<!-- Page 48 -->
Emitted when an MCP tool call has completed successfully.
The index of the output item in the response's output array.
output_index integer
The sequence number of this event.
sequence_number integer
The type of the event. Always 'response.mcp_call_arguments.done'.
type string
 
"sequence number": 1
response.mcp_call.completed
The ID of the MCP tool call item that completed.
item_id string
The index of the output item that completed.
output_index integer
The sequence number of this event.
sequence_number integer
The type of the event. Always 'response.mcp_call.completed'.
type string
OBJECT response.mcp_call.completed
 
 
{
  "type": "response.mcp_call.completed",
  "sequence_number": 1,
  "item_id": "mcp_682d437d90a88191bf88cd03aae0c3e
  "output_index": 0
}


<!-- Page 49 -->
Emitted when an MCP tool call has failed.
Emitted when an MCP tool call is in progress.
response.mcp_call.failed
The ID of the MCP tool call item that failed.
item_id string
The index of the output item that failed.
output_index integer
The sequence number of this event.
sequence_number integer
The type of the event. Always 'response.mcp_call.failed'.
type string
OBJECT response.mcp_call.failed
 
 
{
  "type": "response.mcp_call.failed",
  "sequence_number": 1,
  "item_id": "mcp_682d437d90a88191bf88cd03aae0c3e
  "output_index": 0
}

response.mcp_call.in_progress
The unique identifier of the MCP tool call item being processed.
item_id string
The index of the output item in the response's output array.
output_index integer
The sequence number of this event.
sequence_number integer
OBJECT response.mcp_call.in_progress
 
 
{
  "type": "response.mcp_call.in_progress",
  "sequence_number": 1,
  "output_index": 0,
  "item_id": "mcp_682d437d90a88191bf88cd03aae0c3e
}


<!-- Page 50 -->
## Emitted when the list of available MCP tools has been successfully
retrieved.
Emitted when the attempt to list available MCP tools has failed.
The type of the event. Always 'response.mcp_call.in_progress'.
type string
response.mcp_list_tools.completed
The ID of the MCP tool call item that produced this output.
item_id string
The index of the output item that was processed.
output_index integer
The sequence number of this event.
sequence_number integer
The type of the event. Always 'response.mcp_list_tools.completed'.
type string
OBJECT response.mcp_list_tools.completed
 
 
{
  "type": "response.mcp_list_tools.completed",
  "sequence_number": 1,
  "output_index": 0,
  "item_id": "mcpl_682d4379df088191886b70f4ec39f9
}

response.mcp_list_tools.failed
OBJECT response.mcp_list_tools.failed

<!-- Page 51 -->
The ID of the MCP tool call item that failed.
item_id string
The index of the output item that failed.
output_index integer
The sequence number of this event.
sequence_number integer
The type of the event. Always 'response.mcp_list_tools.failed'.
type string
 
 
{
  "type": "response.mcp_list_tools.failed",
  "sequence_number": 1,
  "output_index": 0,
  "item_id": "mcpl_682d4379df088191886b70f4ec39f9
}

response.mcp_list_tools.in_progress

<!-- Page 52 -->
## Emitted when the system is in the process of retrieving the list of available
MCP tools.
Emitted when a code interpreter call is in progress.
The ID of the MCP tool call item that is being processed.
item_id string
The index of the output item that is being processed.
output_index integer
The sequence number of this event.
sequence_number integer
The type of the event. Always 'response.mcp_list_tools.in_progress'.
type string
OBJECT response.mcp_list_tools.in_progress
 
 
{
  "type": "response.mcp_list_tools.in_progress",
  "sequence_number": 1,
  "output_index": 0,
  "item_id": "mcpl_682d4379df088191886b70f4ec39f9
}

response.code_interpreter_call.in_progress
The unique identifier of the code interpreter tool call item.
item_id string
## The index of the output item in the response for which the code interpreter call is in
progress.
output_index integer
sequence_number integer
OBJECT response.code_interpreter_call.in_progress
 
 
{
  "type": "response.code_interpreter_call.in_prog
  "output_index": 0,
  "item_id": "ci_12345",
  "sequence_number": 1
}


<!-- Page 53 -->
Emitted when the code interpreter is actively interpreting the code snippet.
Emitted when the code interpreter call is completed.
The sequence number of this event, used to order streaming events.
The type of the event. Always response.code_interpreter_call.in_progress .
type string
response.code_interpreter_call.interpreting
The unique identifier of the code interpreter tool call item.
item_id string
## The index of the output item in the response for which the code interpreter is
interpreting code.
output_index integer
The sequence number of this event, used to order streaming events.
sequence_number integer
The type of the event. Always response.code_interpreter_call.interpreting .
type string
OBJECT response.code_interpreter_call.interpreting
 
 
{
  "type": "response.code_interpreter_call.interpr
  "output_index": 4,
  "item_id": "ci_12345",
  "sequence_number": 1
}

response.code_interpreter_call.completed
OBJECT response.code_interpreter_call.completed

<!-- Page 54 -->
Emitted when a partial code snippet is streamed by the code interpreter.
The unique identifier of the code interpreter tool call item.
item_id string
## The index of the output item in the response for which the code interpreter call is
completed.
output_index integer
The sequence number of this event, used to order streaming events.
sequence_number integer
The type of the event. Always response.code_interpreter_call.completed .
type string
{
  "type": "response.code_interpreter_call.complet
  "output_index": 5,
  "item_id": "ci_12345",
  "sequence_number": 1
}

response.code_interpreter_call_code.delta
The partial code snippet being streamed by the code interpreter.
delta string
The unique identifier of the code interpreter tool call item.
item_id string
The index of the output item in the response for which the code is being streamed.
output_index integer
sequence_number integer
OBJECT response.code_interpreter_call_code.delta
 
 
{
  "type": "response.code_interpreter_call_code.de
  "output_index": 0,
  "item_id": "ci_12345",
  "delta": "print('Hello, world')",
  "sequence_number": 1
}


<!-- Page 55 -->
Emitted when the code snippet is finalized by the code interpreter.
The sequence number of this event, used to order streaming events.
The type of the event. Always response.code_interpreter_call_code.delta .
type string
response.code_interpreter_call_code.done
The final code snippet output by the code interpreter.
code string
The unique identifier of the code interpreter tool call item.
item_id string
The index of the output item in the response for which the code is finalized.
output_index integer
The sequence number of this event, used to order streaming events.
sequence_number integer
The type of the event. Always response.code_interpreter_call_code.done .
type string
OBJECT response.code_interpreter_call_code.done
 
 
{
  "type": "response.code_interpreter_call_code.do
  "output_index": 3,
  "item_id": "ci_12345",
  "code": "print('done')",
  "sequence_number": 1
}


<!-- Page 56 -->
Emitted when an annotation is added to output text content.
Emitted when a response is queued and waiting to be processed.
response.output_text.annotation.added
The annotation object being added. (See annotation schema for details.)
annotation object
The index of the annotation within the content part.
annotation_index integer
The index of the content part within the output item.
content_index integer
The unique identifier of the item to which the annotation is being added.
item_id string
The index of the output item in the response's output array.
output_index integer
The sequence number of this event.
sequence_number integer
The type of the event. Always 'response.output_text.annotation.added'.
type string
OBJECT response.output_text.annotation.added
 
 
{
  "type": "response.output_text.annotation.added
  "item_id": "item-abc",
  "output_index": 0,
  "content_index": 0,
  "annotation_index": 0,
  "annotation": {
    "type": "text_annotation",
    "text": "This is a test annotation",
    "start": 0,
    "end": 10
  },
  "sequence_number": 1
}

response.queued
OBJECT response.queued

<!-- Page 57 -->
Event representing a delta (partial update) to the input of a custom tool call.
The full response object that is queued.
## Show properties
response object
The sequence number for this event.
sequence_number integer
The type of the event. Always 'response.queued'.
type string
{
  "type": "response.queued",
  "response": {
    "id": "res_123",
    "status": "queued",
    "created_at": "2021-01-01T00:00:00Z",
    "updated_at": "2021-01-01T00:00:00Z"
  },
  "sequence number": 1

response.custom_tool_call_input.delta
The incremental input data (delta) for the custom tool call.
delta string
Unique identifier for the API item associated with this event.
item_id string
The index of the output this delta applies to.
output_index integer
The sequence number of this event.
sequence_number integer
type string
OBJECT response.custom_tool_call_input.delta
 
 
{
  "type": "response.custom_tool_call_input.delta"
  "output_index": 0,
  "item_id": "ctc_1234567890abcdef",
  "delta": "partial input text"
}


<!-- Page 58 -->
Event indicating that input for a custom tool call is complete.
Emitted when an error occurs.
The event type identifier.
response.custom_tool_call_input.done
The complete input data for the custom tool call.
input string
Unique identifier for the API item associated with this event.
item_id string
The index of the output this event applies to.
output_index integer
The sequence number of this event.
sequence_number integer
The event type identifier.
type string
OBJECT response.custom_tool_call_input.done
 
 
{
  "type": "response.custom_tool_call_input.done",
  "output_index": 0,
  "item_id": "ctc_1234567890abcdef",
  "input": "final complete input text"
}

error
code string or null
## OBJECT error
 
{
  "type": "error",


<!-- Page 59 -->
## Webhooks are HTTP requests sent by OpenAI to a URL you specify when certain events happen during the
course of API usage.
Learn more about webhooks.
Sent when a background response has been completed.
The error code.
The error message.
message string
The error parameter.
param string or null
The sequence number of this event.
sequence_number integer
The type of the event. Always error .
type string
 
  "code": "ERR_SOMETHING",
  "message": "Something went wrong",
  "param": null,
  "sequence_number": 1
}

## Webhook Events
response.completed
OBJECT response.completed

<!-- Page 60 -->
Sent when a background response has been cancelled.
The Unix timestamp (in seconds) of when the model response was completed.
created_at integer
Event data payload.
## Show properties
data object
The unique ID of the event.
id string
The object of the event. Always event .
object string
The type of the event. Always response.completed .
type string
{
  "id": "evt_abc123",
  "type": "response.completed",
  "created_at": 1719168000,
  "data": {
    "id": "resp_abc123"
  }
}

response.cancelled
The Unix timestamp (in seconds) of when the model response was cancelled.
created_at integer
Event data payload.
## Show properties
data object
id string
OBJECT response.cancelled
{
  "id": "evt_abc123",
  "type": "response.cancelled",
  "created_at": 1719168000,
  "data": {
    "id": "resp_abc123"
  }
}


<!-- Page 61 -->
Sent when a background response has failed.
The unique ID of the event.
The object of the event. Always event .
object string
The type of the event. Always response.cancelled .
type string
response.failed
The Unix timestamp (in seconds) of when the model response failed.
created_at integer
Event data payload.
## Show properties
data object
The unique ID of the event.
id string
The object of the event. Always event .
object string
The type of the event. Always response.failed .
type string
OBJECT response.failed
{
  "id": "evt_abc123",
  "type": "response.failed",
  "created_at": 1719168000,
  "data": {
    "id": "resp_abc123"
  }
}


<!-- Page 62 -->
Sent when a background response has been interrupted.
Sent when a batch API request has been completed.
response.incomplete
The Unix timestamp (in seconds) of when the model response was interrupted.
created_at integer
Event data payload.
## Show properties
data object
The unique ID of the event.
id string
The object of the event. Always event .
object string
The type of the event. Always response.incomplete .
type string
OBJECT response.incomplete
{
  "id": "evt_abc123",
  "type": "response.incomplete",
  "created_at": 1719168000,
  "data": {
    "id": "resp_abc123"
  }
}

batch.completed
The Unix timestamp (in seconds) of when the batch API request was completed.
created_at integer
OBJECT batch.completed
 
{
  "id": "evt_abc123",
  "type": "batch.completed",


<!-- Page 63 -->
Sent when a batch API request has been cancelled.
Event data payload.
## Show properties
data object
The unique ID of the event.
id string
The object of the event. Always event .
object string
The type of the event. Always batch.completed .
type string
 
  "created_at": 1719168000,
  "data": {
    "id": "batch_abc123"
  }
}

batch.cancelled
The Unix timestamp (in seconds) of when the batch API request was cancelled.
created_at integer
Event data payload.
## Show properties
data object
The unique ID of the event.
id string
object string
OBJECT batch.cancelled
{
  "id": "evt_abc123",
  "type": "batch.cancelled",
  "created_at": 1719168000,
  "data": {
    "id": "batch_abc123"
  }
}


<!-- Page 64 -->
Sent when a batch API request has expired.
The object of the event. Always event .
The type of the event. Always batch.cancelled .
type string
batch.expired
The Unix timestamp (in seconds) of when the batch API request expired.
created_at integer
Event data payload.
## Show properties
data object
The unique ID of the event.
id string
The object of the event. Always event .
object string
The type of the event. Always batch.expired .
type string
OBJECT batch.expired
{
  "id": "evt_abc123",
  "type": "batch.expired",
  "created_at": 1719168000,
  "data": {
    "id": "batch_abc123"
  }
}

batch.failed

<!-- Page 65 -->
Sent when a batch API request has failed.
Sent when a fine-tuning job has succeeded.
The Unix timestamp (in seconds) of when the batch API request failed.
created_at integer
Event data payload.
## Show properties
data object
The unique ID of the event.
id string
The object of the event. Always event .
object string
The type of the event. Always batch.failed .
type string
OBJECT batch.failed
{
  "id": "evt_abc123",
  "type": "batch.failed",
  "created_at": 1719168000,
  "data": {
    "id": "batch_abc123"
  }
}

fine_tuning.job.succeeded
The Unix timestamp (in seconds) of when the fine-tuning job succeeded.
created_at integer
OBJECT fine_tuning.job.succeeded
 
{
  "id": "evt_abc123",
  "type": "fine_tuning.job.succeeded",
  "created_at": 1719168000,


<!-- Page 66 -->
Sent when a fine-tuning job has failed.
Event data payload.
## Show properties
data object
The unique ID of the event.
id string
The object of the event. Always event .
object string
The type of the event. Always fine_tuning.job.succeeded .
type string
 
  "data": {
    "id": "ftjob_abc123"
  }
}

fine_tuning.job.failed
The Unix timestamp (in seconds) of when the fine-tuning job failed.
created_at integer
Event data payload.
## Show properties
data object
The unique ID of the event.
id string
object string
OBJECT fine_tuning.job.failed
{
  "id": "evt_abc123",
  "type": "fine_tuning.job.failed",
  "created_at": 1719168000,
  "data": {
    "id": "ftjob_abc123"
  }
}


<!-- Page 67 -->
Sent when a fine-tuning job has been cancelled.
The object of the event. Always event .
The type of the event. Always fine_tuning.job.failed .
type string
fine_tuning.job.cancelled
The Unix timestamp (in seconds) of when the fine-tuning job was cancelled.
created_at integer
Event data payload.
## Show properties
data object
The unique ID of the event.
id string
The object of the event. Always event .
object string
The type of the event. Always fine_tuning.job.cancelled .
type string
OBJECT fine_tuning.job.cancelled
{
  "id": "evt_abc123",
  "type": "fine_tuning.job.cancelled",
  "created_at": 1719168000,
  "data": {
    "id": "ftjob_abc123"
  }
}


<!-- Page 68 -->
Sent when an eval run has succeeded.
Sent when an eval run has failed.
eval.run.succeeded
The Unix timestamp (in seconds) of when the eval run succeeded.
created_at integer
Event data payload.
## Show properties
data object
The unique ID of the event.
id string
The object of the event. Always event .
object string
The type of the event. Always eval.run.succeeded .
type string
OBJECT eval.run.succeeded
{
  "id": "evt_abc123",
  "type": "eval.run.succeeded",
  "created_at": 1719168000,
  "data": {
    "id": "evalrun_abc123"
  }
}

eval.run.failed
The Unix timestamp (in seconds) of when the eval run failed.
created_at integer
OBJECT eval.run.failed
 
{
  "id": "evt_abc123",
  "type": "eval.run.failed",


<!-- Page 69 -->
Sent when an eval run has been canceled.
Event data payload.
## Show properties
data object
The unique ID of the event.
id string
The object of the event. Always event .
object string
The type of the event. Always eval.run.failed .
type string
 
  "created_at": 1719168000,
  "data": {
    "id": "evalrun_abc123"
  }
}

eval.run.canceled
The Unix timestamp (in seconds) of when the eval run was canceled.
created_at integer
Event data payload.
## Show properties
data object
The unique ID of the event.
id string
object string
OBJECT eval.run.canceled
{
  "id": "evt_abc123",
  "type": "eval.run.canceled",
  "created_at": 1719168000,
  "data": {
    "id": "evalrun_abc123"
  }
}


<!-- Page 70 -->
Learn how to turn audio into text or text into audio.
Related guide: Speech to text
POST https://api.openai.com/v1/audio/speech
Generates audio from the input text.
## Request body
The object of the event. Always event .
The type of the event. Always eval.run.canceled .
type string
## Audio
Create speech
The text to generate audio for. The maximum length is 4096 characters.
input string
## Required
One of the available TTS models: tts-1 , tts-1-hd  or gpt-4o-mini-tts .
model string
## Required
voice string
Required
Default
SSE Stream Format
Example request
python
 
 
from pathlib import Path
import openai
speech_file_path = Path(__file__).parent / "spee
with openai.audio.speech.with_streaming_response
  model="gpt-4o-mini-tts",
  voice="alloy",
  input="The quick brown fox jumped over the laz
) as response:
  response.stream_to_file(speech_file_path)


<!-- Page 71 -->
## Returns
POST https://api.openai.com/v1/audio/transcriptions
Transcribes audio into the input language.
The voice to use when generating the audio. Supported voices are alloy , ash ,
ballad , coral , echo , fable , onyx , nova , sage , shimmer , and verse .
Previews of the voices are available in the Text to speech guide.
Control the voice of your generated audio with additional instructions. Does not work
with tts-1  or tts-1-hd .
instructions string
## Optional
The format to audio in. Supported formats are mp3 , opus , aac , flac , wav ,
and pcm .
response_format string
## Optional
Defaults to mp3
The speed of the generated audio. Select a value from 0.25  to 4.0 . 1.0  is the
default.
speed number
## Optional
Defaults to 1
The format to stream the audio in. Supported formats are sse  and audio . sse  is
not supported for tts-1  or tts-1-hd .
stream_format string
## Optional
Defaults to audio
The audio file content or a stream of audio events.
## Create transcription
Default
Streaming
Logprobs
Word timestamps

<!-- Page 72 -->
## Request body
The audio file object (not file name) to transcribe, in one of these formats: flac, mp3,
mp4, mpeg, mpga, m4a, ogg, wav, or webm.
file
file
## Required
ID of the model to use. The options are gpt-4o-transcribe ,
gpt-4o-mini-transcribe , and whisper-1  (which is powered by our open source
Whisper V2 model).
model string
## Required
Controls how the audio is cut into chunks. When set to "auto" , the server first
normalizes loudness and then uses voice activity detection (VAD) to choose
boundaries. server_vad  object can be provided to tweak VAD detection parameters
manually. If unset, the audio is transcribed as a single block.
## Show possible types
chunking_strategy
"auto" or object
## Optional
Additional information to include in the transcription response. logprobs  will return
the log probabilities of the tokens in the response to understand the model's
confidence in the transcription. logprobs  only works with response_format set to
json  and only with the models gpt-4o-transcribe  and
gpt-4o-mini-transcribe .
include[]
array
## Optional
The language of the input audio. Supplying the input language in ISO-639-1 (e.g. en )
format will improve accuracy and latency.
language string
## Optional
An optional text to guide the model's style or continue a previous audio segment. The
prompt should match the audio language.
prompt string
## Optional
response_format string
## Optional
Defaults to json
Example request
python
from openai import OpenAI
client = OpenAI()
audio_file = open("speech.mp3", "rb")
transcript = client.audio.transcriptions.create(
  model="gpt-4o-transcribe",
  file=audio_file
)

## Response
 
 
{
  "text": "Imagine the wildest idea that you've 
  "usage": {
    "type": "tokens",
    "input_tokens": 14,
    "input_token_details": {
      "text_tokens": 0,
      "audio_tokens": 14
    },
    "output_tokens": 45,
    "total_tokens": 59
  }
}


<!-- Page 73 -->
## Returns
The format of the output, in one of these options: json , text , srt ,
verbose_json , or vtt . For gpt-4o-transcribe  and gpt-4o-mini-transcribe ,
the only supported format is json .
If set to true, the model response data will be streamed to the client as it is generated
using server-sent events. See the Streaming section of the Speech-to-Text guide for
more information.
Note: Streaming is not supported for the whisper-1  model and will be ignored.
stream boolean or null
## Optional
Defaults to false
The sampling temperature, between 0 and 1. Higher values like 0.8 will make the output
more random, while lower values like 0.2 will make it more focused and deterministic. If
set to 0, the model will use log probability to automatically increase the temperature
until certain thresholds are hit.
temperature number
## Optional
Defaults to 0
The timestamp granularities to populate for this transcription. response_format
must be set verbose_json  to use timestamp granularities. Either or both of these
options are supported: word , or segment . Note: There is no additional latency for
segment timestamps, but generating word timestamps incurs additional latency.
timestamp_granularities[]
array
## Optional
Defaults to segment
The transcription object, a verbose transcription object or a
stream of transcript events.

<!-- Page 74 -->
POST https://api.openai.com/v1/audio/translations
Translates audio into English.
## Request body
Returns
Create translation
The audio file object (not file name) translate, in one of these formats: flac, mp3, mp4,
mpeg, mpga, m4a, ogg, wav, or webm.
file
file
## Required
ID of the model to use. Only whisper-1  (which is powered by our open source
Whisper V2 model) is currently available.
model string or "whisper-1"
## Required
An optional text to guide the model's style or continue a previous audio segment. The
prompt should be in English.
prompt string
## Optional
The format of the output, in one of these options: json , text , srt ,
verbose_json , or vtt .
response_format string
## Optional
Defaults to json
The sampling temperature, between 0 and 1. Higher values like 0.8 will make the output
more random, while lower values like 0.2 will make it more focused and deterministic. If
set to 0, the model will use log probability to automatically increase the temperature
until certain thresholds are hit.
temperature number
## Optional
Defaults to 0
## Example request
python
from openai import OpenAI
client = OpenAI()
audio_file = open("speech.mp3", "rb")
transcript = client.audio.translations.create(
  model="whisper-1",
  file=audio_file
)

## Response
 
 
{
  "text": "Hello, my name is Wolfgang and I come 
}


<!-- Page 75 -->
Represents a transcription response returned by model, based on the
provided input.
Represents a verbose json transcription response returned by model, based
on the provided input.
The translated text.
The transcription object (JSON)
The log probabilities of the tokens in the transcription. Only returned with the models
gpt-4o-transcribe  and gpt-4o-mini-transcribe  if logprobs  is added to the
include  array.
## Show properties
logprobs array
The transcribed text.
text string
Token usage statistics for the request.
## Show possible types
usage object
OBJECT The transcription object (JSON)
 
 
{
  "text": "Imagine the wildest idea that you've 
  "usage": {
    "type": "tokens",
    "input_tokens": 14,
    "input_token_details": {
      "text_tokens": 10,
      "audio_tokens": 4
    },
    "output_tokens": 101,
    "total_tokens": 115
  }
}

The transcription object (Verbose JSON)
OBJECT The transcription object (Verbose JSON)

<!-- Page 76 -->
Emitted for each chunk of audio data generated during speech synthesis.
The duration of the input audio.
duration number
The language of the input audio.
language string
Segments of the transcribed text and their corresponding details.
## Show properties
segments array
The transcribed text.
text string
Usage statistics for models billed by audio input duration.
## Show properties
usage object
Extracted words and their corresponding timestamps.
## Show properties
words array
{
  "task": "transcribe",
  "language": "english",
  "duration": 8.470000267028809,
  "text": "The beach was a popular spot on a hot
  "segments": [
    {
      "id": 0,
      "seek": 0,
      "start": 0.0,
      "end": 3.319999933242798,
      "text": " The beach was a popular spot on 
      "tokens": [
        50364, 440, 7534, 390, 257, 3743, 4008, 
      ],
      "temperature": 0.0,
      "avg_logprob": -0.2860786020755768,
      "compression_ratio": 1.2363636493682861,
      "no_speech_prob": 0.00985979475080967
    },
    ...
  ],
  "usage": {
    "type": "duration",
    "seconds": 9
  }

Stream Event (speech.audio.delta)
A chunk of Base64-encoded audio data.
audio string
type string
OBJECT Stream Event (speech.audio.delta)
{
  "type": "speech.audio.delta",
  "audio": "base64-encoded-audio-data"
}


<!-- Page 77 -->
## Emitted when the speech synthesis is complete and all audio has been
streamed.
Emitted when there is an additional text delta. This is also the first event
emitted when the transcription starts. Only emitted when you
create a transcription with the Stream  parameter set to true .
The type of the event. Always speech.audio.delta .
Stream Event (speech.audio.done)
The type of the event. Always speech.audio.done .
type string
Token usage statistics for the request.
## Show properties
usage object
OBJECT Stream Event (speech.audio.done)
{
  "type": "speech.audio.done",
  "usage": {
    "input_tokens": 14,
    "output_tokens": 101,
    "total_tokens": 115
  }
}

Stream Event (transcript.text.delta)
The text delta that was additionally transcribed.
delta string
The log probabilities of the delta. Only included if you create a transcription with the
include[]  parameter set to logprobs .
logprobs array
OBJECT Stream Event (transcript.text.delta)
{
  "type": "transcript.text.delta",
  "delta": " wonderful"
}


<!-- Page 78 -->
Emitted when the transcription is complete. Contains the complete
transcription text. Only emitted when you create a transcription with the
Stream  parameter set to true .
## Show properties
The type of the event. Always transcript.text.delta .
type string
Stream Event (transcript.text.done)
The log probabilities of the individual tokens in the transcription. Only included if you
create a transcription with the include[]  parameter set to logprobs .
## Show properties
logprobs array
The text that was transcribed.
text string
The type of the event. Always transcript.text.done .
type string
Usage statistics for models billed by token usage.
## Show properties
usage object
OBJECT Stream Event (transcript.text.done)
 
 
{
  "type": "transcript.text.done",
  "text": "I see skies of blue and clouds of whi
  "usage": {
    "type": "tokens",
    "input_tokens": 14,
    "input_token_details": {
      "text_tokens": 10,
      "audio_tokens": 4
    },
    "output_tokens": 31,
    "total_tokens": 45
  }
}


<!-- Page 79 -->
Given a prompt and/or an input image, the model will generate a new image. Related guide:
## Image generation
POST https://api.openai.com/v1/images/generations
Creates an image given a prompt. Learn more.
## Request body
Images
Create image
A text description of the desired image(s). The maximum length is 32000 characters
for gpt-image-1 , 1000 characters for dall-e-2  and 4000 characters for
dall-e-3 .
prompt string
## Required
Allows to set transparency for the background of the generated image(s). This
parameter is only supported for gpt-image-1 . Must be one of transparent ,
opaque  or auto  (default value). When auto  is used, the model will automatically
determine the best background for the image.
If transparent , the output format needs to support transparency, so it should be set
to either png  (default value) or webp .
background string or null
## Optional
Defaults to auto
model string
Optional
Defaults to dall-e-2
## Generate image
Streaming
Example request
python
 
 
import base64
from openai import OpenAI
client = OpenAI()
img = client.images.generate(
    model="gpt-image-1",
    prompt="A cute baby sea otter",
    n=1,
    size="1024x1024"
)
image_bytes = base64.b64decode(img.data[0].b64_j
with open("output.png", "wb") as f:
    f.write(image_bytes)

## Response
 
{
  "created": 1713833628,
  "data": [
    {
      "b64_json": "..."


<!-- Page 80 -->
The model to use for image generation. One of dall-e-2 , dall-e-3 , or
gpt-image-1 . Defaults to dall-e-2  unless a parameter specific to gpt-image-1
is used.
Control the content-moderation level for images generated by gpt-image-1 . Must be
either low  for less restrictive filtering or auto  (default value).
moderation string or null
## Optional
Defaults to auto
The number of images to generate. Must be between 1 and 10. For dall-e-3 , only
n=1  is supported.
n integer or null
## Optional
Defaults to 1
The compression level (0-100%) for the generated images. This parameter is only
supported for gpt-image-1  with the webp  or jpeg  output formats, and defaults
to 100.
output_compression integer or null
## Optional
Defaults to 100
The format in which the generated images are returned. This parameter is only
supported for gpt-image-1 . Must be one of png , jpeg , or webp .
output_format string or null
## Optional
Defaults to png
The number of partial images to generate. This parameter is used for streaming
responses that return partial images. Value must be between 0 and 3. When set to 0,
the response will be a single image sent in one streaming event.
## Note that the final image may be sent before the full number of partial images are
generated if the full image is generated more quickly.
partial_images integer or null
## Optional
Defaults to 0
The quality of the image that will be generated.
quality string or null
## Optional
Defaults to auto
auto  (default value) will automatically select the best quality for the given
model.
high , medium  and low  are supported for gpt-image-1 .
 
    }
  ],
  "usage": {
    "total_tokens": 100,
    "input_tokens": 50,
    "output_tokens": 50,
    "input_tokens_details": {
      "text_tokens": 10,
      "image_tokens": 40
    }
  }
}


<!-- Page 81 -->
## Returns
hd  and standard  are supported for dall-e-3 .
standard  is the only option for dall-e-2 .
The format in which generated images with dall-e-2  and dall-e-3  are returned.
Must be one of url  or b64_json . URLs are only valid for 60 minutes after the
image has been generated. This parameter isn't supported for gpt-image-1  which
will always return base64-encoded images.
response_format string or null
## Optional
Defaults to url
The size of the generated images. Must be one of 1024x1024 , 1536x1024
(landscape), 1024x1536  (portrait), or auto  (default value) for gpt-image-1 , one of
256x256 , 512x512 , or 1024x1024  for dall-e-2 , and one of 1024x1024 ,
1792x1024 , or 1024x1792  for dall-e-3 .
size string or null
## Optional
Defaults to auto
Generate the image in streaming mode. Defaults to false . See the
Image generation guide for more information. This parameter is only supported for
gpt-image-1 .
stream boolean or null
## Optional
Defaults to false
The style of the generated images. This parameter is only supported for dall-e-3 .
Must be one of vivid  or natural . Vivid causes the model to lean towards
generating hyper-real and dramatic images. Natural causes the model to produce
more natural, less hyper-real looking images.
style string or null
## Optional
Defaults to vivid
A unique identifier representing your end-user, which can help OpenAI to monitor and
detect abuse. Learn more.
user string
## Optional

<!-- Page 82 -->
POST https://api.openai.com/v1/images/edits
## Creates an edited or extended image given one or more source images and
a prompt. This endpoint only supports gpt-image-1  and dall-e-2 .
## Request body
Returns an image object.
## Create image edit
The image(s) to edit. Must be a supported image file or an array of images.
For gpt-image-1 , each image should be a png , webp , or jpg  file less than
50MB. You can provide up to 16 images.
For dall-e-2 , you can only provide one image, and it should be a square png  file
less than 4MB.
image string or array
## Required
A text description of the desired image(s). The maximum length is 1000 characters for
dall-e-2 , and 32000 characters for gpt-image-1 .
prompt string
## Required
Allows to set transparency for the background of the generated image(s). This
parameter is only supported for gpt-image-1 . Must be one of transparent ,
opaque  or auto  (default value). When auto  is used, the model will automatically
determine the best background for the image.
background string or null
## Optional
Defaults to auto
Edit image
Streaming
Example request
python
 
import base64
from openai import OpenAI
client = OpenAI()
prompt = """
## Generate a photorealistic image of a gift baske
labeled 'Relax & Unwind' with a ribbon and hand
containing all the items in the reference pictu
"""
result = client.images.edit(
    model="gpt-image-1",
    image=[
        open("body-lotion.png", "rb"),
        open("bath-bomb.png", "rb"),
        open("incense-kit.png", "rb"),
        open("soap.png", "rb"),
    ],
    prompt=prompt
)
image_base64 = result.data[0].b64_json
image_bytes = base64.b64decode(image_base64)
# Save the image to a file


<!-- Page 83 -->
If transparent , the output format needs to support transparency, so it should be set
to either png  (default value) or webp .
Control how much effort the model will exert to match the style and features,
especially facial features, of input images. This parameter is only supported for
gpt-image-1 . Supports high  and low . Defaults to low .
input_fidelity string or null
## Optional
Defaults to low
An additional image whose fully transparent areas (e.g. where alpha is zero) indicate
where image  should be edited. If there are multiple images provided, the mask will be
applied on the first image. Must be a valid PNG file, less than 4MB, and have the same
dimensions as image .
mask
file
## Optional
The model to use for image generation. Only dall-e-2  and gpt-image-1  are
supported. Defaults to dall-e-2  unless a parameter specific to gpt-image-1  is
used.
model string
## Optional
Defaults to dall-e-2
The number of images to generate. Must be between 1 and 10.
n integer or null
## Optional
Defaults to 1
The compression level (0-100%) for the generated images. This parameter is only
supported for gpt-image-1  with the webp  or jpeg  output formats, and defaults
to 100.
output_compression integer or null
## Optional
Defaults to 100
The format in which the generated images are returned. This parameter is only
supported for gpt-image-1 . Must be one of png , jpeg , or webp . The default
value is png .
output_format string or null
## Optional
Defaults to png
partial_images integer or null
## Optional
Defaults to 0
 
with open("gift-basket.png", "wb") as f:
    f.write(image_bytes)


<!-- Page 84 -->
## Returns
The number of partial images to generate. This parameter is used for streaming
responses that return partial images. Value must be between 0 and 3. When set to 0,
the response will be a single image sent in one streaming event.
## Note that the final image may be sent before the full number of partial images are
generated if the full image is generated more quickly.
The quality of the image that will be generated. high , medium  and low  are only
supported for gpt-image-1 . dall-e-2  only supports standard  quality. Defaults
to auto .
quality string or null
## Optional
Defaults to auto
The format in which the generated images are returned. Must be one of url  or
b64_json . URLs are only valid for 60 minutes after the image has been generated.
This parameter is only supported for dall-e-2 , as gpt-image-1  will always return
base64-encoded images.
response_format string or null
## Optional
Defaults to url
The size of the generated images. Must be one of 1024x1024 , 1536x1024
(landscape), 1024x1536  (portrait), or auto  (default value) for gpt-image-1 , and
one of 256x256 , 512x512 , or 1024x1024  for dall-e-2 .
size string or null
## Optional
Defaults to 1024x1024
Edit the image in streaming mode. Defaults to false . See the
Image generation guide for more information.
stream boolean or null
## Optional
Defaults to false
A unique identifier representing your end-user, which can help OpenAI to monitor and
detect abuse. Learn more.
user string
## Optional

<!-- Page 85 -->
POST https://api.openai.com/v1/images/variations
Creates a variation of a given image. This endpoint only supports dall-e-2 .
## Request body
Returns an image object.
## Create image variation
The image to use as the basis for the variation(s). Must be a valid PNG file, less than
4MB, and square.
image
file
## Required
The model to use for image generation. Only dall-e-2  is supported at this time.
model string or "dall-e-2"
## Optional
Defaults to dall-e-2
The number of images to generate. Must be between 1 and 10.
n integer or null
## Optional
Defaults to 1
The format in which the generated images are returned. Must be one of url  or
b64_json . URLs are only valid for 60 minutes after the image has been generated.
response_format string or null
## Optional
Defaults to url
The size of the generated images. Must be one of 256x256 , 512x512 , or
1024x1024 .
size string or null
## Optional
Defaults to 1024x1024
user string
## Optional
Example request
python
from openai import OpenAI
client = OpenAI()
response = client.images.create_variation(
  image=open("image_edit_original.png", "rb"),
  n=2,
  size="1024x1024"
)

## Response
{
  "created": 1589478378,
  "data": [
    {
      "url": "https://..."
    },
    {
      "url": "https://..."
    }
  ]
}


<!-- Page 86 -->
## Returns
The response from the image generation endpoint.
A unique identifier representing your end-user, which can help OpenAI to monitor and
detect abuse. Learn more.
Returns a list of image objects.
## The image generation response
The background parameter used for the image generation. Either transparent  or
opaque .
background string
The Unix timestamp (in seconds) of when the image was created.
created integer
The list of generated images.
## Show properties
data array
The output format of the image generation. Either png , webp , or jpeg .
output_format string
The quality of the image generated. Either low , medium , or high .
quality string
## OBJECT The image generation response
{
  "created": 1713833628,
  "data": [
    {
      "b64_json": "..."
    }
  ],
  "background": "transparent",
  "output_format": "png",
  "size": "1024x1024",
  "quality": "high",
  "usage": {
    "total_tokens": 100,
    "input_tokens": 50,
    "output_tokens": 50,
    "input_tokens_details": {
      "text_tokens": 10,
      "image_tokens": 40
    }
  }
}


<!-- Page 87 -->
Stream image generation and editing in real time with server-sent events.
Learn more about image streaming.
## Emitted when a partial image is available during image generation
streaming.
The size of the image generated. Either 1024x1024 , 1024x1536 , or 1536x1024 .
size string
For gpt-image-1  only, the token usage information for the image generation.
## Show properties
usage object
Image Streaming
image_generation.partial_image
Base64-encoded partial image data, suitable for rendering as an image.
b64_json string
The background setting for the requested image.
background string
created_at integer
OBJECT image_generation.partial_image
 
{
  "type": "image_generation.partial_image",
  "b64_json": "...",
  "created_at": 1620000000,
  "size": "1024x1024",
  "quality": "high",
  "background": "transparent",
  "output_format": "png",


<!-- Page 88 -->
## Emitted when image generation has completed and the final image is
available.
The Unix timestamp when the event was created.
The output format for the requested image.
output_format string
0-based index for the partial image (streaming).
partial_image_index integer
The quality setting for the requested image.
quality string
The size of the requested image.
size string
The type of the event. Always image_generation.partial_image .
type string
 
  "partial image index": 0

image_generation.completed
Base64-encoded image data, suitable for rendering as an image.
b64_json string
The background setting for the generated image.
background string
created_at integer
OBJECT image_generation.completed
 
{
  "type": "image_generation.completed",
  "b64_json": "...",
  "created_at": 1620000000,
  "size": "1024x1024",
  "quality": "high",
  "background": "transparent",
  "output_format": "png",
  "usage": {
    "total_tokens": 100,
    "input_tokens": 50,


<!-- Page 89 -->
Emitted when a partial image is available during image editing streaming.
The Unix timestamp when the event was created.
The output format for the generated image.
output_format string
The quality setting for the generated image.
quality string
The size of the generated image.
size string
The type of the event. Always image_generation.completed .
type string
For gpt-image-1  only, the token usage information for the image generation.
## Show properties
usage object
 
    "output_tokens": 50,
    "input_tokens_details": {
      "text_tokens": 10,
      "image_tokens": 40
    }
  }
}

image_edit.partial_image
Base64-encoded partial image data, suitable for rendering as an image.
b64_json string
background string
OBJECT image_edit.partial_image
 
{
  "type": "image_edit.partial_image",
  "b64_json": "...",
  "created_at": 1620000000,
  "size": "1024x1024",
  "quality": "high",


<!-- Page 90 -->
Emitted when image editing has completed and the final image is available.
The background setting for the requested edited image.
The Unix timestamp when the event was created.
created_at integer
The output format for the requested edited image.
output_format string
0-based index for the partial image (streaming).
partial_image_index integer
The quality setting for the requested edited image.
quality string
The size of the requested edited image.
size string
The type of the event. Always image_edit.partial_image .
type string
 
  "background": "transparent",
  "output_format": "png",
  "partial_image_index": 0
}

image_edit.completed
Base64-encoded final edited image data, suitable for rendering as an image.
b64_json string
The background setting for the edited image.
background string
OBJECT image_edit.completed
 
{
  "type": "image_edit.completed",
  "b64_json": "...",
  "created_at": 1620000000,
  "size": "1024x1024",
  "quality": "high",
  "background": "transparent",


<!-- Page 91 -->
## Get a vector representation of a given input that can be easily consumed by machine learning models and
algorithms. Related guide: Embeddings
The Unix timestamp when the event was created.
created_at integer
The output format for the edited image.
output_format string
The quality setting for the edited image.
quality string
The size of the edited image.
size string
The type of the event. Always image_edit.completed .
type string
For gpt-image-1  only, the token usage information for the image generation.
## Show properties
usage object
 
  "output_format": "png",
  "usage": {
    "total_tokens": 100,
    "input_tokens": 50,
    "output_tokens": 50,
    "input_tokens_details": {
      "text_tokens": 10,
      "image_tokens": 40
    }
  }
}

## Embeddings
Create embeddings

<!-- Page 92 -->
POST https://api.openai.com/v1/embeddings
Creates an embedding vector representing the input text.
## Request body
Returns
Input text to embed, encoded as a string or array of tokens. To embed multiple inputs
in a single request, pass an array of strings or array of token arrays. The input must not
exceed the max input tokens for the model (8192 tokens for all embedding models),
cannot be an empty string, and any array must be 2048 dimensions or less.
Example Python code for counting tokens. In addition to the per-input token limit, all
embedding models enforce a maximum of 300,000 tokens summed across all inputs
in a single request.
input string or array
## Required
ID of the model to use. You can use the List models API to see all of your available
models, or see our Model overview for descriptions of them.
model string
## Required
The number of dimensions the resulting output embeddings should have. Only
supported in text-embedding-3  and later models.
dimensions integer
## Optional
The format to return the embeddings in. Can be either float  or base64 .
encoding_format string
## Optional
Defaults to float
A unique identifier representing your end-user, which can help OpenAI to monitor and
detect abuse. Learn more.
user string
## Optional
Example request
python
 
 
from openai import OpenAI
client = OpenAI()
client.embeddings.create(
  model="text-embedding-ada-002",
  input="The food was delicious and the waiter...
  encoding_format="float"
)

## Response
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [
        0.0023064255,
        -0.009327292,
        .... (1536 floats total for ada-002)
        -0.0028842222,
      ],
      "index": 0
    }
  ],
  "model": "text-embedding-ada-002",
  "usage": {
    "prompt_tokens": 8,
    "total_tokens": 8
  }
}


<!-- Page 93 -->
Represents an embedding vector returned by embedding endpoint.
Create, manage, and run evals in the OpenAI platform. Related guide: Evals
A list of embedding objects.
## The embedding object
The embedding vector, which is a list of floats. The length of vector depends on the
model as listed in the embedding guide.
embedding array
The index of the embedding in the list of embeddings.
index integer
The object type, which is always "embedding".
object string
## OBJECT The embedding object
{
  "object": "embedding",
  "embedding": [
    0.0023064255,
    -0.009327292,
    .... (1536 floats total for ada-002)
    -0.0028842222,
  ],
  "index": 0
}

## Evals
Create eval

<!-- Page 94 -->
POST https://api.openai.com/v1/evals
Create the structure of an evaluation that can be used to test a model's
performance. An evaluation is a set of testing criteria and the config for a
data source, which dictates the schema of the data used in the evaluation.
After creating an evaluation, you can run it on different models and model
parameters. We support several types of graders and datasources. For more
information, see the Evals guide.
## Request body
The configuration for the data source used for the evaluation runs. Dictates the
schema of the data used in the evaluation.
## Show possible types
data_source_config object
## Required
A list of graders for all eval runs in this group. Graders can reference variables in the
data source using double curly braces notation, like {{item.variable_name}} . To
reference the model's output, use the sample  namespace (ie,
{{sample.output_text}} ).
## Show possible types
testing_criteria array
## Required
Set of 16 key-value pairs that can be attached to an object. This can be useful for
storing additional information about the object in a structured format, and querying for objects via API or the dashboard.
Keys are strings with a maximum length of 64 characters. Values are strings with a
maximum length of 512 characters.
metadata
map
## Optional
The name of the evaluation.
name string
## Optional
Example request
python
 
 
from openai import OpenAI
client = OpenAI()
eval_obj = client.evals.create(
  name="Sentiment",
  data_source_config={
    "type": "stored_completions",
    "metadata": {"usecase": "chatbot"}
  },
  testing_criteria=[
    {
      "type": "label_model",
      "model": "o3-mini",
      "input": [
        {"role": "developer", "content": "Classi
        {"role": "user", "content": "Statement: 
      ],
      "passing_labels": ["positive"],
      "labels": ["positive", "neutral", "negativ
      "name": "Example label grader"
    }
  ]
)
print(eval_obj)

## Response
 
{
  "object": "eval",
  "id": "eval_67b7fa9a81a88190ab4aa417e397ea21"
  "data_source_config": {
    "type": "stored_completions",
    "metadata": {
      "usecase": "chatbot"
    },
    "schema": {
      "type": "object",


<!-- Page 95 -->
## Returns
GET https://api.openai.com/v1/evals/{eval_id}
Get an evaluation by ID.
## Path parameters
Returns
The created Eval object.
 
      "properties": {
        "item": {
          "type": "object"
        },
        "sample": {
          "type": "object"
        }
      },
      "required": [
        "item",
        "sample"
      ]
  },
  "testing_criteria": [
    {
      "name": "Example label grader",
      "type": "label_model",
      "model": "o3-mini",
      "input": [
        {
          "type": "message",
          "role": "developer",
          "content": {
            "type": "input_text",
            "text": "Classify the sentiment of 
          }
        },
        {
          "type": "message",
          "role": "user",
          "content": {
            "type": "input_text",
            "text": "Statement: {{item.input}}"
          }
        }
      ],
      "passing_labels": [
        "positive"
      ],
      "labels": [

## Get an eval
The ID of the evaluation to retrieve.
eval_id string
## Required
The Eval object matching the specified ID.
## Example request
python
 
 
from openai import OpenAI
client = OpenAI()
eval_obj = client.evals.retrieve("eval_67abd54d9b
print(eval_obj)

## Response
 
{
  "object": "eval",
  "id": "eval_67abd54d9b0081909a86353f6fb9317a"
  "data_source_config": {
    "type": "custom",
    "schema": {
      "type": "object",
      "properties": {
        "item": {
          "type": "object",
          "properties": {
            "input": {
              "type": "string"
            },
            "ground_truth": {
              "type": "string"


<!-- Page 96 -->
POST https://api.openai.com/v1/evals/{eval_id}
Update certain properties of an evaluation.
## Path parameters
Request body
 
        "positive",
        "neutral",
        "negative"
      ]
    }
  ],
  "name": "Sentiment",
  "created_at": 1740110490,
  "metadata": {
    "description": "An eval for sentiment analy
  }
}

            }
          },
          "required": [
            "input",
            "ground_truth"
          ]
        }
      },
      "required": [
        "item"
      ]
    }
  },
  "testing_criteria": [
    {
      "name": "String check",
      "id": "String check-2eaf2d8d-d649-4335-81
      "type": "string_check",
      "input": "{{item.input}}",
      "reference": "{{item.ground_truth}}",
      "operation": "eq"
    }
  ],
  "name": "External Data Eval",
  "created_at": 1739314509,
  "metadata": {},
}

## Update an eval
The ID of the evaluation to update.
eval_id string
## Required
Set of 16 key-value pairs that can be attached to an object. This can be useful for
storing additional information about the object in a structured format, and querying for objects via API or the dashboard.
Keys are strings with a maximum length of 64 characters. Values are strings with a
maximum length of 512 characters.
metadata
map
## Optional
name string
Optional
Example request
python
 
 
from openai import OpenAI
client = OpenAI()
updated_eval = client.evals.update(
  "eval_67abd54d9b0081909a86353f6fb9317a",
  name="Updated Eval",
  metadata={"description": "Updated description"}
)
print(updated_eval)

## Response
 
{
  "object": "eval",
  "id": "eval_67abd54d9b0081909a86353f6fb9317a"
  "data_source_config": {
    "type": "custom",
    "schema": {
      "type": "object",
      "properties": {
        "item": {


<!-- Page 97 -->
## Returns
DELETE https://api.openai.com/v1/evals/{eval_id}
Delete an evaluation.
## Path parameters
Returns
Rename the evaluation.
The Eval object matching the updated version.
 
          "type": "object",
          "properties": {
            "input": {
              "type": "string"
            },
            "ground_truth": {
              "type": "string"
            }
          },
          "required": [
            "input",
            "ground_truth"
          ]
        }
      },
      "required": [
        "item"
      ]
    }
  },
  "testing_criteria": [
    {
      "name": "String check",
      "id": "String check-2eaf2d8d-d649-4335-81
      "type": "string_check",
      "input": "{{item.input}}",
      "reference": "{{item.ground_truth}}",
      "operation": "eq"
    }
  ],
  "name": "Updated Eval",
  "created_at": 1739314509,
  "metadata": {"description": "Updated descript
}

## Delete an eval
The ID of the evaluation to delete.
eval_id string
## Required
A deletion confirmation object.
## Example request
python
from openai import OpenAI
client = OpenAI()
deleted = client.evals.delete("eval_abc123")
print(deleted)

## Response
{
  "object": "eval.deleted",
  "deleted": true,
  "eval_id": "eval_abc123"
}


<!-- Page 98 -->
GET https://api.openai.com/v1/evals
List evaluations for a project.
## Query parameters
Returns
List evals
Identifier for the last eval from the previous pagination request.
after string
## Optional
Number of evals to retrieve.
limit integer
## Optional
Defaults to 20
Sort order for evals by timestamp. Use asc  for ascending order or desc  for
descending order.
order string
## Optional
Defaults to asc
Evals can be ordered by creation time or last updated time. Use created_at  for
creation time or updated_at  for last updated time.
order_by string
## Optional
Defaults to created_at
A list of evals matching the specified filters.
## Example request
python
from openai import OpenAI
client = OpenAI()
evals = client.evals.list(limit=1)
print(evals)

## Response
 
{
  "object": "list",
  "data": [
    {
      "id": "eval_67abd54d9b0081909a86353f6fb93
      "object": "eval",
      "data_source_config": {
        "type": "stored_completions",
        "metadata": {
          "usecase": "push_notifications_summar
        },
        "schema": {
          "type": "object",
          "properties": {
            "item": {
              "type": "object"
            },
            "sample": {
              "type": "object"
            }
          },
          "required": [
            "item",
            "sample"
          ]
        }


<!-- Page 99 -->
GET https://api.openai.com/v1/evals/{eval_id}/runs
Get a list of runs for an evaluation.
## Path parameters
Query parameters
Returns
 
      },
      "testing_criteria": [
        {
          "name": "Push Notification Summary Gr
          "id": "Push Notification Summary Grad
          "type": "label_model",
          "model": "o3-mini",
          "input": [
            {
              "type": "message",
              "role": "developer",
              "content": {
                "type": "input_text",
                "text": "\nLabel the following 
              }
            },
            {
              "type": "message",
              "role": "user",
              "content": {
                "type": "input_text",
                "text": "\nPush notifications: 
              }
            }
          ],
          "passing_labels": [
            "correct"
          ],
          "labels": [
            "correct",
            "incorrect"
          ],
          "sampling_params": null
        }
      ],
      "name": "Push Notification Summary Grader
      "created_at": 1739314509,
      "metadata": {
        "description": "A stored completions ev
      }

## Get eval runs
The ID of the evaluation to retrieve runs for.
eval_id string
## Required
Identifier for the last run from the previous pagination request.
after string
## Optional
Number of runs to retrieve.
limit integer
## Optional
Defaults to 20
Sort order for runs by timestamp. Use asc  for ascending order or desc  for
descending order. Defaults to asc .
order string
## Optional
Defaults to asc
Filter runs by status. One of queued  | in_progress  | failed  | completed  |
canceled .
status string
## Optional
Example request
python
 
 
from openai import OpenAI
client = OpenAI()
runs = client.evals.runs.list("egroup_67abd54d9b0
print(runs)

## Response
 
{
  "object": "list",
  "data": [
    {
      "object": "eval.run",
      "id": "evalrun_67e0c7d31560819090d60c0780
      "eval_id": "eval_67e0c726d560819083f19a95
      "report_url": "https://platform.openai.co
      "status": "completed",
      "model": "o3-mini",
      "name": "bulk_with_negative_examples_o3-m
      "created_at": 1742784467,
      "result_counts": {
        "total": 1,
        "errored": 0,
        "failed": 0,
        "passed": 1
      },
      "per_model_usage": [
        {
          "model_name": "o3-mini",
          "invocation_count": 1,


<!-- Page 100 -->
GET https://api.openai.com/v1/evals/{eval_id}/runs/{run_id}
Get an evaluation run by ID.
## Path parameters
Returns
 
    }
  ],
  "first_id": "eval_67abd54d9b0081909a86353f6fb
  "last_id": "eval_67aa884cf6688190b58f657d4441
  "has_more": true

A list of EvalRun objects matching the specified ID.
 
          "prompt_tokens": 563,
          "completion_tokens": 874,
          "total_tokens": 1437,
          "cached_tokens": 0
        }
      ],
      "per_testing_criteria_results": [
        {
          "testing_criteria": "Push Notificatio
          "passed": 1,
          "failed": 0
        }
      ],
      "data_source": {
        "type": "completions",
        "source": {
          "type": "file_content",
          "content": [
            {
              "item": {
                "notifications": "\n- New messa
              }
            }
          ]
        },
        "input_messages": {
          "type": "template",
          "template": [
            {
              "type": "message",
              "role": "developer",
              "content": {
                "type": "input_text",
                "text": "\n\n\n\nYou are a help
              }
            },
            {
              "type": "message",
              "role": "user",
              "content": {

## Get an eval run
The ID of the evaluation to retrieve runs for.
eval_id string
## Required
The ID of the run to retrieve.
run_id string
## Required
The EvalRun object matching the specified ID.
## Example request
python
from openai import OpenAI
client = OpenAI()
run = client.evals.runs.retrieve(
  "eval_67abd54d9b0081909a86353f6fb9317a",
  "evalrun_67abd54d60ec8190832b46859da808f7"
)
print(run)

## Response
 
{
  "object": "eval.run",
  "id": "evalrun_67abd54d60ec8190832b46859da80
  "eval_id": "eval_67abd54d9b0081909a86353f6fb
  "report_url": "https://platform.openai.com/e
  "status": "queued",
  "model": "gpt-4o-mini",
  "name": "gpt-4o-mini",
  "created_at": 1743092069,
  "result_counts": {
    "total": 0,
    "errored": 0,
    "failed": 0,
    "passed": 0
  },


<!-- Page 101 -->
POST https://api.openai.com/v1/evals/{eval_id}/runs
Kicks off a new run for a given evaluation, specifying the data source, and
what model configuration to use to test. The datasource will be validated
against the schema specified in the config of the evaluation.
## Path parameters
Request body
 
                "type": "input_text",
                "text": "<push_notifications>{{
              }
            }
          ]
        },
        "model": "o3-mini",
        "sampling_params": null
      },
      "error": null,
      "metadata": {}
    }
  ],
  "first_id": "evalrun_67e0c7d31560819090d60c07
  "last_id": "evalrun_67e0c7d31560819090d60c078
  "has_more": true
}

  "per_model_usage": null,
  "per_testing_criteria_results": null,
  "data_source": {
    "type": "completions",
    "source": {
      "type": "file_content",
      "content": [
        {
          "item": {
            "input": "Tech Company Launches Ad
            "ground_truth": "Technology"
          }
        },
        {
          "item": {
            "input": "Central Bank Increases I
            "ground_truth": "Markets"
          }
        },
        {
          "item": {
            "input": "International Summit Add
            "ground_truth": "World"
          }
        },
        {
          "item": {
            "input": "Major Retailer Reports R
            "ground_truth": "Business"
          }
        },
        {
          "item": {
            "input": "National Team Qualifies 
            "ground_truth": "Sports"
          }
        },
        {
          "item": {
            "input": "Stock Markets Rally Afte

## Create eval run
The ID of the evaluation to create a run for.
eval_id string
## Required
Details about the run's data source.
## Show possible types
data_source object
## Required
Set of 16 key-value pairs that can be attached to an object. This can be useful for
storing additional information about the object in a structured format, and querying for objects via API or the dashboard.
metadata
map
## Optional
Example request
python
 
from openai import OpenAI
client = OpenAI()
run = client.evals.runs.create(
  "eval_67e579652b548190aaa83ada4b125f47",
  name="gpt-4o-mini",
  data_source={
    "type": "completions",
    "input_messages": {
      "type": "template",
      "template": [
        {
          "role": "developer",
          "content": "Categorize a given news h
        },
        {
          "role": "user",
          "content": "{{item.input}}"
        }
      ]
    },
    "sampling_params": {
      "temperature": 1,
      "max_completions_tokens": 2048,
      "top_p": 1,


<!-- Page 102 -->
## Returns
POST https://api.openai.com/v1/evals/{eval_id}/runs/{run_id}
Cancel an ongoing evaluation run.
## Path parameters
Returns
 
            "ground_truth": "Markets"
          }
        },
        {
          "item": {
            "input": "Global Manufacturer Anno
            "ground_truth": "Business"
          }
        },
        {
          "item": {
            "input": "Breakthrough in Renewabl
            "ground_truth": "Technology"
          }
        },
        {
          "item": {
            "input": "World Leaders Sign Histo
            "ground_truth": "World"
          }
        },
        {
          "item": {
            "input": "Professional Athlete Set
            "ground_truth": "Sports"
          }
        },
        {
          "item": {
            "input": "Financial Institutions A
            "ground_truth": "Business"
          }
        },
        {
          "item": {
            "input": "Tech Conference Showcase
            "ground_truth": "Technology"
          }
        },
        {

Keys are strings with a maximum length of 64 characters. Values are strings with a
maximum length of 512 characters.
The name of the run.
name string
## Optional
The EvalRun object matching the specified ID.
 
      "seed": 42
    },
    "model": "gpt-4o-mini",
    "source": {
      "type": "file_content",
      "content": [
        {
          "item": {
            "input": "Tech Company Launches Adv
            "ground_truth": "Technology"
          }
        }
      ]
    }
  }
)
print(run)

## Response
 
{
  "object": "eval.run",
  "id": "evalrun_67e57965b480819094274e3a32235e
  "eval_id": "eval_67e579652b548190aaa83ada4b12
  "report_url": "https://platform.openai.com/ev
  "status": "queued",
  "model": "gpt-4o-mini",
  "name": "gpt-4o-mini",
  "created_at": 1743092069,
  "result_counts": {
    "total": 0,
    "errored": 0,
    "failed": 0,
    "passed": 0
  },
  "per_model_usage": null,
  "per_testing_criteria_results": null,
  "data_source": {
    "type": "completions",

## Cancel eval run
The ID of the evaluation whose run you want to cancel.
eval_id string
## Required
The ID of the run to cancel.
run_id string
## Required
Example request
python
from openai import OpenAI
client = OpenAI()
canceled_run = client.evals.runs.cancel(
  "eval_67abd54d9b0081909a86353f6fb9317a",
  "evalrun_67abd54d60ec8190832b46859da808f7"
)
print(canceled_run)

## Response
 
{
  "object": "eval.run",
  "id": "evalrun_67abd54d60ec8190832b46859da80
  "eval_id": "eval_67abd54d9b0081909a86353f6fb
  "report_url": "https://platform.openai.com/e
  "status": "canceled",


<!-- Page 103 -->
DELETE https://api.openai.com/v1/evals/{eval_id}/runs/{run_id}
Delete an eval run.
## Path parameters
Returns
 
          "item": {
            "input": "Global Markets Respond t
            "ground_truth": "Markets"
          }
        },
        {
          "item": {
            "input": "International Cooperatio
            "ground_truth": "World"
          }
        },
        {
          "item": {
            "input": "Sports League Announces 
            "ground_truth": "Sports"
          }
        }
      ]
    },
    "input_messages": {
      "type": "template",
      "template": [
        {
          "type": "message",
          "role": "developer",
          "content": {
            "type": "input_text",
            "text": "Categorize a given news h
          }
        },
        {
          "type": "message",
          "role": "user",
          "content": {
            "type": "input_text",
            "text": "{{item.input}}"
          }
        }
      ]
    },

    "source": {
      "type": "file_content",
      "content": [
        {
          "item": {
            "input": "Tech Company Launches Adv
            "ground_truth": "Technology"
          }
        }
      ]
    },
    "input_messages": {
      "type": "template",
      "template": [
        {
          "type": "message",
          "role": "developer",
          "content": {
            "type": "input_text",
            "text": "Categorize a given news he
          }
        },
        {
          "type": "message",
          "role": "user",
          "content": {
            "type": "input_text",
            "text": "{{item.input}}"
          }
        }
      ]
    },
    "model": "gpt-4o-mini",
    "sampling_params": {
      "seed": 42,
      "temperature": 1.0,
      "top_p": 1.0,
      "max_completions_tokens": 2048
    }
  },

The updated EvalRun object reflecting that the run is canceled.
 
  "model": "gpt-4o-mini",
  "name": "gpt-4o-mini",
  "created_at": 1743092069,
  "result_counts": {
    "total": 0,
    "errored": 0,
    "failed": 0,
    "passed": 0
  },
  "per_model_usage": null,
  "per_testing_criteria_results": null,
  "data_source": {
    "type": "completions",
    "source": {
      "type": "file_content",
      "content": [
        {
          "item": {
            "input": "Tech Company Launches Ad
            "ground_truth": "Technology"
          }
        },
        {
          "item": {
            "input": "Central Bank Increases I
            "ground_truth": "Markets"
          }
        },
        {
          "item": {
            "input": "International Summit Add
            "ground_truth": "World"
          }
        },
        {
          "item": {
            "input": "Major Retailer Reports R
            "ground_truth": "Business"
          }
        },

## Delete eval run
The ID of the evaluation to delete the run from.
eval_id string
## Required
The ID of the run to delete.
run_id string
## Required
Example request
python
from openai import OpenAI
client = OpenAI()
deleted = client.evals.runs.delete(
  "eval_123abc",
  "evalrun_abc456"
)
print(deleted)

## Response
 
{
  "object": "eval.run.deleted",
  "deleted": true,


<!-- Page 104 -->
GET https://api.openai.com/v1/evals/{eval_id}/runs/{run_id}/output_items/
{output_item_id}
Get an evaluation run output item by ID.
## Path parameters
Returns
 
    "model": "gpt-4o-mini",
    "sampling_params": {
      "seed": 42,
      "temperature": 1.0,
      "top_p": 1.0,
      "max_completions_tokens": 2048
    }
  },
"
"
ll

  "error": null,
  "metadata": {}

        {
          "item": {
            "input": "National Team Qualifies 
            "ground_truth": "Sports"
          }
        },
        {
          "item": {
            "input": "Stock Markets Rally Afte
            "ground_truth": "Markets"
          }
        },
        {
          "item": {
            "input": "Global Manufacturer Anno
            "ground_truth": "Business"
          }
        },
        {
          "item": {
            "input": "Breakthrough in Renewabl
            "ground_truth": "Technology"
          }
        },
        {
          "item": {
            "input": "World Leaders Sign Histo
            "ground_truth": "World"
          }
        },
        {
          "item": {
            "input": "Professional Athlete Set
            "ground_truth": "Sports"
          }
        },
        {
          "item": {
            "input": "Financial Institutions A
            "ground_truth": "Business"

An object containing the status of the delete operation.
 
  "run_id": "evalrun_abc456"
}

## Get an output item of an eval run
The ID of the evaluation to retrieve runs for.
eval_id string
## Required
The ID of the output item to retrieve.
output_item_id string
## Required
The ID of the run to retrieve.
run_id string
## Required
The EvalRunOutputItem object matching the specified ID.
## Example request
python
 
 
from openai import OpenAI
client = OpenAI()
output_item = client.evals.runs.output_items.retr
  "eval_67abd54d9b0081909a86353f6fb9317a",
  "evalrun_67abd54d60ec8190832b46859da808f7",
  "outputitem_67abd55eb6548190bb580745d5644a33"
)
print(output_item)

## Response
 
{
  "object": "eval.run.output_item",
  "id": "outputitem_67e5796c28e081909917bf79f6e
  "created_at": 1743092076,
  "run_id": "evalrun_67abd54d60ec8190832b46859d
  "eval_id": "eval_67abd54d9b0081909a86353f6fb9
  "status": "pass",
  "datasource_item_id": 5,
  "datasource_item": {
    "input": "Stock Markets Rally After Positiv
    "ground_truth": "Markets"
  },
  "results": [
    {


<!-- Page 105 -->
GET https://api.openai.com/v1/evals/{eval_id}/runs/{run_id}/output_items
Get a list of output items for an evaluation run.
## Path parameters
Query parameters
 
          }
        },
        {
          "item": {
            "input": "Tech Conference Showcase
            "ground_truth": "Technology"
          }
        },
        {
          "item": {
            "input": "Global Markets Respond t
            "ground_truth": "Markets"
          }
        },
        {
          "item": {
            "input": "International Cooperatio
            "ground_truth": "World"
          }
        },
        {
          "item": {
            "input": "Sports League Announces 
            "ground_truth": "Sports"
          }
        }
      ]
    },
    "input_messages": {
      "type": "template",
      "template": [
        {
          "type": "message",
          "role": "developer",
          "content": {
            "type": "input_text",
            "text": "Categorize a given news h
          }
        },
        {

      "name": "String check-a2486074-d803-4445-
      "sample": null,
      "passed": true,
      "score": 1.0
    }
  ],
  "sample": {
    "input": [
      {
        "role": "developer",
        "content": "Categorize a given news hea
        "tool_call_id": null,
        "tool_calls": null,
        "function_call": null
      },
      {
        "role": "user",
        "content": "Stock Markets Rally After P
        "tool_call_id": null,
        "tool_calls": null,
        "function_call": null
      }
    ],
    "output": [
      {
        "role": "assistant",
        "content": "Markets",
        "tool_call_id": null,
        "tool_calls": null,
        "function_call": null
      }
    ],
    "finish_reason": "stop",
    "model": "gpt-4o-mini-2024-07-18",
    "usage": {
      "total_tokens": 325,
      "completion_tokens": 2,
      "prompt_tokens": 323,
      "cached_tokens": 0
    },

## Get eval run output items
The ID of the evaluation to retrieve runs for.
eval_id string
## Required
The ID of the run to retrieve output items for.
run_id string
## Required
Identifier for the last output item from the previous pagination request.
after string
## Optional
Number of output items to retrieve.
limit integer
## Optional
Defaults to 20
order string
## Optional
Defaults to asc
Example request
python
 
 
from openai import OpenAI
client = OpenAI()
output_items = client.evals.runs.output_items.lis
  "egroup_67abd54d9b0081909a86353f6fb9317a",
  "erun_67abd54d60ec8190832b46859da808f7"
)
print(output_items)

## Response
 
{
  "object": "list",
  "data": [
    {
      "object": "eval.run.output_item",
      "id": "outputitem_67e5796c28e081909917bf7
      "created_at": 1743092076,
      "run_id": "evalrun_67abd54d60ec8190832b46
      "eval_id": "eval_67abd54d9b0081909a86353f
      "status": "pass",
      "datasource_item_id": 5,
      "datasource_item": {
        "input": "Stock Markets Rally After Pos


<!-- Page 106 -->
## Returns
An Eval object with a data source config and testing criteria. An Eval
represents a task to be done for your LLM integration. Like:
 
          "type": "message",
          "role": "user",
          "content": {
            "type": "input_text",
            "text": "{{item.input}}"
          }
        }
      ]
    },
    "model": "gpt-4o-mini",
    "sampling_params": {
      "seed": 42,
      "temperature": 1.0,
      "top_p": 1.0,
      "max_completions_tokens": 2048
    }
  },
  "error": null,

    "error": null,
    "temperature": 1.0,
    "max_completion_tokens": 2048,
    "top_p": 1.0,
    "seed": 42
  }

Sort order for output items by timestamp. Use asc  for ascending order or desc  for
descending order. Defaults to asc .
Filter output items by status. Use failed  to filter by failed output items or pass  to
filter by passed output items.
status string
## Optional
A list of EvalRunOutputItem objects matching the specified ID.
 
        "ground_truth": "Markets"
      },
      "results": [
        {
          "name": "String check-a2486074-d803-4
          "sample": null,
          "passed": true,
          "score": 1.0
        }
      ],
      "sample": {
        "input": [
          {
            "role": "developer",
            "content": "Categorize a given news
            "tool_call_id": null,
            "tool_calls": null,
            "function_call": null
          },
          {
            "role": "user",
            "content": "Stock Markets Rally Aft
            "tool_call_id": null,
            "tool_calls": null,
            "function_call": null
          }
        ],
        "output": [
          {
            "role": "assistant",
            "content": "Markets",
            "tool_call_id": null,
            "tool_calls": null,
            "function_call": null
          }
        ],
        "finish_reason": "stop",
        "model": "gpt-4o-mini-2024-07-18",
        "usage": {
          "total_tokens": 325,

## The eval object
Improve the quality of my chatbot
See how well my chatbot handles customer support
Check if o4-mini is better at my usecase than gpt-4o
The Unix timestamp (in seconds) for when the eval was created.
created_at integer
Configuration of data sources used in runs of the evaluation.
## Show possible types
data_source_config object
id string
## OBJECT The eval object
 
{
  "object": "eval",
  "id": "eval_67abd54d9b0081909a86353f6fb9317a"
  "data_source_config": {
    "type": "custom",
    "item_schema": {
      "type": "object",
      "properties": {
        "label": {"type": "string"},
      },
      "required": ["label"]
    },
    "include_sample_schema": true
  },
  "testing_criteria": [
    {


<!-- Page 107 -->
A schema representing an evaluation run.
 
          "completion_tokens": 2,
          "prompt_tokens": 323,
          "cached_tokens": 0
        },
        "error": null,
        "temperature": 1.0,
        "max_completion_tokens": 2048,
        "top_p": 1.0,
        "seed": 42
      }
    }
  ],
  "first_id": "outputitem_67e5796c28e081909917b
  "last_id": "outputitem_67e5796c28e081909917bf
  "has_more": true
}

Unique identifier for the evaluation.
Set of 16 key-value pairs that can be attached to an object. This can be useful for
storing additional information about the object in a structured format, and querying for objects via API or the dashboard.
Keys are strings with a maximum length of 64 characters. Values are strings with a
maximum length of 512 characters.
metadata
map
The name of the evaluation.
name string
The object type.
object string
A list of testing criteria.
## Show possible types
testing_criteria array
 
      "name": "My string check grader",
      "type": "string_check",
      "input": "{{sample.output_text}}",
      "reference": "{{item.label}}",
      "operation": "eq",
    }
  ],
  "name": "External Data Eval",
  "created_at": 1739314509,
  "metadata": {
    "test": "synthetics",
  }
}

## The eval run object
Unix timestamp (in seconds) when the evaluation run was created.
created_at integer
Information about the run's data source.
## Show possible types
data_source object
## OBJECT The eval run object
 
{
  "object": "eval.run",
  "id": "evalrun_67e57965b480819094274e3a32235
  "eval_id": "eval_67e579652b548190aaa83ada4b1
  "report_url": "https://platform.openai.com/e
  "status": "queued",
  "model": "gpt-4o-mini",
  "name": "gpt-4o-mini",


<!-- Page 108 -->
An object representing an error response from the Eval API.
## Show properties
error object
The identifier of the associated evaluation.
eval_id string
Unique identifier for the evaluation run.
id string
Set of 16 key-value pairs that can be attached to an object. This can be useful for
storing additional information about the object in a structured format, and querying for objects via API or the dashboard.
Keys are strings with a maximum length of 64 characters. Values are strings with a
maximum length of 512 characters.
metadata
map
The model that is evaluated, if applicable.
model string
The name of the evaluation run.
name string
The type of the object. Always "eval.run".
object string
Usage statistics for each model during the evaluation run.
## Show properties
per_model_usage array
Results per testing criteria applied during the evaluation run.
per_testing_criteria_results array
 
  "created_at": 1743092069,
  "result_counts": {
    "total": 0,
    "errored": 0,
    "failed": 0,
    "passed": 0
  },
  "per_model_usage": null,
  "per_testing_criteria_results": null,
  "data_source": {
    "type": "completions",
    "source": {
      "type": "file_content",
      "content": [
        {
          "item": {
            "input": "Tech Company Launches Ad
            "ground_truth": "Technology"
          }
        },
        {
          "item": {
            "input": "Central Bank Increases I
            "ground_truth": "Markets"
          }
        },
        {
          "item": {
            "input": "International Summit Add
            "ground_truth": "World"
          }
        },
        {
          "item": {
            "input": "Major Retailer Reports R
            "ground_truth": "Business"
          }
        },
        {
          "item": {


<!-- Page 109 -->
A schema representing an evaluation run output item.
## Show properties
The URL to the rendered evaluation run report on the UI dashboard.
report_url string
Counters summarizing the outcomes of the evaluation run.
## Show properties
result_counts object
The status of the evaluation run.
status string
 
            "input": "National Team Qualifies 
            "ground_truth": "Sports"
          }
        },
        {
          "item": {
            "input": "Stock Markets Rally Afte
            "ground_truth": "Markets"
          }
        },
        {
          "item": {
            "input": "Global Manufacturer Anno
            "ground_truth": "Business"
          }
        },
        {
          "item": {
            "input": "Breakthrough in Renewabl
            "ground_truth": "Technology"
          }
        },
        {
          "item": {
            "input": "World Leaders Sign Histo
            "ground_truth": "World"
          }
        },
        {
          "item": {
            "input": "Professional Athlete Set
            "ground_truth": "Sports"
          }
        },
        {
          "item": {
            "input": "Financial Institutions A
            "ground_truth": "Business"
          }
        },

## The eval run output item object
Unix timestamp (in seconds) when the evaluation run was created.
created_at integer
Details of the input data source item.
datasource_item object
The identifier for the data source item.
datasource_item_id integer
The identifier of the evaluation group.
eval_id string
id string
## OBJECT The eval run output item object
 
{
  "object": "eval.run.output_item",
  "id": "outputitem_67abd55eb6548190bb580745d56
  "run_id": "evalrun_67abd54d60ec8190832b46859d
  "eval_id": "eval_67abd54d9b0081909a86353f6fb9
  "created_at": 1739314509,
  "status": "pass",
  "datasource_item_id": 137,
  "datasource_item": {
      "teacher": "To grade essays, I only check
      "student": "I am a student who is trying 
  },
  "results": [
    {
      "name": "String Check Grader",
      "type": "string-check-grader",
      "score": 1.0,


<!-- Page 110 -->
Manage fine-tuning jobs to tailor a model to your specific training data. Related guide: Fine-tune models
POST https://api.openai.com/v1/fine_tuning/jobs
        {
          "item": {
            "input": "Tech Conference Showcase
            "ground_truth": "Technology"
          }
        },
        {
          "item": {
            "input": "Global Markets Respond t
            "ground_truth": "Markets"
          }
        },
        {
          "item": {
            "input": "International Cooperatio
            "ground_truth": "World"
          }
        },
        {
          "item": {
            "input": "Sports League Announces 
            "ground_truth": "Sports"
          }
        }
      ]
    },
    "input_messages": {
      "type": "template",
      "template": [
        {
          "type": "message",
          "role": "developer",
          "content": {
            "type": "input_text",
            "text": "Categorize a given news h
          }
        },
        {
          "type": "message",
          "role": "user",

Unique identifier for the evaluation run output item.
The type of the object. Always "eval.run.output_item".
object string
A list of results from the evaluation run.
## Show properties
results array
The identifier of the evaluation run associated with this output item.
run_id string
A sample containing the input and output of the evaluation run.
## Show properties
sample object
The status of the evaluation run.
status string
 
      "passed": true,
    }
  ],
  "sample": {
    "input": [
      {
        "role": "system",
        "content": "You are an evaluator bot...
      },
      {
        "role": "user",
        "content": "You are assessing..."
      }
    ],
    "output": [
      {
        "role": "assistant",
        "content": "The rubric is not clear nor
      }
    ],
    "finish_reason": "stop",
    "model": "gpt-4o-2024-08-06",
    "usage": {
      "total_tokens": 521,
      "completion_tokens": 2,
      "prompt_tokens": 519,
      "cached_tokens": 0
    },
    "error": null,
    "temperature": 1.0,
    "max_completion_tokens": 2048,
    "top_p": 1.0,
    "seed": 42
  }
}

## Fine-tuning
Create fine-tuning job
Default
Epochs
DPO
Reinforcement
Validation 

<!-- Page 111 -->
## Creates a fine-tuning job which begins the process of creating a new model
from a given dataset.
## Response includes details of the enqueued job including job status and the
name of the fine-tuned models once complete.
## Learn more about fine-tuning
Request body
 
          "content": {
            "type": "input_text",
            "text": "{{item.input}}"
          }
        }
      ]
    },
    "model": "gpt-4o-mini",
    "sampling_params": {
      "seed": 42,
      "temperature": 1.0,
      "top_p": 1.0,
      "max_completions_tokens": 2048
    }
  },
"error": null

The name of the model to fine-tune. You can select one of the supported models.
model string
## Required
The ID of an uploaded file that contains training data.
See upload file for how to upload a file.
Your dataset must be formatted as a JSONL file. Additionally, you must upload your file
with the purpose fine-tune .
The contents of the file should differ depending on if the model uses the chat,
completions format, or if the fine-tuning method uses the preference format.
See the fine-tuning guide for more details.
training_file string
## Required
The hyperparameters used for the fine-tuning job. This value is now deprecated in
favor of method , and should be passed in under the method  parameter.
## Show properties
hyperparameters
Deprecated object
Optional
A list of integrations to enable for your fine-tuning job.
## Show properties
integrations array or null
Optional
Set of 16 key-value pairs that can be attached to an object. This can be useful for
storing additional information about the object in a structured format, and querying for
metadata
map
## Optional
Example request
python
from openai import OpenAI
client = OpenAI()
client.fine_tuning.jobs.create(
  training_file="file-abc123",
  model="gpt-4o-mini"
)

## Response
{
  "object": "fine_tuning.job",
  "id": "ftjob-abc123",
  "model": "gpt-4o-mini-2024-07-18",
  "created_at": 1721764800,
  "fine_tuned_model": null,
  "organization_id": "org-123",
  "result_files": [],
  "status": "queued",
  "validation_file": null,
  "training_file": "file-abc123",
  "method": {
    "type": "supervised",
    "supervised": {
      "hyperparameters": {
        "batch_size": "auto",
        "learning_rate_multiplier": "auto",
        "n_epochs": "auto",
      }
    }
  },
  "metadata": null
}


<!-- Page 112 -->
## Returns objects via API or the dashboard.
Keys are strings with a maximum length of 64 characters. Values are strings with a
maximum length of 512 characters.
The method used for fine-tuning.
## Show properties
method object
Optional
The seed controls the reproducibility of the job. Passing in the same seed and job
parameters should produce the same results, but may differ in rare cases. If a seed is
not specified, one will be generated for you.
seed integer or null
## Optional
A string of up to 64 characters that will be added to your fine-tuned model name.
For example, a suffix  of "custom-model-name" would produce a model name like
ft:gpt-4o-mini:openai:custom-model-name:7p4lURel .
suffix string or null
## Optional
Defaults to null
The ID of an uploaded file that contains validation data.
If you provide this file, the data is used to generate validation metrics periodically
during fine-tuning. These metrics can be viewed in the fine-tuning results file. The
same data should not be present in both train and validation files.
Your dataset must be formatted as a JSONL file. You must upload your file with the
purpose fine-tune .
See the fine-tuning guide for more details.
validation_file string or null
## Optional
A fine-tuning.job object.

<!-- Page 114 -->
GET https://api.openai.com/v1/fine_tuning/jobs
List your organization's fine-tuning jobs
## Query parameters
Returns
Identifier for the last job from the previous pagination request.
after string
## Optional
Number of fine-tuning jobs to retrieve.
limit integer
## Optional
Defaults to 20
Optional metadata filter. To filter, use the syntax metadata[k]=v . Alternatively, set
metadata=null  to indicate no metadata.
metadata object or null
## Optional
A list of paginated fine-tuning job objects.
## Example request
python
from openai import OpenAI
client = OpenAI()
client.fine_tuning.jobs.list()

## Response
{
  "object": "list",
  "data": [
    {
      "object": "fine_tuning.job",
      "id": "ftjob-abc123",
      "model": "gpt-4o-mini-2024-07-18",
      "created_at": 1721764800,
      "fine_tuned_model": null,
      "organization_id": "org-123",
      "result_files": [],
      "status": "queued",
      "validation_file": null,
      "training_file": "file-abc123",
      "metadata": {
        "key": "value"
      }
    },
    { ... },
    { ... }
  ], "has_more": true
}

## List fine-tuning events

<!-- Page 115 -->
GET https://api.openai.com/v1/fine_tuning/jobs/{fine_tuning_job_id}/events
Get status updates for a fine-tuning job.
## Path parameters
Query parameters
Returns
The ID of the fine-tuning job to get events for.
fine_tuning_job_id string
## Required
Identifier for the last event from the previous pagination request.
after string
## Optional
Number of events to retrieve.
limit integer
## Optional
Defaults to 20
A list of fine-tuning event objects.
## Example request
python
from openai import OpenAI
client = OpenAI()
client.fine_tuning.jobs.list_events(
  fine_tuning_job_id="ftjob-abc123",
  limit=2
)

## Response
 
 
{
  "object": "list",
  "data": [
    {
      "object": "fine_tuning.job.event",
      "id": "ft-event-ddTJfwuMVpfLXseO0Am0Gqjm",
      "created_at": 1721764800,
      "level": "info",
      "message": "Fine tuning job successfully c
      "data": null,
      "type": "message"
    },
    {
      "object": "fine_tuning.job.event",
      "id": "ft-event-tyiGuB72evQncpH87xe505Sv",
      "created_at": 1721764800,
      "level": "info",
      "message": "New fine-tuned model created: 
      "data": null,
      "type": "message"
    }
  ],
  "has_more": true
}


<!-- Page 116 -->
GET https://api.openai.com/v1/fine_tuning/jobs/{fine_tuning_job_id}/checkp
oints
List checkpoints for a fine-tuning job.
## Path parameters
Query parameters
Returns
List fine-tuning checkpoints
The ID of the fine-tuning job to get checkpoints for.
fine_tuning_job_id string
## Required
Identifier for the last checkpoint ID from the previous pagination request.
after string
## Optional
Number of checkpoints to retrieve.
limit integer
## Optional
Defaults to 10
A list of fine-tuning checkpoint objects for a fine-tuning job.
## Example request
curl
 
 
curl https://api.openai.com/v1/fine_tuning/jobs/f
  -H "Authorization: Bearer $OPENAI_API_KEY"

## Response
 
{
  "object": "list",
  "data": [
    {
      "object": "fine_tuning.job.checkpoint",
      "id": "ftckpt_zc4Q7MP6XxulcVzj4MZdwsAB",
      "created_at": 1721764867,
      "fine_tuned_model_checkpoint": "ft:gpt-4o
      "metrics": {
        "full_valid_loss": 0.134,
        "full_valid_mean_token_accuracy": 0.874
      },
      "fine_tuning_job_id": "ftjob-abc123",
      "step_number": 2000
    },
    {
      "object": "fine_tuning.job.checkpoint",
      "id": "ftckpt_enQCFmOTGj3syEpYVhBRLTSy",
      "created_at": 1721764800,
      "fine_tuned_model_checkpoint": "ft:gpt-4o
      "metrics": {
        "full_valid_loss": 0.167,
        "full_valid_mean_token_accuracy": 0.781
      },
      "fine_tuning_job_id": "ftjob-abc123",
      "step_number": 1000
    }
  ],
  "first_id": "ftckpt_zc4Q7MP6XxulcVzj4MZdwsAB"


<!-- Page 117 -->
GET https://api.openai.com/v1/fine_tuning/checkpoints/{fine_tuned_model_ch
eckpoint}/permissions
NOTE: This endpoint requires an admin API key.
## Organization owners can use this endpoint to view all permissions for a fine-
tuned model checkpoint.
## Path parameters
Query parameters
 
  "last_id": "ftckpt_enQCFmOTGj3syEpYVhBRLTSy",
  "has_more": true
}

## List checkpoint permissions
The ID of the fine-tuned model checkpoint to get permissions for.
fine_tuned_model_checkpoint string
## Required
Identifier for the last permission ID from the previous pagination request.
after string
## Optional
Number of permissions to retrieve.
limit integer
## Optional
Defaults to 10
The order in which to retrieve permissions.
order string
## Optional
Defaults to descending
The ID of the project to get permissions for.
project_id string
## Optional
Example request
curl
 
 
curl https://api.openai.com/v1/fine_tuning/checkp
  -H "Authorization: Bearer $OPENAI_API_KEY"

## Response
 
 
{
  "object": "list",
  "data": [
    {
      "object": "checkpoint.permission",
      "id": "cp_zc4Q7MP6XxulcVzj4MZdwsAB",
      "created_at": 1721764867,
      "project_id": "proj_abGMw1llN8IrBb6SvvY5A1
    },
    {
      "object": "checkpoint.permission",
      "id": "cp_enQCFmOTGj3syEpYVhBRLTSy",
      "created_at": 1721764800,
      "project_id": "proj_iqGMw1llN8IrBb6SvvY5A1
    },
  ],
  "first_id": "cp_zc4Q7MP6XxulcVzj4MZdwsAB",
  "last_id": "cp_enQCFmOTGj3syEpYVhBRLTSy",
  "has_more": false
}


<!-- Page 118 -->
## Returns
POST https://api.openai.com/v1/fine_tuning/checkpoints/{fine_tuned_model_c
heckpoint}/permissions
NOTE: Calling this endpoint requires an admin API key.
## This enables organization owners to share fine-tuned models with other
projects in their organization.
## Path parameters
Request body
Returns
A list of fine-tuned model checkpoint permission objects for a fine-tuned model
checkpoint.
## Create checkpoint permissions
The ID of the fine-tuned model checkpoint to create a permission for.
fine_tuned_model_checkpoint string
## Required
The project identifiers to grant access to.
project_ids array
## Required
Example request
curl
 
 
curl https://api.openai.com/v1/fine_tuning/checkp
  -H "Authorization: Bearer $OPENAI_API_KEY"
  -d '{"project_ids": ["proj_abGMw1llN8IrBb6SvvY5

## Response
 
 
{
  "object": "list",
  "data": [
    {
      "object": "checkpoint.permission",
      "id": "cp_zc4Q7MP6XxulcVzj4MZdwsAB",
      "created_at": 1721764867,
      "project_id": "proj_abGMw1llN8IrBb6SvvY5A1
    }
  ],
  "first_id": "cp_zc4Q7MP6XxulcVzj4MZdwsAB",
  "last_id": "cp_zc4Q7MP6XxulcVzj4MZdwsAB",
  "has_more": false
}


<!-- Page 119 -->
DELETE https://api.openai.com/v1/fine_tuning/checkpoints/{fine_tuned_model
_checkpoint}/permissions/{permission_id}
NOTE: This endpoint requires an admin API key.
## Organization owners can use this endpoint to delete a permission for a fine-
tuned model checkpoint.
## Path parameters
Returns
A list of fine-tuned model checkpoint permission objects for a fine-tuned model
checkpoint.
## Delete checkpoint permission
The ID of the fine-tuned model checkpoint to delete a permission for.
fine_tuned_model_checkpoint string
## Required
The ID of the fine-tuned model checkpoint permission to delete.
permission_id string
## Required
The deletion status of the fine-tuned model checkpoint permission object.
## Example request
curl
 
 
curl https://api.openai.com/v1/fine_tuning/checkp
  -H "Authorization: Bearer $OPENAI_API_KEY"

## Response
{
  "object": "checkpoint.permission",
  "id": "cp_zc4Q7MP6XxulcVzj4MZdwsAB",
  "deleted": true
}


<!-- Page 120 -->
GET https://api.openai.com/v1/fine_tuning/jobs/{fine_tuning_job_id}
Get info about a fine-tuning job.
## Learn more about fine-tuning
Path parameters
Returns
Retrieve fine-tuning job
The ID of the fine-tuning job.
fine_tuning_job_id string
## Required
The fine-tuning object with the given ID.
## Example request
python
 
 
from openai import OpenAI
client = OpenAI()
client.fine_tuning.jobs.retrieve("ftjob-abc123")

## Response
 
{
  "object": "fine_tuning.job",
  "id": "ftjob-abc123",
  "model": "davinci-002",
  "created_at": 1692661014,
  "finished_at": 1692661190,
  "fine_tuned_model": "ft:davinci-002:my-org:cu
  "organization_id": "org-123",
  "result_files": [
      "file-abc123"
  ],
  "status": "succeeded",
  "validation_file": null,
  "training_file": "file-abc123",
  "hyperparameters": {
      "n_epochs": 4,
      "batch_size": 1,
      "learning_rate_multiplier": 1.0
  },
  "trained_tokens": 5768,
  "integrations": [],
  "seed": 0,
  "estimated_finish": 0,
  "method": {
    "type": "supervised",
    "supervised": {
      "hyperparameters": {


<!-- Page 121 -->
POST https://api.openai.com/v1/fine_tuning/jobs/{fine_tuning_job_id}/cance
l
Immediately cancel a fine-tune job.
## Path parameters
Returns
POST https://api.openai.com/v1/fine_tuning/jobs/{fine_tuning_job_id}/resum
e
 
        "n_epochs": 4,
        "batch_size": 1,
        "learning_rate_multiplier": 1.0
      }
    }
  }
}

## Cancel fine-tuning
The ID of the fine-tuning job to cancel.
fine_tuning_job_id string
## Required
The cancelled fine-tuning object.
## Example request
python
from openai import OpenAI
client = OpenAI()
client.fine_tuning.jobs.cancel("ftjob-abc123")

## Response
{
  "object": "fine_tuning.job",
  "id": "ftjob-abc123",
  "model": "gpt-4o-mini-2024-07-18",
  "created_at": 1721764800,
  "fine_tuned_model": null,
  "organization_id": "org-123",
  "result_files": [],
  "status": "cancelled",
  "validation_file": "file-abc123",
  "training_file": "file-abc123"
}

## Resume fine-tuning
Example request
python

<!-- Page 122 -->
Resume a fine-tune job.
## Path parameters
Returns
POST https://api.openai.com/v1/fine_tuning/jobs/{fine_tuning_job_id}/pause
Pause a fine-tune job.
## Path parameters
Returns
The ID of the fine-tuning job to resume.
fine_tuning_job_id string
## Required
The resumed fine-tuning object.
from openai import OpenAI
client = OpenAI()

## Response
{
  "object": "fine_tuning.job",
  "id": "ftjob-abc123",
  "model": "gpt-4o-mini-2024-07-18",
  "created_at": 1721764800,
  "fine_tuned_model": null,
  "organization_id": "org-123",
  "result_files": [],
  "status": "queued",
  "validation_file": "file-abc123",
  "training_file": "file-abc123"
}

## Pause fine-tuning
The ID of the fine-tuning job to pause.
fine_tuning_job_id string
## Required
Example request
python
from openai import OpenAI
client = OpenAI()
client.fine_tuning.jobs.pause("ftjob-abc123")

## Response
 
{
  "object": "fine_tuning.job",
  "id": "ftjob-abc123",
  "model": "gpt-4o-mini-2024-07-18",
  "created_at": 1721764800,


<!-- Page 123 -->
## The per-line training example of a fine-tuning input file for chat models
using the supervised method. Input messages may contain text or image
content only. Audio and file input messages are not currently supported for
fine-tuning.
The paused fine-tuning object.
 
  "fine_tuned_model": null,
  "organization_id": "org-123",
  "result_files": [],
  "status": "paused",
  "validation_file": "file-abc123",
  "training_file": "file-abc123"
}

## Training format for chat models using the supervised method
A list of functions the model may generate JSON inputs for.
## Show properties
functions
Deprecated array
Show possible types
messages array
Whether to enable parallel function calling during tool use.
parallel_tool_calls boolean
A list of tools the model may generate JSON inputs for.
## Show properties
tools array
OBJECT Training format for chat models using the ...
{
  "messages": [
    { "role": "user", "content": "What is the we
    {
      "role": "assistant",
      "tool_calls": [
        {
          "id": "call_id",
          "type": "function",
          "function": {
            "name": "get_current_weather",
            "arguments": "{\"location\": \"San F
          }
        }
      ]
    }
  ],
  "parallel_tool_calls": false,
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_current_weather",
        "description": "Get the current weather"


<!-- Page 124 -->
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
                "type": "string",
                "description": "The city and cou
            },
            "format": { "type": "string", "enum"
          },
          "required": ["location", "format"]
        }
      }
    }
  ]
}

## Training format for chat models using the preference method

<!-- Page 125 -->
## The per-line training example of a fine-tuning input file for chat models
using the dpo method. Input messages may contain text or image content
only. Audio and file input messages are not currently supported for fine-
tuning.
Per-line training example for reinforcement fine-tuning. Note that messages
and tools  are the only reserved keywords. Any other arbitrary key-value
data can be included on training datapoints and will be available to
reference during grading under the {{ item.XXX }}  template variable. Input
messages may contain text or image content only. Audio and file input
messages are not currently supported for fine-tuning.
## Show properties
input object
The non-preferred completion message for the output.
## Show possible types
non_preferred_output array
The preferred completion message for the output.
## Show possible types
preferred_output array
OBJECT Training format for chat models using the ...
 
 
{
  "input": {
    "messages": [
      { "role": "user", "content": "What is the 
    ]
  },
  "preferred_output": [
    {
      "role": "assistant",
      "content": "The weather in San Francisco i
    }
  ],
  "non_preferred_output": [
    {
      "role": "assistant",
      "content": "The weather in San Francisco i
    }
  ]
}

## Training format for reasoning models using the reinforcement method
messages array
OBJECT Training format for reasoning models using...
 
{
  "messages": [
    {
      "role": "user",
      "content": "Your task is to take a chemic
    },
  ],
  # Any other JSON data can be inserted into an


<!-- Page 126 -->
The fine_tuning.job  object represents a fine-tuning job that has been
created through the API.
## Show possible types
A list of tools the model may generate JSON inputs for.
## Show properties
tools array
 
  "reference_answer": {
    "donor_bond_counts": 5,
    "acceptor_bond_counts": 7
  }
}

## The fine-tuning job object
The Unix timestamp (in seconds) for when the fine-tuning job was created.
created_at integer
For fine-tuning jobs that have failed , this will contain more information on the
cause of the failure.
## Show properties
error object or null
The Unix timestamp (in seconds) for when the fine-tuning job is estimated to finish.
The value will be null if the fine-tuning job is not running.
estimated_finish integer or null
The name of the fine-tuned model that is being created. The value will be null if the
fine-tuning job is still running.
fine_tuned_model string or null
finished_at integer or null
## OBJECT The fine-tuning job object
 
{
  "object": "fine_tuning.job",
  "id": "ftjob-abc123",
  "model": "davinci-002",
  "created_at": 1692661014,
  "finished_at": 1692661190,
  "fine_tuned_model": "ft:davinci-002:my-org:cu
  "organization_id": "org-123",
  "result_files": [
      "file-abc123"
  ],
  "status": "succeeded",
  "validation_file": null,
  "training_file": "file-abc123",
  "hyperparameters": {
      "n_epochs": 4,
      "batch_size": 1,
      "learning_rate_multiplier": 1.0
  },
  "trained_tokens": 5768,
  "integrations": [],
  "seed": 0,
  "estimated_finish": 0,
  "method": {


<!-- Page 127 -->
The Unix timestamp (in seconds) for when the fine-tuning job was finished. The value
will be null if the fine-tuning job is still running.
The hyperparameters used for the fine-tuning job. This value will only be returned when
running supervised  jobs.
## Show properties
hyperparameters object
The object identifier, which can be referenced in the API endpoints.
id string
A list of integrations to enable for this fine-tuning job.
## Show possible types
integrations array or null
Set of 16 key-value pairs that can be attached to an object. This can be useful for
storing additional information about the object in a structured format, and querying for objects via API or the dashboard.
Keys are strings with a maximum length of 64 characters. Values are strings with a
maximum length of 512 characters.
metadata
map
The method used for fine-tuning.
## Show properties
method object
The base model that is being fine-tuned.
model string
The object type, which is always "fine_tuning.job".
object string
organization_id string
 
    "type": "supervised",
    "supervised": {
      "hyperparameters": {
        "n_epochs": 4,
        "batch_size": 1,
        "learning_rate_multiplier": 1.0
      }
    }
  },
  "metadata": {
    "key": "value"
  }
}


<!-- Page 128 -->
## Fine-tuning job event object
The organization that owns the fine-tuning job.
The compiled results file ID(s) for the fine-tuning job. You can retrieve the results with
the Files API.
result_files array
The seed used for the fine-tuning job.
seed integer
The current status of the fine-tuning job, which can be either validating_files ,
queued , running , succeeded , failed , or cancelled .
status string
The total number of billable tokens processed by this fine-tuning job. The value will be null if the fine-tuning job is still running.
trained_tokens integer or null
The file ID used for training. You can retrieve the training data with the Files API.
training_file string
The file ID used for validation. You can retrieve the validation results with the Files API.
validation_file string or null
## The fine-tuning job event object
The Unix timestamp (in seconds) for when the fine-tuning job was created.
created_at integer
## OBJECT The fine-tuning job event object
 
{
  "object": "fine_tuning.job.event",
  "id": "ftevent-abc123"
  "created_at": 1677610602,


<!-- Page 129 -->
The fine_tuning.job.checkpoint  object represents a model checkpoint for a
fine-tuning job that is ready to use.
The data associated with the event.
data object
The object identifier.
id string
The log level of the event.
level string
The message of the event.
message string
The object type, which is always "fine_tuning.job.event".
object string
The type of event.
type string
 
  "level": "info",
  "message": "Created fine-tuning job",
  "data": {},
  "type": "message"
}

## The fine-tuning job checkpoint object
The Unix timestamp (in seconds) for when the checkpoint was created.
created_at integer
The name of the fine-tuned checkpoint model that is created.
fine_tuned_model_checkpoint string
## OBJECT The fine-tuning job checkpoint object
 
{
  "object": "fine_tuning.job.checkpoint",
  "id": "ftckpt_qtZ5Gyk4BLq1SfLFWp3RtO3P",
  "created_at": 1712211699,
  "fine_tuned_model_checkpoint": "ft:gpt-4o-min
  "fine_tuning_job_id": "ftjob-fpbNQ3H1GrMehXRf
  "metrics": {
    "step": 88,
    "train_loss": 0.478,


<!-- Page 130 -->
The checkpoint.permission  object represents a permission for a fine-tuned
model checkpoint.
The name of the fine-tuning job that this checkpoint was created from.
fine_tuning_job_id string
The checkpoint identifier, which can be referenced in the API endpoints.
id string
Metrics at the step number during the fine-tuning job.
## Show properties
metrics object
The object type, which is always "fine_tuning.job.checkpoint".
object string
The step number that the checkpoint was created at.
step_number integer
 
    "train_mean_token_accuracy": 0.924,
    "valid_loss": 10.112,
    "valid_mean_token_accuracy": 0.145,
    "full_valid_loss": 0.567,
    "full_valid_mean_token_accuracy": 0.944
  },
  "step_number": 88
}

## The fine-tuned model checkpoint permission object
The Unix timestamp (in seconds) for when the permission was created.
created_at integer
The permission identifier, which can be referenced in the API endpoints.
id string
The object type, which is always "checkpoint.permission".
object string
OBJECT The fine-tuned model checkpoint permission...
 
 
{
  "object": "checkpoint.permission",
  "id": "cp_zc4Q7MP6XxulcVzj4MZdwsAB",
  "created_at": 1712211699,
  "project_id": "proj_abGMw1llN8IrBb6SvvY5A1iH"
}


<!-- Page 131 -->
Manage and run graders in the OpenAI platform. Related guide: Graders
The project identifier that the permission is for.
project_id string
## Graders
String Check Grader

<!-- Page 132 -->
## A StringCheckGrader object that performs a string comparison between
input and reference using a specified operation.
A TextSimilarityGrader object which grades text based on similarity metrics.
The input text. This may include template strings.
input string
The name of the grader.
name string
The string check operation to perform. One of eq , ne , like , or ilike .
operation string
The reference text. This may include template strings.
reference string
The object type, which is always string_check .
type string
## OBJECT String Check Grader
{
  "type": "string_check",
  "name": "Example string check grader",
  "input": "{{sample.output_text}}",
  "reference": "{{item.label}}",
  "operation": "eq"
}

## Text Similarity Grader
The evaluation metric to use. One of fuzzy_match , bleu , gleu , meteor ,
rouge_1 , rouge_2 , rouge_3 , rouge_4 , rouge_5 , or rouge_l .
evaluation_metric string
The text being graded.
input string
name string
## OBJECT Text Similarity Grader
{
  "type": "text_similarity",
  "name": "Example text similarity grader",
  "input": "{{sample.output_text}}",
  "reference": "{{item.label}}",
  "evaluation_metric": "fuzzy_match"
}


<!-- Page 133 -->
## A ScoreModelGrader object that uses a model to assign a score to the
input.
The name of the grader.
The text being graded against.
reference string
The type of grader.
type string
## Score Model Grader
The input text. This may include template strings.
## Show properties
input array
The model to use for the evaluation.
model string
The name of the grader.
name string
The range of the score. Defaults to [0, 1] .
range array
The sampling parameters for the model.
sampling_params object
type string
## OBJECT Score Model Grader
 
 
 
 
{
    "type": "score_model",
    "name": "Example score model grader",
    "input": [
        {
            "role": "user",
            "content": (
                "Score how close the reference 
                " Return just a floating point 
                " Reference answer: {{item.labe
                " Model answer: {{sample.output
            ),
        }
    ],
    "model": "gpt-4o-2024-08-06",
    "sampling_params": {
        "temperature": 1,
        "top_p": 1,
        "seed": 42,


<!-- Page 134 -->
## A LabelModelGrader object which uses a model to assign labels to each
item in the evaluation.
The object type, which is always score_model .
## Label Model Grader
Show properties
input array
The labels to assign to each item in the evaluation.
labels array
The model to use for the evaluation. Must support structured outputs.
model string
The name of the grader.
name string
The labels that indicate a passing result. Must be a subset of labels.
passing_labels array
The object type, which is always label_model .
type string
## OBJECT Label Model Grader
 
{
  "name": "First label grader",
  "type": "label_model",
  "model": "gpt-4o-2024-08-06",
  "input": [
    {
      "type": "message",
      "role": "system",
      "content": {
        "type": "input_text",
        "text": "Classify the sentiment of the 
      }
    },
    {
      "type": "message",
      "role": "user",
      "content": {
        "type": "input_text",
        "text": "Statement: {{item.response}}"
      }
    }
  ],
  "passing_labels": [
    "positive"
  ],
  "labels": [
    "positive",
    "neutral",
    "negative"


<!-- Page 135 -->
A PythonGrader object that runs a python script on the input.
## A MultiGrader object combines the output of multiple graders to produce a
single score.
 
  ]
}

## Python Grader
The image tag to use for the python script.
image_tag string
The name of the grader.
name string
The source code of the python script.
source string
The object type, which is always python .
type string
## OBJECT Python Grader
 
 
{
  "type": "python",
  "name": "Example python grader",
  "image_tag": "2025-05-08",
  "source": """
def grade(sample: dict, item: dict) -> float:
    \"""
    Returns 1.0 if `output_text` equals `label`,
    \"""
    output = sample.get("output_text")
    label = item.get("label")
    return 1.0 if output == label else 0.0
""",
}

## Multi Grader
calculate_output string
## OBJECT Multi Grader
{
  "type": "multi",
  "name": "example multi grader",
  "graders": [


<!-- Page 136 -->
POST https://api.openai.com/v1/fine_tuning/alpha/graders/run
Run a grader.
## Request body
A formula to calculate the output based on grader results.
## Show possible types
graders object
The name of the grader.
name string
The object type, which is always multi .
type string
    {
      "type": "text_similarity",
      "name": "example text similarity grader",
      "input": "The graded text",
      "reference": "The reference text",
      "evaluation_metric": "fuzzy_match"
    },
    {
      "type": "string_check",
      "name": "Example string check grader",
      "input": "{{sample.output_text}}",
      "reference": "{{item.label}}",
      "operation": "eq"
    }
  ],
  "calculate_output": "0.5 * text_similarity_sco
}

## Run grader
Beta
The grader used for the fine-tuning job.
## Show possible types
grader object
Required
model_sample string
## Required
Example request
curl
 
curl -X POST https://api.openai.com/v1/fine_tun
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "grader": {
      "type": "score_model",
      "name": "Example score model grader",
      "input": [
        {
          "role": "user",
          "content": "Score how close the refer
        }
      ],


<!-- Page 137 -->
## Returns
POST https://api.openai.com/v1/fine_tuning/alpha/graders/validate
Validate a grader.
## Request body
The model sample to be evaluated. This value will be used to populate the sample
namespace. See the guide for more details. The output_json  variable will be
populated if the model sample is a valid JSON string.
The dataset item provided to the grader. This will be used to populate the item
namespace. See the guide for more details.
item object
## Optional
The results from the grader run.
 
      "model": "gpt-4o-2024-08-06",
      "sampling_params": {
        "temperature": 1,
        "top_p": 1,
        "seed": 42
      }
    },
    "item": {
      "reference_answer": "fuzzy wuzzy was a be
    },
    "model_sample": "fuzzy wuzzy was a bear"
  }'

## Response
 
{
  "reward": 1.0,
  "metadata": {
    "name": "Example score model grader",
    "type": "score_model",
    "errors": {
      "formula_parse_error": false,
      "sample_parse_error": false,
      "truncated_observation_error": false,
      "unresponsive_reward_error": false,
      "invalid_variable_error": false,
      "other_error": false,
      "python_grader_server_error": false,
      "python_grader_server_error_type": null,
      "python_grader_runtime_error": false,
      "python_grader_runtime_error_details": nu
      "model_grader_server_error": false,
      "model_grader_refusal_error": false,
      "model_grader_parse_error": false,
      "model_grader_server_error_details": null
    },
    "execution_time": 4.365238428115845,
    "scores": {},
    "token_usage": {

## Validate grader
Beta
The grader used for the fine-tuning job.
## Show possible types
grader object
Required
Example request
curl
 
curl https://api.openai.com/v1/fine_tuning/alph
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "grader": {
      "type": "string_check",
      "name": "Example string check grader",
      "input": "{{sample.output_text}}",
      "reference": "{{item.label}}",
      "operation": "eq"


<!-- Page 138 -->
## Returns
Create large batches of API requests for asynchronous processing. The Batch API returns completions
within 24 hours for a 50% discount. Related guide: Batch
POST https://api.openai.com/v1/batches
## Creates and executes a batch from an uploaded file of requests
Request body
 
      "prompt_tokens": 190,
      "total_tokens": 324,
      "completion_tokens": 134,
      "cached_tokens": 0
    },
    "sampled_model_name": "gpt-4o-2024-08-06"
  },
  "sub_rewards": {},
  "model_grader_token_usage_per_model": {
    "gpt-4o-2024-08-06": {
      "prompt_tokens": 190,
      "total_tokens": 324,
      "completion_tokens": 134,
      "cached_tokens": 0
    }
  }
}

The validated grader object.
 
    }
  }'

## Response
{
  "grader": {
    "type": "string_check",
    "name": "Example string check grader",
    "input": "{{sample.output_text}}",
    "reference": "{{item.label}}",
    "operation": "eq"
  }
}

## Batch
Create batch
completion_window string
## Required
Example request
python
 
from openai import OpenAI
client = OpenAI()
client.batches.create(
  input_file_id="file-abc123",
  endpoint="/v1/chat/completions",


<!-- Page 139 -->
## Returns
The time frame within which the batch should be processed. Currently only 24h  is
supported.
The endpoint to be used for all requests in the batch. Currently /v1/responses ,
/v1/chat/completions , /v1/embeddings , and /v1/completions  are supported.
Note that /v1/embeddings  batches are also restricted to a maximum of 50,000
embedding inputs across all requests in the batch.
endpoint string
## Required
The ID of an uploaded file that contains requests for the new batch.
See upload file for how to upload a file.
Your input file must be formatted as a JSONL file, and must be uploaded with the
purpose batch . The file can contain up to 50,000 requests, and can be up to 200 MB
in size.
input_file_id string
## Required
Set of 16 key-value pairs that can be attached to an object. This can be useful for
storing additional information about the object in a structured format, and querying for objects via API or the dashboard.
Keys are strings with a maximum length of 64 characters. Values are strings with a
maximum length of 512 characters.
metadata
map
## Optional
The created Batch object.
 
  completion_window="24h"
)

## Response
{
  "id": "batch_abc123",
  "object": "batch",
  "endpoint": "/v1/chat/completions",
  "errors": null,
  "input_file_id": "file-abc123",
  "completion_window": "24h",
  "status": "validating",
  "output_file_id": null,
  "error_file_id": null,
  "created_at": 1711471533,
  "in_progress_at": null,
  "expires_at": null,
  "finalizing_at": null,
  "completed_at": null,
  "failed_at": null,
  "expired_at": null,
  "cancelling_at": null,
  "cancelled_at": null,
  "request_counts": {
    "total": 0,
    "completed": 0,
    "failed": 0
  },
  "metadata": {
    "customer_id": "user_123456789",
    "batch_description": "Nightly eval job",
  }
}

## Retrieve batch

<!-- Page 140 -->
GET https://api.openai.com/v1/batches/{batch_id}
Retrieves a batch.
## Path parameters
Returns
The ID of the batch to retrieve.
batch_id string
## Required
The Batch object matching the specified ID.
## Example request
python
from openai import OpenAI
client = OpenAI()
client.batches.retrieve("batch_abc123")

## Response
{
  "id": "batch_abc123",
  "object": "batch",
  "endpoint": "/v1/completions",
  "errors": null,
  "input_file_id": "file-abc123",
  "completion_window": "24h",
  "status": "completed",
  "output_file_id": "file-cvaTdG",
  "error_file_id": "file-HOWS94",
  "created_at": 1711471533,
  "in_progress_at": 1711471538,
  "expires_at": 1711557933,
  "finalizing_at": 1711493133,
  "completed_at": 1711493163,
  "failed_at": null,
  "expired_at": null,
  "cancelling_at": null,
  "cancelled_at": null,
  "request_counts": {
    "total": 100,
    "completed": 95,
    "failed": 5
  },
  "metadata": {
    "customer_id": "user_123456789",
    "batch_description": "Nightly eval job",
  }
}


<!-- Page 141 -->
POST https://api.openai.com/v1/batches/{batch_id}/cancel
Cancels an in-progress batch. The batch will be in status cancelling  for up
to 10 minutes, before changing to cancelled , where it will have partial
results (if any) available in the output file.
## Path parameters
Returns
Cancel batch
The ID of the batch to cancel.
batch_id string
## Required
The Batch object matching the specified ID.
## Example request
python
from openai import OpenAI
client = OpenAI()
client.batches.cancel("batch_abc123")

## Response
 
{
  "id": "batch_abc123",
  "object": "batch",
  "endpoint": "/v1/chat/completions",
  "errors": null,
  "input_file_id": "file-abc123",
  "completion_window": "24h",
  "status": "cancelling",
  "output_file_id": null,
  "error_file_id": null,
  "created_at": 1711471533,
  "in_progress_at": 1711471538,
  "expires_at": 1711557933,
  "finalizing_at": null,
  "completed_at": null,
  "failed_at": null,
  "expired_at": null,
  "cancelling_at": 1711475133,
  "cancelled_at": null,
  "request_counts": {
    "total": 100,
    "completed": 23,
    "failed": 1
  },
  "metadata": {
    "customer_id": "user_123456789",
    "batch_description": "Nightly eval job",


<!-- Page 142 -->
GET https://api.openai.com/v1/batches
List your organization's batches.
## Query parameters
Returns
 
  }
}

## List batch
A cursor for use in pagination. after  is an object ID that defines your place in the list.
For instance, if you make a list request and receive 100 objects, ending with obj_foo,
your subsequent call can include after=obj_foo in order to fetch the next page of the
list.
after string
## Optional
A limit on the number of objects to be returned. Limit can range between 1 and 100,
and the default is 20.
limit integer
## Optional
Defaults to 20
A list of paginated Batch objects.
## Example request
python
from openai import OpenAI
client = OpenAI()
client.batches.list()

## Response
 
{
  "object": "list",
  "data": [
    {
      "id": "batch_abc123",
      "object": "batch",
      "endpoint": "/v1/chat/completions",
      "errors": null,
      "input_file_id": "file-abc123",
      "completion_window": "24h",
      "status": "completed",
      "output_file_id": "file-cvaTdG",
      "error_file_id": "file-HOWS94",
      "created_at": 1711471533,
      "in_progress_at": 1711471538,
      "expires_at": 1711557933,
      "finalizing_at": 1711493133,
      "completed_at": 1711493163,
      "failed_at": null,
      "expired_at": null,
      "cancelling_at": null,
      "cancelled_at": null,
      "request_counts": {


<!-- Page 143 -->
 
        "total": 100,
        "completed": 95,
        "failed": 5
      },
      "metadata": {
        "customer_id": "user_123456789",
        "batch_description": "Nightly job",
      }
    },
    { ... },
  ],
  "first_id": "batch_abc123",
  "last_id": "batch_abc456",
  "has_more": true
}

## The batch object
The Unix timestamp (in seconds) for when the batch was cancelled.
cancelled_at integer
The Unix timestamp (in seconds) for when the batch started cancelling.
cancelling_at integer
The Unix timestamp (in seconds) for when the batch was completed.
completed_at integer
The time frame within which the batch should be processed.
completion_window string
The Unix timestamp (in seconds) for when the batch was created.
created_at integer
The OpenAI API endpoint used by the batch.
endpoint string
The ID of the file containing the outputs of requests with errors.
error_file_id string
## Show properties
errors object
expired_at integer
## OBJECT The batch object
 
{
  "id": "batch_abc123",
  "object": "batch",
  "endpoint": "/v1/completions",
  "errors": null,
  "input_file_id": "file-abc123",
  "completion_window": "24h",
  "status": "completed",
  "output_file_id": "file-cvaTdG",
  "error_file_id": "file-HOWS94",
  "created_at": 1711471533,
  "in_progress_at": 1711471538,
  "expires_at": 1711557933,
  "finalizing_at": 1711493133,
  "completed_at": 1711493163,
  "failed_at": null,
  "expired_at": null,
  "cancelling_at": null,
  "cancelled_at": null,
  "request_counts": {
    "total": 100,
    "completed": 95,
    "failed": 5
  },
  "metadata": {
    "customer_id": "user_123456789",
    "batch_description": "Nightly eval job",


<!-- Page 144 -->
The Unix timestamp (in seconds) for when the batch expired.
The Unix timestamp (in seconds) for when the batch will expire.
expires_at integer
The Unix timestamp (in seconds) for when the batch failed.
failed_at integer
The Unix timestamp (in seconds) for when the batch started finalizing.
finalizing_at integer
id string
The Unix timestamp (in seconds) for when the batch started processing.
in_progress_at integer
The ID of the input file for the batch.
input_file_id string
Set of 16 key-value pairs that can be attached to an object. This can be useful for
storing additional information about the object in a structured format, and querying for objects via API or the dashboard.
Keys are strings with a maximum length of 64 characters. Values are strings with a
maximum length of 512 characters.
metadata
map
The object type, which is always batch .
object string
The ID of the file containing the outputs of successfully executed requests.
output_file_id string
The request counts for different statuses within the batch.
request_counts object
 
}

<!-- Page 145 -->
## The per-line object of the batch input file
The per-line object of the batch output and error files
Show properties
The current status of the batch.
status string
## The request input object
A developer-provided per-request id that will be used to match outputs to inputs. Must
be unique for each request in a batch.
custom_id string
The HTTP method to be used for the request. Currently only POST  is supported.
method string
The OpenAI API relative URL to be used for the request. Currently
/v1/chat/completions , /v1/embeddings , and /v1/completions  are supported.
url string
## OBJECT The request input object
 
 
{"custom_id": "request-1", "method": "POST", "url": 
## The request output object
A developer-provided per-request id that will be used to match outputs to inputs.
custom_id string
## OBJECT The request output object
 
 
{"id": "batch_req_wnaDys", "custom_id": "request-2",

<!-- Page 146 -->
Files are used to upload documents that can be used with features like Assistants, Fine-tuning, and
Batch API.
POST https://api.openai.com/v1/files
Upload a file that can be used across various endpoints. Individual files can
be up to 512 MB, and the size of all files uploaded by one organization can
be up to 100 GB.
The Assistants API supports files up to 2 million tokens and of specific file
types. See the Assistants Tools guide for details.
The Fine-tuning API only supports .jsonl  files. The input also has certain
required formats for fine-tuning chat or completions models.
For requests that failed with a non-HTTP error, this will contain more information on
the cause of the failure.
## Show properties
error object or null
id string
Show properties
response object or null
Files
Upload file
Example request
python
from openai import OpenAI
client = OpenAI()
client.files.create(
  file=open("mydata.jsonl", "rb"),
  purpose="fine-tune"
)


<!-- Page 147 -->
The Batch API only supports .jsonl  files up to 200 MB in size. The input
also has a specific required format.
Please contact us if you need to increase these storage limits.
## Request body
Returns
GET https://api.openai.com/v1/files
Returns a list of files.
## Query parameters
The File object (not file name) to be uploaded.
file
file
## Required
The intended purpose of the uploaded file. One of: - assistants : Used in the
Assistants API - batch : Used in the Batch API - fine-tune : Used for fine-tuning -
vision : Images used for vision fine-tuning - user_data : Flexible file type for any
purpose - evals : Used for eval data sets
purpose string
## Required
The uploaded File object.
## Response
{
  "id": "file-abc123",
  "object": "file",
  "bytes": 120000,
  "created_at": 1677610602,
  "filename": "mydata.jsonl",
  "purpose": "fine-tune",
}

## List files
after string
Optional
Example request
python
from openai import OpenAI
client = OpenAI()
client.files.list()


<!-- Page 148 -->
## Returns
GET https://api.openai.com/v1/files/{file_id}
Returns information about a specific file.
## Path parameters
A cursor for use in pagination. after  is an object ID that defines your place in the list.
For instance, if you make a list request and receive 100 objects, ending with obj_foo,
your subsequent call can include after=obj_foo in order to fetch the next page of the
list.
A limit on the number of objects to be returned. Limit can range between 1 and 10,000,
and the default is 10,000.
limit integer
## Optional
Defaults to 10000
Sort order by the created_at  timestamp of the objects. asc  for ascending order
and desc  for descending order.
order string
## Optional
Defaults to desc
Only return files with the given purpose.
purpose string
## Optional
A list of File objects.
## Response
{
  "object": "list",
  "data": [
    {
      "id": "file-abc123",
      "object": "file",
      "bytes": 175,
      "created_at": 1613677385,
      "filename": "salesOverview.pdf",
      "purpose": "assistants",
    },
    {
      "id": "file-abc456",
      "object": "file",
      "bytes": 140,
      "created_at": 1613779121,
      "filename": "puppy.jsonl",
      "purpose": "fine-tune",
    }
  ],
  "first_id": "file-abc123",
  "last_id": "file-abc456",
  "has_more": false
}

## Retrieve file
Example request
python
from openai import OpenAI
client = OpenAI()
client.files.retrieve("file-abc123")


<!-- Page 149 -->
## Returns
DELETE https://api.openai.com/v1/files/{file_id}
Delete a file.
## Path parameters
Returns
The ID of the file to use for this request.
file_id string
## Required
The File object matching the specified ID.
## Response
{
  "id": "file-abc123",
  "object": "file",
  "bytes": 120000,
  "created_at": 1677610602,
  "filename": "mydata.jsonl",
  "purpose": "fine-tune",
}

## Delete file
The ID of the file to use for this request.
file_id string
## Required
Deletion status.
## Example request
python
from openai import OpenAI
client = OpenAI()
client.files.delete("file-abc123")

## Response
{
  "id": "file-abc123",
  "object": "file",
  "deleted": true
}


<!-- Page 150 -->
GET https://api.openai.com/v1/files/{file_id}/content
Returns the contents of the specified file.
## Path parameters
Returns
The File  object represents a document that has been uploaded to
OpenAI.
## Retrieve file content
The ID of the file to use for this request.
file_id string
## Required
The file content.
## Example request
python
from openai import OpenAI
client = OpenAI()
content = client.files.content("file-abc123")

## The file object
The size of the file, in bytes.
bytes integer
The Unix timestamp (in seconds) for when the file was created.
created_at integer
expires_at integer
## OBJECT The file object
{
  "id": "file-abc123",
  "object": "file",
  "bytes": 120000,
  "created_at": 1677610602,
  "expires_at": 1680202602,
  "filename": "salesOverview.pdf",
  "purpose": "assistants",
}


<!-- Page 151 -->
Allows you to upload large files in multiple parts.
The Unix timestamp (in seconds) for when the file will expire.
The name of the file.
filename string
The file identifier, which can be referenced in the API endpoints.
id string
The object type, which is always file .
object string
The intended purpose of the file. Supported values are assistants ,
assistants_output , batch , batch_output , fine-tune , fine-tune-results ,
vision , and user_data .
purpose string
Deprecated. The current status of the file, which can be either uploaded ,
processed , or error .
status
## Deprecated string
Deprecated. For details on why a fine-tuning training file failed validation, see the
error  field on fine_tuning.job .
status_details
## Deprecated string
Uploads

<!-- Page 152 -->
POST https://api.openai.com/v1/uploads
Creates an intermediate Upload object that you can add Parts to. Currently,
an Upload can accept at most 8 GB in total and expires after an hour after
you create it.
Once you complete the Upload, we will create a File object that contains all
the parts you uploaded. This File is usable in the rest of our platform as a
regular File object.
For certain purpose  values, the correct mime_type  must be specified.
## Please refer to documentation for the
supported MIME types for your use case.
For guidance on the proper filename extensions for each purpose, please
follow the documentation on creating a File.
## Request body
Create upload
The number of bytes in the file you are uploading.
bytes integer
## Required
The name of the file to upload.
filename string
## Required
The MIME type of the file.
This must fall within the supported MIME types for your file purpose. See the
supported MIME types for assistants and vision.
mime_type string
## Required
The intended purpose of the uploaded file.
See the documentation on File purposes.
purpose string
## Required
Example request
curl
curl https://api.openai.com/v1/uploads \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "purpose": "fine-tune",
    "filename": "training_examples.jsonl",
    "bytes": 2147483648,
    "mime_type": "text/jsonl"
  }'

## Response
{
  "id": "upload_abc123",
  "object": "upload",
  "bytes": 2147483648,
  "created_at": 1719184911,
  "filename": "training_examples.jsonl",
  "purpose": "fine-tune",
  "status": "pending",
  "expires_at": 1719127296
}


<!-- Page 153 -->
## Returns
POST https://api.openai.com/v1/uploads/{upload_id}/parts
Adds a Part to an Upload object. A Part represents a chunk of bytes from
the file you are trying to upload.
Each Part can be at most 64 MB, and you can add Parts until you hit the
Upload maximum of 8 GB.
It is possible to add multiple Parts in parallel. You can decide the intended
order of the Parts when you complete the Upload.
## Path parameters
Request body
Returns
The Upload object with status pending .
## Add upload part
The ID of the Upload.
upload_id string
## Required
The chunk of bytes for this Part.
data
file
## Required
Example request
curl
 
 
curl https://api.openai.com/v1/uploads/upload_abc
  -F data="aHR0cHM6Ly9hcGkub3BlbmFpLmNvbS92MS91cG

## Response
{
  "id": "part_def456",
  "object": "upload.part",
  "created_at": 1719185911,
  "upload_id": "upload_abc123"
}


<!-- Page 154 -->
POST https://api.openai.com/v1/uploads/{upload_id}/complete
Completes the Upload.
Within the returned Upload object, there is a nested File object that is ready
to use in the rest of the platform.
## You can specify the order of the Parts by passing in an ordered list of the
Part IDs.
## The number of bytes uploaded upon completion must match the number of
bytes initially specified when creating the Upload object. No Parts may be
added after an Upload is completed.
## Path parameters
Request body
The upload Part object.
## Complete upload
The ID of the Upload.
upload_id string
## Required
The ordered list of Part IDs.
part_ids array
## Required
md5 string
## Optional
Example request
curl
 
 
curl https://api.openai.com/v1/uploads/upload_abc
  -d '{
    "part_ids": ["part_def456", "part_ghi789"]
  }'

## Response
{
  "id": "upload_abc123",
  "object": "upload",
  "bytes": 2147483648,
  "created_at": 1719184911,
  "filename": "training_examples.jsonl",
  "purpose": "fine-tune",
  "status": "completed",
  "expires_at": 1719127296,
  "file": {
    "id": "file-xyz321",
    "object": "file",
    "bytes": 2147483648,
    "created_at": 1719186911,
    "filename": "training_examples.jsonl",
    "purpose": "fine-tune",
  }
}


<!-- Page 155 -->
## Returns
POST https://api.openai.com/v1/uploads/{upload_id}/cancel
Cancels the Upload. No Parts may be added after an Upload is cancelled.
## Path parameters
Returns
The optional md5 checksum for the file contents to verify if the bytes uploaded
matches what you expect.
## The Upload object with status completed  with an additional file  property
containing the created usable File object.
## Cancel upload
The ID of the Upload.
upload_id string
## Required
The Upload object with status cancelled .
## Example request
curl
 
 
curl https://api.openai.com/v1/uploads/upload_abc123
## Response
{
  "id": "upload_abc123",
  "object": "upload",
  "bytes": 2147483648,
  "created_at": 1719184911,
  "filename": "training_examples.jsonl",
  "purpose": "fine-tune",
  "status": "cancelled",
  "expires_at": 1719127296
}


<!-- Page 156 -->
The Upload object can accept byte chunks in the form of Parts.
## The upload object
The intended number of bytes to be uploaded.
bytes integer
The Unix timestamp (in seconds) for when the Upload was created.
created_at integer
The Unix timestamp (in seconds) for when the Upload will expire.
expires_at integer
The ready File object after the Upload is completed.
file
undefined or null
The name of the file to be uploaded.
filename string
The Upload unique identifier, which can be referenced in API endpoints.
id string
The object type, which is always "upload".
object string
The intended purpose of the file. Please refer here for acceptable values.
purpose string
The status of the Upload.
status string
## OBJECT The upload object
{
  "id": "upload_abc123",
  "object": "upload",
  "bytes": 2147483648,
  "created_at": 1719184911,
  "filename": "training_examples.jsonl",
  "purpose": "fine-tune",
  "status": "completed",
  "expires_at": 1719127296,
  "file": {
    "id": "file-xyz321",
    "object": "file",
    "bytes": 2147483648,
    "created_at": 1719186911,
    "filename": "training_examples.jsonl",
    "purpose": "fine-tune",
  }
}


<!-- Page 157 -->
## The upload Part represents a chunk of bytes we can add to an Upload object.
List and describe the various models available in the API. You can refer to the Models documentation to
understand what models are available and the differences between them.
GET https://api.openai.com/v1/models
## The upload part object
The Unix timestamp (in seconds) for when the Part was created.
created_at integer
The upload Part unique identifier, which can be referenced in API endpoints.
id string
The object type, which is always upload.part .
object string
The ID of the Upload object that this Part was added to.
upload_id string
## OBJECT The upload part object
{
    "id": "part_def456",
    "object": "upload.part",
    "created_at": 1719186911,
    "upload_id": "upload_abc123"
}

## Models
List models
Example request
python

<!-- Page 158 -->
Lists the currently available models, and provides basic information about
each one such as the owner and availability.
## Returns
GET https://api.openai.com/v1/models/{model}
A list of model objects.
from openai import OpenAI
client = OpenAI()

## Response
{
  "object": "list",
  "data": [
    {
      "id": "model-id-0",
      "object": "model",
      "created": 1686935002,
      "owned_by": "organization-owner"
    },
    {
      "id": "model-id-1",
      "object": "model",
      "created": 1686935002,
      "owned_by": "organization-owner",
    },
    {
      "id": "model-id-2",
      "object": "model",
      "created": 1686935002,
      "owned_by": "openai"
    },
  ],
  "object": "list"
}

## Retrieve model
Example request
gpt-5
python

<!-- Page 159 -->
Retrieves a model instance, providing basic information about the model
such as the owner and permissioning.
## Path parameters
Returns
DELETE https://api.openai.com/v1/models/{model}
Delete a fine-tuned model. You must have the Owner role in your
organization to delete a model.
## Path parameters
Returns
The ID of the model to use for this request
model string
Required
The model object matching the specified ID.
from openai import OpenAI
client = OpenAI()
client.models.retrieve("gpt-5")

## Response
{
  "id": "gpt-5",
  "object": "model",
  "created": 1686935002,
  "owned_by": "openai"
}

## Delete a fine-tuned model
The model to delete
model string
Required
Example request
python
 
 
from openai import OpenAI
client = OpenAI()
client.models.delete("ft:gpt-4o-mini:acemeco:suff

## Response
 
 
{
  "id": "ft:gpt-4o-mini:acemeco:suffix:abc123",
  "object": "model",
  "deleted": true
}


<!-- Page 160 -->
Describes an OpenAI model offering that can be used with the API.
Given text and/or image inputs, classifies if those inputs are potentially harmful across several categories.
Related guide: Moderations
Deletion status.
## The model object
The Unix timestamp (in seconds) when the model was created.
created integer
The model identifier, which can be referenced in the API endpoints.
id string
The object type, which is always "model".
object string
The organization that owns the model.
owned_by string
## OBJECT The model object
{
  "id": "gpt-5",
  "object": "model",
  "created": 1686935002,
  "owned_by": "openai"
}

## Moderations

<!-- Page 161 -->
POST https://api.openai.com/v1/moderations
Classifies if text and/or image inputs are potentially harmful. Learn more in
the moderation guide.
## Request body
Returns
Create moderation
Input (or inputs) to classify. Can be a single string, an array of strings, or an array of
multi-modal input objects similar to other models.
## Show possible types
input string or array
Required
The content moderation model you would like to use. Learn more in
the moderation guide, and learn about available models here.
model string
## Optional
Defaults to omni-moderation-latest
A moderation object.
## Single string
Image and text
Example request
python
 
 
from openai import OpenAI
client = OpenAI()
moderation = client.moderations.create(input="I w
print(moderation)

## Response
 
{
  "id": "modr-AB8CjOTu2jiq12hp1AQPfeqFWaORR",
  "model": "text-moderation-007",
  "results": [
    {
      "flagged": true,
      "categories": {
        "sexual": false,
        "hate": false,
        "harassment": true,
        "self-harm": false,
        "sexual/minors": false,
        "hate/threatening": false,
        "violence/graphic": false,
        "self-harm/intent": false,
        "self-harm/instructions": false,
        "harassment/threatening": true,
        "violence": true
      },
      "category_scores": {
        "sexual": 0.000011726012417057063,
        "hate": 0.22706663608551025,
        "harassment": 0.5215635299682617,
        "self-harm": 2.227119921371923e-6,


<!-- Page 162 -->
Represents if a given text input is potentially harmful.
 
        "sexual/minors": 7.107352217872176e-8,
        "hate/threatening": 0.02354732900857925
        "violence/graphic": 0.00003391829886822
        "self-harm/intent": 1.646940972932498e-
        "self-harm/instructions": 1.11987552564
        "harassment/threatening": 0.56947457790
        "violence": 0.9971134662628174
      }
    }
  ]
}

## The moderation object
The unique identifier for the moderation request.
id string
The model used to generate the moderation results.
model string
A list of moderation objects.
## Show properties
results array
OBJECT The moderation object
 
{
  "id": "modr-0d9740456c391e43c445bf0f010940c7"
  "model": "omni-moderation-latest",
  "results": [
    {
      "flagged": true,
      "categories": {
        "harassment": true,
        "harassment/threatening": true,
        "sexual": false,
        "hate": false,
        "hate/threatening": false,
        "illicit": false,
        "illicit/violent": false,
        "self-harm/intent": false,
        "self-harm/instructions": false,
        "self-harm": false,
        "sexual/minors": false,
        "violence": true,
        "violence/graphic": true
      },
      "category_scores": {
        "harassment": 0.8189693396524255,
        "harassment/threatening": 0.80498542069
        "sexual": 1.573112165348997e-6,
        "hate": 0.007562942636942845,
        "hate/threatening": 0.00420885459183547
        "illicit": 0.030535955153511665,
        "illicit/violent": 0.008925306722380033
        "self-harm/intent": 0.00023023930975076
        "self-harm/instructions": 0.00022938692


<!-- Page 163 -->
Vector stores power semantic search for the Retrieval API and the file_search  tool in the Responses and
Assistants APIs.
Related guide: File Search
POST https://api.openai.com/v1/vector_stores
Create a vector store.
## Request body
 
        "self-harm": 0.012598046106750154,
        "sexual/minors": 2.212566909570261e-8,
        "violence": 0.9999992735124786,
        "violence/graphic": 0.843064871157054
      },
      "category_applied_input_types": {
        "harassment": [
          "text"
        ],
        "harassment/threatening": [
          "text"
        ],
        "sexual": [
          "text",
          "image"
        ],
        "hate": [
          "text"
        ],
        "hate/threatening": [
          "text"
        ],
        "illicit": [
          "text"
        ],
        "illicit/violent": [
          "text"
        ],
        "self-harm/intent": [
          "text",
          "image"
        ],
        "self-harm/instructions": [
          "text",
          "image"
        ],
        "self-harm": [
          "text",
          "image"
        ],

## Vector stores
Create vector store
The chunking strategy used to chunk the file(s). If not set, will use the auto  strategy.
Only applicable if file_ids  is non-empty.
## Show possible types
chunking_strategy object
## Optional
The expiration policy for a vector store.
## Show properties
expires_after object
## Optional
file_ids array
## Optional
Example request
python
from openai import OpenAI
client = OpenAI()
vector_store = client.vector_stores.create(
  name="Support FAQ"
)
print(vector_store)

## Response
 
{
  "id": "vs_abc123",
  "object": "vector_store",
  "created_at": 1699061776,
  "name": "Support FAQ",
  "bytes": 139920,


<!-- Page 164 -->
## Returns
GET https://api.openai.com/v1/vector_stores
Returns a list of vector stores.
## Query parameters
 
        "sexual/minors": [
          "text"
        ],
        "violence": [
          "text",
          "image"
        ],
        "violence/graphic": [
          "text",
          "image"
        ]
      }
    }
  ]
}

A list of File IDs that the vector store should use. Useful for tools like file_search
that can access files.
Set of 16 key-value pairs that can be attached to an object. This can be useful for
storing additional information about the object in a structured format, and querying for objects via API or the dashboard.
Keys are strings with a maximum length of 64 characters. Values are strings with a
maximum length of 512 characters.
metadata
map
## Optional
The name of the vector store.
name string
## Optional
A vector store object.
 
  "file_counts": {
    "in_progress": 0,
    "completed": 3,
    "failed": 0,
    "cancelled": 0,
    "total": 3
  }
}

## List vector stores
A cursor for use in pagination. after  is an object ID that defines your place in the list.
For instance, if you make a list request and receive 100 objects, ending with obj_foo,
after string
## Optional
Example request
python
from openai import OpenAI
client = OpenAI()
vector_stores = client.vector_stores.list()
print(vector_stores)

## Response

<!-- Page 165 -->
## Returns
GET https://api.openai.com/v1/vector_stores/{vector_store_id}
Retrieves a vector store.
## Path parameters
your subsequent call can include after=obj_foo in order to fetch the next page of the
list.
A cursor for use in pagination. before  is an object ID that defines your place in the
list. For instance, if you make a list request and receive 100 objects, starting with
obj_foo, your subsequent call can include before=obj_foo in order to fetch the previous
page of the list.
before string
## Optional
A limit on the number of objects to be returned. Limit can range between 1 and 100,
and the default is 20.
limit integer
## Optional
Defaults to 20
Sort order by the created_at  timestamp of the objects. asc  for ascending order
and desc  for descending order.
order string
## Optional
Defaults to desc
A list of vector store objects.
{
  "object": "list",
  "data": [
    {
      "id": "vs_abc123",
      "object": "vector_store",
      "created_at": 1699061776,
      "name": "Support FAQ",
      "bytes": 139920,
      "file_counts": {
        "in_progress": 0,
        "completed": 3,
        "failed": 0,
        "cancelled": 0,
        "total": 3
      }
    },
    {
      "id": "vs_abc456",
      "object": "vector_store",
      "created_at": 1699061776,
      "name": "Support FAQ v2",
      "bytes": 139920,
      "file_counts": {
        "in_progress": 0,
        "completed": 3,
        "failed": 0,
        "cancelled": 0,
        "total": 3
      }
    }
  ],
  "first_id": "vs_abc123",
  "last_id": "vs_abc456",
  "has_more": false

## Retrieve vector store
Example request
python
 
from openai import OpenAI
client = OpenAI()
vector_store = client.vector_stores.retrieve(
  vector_store_id="vs_abc123"


<!-- Page 166 -->
## Returns
POST https://api.openai.com/v1/vector_stores/{vector_store_id}
Modifies a vector store.
## Path parameters
Request body
The ID of the vector store to retrieve.
vector_store_id string
## Required
The vector store object matching the specified ID.
 
)
print(vector_store)

## Response
{
  "id": "vs_abc123",
  "object": "vector_store",
  "created_at": 1699061776
}

## Modify vector store
The ID of the vector store to modify.
vector_store_id string
## Required
The expiration policy for a vector store.
## Show properties
expires_after object or null
## Optional
metadata
map
Optional
Example request
python
from openai import OpenAI
client = OpenAI()
vector_store = client.vector_stores.update(
  vector_store_id="vs_abc123",
  name="Support FAQ"
)
print(vector_store)

## Response
 
{
  "id": "vs_abc123",
  "object": "vector_store",
  "created_at": 1699061776,
  "name": "Support FAQ",
  "bytes": 139920,
  "file_counts": {
    "in_progress": 0,


<!-- Page 167 -->
## Returns
DELETE https://api.openai.com/v1/vector_stores/{vector_store_id}
Delete a vector store.
## Path parameters
Returns
Set of 16 key-value pairs that can be attached to an object. This can be useful for
storing additional information about the object in a structured format, and querying for objects via API or the dashboard.
Keys are strings with a maximum length of 64 characters. Values are strings with a
maximum length of 512 characters.
The name of the vector store.
name string or null
## Optional
The modified vector store object.
 
    "completed": 3,
    "failed": 0,
    "cancelled": 0,
    "total": 3
  }
}

## Delete vector store
The ID of the vector store to delete.
vector_store_id string
## Required
Example request
python
 
 
from openai import OpenAI
client = OpenAI()
deleted_vector_store = client.vector_stores.delet
  vector_store_id="vs_abc123"
)
print(deleted_vector_store)

## Response
 
{
  id: "vs_abc123",
  object: "vector_store.deleted",


<!-- Page 168 -->
POST https://api.openai.com/v1/vector_stores/{vector_store_id}/search
## Search a vector store for relevant chunks based on a query and file
attributes filter.
## Path parameters
Request body
Deletion status
 
  deleted: true
}

## Search vector store
The ID of the vector store to search.
vector_store_id string
## Required
A query string for a search
query string or array
Required
A filter to apply based on file attributes.
## Show possible types
filters object
Optional
The maximum number of results to return. This number should be between 1 and 50
inclusive.
max_num_results integer
## Optional
Defaults to 10
ranking_options object
## Optional
Example request
curl
 
 
curl -X POST \
https://api.openai.com/v1/vector_stores/vs_abc123
-H "Authorization: Bearer $OPENAI_API_KEY" \
-H "Content-Type: application/json" \
-d '{"query": "What is the return policy?", "filt

## Response
 
{
  "object": "vector_store.search_results.page",
  "search_query": "What is the return policy?",
  "data": [
    {
      "file_id": "file_123",
      "filename": "document.pdf",
      "score": 0.95,
      "attributes": {
        "author": "John Doe",
        "date": "2023-01-01"
      },
      "content": [
        {
          "type": "text",
          "text": "Relevant chunk"
        }
      ]


<!-- Page 169 -->
## Returns
A vector store is a collection of processed files can be used by the
file_search  tool.
Ranking options for search.
## Show properties
Whether to rewrite the natural language query for vector search.
rewrite_query boolean
## Optional
Defaults to false
A page of search results from the vector store.
 
    },
    {
      "file_id": "file_456",
      "filename": "notes.txt",
      "score": 0.89,
      "attributes": {
        "author": "Jane Smith",
        "date": "2023-01-02"
      },
      "content": [
        {
          "type": "text",
          "text": "Sample text content from the
        }
      ]
    }
  ],
  "has_more": false,
  "next_page": null
}

## The vector store object
The Unix timestamp (in seconds) for when the vector store was created.
created_at integer
The expiration policy for a vector store.
## Show properties
expires_after object
The Unix timestamp (in seconds) for when the vector store will expire.
expires_at integer or null
## Show properties
file_counts object
## OBJECT The vector store object
{
  "id": "vs_123",
  "object": "vector_store",
  "created_at": 1698107661,
  "usage_bytes": 123456,
  "last_active_at": 1698107661,
  "name": "my_vector_store",
  "status": "completed",
  "file_counts": {
    "in_progress": 0,
    "completed": 100,
    "cancelled": 0,
    "failed": 0,
    "total": 100
  },
  "last_used_at": 1698107661
}


<!-- Page 170 -->
Vector store files represent files inside a vector store.
The identifier, which can be referenced in API endpoints.
id string
The Unix timestamp (in seconds) for when the vector store was last active.
last_active_at integer or null
Set of 16 key-value pairs that can be attached to an object. This can be useful for
storing additional information about the object in a structured format, and querying for objects via API or the dashboard.
Keys are strings with a maximum length of 64 characters. Values are strings with a
maximum length of 512 characters.
metadata
map
The name of the vector store.
name string
The object type, which is always vector_store .
object string
The status of the vector store, which can be either expired , in_progress , or
completed . A status of completed  indicates that the vector store is ready for use.
status string
The total number of bytes used by the files in the vector store.
usage_bytes integer
## Vector store files

<!-- Page 171 -->
Related guide: File Search
POST https://api.openai.com/v1/vector_stores/{vector_store_id}/files
Create a vector store file by attaching a File to a vector store.
## Path parameters
Request body
Create vector store file
The ID of the vector store for which to create a File.
vector_store_id string
## Required
A File ID that the vector store should use. Useful for tools like file_search  that can
access files.
file_id string
## Required
Set of 16 key-value pairs that can be attached to an object. This can be useful for
storing additional information about the object in a structured format, and querying for objects via API or the dashboard. Keys are strings with a maximum length of 64
characters. Values are strings with a maximum length of 512 characters, booleans, or numbers.
attributes
map
## Optional
The chunking strategy used to chunk the file(s). If not set, will use the auto  strategy.
## Show possible types
chunking_strategy object
## Optional
Example request
python
 
 
from openai import OpenAI
client = OpenAI()
vector_store_file = client.vector_stores.files.cr
  vector_store_id="vs_abc123",
  file_id="file-abc123"
)
print(vector_store_file)

## Response
{
  "id": "file-abc123",
  "object": "vector_store.file",
  "created_at": 1699061776,
  "usage_bytes": 1234,
  "vector_store_id": "vs_abcd",
  "status": "completed",
  "last_error": null
}


<!-- Page 172 -->
## Returns
GET https://api.openai.com/v1/vector_stores/{vector_store_id}/files
Returns a list of vector store files.
## Path parameters
Query parameters
A vector store file object.
## List vector store files
The ID of the vector store that the files belong to.
vector_store_id string
## Required
A cursor for use in pagination. after  is an object ID that defines your place in the list.
For instance, if you make a list request and receive 100 objects, ending with obj_foo,
your subsequent call can include after=obj_foo in order to fetch the next page of the
list.
after string
## Optional
A cursor for use in pagination. before  is an object ID that defines your place in the
list. For instance, if you make a list request and receive 100 objects, starting with
before string
## Optional
Example request
python
 
 
from openai import OpenAI
client = OpenAI()
vector_store_files = client.vector_stores.files.l
  vector_store_id="vs_abc123"
)
print(vector_store_files)

## Response
{
  "object": "list",
  "data": [
    {
      "id": "file-abc123",
      "object": "vector_store.file",
      "created_at": 1699061776,
      "vector_store_id": "vs_abc123"
    },
    {
      "id": "file-abc456",
      "object": "vector_store.file",
      "created_at": 1699061776,
      "vector_store_id": "vs_abc123"


<!-- Page 173 -->
## Returns
GET https://api.openai.com/v1/vector_stores/{vector_store_id}/files/{file_
id}
Retrieves a vector store file.
## Path parameters
obj_foo, your subsequent call can include before=obj_foo in order to fetch the previous
page of the list.
Filter by file status. One of in_progress , completed , failed , cancelled .
filter string
## Optional
A limit on the number of objects to be returned. Limit can range between 1 and 100,
and the default is 20.
limit integer
## Optional
Defaults to 20
Sort order by the created_at  timestamp of the objects. asc  for ascending order
and desc  for descending order.
order string
## Optional
Defaults to desc
A list of vector store file objects.
    }
  ],
  "first_id": "file-abc123",
  "last_id": "file-abc456",
  "has_more": false
}

## Retrieve vector store file
file_id string
## Required
Example request
python
 
from openai import OpenAI
client = OpenAI()
vector_store_file = client.vector_stores.files.r
  vector_store_id="vs_abc123",
  file_id="file-abc123"


<!-- Page 174 -->
## Returns
GET https://api.openai.com/v1/vector_stores/{vector_store_id}/files/{file_
id}/content
Retrieve the parsed contents of a vector store file.
## Path parameters
Returns
The ID of the file being retrieved.
The ID of the vector store that the file belongs to.
vector_store_id string
## Required
The vector store file object.
 
)
## Response
{
  "id": "file-abc123",
  "object": "vector_store.file",
  "created_at": 1699061776,
  "vector_store_id": "vs_abcd",
  "status": "completed",
  "last_error": null
}

## Retrieve vector store file content
The ID of the file within the vector store.
file_id string
## Required
The ID of the vector store.
vector_store_id string
## Required
Example request
curl
 
 
curl \
https://api.openai.com/v1/vector_stores/vs_abc123
-H "Authorization: Bearer $OPENAI_API_KEY"

## Response
{
  "file_id": "file-abc123",
  "filename": "example.txt",
  "attributes": {"key": "value"},
  "content": [
    {"type": "text", "text": "..."},
    ...
  ]
}


<!-- Page 175 -->
POST https://api.openai.com/v1/vector_stores/{vector_store_id}/files/{file
_id}
Update attributes on a vector store file.
## Path parameters
Request body
Returns
The parsed contents of the specified vector store file.
## Update vector store file attributes
The ID of the file to update attributes.
file_id string
## Required
The ID of the vector store the file belongs to.
vector_store_id string
## Required
Set of 16 key-value pairs that can be attached to an object. This can be useful for
storing additional information about the object in a structured format, and querying for objects via API or the dashboard. Keys are strings with a maximum length of 64
characters. Values are strings with a maximum length of 512 characters, booleans, or numbers.
attributes
map
## Required
Example request
curl
 
 
curl https://api.openai.com/v1/vector_stores/{vec
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"attributes": {"key1": "value1", "key2": 2

## Response
{
  "id": "file-abc123",
  "object": "vector_store.file",
  "usage_bytes": 1234,
  "created_at": 1699061776,
  "vector_store_id": "vs_abcd",
  "status": "completed",
  "last_error": null,
  "chunking_strategy": {...},
  "attributes": {"key1": "value1", "key2": 2}
}


<!-- Page 176 -->
DELETE https://api.openai.com/v1/vector_stores/{vector_store_id}/files/{fi
le_id}
Delete a vector store file. This will remove the file from the vector store but
the file itself will not be deleted. To delete the file, use the delete file
endpoint.
## Path parameters
Returns
The updated vector store file object.
## Delete vector store file
The ID of the file to delete.
file_id string
## Required
The ID of the vector store that the file belongs to.
vector_store_id string
## Required
Deletion status
Example request
python
 
 
from openai import OpenAI
client = OpenAI()
deleted_vector_store_file = client.vector_stores.
    vector_store_id="vs_abc123",
    file_id="file-abc123"
)
print(deleted_vector_store_file)

## Response
{
  id: "file-abc123",
  object: "vector_store.file.deleted",
  deleted: true
}


<!-- Page 177 -->
A list of files attached to a vector store.
## The vector store file object
Beta
Set of 16 key-value pairs that can be attached to an object. This can be useful for
storing additional information about the object in a structured format, and querying for objects via API or the dashboard. Keys are strings with a maximum length of 64
characters. Values are strings with a maximum length of 512 characters, booleans, or numbers.
attributes
map
The strategy used to chunk the file.
## Show possible types
chunking_strategy object
The Unix timestamp (in seconds) for when the vector store file was created.
created_at integer
The identifier, which can be referenced in API endpoints.
id string
The last error associated with this vector store file. Will be null  if there are no errors.
## Show properties
last_error object or null
The object type, which is always vector_store.file .
object string
The status of the vector store file, which can be either in_progress , completed ,
cancelled , or failed . The status completed  indicates that the vector store file is
ready for use.
status string
## OBJECT The vector store file object
{
  "id": "file-abc123",
  "object": "vector_store.file",
  "usage_bytes": 1234,
  "created_at": 1698107661,
  "vector_store_id": "vs_abc123",
  "status": "completed",
  "last_error": null,
  "chunking_strategy": {
    "type": "static",
    "static": {
      "max_chunk_size_tokens": 800,
      "chunk_overlap_tokens": 400
    }
  }
}


<!-- Page 178 -->
Vector store file batches represent operations to add multiple files to a vector store. Related guide:
## File Search
POST https://api.openai.com/v1/vector_stores/{vector_store_id}/file_batche
s
Create a vector store file batch.
## Path parameters
Request body
The total vector store usage in bytes. Note that this may be different from the original
file size.
usage_bytes integer
The ID of the vector store that the File is attached to.
vector_store_id string
## Vector store file batches
Create vector store file batch
The ID of the vector store for which to create a File Batch.
vector_store_id string
## Required
Example request
python
 
 
from openai import OpenAI
client = OpenAI()
vector_store_file_batch = client.vector_stores.fi
  vector_store_id="vs_abc123",
  file_ids=["file-abc123", "file-abc456"]
)
print(vector_store_file_batch)

## Response

<!-- Page 179 -->
## Returns
GET https://api.openai.com/v1/vector_stores/{vector_store_id}/file_batche
s/{batch_id}
Retrieves a vector store file batch.
## Path parameters
A list of File IDs that the vector store should use. Useful for tools like file_search
that can access files.
file_ids array
## Required
Set of 16 key-value pairs that can be attached to an object. This can be useful for
storing additional information about the object in a structured format, and querying for objects via API or the dashboard. Keys are strings with a maximum length of 64
characters. Values are strings with a maximum length of 512 characters, booleans, or numbers.
attributes
map
## Optional
The chunking strategy used to chunk the file(s). If not set, will use the auto  strategy.
## Show possible types
chunking_strategy object
## Optional
A vector store file batch object.
{
  "id": "vsfb_abc123",
  "object": "vector_store.file_batch",
  "created_at": 1699061776,
  "vector_store_id": "vs_abc123",
  "status": "in_progress",
  "file_counts": {
    "in_progress": 1,
    "completed": 1,
    "failed": 0,
    "cancelled": 0,
    "total": 0,
  }

## Retrieve vector store file batch
Example request
python
 
from openai import OpenAI
client = OpenAI()
vector_store_file_batch = client.vector_stores.f
  vector_store_id="vs_abc123",
  batch_id="vsfb_abc123"


<!-- Page 180 -->
## Returns
POST https://api.openai.com/v1/vector_stores/{vector_store_id}/file_batche
s/{batch_id}/cancel
Cancel a vector store file batch. This attempts to cancel the processing of
files in this batch as soon as possible.
## Path parameters
The ID of the file batch being retrieved.
batch_id string
## Required
The ID of the vector store that the file batch belongs to.
vector_store_id string
## Required
The vector store file batch object.
 
)
print(vector_store_file_batch)

## Response
{
  "id": "vsfb_abc123",
  "object": "vector_store.file_batch",
  "created_at": 1699061776,
  "vector_store_id": "vs_abc123",
  "status": "in_progress",
  "file_counts": {
    "in_progress": 1,
    "completed": 1,
    "failed": 0,
    "cancelled": 0,
    "total": 0,
  }
}

## Cancel vector store file batch
The ID of the file batch to cancel.
batch_id string
## Required
Example request
python
 
 
from openai import OpenAI
client = OpenAI()
deleted_vector_store_file_batch = client.vector_s
    vector_store_id="vs_abc123",
    file_batch_id="vsfb_abc123"
)
print(deleted_vector_store_file_batch)


<!-- Page 181 -->
## Returns
GET https://api.openai.com/v1/vector_stores/{vector_store_id}/file_batche
s/{batch_id}/files
Returns a list of vector store files in a batch.
## Path parameters
Query parameters
The ID of the vector store that the file batch belongs to.
vector_store_id string
## Required
The modified vector store file batch object.
## Response
{
  "id": "vsfb_abc123",
  "object": "vector_store.file_batch",
  "created_at": 1699061776,
  "vector_store_id": "vs_abc123",
  "status": "in_progress",
  "file_counts": {
    "in_progress": 12,
    "completed": 3,
    "failed": 0,
    "cancelled": 0,
    "total": 15,
  }

## List vector store files in a batch
The ID of the file batch that the files belong to.
batch_id string
## Required
The ID of the vector store that the files belong to.
vector_store_id string
## Required
Example request
python
 
 
from openai import OpenAI
client = OpenAI()
vector_store_files = client.vector_stores.file_ba
  vector_store_id="vs_abc123",
  batch_id="vsfb_abc123"
)
print(vector_store_files)

## Response
 
{
  "object": "list",
  "data": [


<!-- Page 182 -->
## Returns
A cursor for use in pagination. after  is an object ID that defines your place in the list.
For instance, if you make a list request and receive 100 objects, ending with obj_foo,
your subsequent call can include after=obj_foo in order to fetch the next page of the
list.
after string
## Optional
A cursor for use in pagination. before  is an object ID that defines your place in the
list. For instance, if you make a list request and receive 100 objects, starting with
obj_foo, your subsequent call can include before=obj_foo in order to fetch the previous
page of the list.
before string
## Optional
Filter by file status. One of in_progress , completed , failed , cancelled .
filter string
## Optional
A limit on the number of objects to be returned. Limit can range between 1 and 100,
and the default is 20.
limit integer
## Optional
Defaults to 20
Sort order by the created_at  timestamp of the objects. asc  for ascending order
and desc  for descending order.
order string
## Optional
Defaults to desc
A list of vector store file objects.
 
    {
      "id": "file-abc123",
      "object": "vector_store.file",
      "created_at": 1699061776,
      "vector_store_id": "vs_abc123"
    },
    {
      "id": "file-abc456",
      "object": "vector_store.file",
      "created_at": 1699061776,
      "vector_store_id": "vs_abc123"
    }
  ],
  "first_id": "file-abc123",
  "last_id": "file-abc456",
  "has_more": false
}

## The vector store files batch object
Beta

<!-- Page 183 -->
A batch of files attached to a vector store.
Create and manage containers for use with the Code Interpreter tool.
The Unix timestamp (in seconds) for when the vector store files batch was created.
created_at integer
## Show properties
file_counts object
The identifier, which can be referenced in API endpoints.
id string
The object type, which is always vector_store.file_batch .
object string
The status of the vector store files batch, which can be either in_progress ,
completed , cancelled  or failed .
status string
The ID of the vector store that the File is attached to.
vector_store_id string
## OBJECT The vector store files batch object
{
  "id": "vsfb_123",
  "object": "vector_store.files_batch",
  "created_at": 1698107661,
  "vector_store_id": "vs_abc123",
  "status": "completed",
  "file_counts": {
    "in_progress": 0,
    "completed": 100,
    "failed": 0,
    "cancelled": 0,
    "total": 100
  }
}

## Containers
Create container

<!-- Page 184 -->
POST https://api.openai.com/v1/containers
## Create Container
Request body
Returns
GET https://api.openai.com/v1/containers
## List Containers
Query parameters
Name of the container to create.
name string
## Required
Container expiration time in seconds relative to the 'anchor' time.
## Show properties
expires_after object
## Optional
IDs of files to copy to the container.
file_ids array
## Optional
The created container object.
## Example request
curl
curl https://api.openai.com/v1/containers \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
        "name": "My Container"
      }'

## Response
 
 
{
    "id": "cntr_682e30645a488191b6363a0cbefc0f0a
    "object": "container",
    "created_at": 1747857508,
    "status": "running",
    "expires_after": {
        "anchor": "last_active_at",
        "minutes": 20
    },
    "last_active_at": 1747857508,
    "name": "My Container"
}

## List containers
Example request
curl
curl https://api.openai.com/v1/containers \
  -H "Authorization: Bearer $OPENAI_API_KEY"

## Response

<!-- Page 185 -->
## Returns
GET https://api.openai.com/v1/containers/{container_id}
## Retrieve Container
Path parameters
A cursor for use in pagination. after  is an object ID that defines your place in the list.
For instance, if you make a list request and receive 100 objects, ending with obj_foo,
your subsequent call can include after=obj_foo in order to fetch the next page of the
list.
after string
## Optional
A limit on the number of objects to be returned. Limit can range between 1 and 100,
and the default is 20.
limit integer
## Optional
Defaults to 20
Sort order by the created_at  timestamp of the objects. asc  for ascending order
and desc  for descending order.
order string
## Optional
Defaults to desc
a list of container objects.
 
 
{
  "object": "list",
  "data": [
    {
        "id": "cntr_682dfebaacac8198bbfe9c2474fb
        "object": "container",
        "created_at": 1747844794,
        "status": "running",
        "expires_after": {
            "anchor": "last_active_at",
            "minutes": 20
        },
        "last_active_at": 1747844794,
        "name": "My Container"
    }
  ],
  "first_id": "container_123",
  "last_id": "container_123",
  "has_more": false
}

## Retrieve container
container_id string
## Required
Example request
curl
 
 
curl https://api.openai.com/v1/containers/cntr_68
  -H "Authorization: Bearer $OPENAI_API_KEY"

## Response

<!-- Page 186 -->
## Returns
DELETE https://api.openai.com/v1/containers/{container_id}
## Delete Container
Path parameters
Returns
The container object.
{
    "id": "cntr_682dfebaacac8198bbfe9c2474fb6f4a
    "object": "container",
    "created_at": 1747844794,
    "status": "running",
    "expires_after": {
        "anchor": "last_active_at",
        "minutes": 20
    },
    "last_active_at": 1747844794,
    "name": "My Container"

## Delete a container
The ID of the container to delete.
container_id string
## Required
Deletion Status
Example request
curl
 
 
curl -X DELETE https://api.openai.com/v1/containe
  -H "Authorization: Bearer $OPENAI_API_KEY"

## Response
 
 
{
    "id": "cntr_682dfebaacac8198bbfe9c2474fb6f4a0
    "object": "container.deleted",
    "deleted": true
}


<!-- Page 187 -->
Create and manage container files for use with the Code Interpreter tool.
## The container object
Unix timestamp (in seconds) when the container was created.
created_at integer
The container will expire after this time period. The anchor is the reference point for the
expiration. The minutes is the number of minutes after the anchor before the container
expires.
## Show properties
expires_after object
Unique identifier for the container.
id string
Name of the container.
name string
The type of this object.
object string
Status of the container (e.g., active, deleted).
status string
## OBJECT The container object
 
 
{
   "id": "cntr_682dfebaacac8198bbfe9c2474fb6f4a0
   "object": "container",
   "created_at": 1747844794,
   "status": "running",
   "expires_after": {
     "anchor": "last_active_at",
     "minutes": 20
   },
   "last_active_at": 1747844794,
   "name": "My Container"
}

## Container Files

<!-- Page 188 -->
POST https://api.openai.com/v1/containers/{container_id}/files
## Create a Container File
You can send either a multipart/form-data request with the raw file content,
or a JSON request with a file ID.
## Path parameters
Request body
Returns
GET https://api.openai.com/v1/containers/{container_id}/files
## Create container file
container_id string
## Required
The File object (not file name) to be uploaded.
file
file
## Optional
Name of the file to create.
file_id string
## Optional
The created container file object.
## Example request
curl
 
 
curl https://api.openai.com/v1/containers/cntr_68
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -F file="@example.txt"

## Response
 
 
{
  "id": "cfile_682e0e8a43c88191a7978f477a09bdf5",
  "object": "container.file",
  "created_at": 1747848842,
  "bytes": 880,
  "container_id": "cntr_682e0e7318108198aa783fd92
  "path": "/mnt/data/88e12fa445d32636f190a0b33dae
  "source": "user"
}

## List container files

<!-- Page 189 -->
## List Container files
Path parameters
Query parameters
Returns
container_id string
## Required
A cursor for use in pagination. after  is an object ID that defines your place in the list.
For instance, if you make a list request and receive 100 objects, ending with obj_foo,
your subsequent call can include after=obj_foo in order to fetch the next page of the
list.
after string
## Optional
A limit on the number of objects to be returned. Limit can range between 1 and 100,
and the default is 20.
limit integer
## Optional
Defaults to 20
Sort order by the created_at  timestamp of the objects. asc  for ascending order
and desc  for descending order.
order string
## Optional
Defaults to desc
a list of container file objects.
## Example request
curl
curl https://api.openai.com/v1/containers/cntr_68
H "Authorization: Bearer $OPENAI API KEY"

## Response
 
 
{
    "object": "list",
    "data": [
        {
            "id": "cfile_682e0e8a43c88191a7978f4
            "object": "container.file",
            "created_at": 1747848842,
            "bytes": 880,
            "container_id": "cntr_682e0e73181081
            "path": "/mnt/data/88e12fa445d32636f
            "source": "user"
        }
    ],
    "first_id": "cfile_682e0e8a43c88191a7978f477
    "has_more": false,
    "last_id": "cfile_682e0e8a43c88191a7978f477a
}

## Retrieve container file

<!-- Page 190 -->
GET https://api.openai.com/v1/containers/{container_id}/files/{file_id}
## Retrieve Container File
Path parameters
Returns
GET https://api.openai.com/v1/containers/{container_id}/files/{file_id}/co
ntent
## Retrieve Container File Content
Path parameters
container_id string
## Required
file_id string
## Required
The container file object.
## Example request
curl
 
 
curl https://api.openai.com/v1/containers/contain
  -H "Authorization: Bearer $OPENAI_API_KEY"

## Response
 
 
{
    "id": "cfile_682e0e8a43c88191a7978f477a09bdf5
    "object": "container.file",
    "created_at": 1747848842,
    "bytes": 880,
    "container_id": "cntr_682e0e7318108198aa783fd
    "path": "/mnt/data/88e12fa445d32636f190a0b33d
    "source": "user"
}

## Retrieve container file content
container_id string
## Required
file_id string
## Required
Example request
curl
 
 
curl https://api.openai.com/v1/containers/contain
  -H "Authorization: Bearer $OPENAI_API_KEY"

## Response
<binary content of the file>

<!-- Page 191 -->
## Returns
DELETE https://api.openai.com/v1/containers/{container_id}/files/{file_id}
## Delete Container File
Path parameters
Returns
The contents of the container file.
## Delete a container file
container_id string
## Required
file_id string
## Required
Deletion Status
Example request
curl
 
 
curl -X DELETE https://api.openai.com/v1/containe
  -H "Authorization: Bearer $OPENAI_API_KEY"

## Response
 
 
{
    "id": "cfile_682e0e8a43c88191a7978f477a09bdf5
    "object": "container.file.deleted",
    "deleted": true
}

## The container file object
bytes integer
OBJECT The container file object

<!-- Page 192 -->
Communicate with a GPT-4o class model in real time using WebRTC or WebSockets. Supports text and
audio inputs and ouputs, along with audio transcriptions. Learn more about the Realtime API.
Size of the file in bytes.
The container this file belongs to.
container_id string
Unix timestamp (in seconds) when the file was created.
created_at integer
Unique identifier for the file.
id string
The type of this object ( container.file ).
object string
Path of the file in the container.
path string
Source of the file (e.g., user , assistant ).
source string
{
    "id": "cfile_682e0e8a43c88191a7978f477a09bdf5
    "object": "container.file",
    "created_at": 1747848842,
    "bytes": 880,
    "container_id": "cntr_682e0e7318108198aa783fd
    "path": "/mnt/data/88e12fa445d32636f190a0b33d
    "source": "user"
}

## Realtime
Beta
Session tokens

<!-- Page 193 -->
REST API endpoint to generate ephemeral session tokens for use in client-side applications.
POST https://api.openai.com/v1/realtime/sessions
## Create an ephemeral API token for use in client-side applications with the
Realtime API. Can be configured with the same session parameters as the
session.update  client event.
It responds with a session object, plus a client_secret  key which contains
a usable ephemeral API token that can be used to authenticate browser
clients for the Realtime API.
## Request body
Create session
Configuration options for the generated client secret.
## Show properties
client_secret object
## Optional
The format of input audio. Options are pcm16 , g711_ulaw , or g711_alaw . For
pcm16 , input audio must be 16-bit PCM at a 24kHz sample rate, single channel
(mono), and little-endian byte order.
input_audio_format string
## Optional
Defaults to pcm16
Configuration for input audio noise reduction. This can be set to null  to turn off.
## Noise reduction filters audio added to the input audio buffer before it is sent to VAD
and the model. Filtering the audio can improve VAD and turn detection accuracy
(reducing false positives) and model performance by improving perception of the input
audio.
input_audio_noise_reduction object
## Optional
Defaults to null
Example request
curl
 
 
curl -X POST https://api.openai.com/v1/realtime/s
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-realtime-preview",
    "modalities": ["audio", "text"],
    "instructions": "You are a friendly assistant
  }'

## Response
 
{
  "id": "sess_001",
  "object": "realtime.session",
  "model": "gpt-4o-realtime-preview",
  "modalities": ["audio", "text"],
  "instructions": "You are a friendly assistant
  "voice": "alloy",
  "input_audio_format": "pcm16",
  "output_audio_format": "pcm16",
  "input_audio_transcription": {
      "model": "whisper-1"
  },
  "turn_detection": null,
  "tools": [],
  "tool_choice": "none",
  "temperature": 0.7,
  "max_response_output_tokens": 200,


<!-- Page 194 -->
## Show properties
Configuration for input audio transcription, defaults to off and can be set to null  to
turn off once on. Input audio transcription is not native to the model, since the model
consumes audio directly. Transcription runs asynchronously through
the /audio/transcriptions endpoint and should be treated as guidance of input audio
content rather than precisely what the model heard. The client can optionally set the
language and prompt for transcription, these offer additional guidance to the
transcription service.
## Show properties
input_audio_transcription object
## Optional
The default system instructions (i.e. system message) prepended to model calls. This
field allows the client to guide the model on desired responses. The model can be
instructed on response content and format, (e.g. "be extremely succinct", "act friendly",
"here are examples of good responses") and on audio behavior (e.g. "talk quickly",
"inject emotion into your voice", "laugh frequently"). The instructions are not
guaranteed to be followed by the model, but they provide guidance to the model on the
desired behavior.
## Note that the server sets default instructions which will be used if this field is not set
and are visible in the session.created  event at the start of the session.
instructions string
## Optional
Maximum number of output tokens for a single assistant response, inclusive of tool
calls. Provide an integer between 1 and 4096 to limit output tokens, or inf  for the
maximum available tokens for a given model. Defaults to inf .
max_response_output_tokens integer or "inf"
## Optional
The set of modalities the model can respond with. To disable audio, set this to ["text"].
modalities
## Optional
The Realtime model used for this session.
model string
## Optional
 
  "speed": 1.1,
  "tracing": "auto",
  "client_secret": {
    "value": "ek_abc123", 
    "expires_at": 1234567890
  }
}


<!-- Page 195 -->
The format of output audio. Options are pcm16 , g711_ulaw , or g711_alaw . For
pcm16 , output audio is sampled at a rate of 24kHz.
output_audio_format string
## Optional
Defaults to pcm16
The speed of the model's spoken response. 1.0 is the default speed. 0.25 is the
minimum speed. 1.5 is the maximum speed. This value can only be changed in between
model turns, not while a response is in progress.
speed number
## Optional
Defaults to 1
Sampling temperature for the model, limited to [0.6, 1.2]. For audio models a
temperature of 0.8 is highly recommended for best performance.
temperature number
## Optional
Defaults to 0.8
How the model chooses tools. Options are auto , none , required , or specify a
function.
tool_choice string
## Optional
Defaults to auto
Tools (functions) available to the model.
## Show properties
tools array
Optional
Configuration options for tracing. Set to null to disable tracing. Once tracing is enabled
for a session, the configuration cannot be modified.
auto  will create a trace for the session with default values for the workflow name,
group id, and metadata.
## Show possible types
tracing
"auto" or object
## Optional
Configuration for turn detection, ether Server VAD or Semantic VAD. This can be set to null  to turn off, in which case the client must manually trigger model response.
## Server VAD means that the model will detect the start and end of speech based on
audio volume and respond at the end of user speech. Semantic VAD is more advanced
turn_detection object
## Optional

<!-- Page 196 -->
## Returns
POST https://api.openai.com/v1/realtime/transcription_sessions
## Create an ephemeral API token for use in client-side applications with the
Realtime API specifically for realtime transcriptions. Can be configured with
the same session parameters as the transcription_session.update  client
event.
It responds with a session object, plus a client_secret  key which contains
a usable ephemeral API token that can be used to authenticate browser
clients for the Realtime API.
and uses a turn detection model (in conjuction with VAD) to semantically estimate
whether the user has finished speaking, then dynamically sets a timeout based on this
probability. For example, if user audio trails off with "uhhm", the model will score a low
probability of turn end and wait longer for the user to continue speaking. This can be
useful for more natural conversations, but may have a higher latency.
## Show properties
The voice the model uses to respond. Voice cannot be changed during the session
once the model has responded with audio at least once. Current voice options are
alloy , ash , ballad , coral , echo , sage , shimmer , and verse .
voice string
## Optional
The created Realtime session object, plus an ephemeral key
## Create transcription session
Example request
curl
 
 
curl -X POST https://api.openai.com/v1/realtime/t
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{}'

## Response
 
{
  "id": "sess_BBwZc7cFV3XizEyKGDCGL",


<!-- Page 197 -->
## Request body
Configuration options for the generated client secret.
## Show properties
client_secret object
## Optional
The set of items to include in the transcription. Current available items are:
null.
include array
## Optional
The format of input audio. Options are pcm16 , g711_ulaw , or g711_alaw . For
pcm16 , input audio must be 16-bit PCM at a 24kHz sample rate, single channel
(mono), and little-endian byte order.
input_audio_format string
## Optional
Defaults to pcm16
Configuration for input audio noise reduction. This can be set to null  to turn off.
## Noise reduction filters audio added to the input audio buffer before it is sent to VAD
and the model. Filtering the audio can improve VAD and turn detection accuracy
(reducing false positives) and model performance by improving perception of the input
audio.
## Show properties
input_audio_noise_reduction object
## Optional
Defaults to null
Configuration for input audio transcription. The client can optionally set the language
and prompt for transcription, these offer additional guidance to the transcription
service.
## Show properties
input_audio_transcription object
## Optional
The set of modalities the model can respond with. To disable audio, set this to ["text"].
modalities
## Optional
turn_detection object
## Optional
 
  "object": "realtime.transcription_session",
  "modalities": ["audio", "text"],
  "turn_detection": {
    "type": "server_vad",
    "threshold": 0.5,
    "prefix_padding_ms": 300,
    "silence_duration_ms": 200
  },
  "input_audio_format": "pcm16",
  "input_audio_transcription": {
    "model": "gpt-4o-transcribe",
    "language": null,
    "prompt": ""
  },
  "client_secret": null
}


<!-- Page 198 -->
## Returns
A new Realtime session configuration, with an ephermeral key. Default TTL
for keys is one minute.
Configuration for turn detection, ether Server VAD or Semantic VAD. This can be set to null  to turn off, in which case the client must manually trigger model response.
## Server VAD means that the model will detect the start and end of speech based on
audio volume and respond at the end of user speech. Semantic VAD is more advanced
and uses a turn detection model (in conjuction with VAD) to semantically estimate
whether the user has finished speaking, then dynamically sets a timeout based on this
probability. For example, if user audio trails off with "uhhm", the model will score a low
probability of turn end and wait longer for the user to continue speaking. This can be
useful for more natural conversations, but may have a higher latency.
## Show properties
The created Realtime transcription session object, plus an ephemeral key
## The session object
Ephemeral key returned by the API.
## Show properties
client_secret object
The format of input audio. Options are pcm16 , g711_ulaw , or g711_alaw .
input_audio_format string
input_audio_transcription object
## OBJECT The session object
 
{
  "id": "sess_001",
  "object": "realtime.session",
  "model": "gpt-4o-realtime-preview",
  "modalities": ["audio", "text"],
  "instructions": "You are a friendly assistant
  "voice": "alloy",
  "input_audio_format": "pcm16",
  "output_audio_format": "pcm16",
  "input_audio_transcription": {
      "model": "whisper-1"
  },


<!-- Page 199 -->
Configuration for input audio transcription, defaults to off and can be set to null  to
turn off once on. Input audio transcription is not native to the model, since the model
consumes audio directly. Transcription runs asynchronously and should be treated as
rough guidance rather than the representation understood by the model.
## Show properties
The default system instructions (i.e. system message) prepended to model calls. This
field allows the client to guide the model on desired responses. The model can be
instructed on response content and format, (e.g. "be extremely succinct", "act friendly",
"here are examples of good responses") and on audio behavior (e.g. "talk quickly",
"inject emotion into your voice", "laugh frequently"). The instructions are not
guaranteed to be followed by the model, but they provide guidance to the model on the
desired behavior.
## Note that the server sets default instructions which will be used if this field is not set
and are visible in the session.created  event at the start of the session.
instructions string
Maximum number of output tokens for a single assistant response, inclusive of tool
calls. Provide an integer between 1 and 4096 to limit output tokens, or inf  for the
maximum available tokens for a given model. Defaults to inf .
max_response_output_tokens integer or "inf"
The set of modalities the model can respond with. To disable audio, set this to ["text"].
modalities
The format of output audio. Options are pcm16 , g711_ulaw , or g711_alaw .
output_audio_format string
The speed of the model's spoken response. 1.0 is the default speed. 0.25 is the
minimum speed. 1.5 is the maximum speed. This value can only be changed in between
model turns, not while a response is in progress.
speed number
temperature number
 
  "turn_detection": null,
  "tools": [],
  "tool_choice": "none",
  "temperature": 0.7,
  "speed": 1.1,
  "tracing": "auto",
  "max_response_output_tokens": 200,
  "client_secret": {
    "value": "ek_abc123", 
    "expires_at": 1234567890
  }
}


<!-- Page 200 -->
Sampling temperature for the model, limited to [0.6, 1.2]. Defaults to 0.8.
How the model chooses tools. Options are auto , none , required , or specify a
function.
tool_choice string
Tools (functions) available to the model.
## Show properties
tools array
Configuration options for tracing. Set to null to disable tracing. Once tracing is enabled
for a session, the configuration cannot be modified.
auto  will create a trace for the session with default values for the workflow name,
group id, and metadata.
## Show possible types
tracing
"auto" or object
Configuration for turn detection. Can be set to null  to turn off. Server VAD means
that the model will detect the start and end of speech based on audio volume and
respond at the end of user speech.
## Show properties
turn_detection object
The voice the model uses to respond. Voice cannot be changed during the session
once the model has responded with audio at least once. Current voice options are
alloy , ash , ballad , coral , echo , sage , shimmer , and verse .
voice string
## The transcription session object

<!-- Page 201 -->
A new Realtime transcription session configuration.
When a session is created on the server via REST API, the session object
also contains an ephemeral key. Default TTL for keys is 10 minutes. This
property is not present when a session is updated via the WebSocket API.
Ephemeral key returned by the API. Only present when the session is created on the
server via REST API.
## Show properties
client_secret object
The format of input audio. Options are pcm16 , g711_ulaw , or g711_alaw .
input_audio_format string
Configuration of the transcription model.
## Show properties
input_audio_transcription object
The set of modalities the model can respond with. To disable audio, set this to ["text"].
modalities
Configuration for turn detection. Can be set to null  to turn off. Server VAD means
that the model will detect the start and end of speech based on audio volume and
respond at the end of user speech.
## Show properties
turn_detection object
## OBJECT The transcription session object
{
  "id": "sess_BBwZc7cFV3XizEyKGDCGL",
  "object": "realtime.transcription_session",
  "expires_at": 1742188264,
  "modalities": ["audio", "text"],
  "turn_detection": {
    "type": "server_vad",
    "threshold": 0.5,
    "prefix_padding_ms": 300,
    "silence_duration_ms": 200
  },
  "input_audio_format": "pcm16",
  "input_audio_transcription": {
    "model": "gpt-4o-transcribe",
    "language": null,
    "prompt": ""
  },
  "client_secret": null
}

## Client events

<!-- Page 202 -->
These are events that the OpenAI Realtime WebSocket server will accept from the client.
Send this event to update the sessions default configuration. The client
may send this event at any time to update any field, except for voice .
However, note that once a session has been initialized with a particular
model , it cant be changed to another model using session.update .
When the server receives a session.update , it will respond with a
session.updated  event showing the full, effective configuration. Only the
fields that are present are updated. To clear a field like instructions , pass
an empty string.
session.update
Optional client-generated ID used to identify this event.
event_id string
Realtime session object configuration.
## Show properties
session object
The event type, must be session.update .
type string
OBJECT session.update
{
    "event_id": "event_123",
    "type": "session.update",
    "session": {
        "modalities": ["text", "audio"],
        "instructions": "You are a helpful assis
        "voice": "sage",
        "input_audio_format": "pcm16",
        "output_audio_format": "pcm16",
        "input_audio_transcription": {
            "model": "whisper-1"
        },
        "turn_detection": {
            "type": "server_vad",
            "threshold": 0.5,
            "prefix_padding_ms": 300,
            "silence_duration_ms": 500,
            "create_response": true
        },
        "tools": [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get the current 
                "parameters": {
                    "type": "object",
                    "properties": {


<!-- Page 203 -->
Send this event to append audio bytes to the input audio buffer. The audio
buffer is temporary storage you can write to and later commit. In Server
VAD mode, the audio buffer is used to detect speech and the server will
decide when to commit. When Server VAD is disabled, you must commit the
audio buffer manually.
## The client may choose how much audio to place in each event up to a
maximum of 15 MiB, for example streaming smaller chunks from the client
may allow the VAD to be more responsive. Unlike made other client events,
the server will not send a confirmation response to this event.
                        "location": { "type": "s
                    },
                    "required": ["location"]
                }
            }
        ],
        "tool_choice": "auto",
        "temperature": 0.8,
        "max_response_output_tokens": "inf",
        "speed": 1.1,
        "tracing": "auto"
    }
}

input_audio_buffer.append
Base64-encoded audio bytes. This must be in the format specified by the
input_audio_format  field in the session configuration.
audio string
Optional client-generated ID used to identify this event.
event_id string
type string
OBJECT input_audio_buffer.append
{
    "event_id": "event_456",
    "type": "input_audio_buffer.append",
    "audio": "Base64EncodedAudioData"
}


<!-- Page 204 -->
Send this event to commit the user input audio buffer, which will create a
new user message item in the conversation. This event will produce an error
if the input audio buffer is empty. When in Server VAD mode, the client does
not need to send this event, the server will commit the audio buffer
automatically.
Committing the input audio buffer will trigger input audio transcription (if
enabled in session configuration), but it will not create a response from the
model. The server will respond with an input_audio_buffer.committed  event.
Send this event to clear the audio bytes in the buffer. The server will
respond with an input_audio_buffer.cleared  event.
The event type, must be input_audio_buffer.append .
input_audio_buffer.commit
Optional client-generated ID used to identify this event.
event_id string
The event type, must be input_audio_buffer.commit .
type string
OBJECT input_audio_buffer.commit
{
    "event_id": "event_789",
    "type": "input_audio_buffer.commit"
}

input_audio_buffer.clear
event_id string
OBJECT input_audio_buffer.clear
 
{
    "event_id": "event_012",


<!-- Page 205 -->
Add a new Item to the Conversation's context, including messages, function
calls, and function call responses. This event can be used both to populate a
"history" of the conversation and to add new items mid-stream, but has the
current limitation that it cannot populate assistant audio messages.
If successful, the server will respond with a conversation.item.created
event, otherwise an error  event will be sent.
Optional client-generated ID used to identify this event.
The event type, must be input_audio_buffer.clear .
type string
 
    "type": "input_audio_buffer.clear"
}

conversation.item.create
Optional client-generated ID used to identify this event.
event_id string
The item to add to the conversation.
## Show properties
item object
The ID of the preceding item after which the new item will be inserted. If not set, the
new item will be appended to the end of the conversation. If set to root , the new
previous_item_id string
OBJECT conversation.item.create
{
    "event_id": "event_345",
    "type": "conversation.item.create",
    "previous_item_id": null,
    "item": {
        "id": "msg_001",
        "type": "message",
        "role": "user",
        "content": [
            {
                "type": "input_text",
                "text": "Hello, how are you?"
            }
        ]
    }
}


<!-- Page 206 -->
Send this event when you want to retrieve the server's representation of a
specific item in the conversation history. This is useful, for example, to
inspect user audio after noise cancellation and VAD. The server will respond
with a conversation.item.retrieved  event, unless the item does not exist in
the conversation history, in which case the server will respond with an error.
item will be added to the beginning of the conversation. If set to an existing ID, it allows
an item to be inserted mid-conversation. If the ID cannot be found, an error will be
returned and the item will not be added.
The event type, must be conversation.item.create .
type string
conversation.item.retrieve
Optional client-generated ID used to identify this event.
event_id string
The ID of the item to retrieve.
item_id string
The event type, must be conversation.item.retrieve .
type string
OBJECT conversation.item.retrieve
{
    "event_id": "event_901",
    "type": "conversation.item.retrieve",
    "item_id": "msg_003"
}

conversation.item.truncate

<!-- Page 207 -->
Send this event to truncate a previous assistant messages audio. The
server will produce audio faster than realtime, so this event is useful when
the user interrupts to truncate audio that has already been sent to the client
but not yet played. This will synchronize the server's understanding of the
audio with the client's playback.
## Truncating audio will delete the server-side text transcript to ensure there is
not text in the context that hasn't been heard by the user.
If successful, the server will respond with a conversation.item.truncated
event.
Inclusive duration up to which audio is truncated, in milliseconds. If the audio_end_ms
is greater than the actual audio duration, the server will respond with an error.
audio_end_ms integer
The index of the content part to truncate. Set this to 0.
content_index integer
Optional client-generated ID used to identify this event.
event_id string
The ID of the assistant message item to truncate. Only assistant message items can be
truncated.
item_id string
The event type, must be conversation.item.truncate .
type string
OBJECT conversation.item.truncate
{
    "event_id": "event_678",
    "type": "conversation.item.truncate",
    "item_id": "msg_002",
    "content_index": 0,
    "audio_end_ms": 1500
}

conversation.item.delete

<!-- Page 208 -->
## Send this event when you want to remove any item from the conversation
history. The server will respond with a conversation.item.deleted  event,
unless the item does not exist in the conversation history, in which case the
server will respond with an error.
This event instructs the server to create a Response, which means
triggering model inference. When in Server VAD mode, the server will create
Responses automatically.
A Response will include at least one Item, and may have two, in which case
the second will be a function call. These Items will be appended to the
conversation history.
The server will respond with a response.created  event, events for Items and
content created, and finally a response.done  event to indicate the
Response is complete.
Optional client-generated ID used to identify this event.
event_id string
The ID of the item to delete.
item_id string
The event type, must be conversation.item.delete .
type string
OBJECT conversation.item.delete
{
    "event_id": "event_901",
    "type": "conversation.item.delete",
    "item_id": "msg_003"
}

response.create
OBJECT response.create
 
{
    "event_id": "event_234",
    "type": "response.create",
    "response": {
        "modalities": ["text", "audio"],
        "instructions": "Please assist the user
        "voice": "sage",
        "output_audio_format": "pcm16",
        "tools": [
            {
                "type": "function",


<!-- Page 209 -->
The response.create  event includes inference configuration like
instructions , and temperature . These fields will override the Session's
configuration for this Response only.
Send this event to cancel an in-progress response. The server will respond
with a response.cancelled  event or an error if there is no response to
cancel.
Optional client-generated ID used to identify this event.
event_id string
## Create a new Realtime response with these parameters
Show properties
response object
The event type, must be response.create .
type string
 
                "name": "calculate_sum",
                "description": "Calculates the 
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": { "type": "number"
                        "b": { "type": "number"
                    },
                    "required": ["a", "b"]
                }
            }
        ],
        "tool_choice": "auto",
        "temperature": 0.8,
        "max_output_tokens": 1024
    }
}

response.cancel
Optional client-generated ID used to identify this event.
event_id string
A specific response ID to cancel - if not provided, will cancel an in-progress response in
the default conversation.
response_id string
type string
OBJECT response.cancel
{
    "event_id": "event_567",
    "type": "response.cancel"
}


<!-- Page 210 -->
The event type, must be response.cancel .
transcription_session.update

<!-- Page 211 -->
Send this event to update a transcription session.
WebRTC Only: Emit to cut off the current audio response. This will trigger
the server to stop generating audio and emit a output_audio_buffer.cleared
Optional client-generated ID used to identify this event.
event_id string
Realtime transcription session object configuration.
## Show properties
session object
The event type, must be transcription_session.update .
type string
OBJECT transcription_session.update
 
 
{
  "type": "transcription_session.update",
  "session": {
    "input_audio_format": "pcm16",
    "input_audio_transcription": {
      "model": "gpt-4o-transcribe",
      "prompt": "",
      "language": ""
    },
    "turn_detection": {
      "type": "server_vad",
      "threshold": 0.5,
      "prefix_padding_ms": 300,
      "silence_duration_ms": 500,
      "create_response": true,
    },
    "input_audio_noise_reduction": {
      "type": "near_field"
    },
    "include": [
      "item.input_audio_transcription.logprobs",
    ]
  }
}

output_audio_buffer.clear
OBJECT output_audio_buffer.clear

<!-- Page 212 -->
event. This event should be preceded by a response.cancel  client event to
stop the generation of the current response. Learn more.
These are events emitted from the OpenAI Realtime WebSocket server to the client.
Returned when an error occurs, which could be a client problem or a server
problem. Most errors are recoverable and the session will stay open, we
recommend to implementors to monitor and log error messages by default.
The unique ID of the client event used for error handling.
event_id string
The event type, must be output_audio_buffer.clear .
type string
{
    "event_id": "optional_client_event_id",
    "type": "output_audio_buffer.clear"
}

## Server events
error
Details of the error.
## Show properties
error object
The unique ID of the server event.
event_id string
type string
## OBJECT error
 
 
{
    "event_id": "event_890",
    "type": "error",
    "error": {
        "type": "invalid_request_error",
        "code": "invalid_event",
        "message": "The 'type' field is missing.
        "param": null,
        "event_id": "event_567"
    }
}


<!-- Page 213 -->
Returned when a Session is created. Emitted automatically when a new
connection is established as the first server event. This event will contain
the default Session configuration.
The event type, must be error .
session.created
The unique ID of the server event.
event_id string
Realtime session object configuration.
## Show properties
session object
The event type, must be session.created .
type string
OBJECT session.created
 
{
    "event_id": "event_1234",
    "type": "session.created",
    "session": {
        "id": "sess_001",
        "object": "realtime.session",
        "model": "gpt-4o-realtime-preview",
        "modalities": ["text", "audio"],
        "instructions": "...model instructions 
        "voice": "sage",
        "input_audio_format": "pcm16",
        "output_audio_format": "pcm16",
        "input_audio_transcription": null,
        "turn_detection": {
            "type": "server_vad",
            "threshold": 0.5,
            "prefix_padding_ms": 300,
            "silence_duration_ms": 200
        },
        "tools": [],
        "tool_choice": "auto",
        "temperature": 0.8,
        "max_response_output_tokens": "inf",
        "speed": 1.1,
        "tracing": "auto"


<!-- Page 214 -->
Returned when a session is updated with a session.update  event, unless
there is an error.
 
    }
}

session.updated
The unique ID of the server event.
event_id string
Realtime session object configuration.
## Show properties
session object
The event type, must be session.updated .
type string
OBJECT session.updated
{
    "event_id": "event_5678",
    "type": "session.updated",
    "session": {
        "id": "sess_001",
        "object": "realtime.session",
        "model": "gpt-4o-realtime-preview",
        "modalities": ["text"],
        "instructions": "New instructions",
        "voice": "sage",
        "input_audio_format": "pcm16",
        "output_audio_format": "pcm16",
        "input_audio_transcription": {
            "model": "whisper-1"
        },
        "turn_detection": null,
        "tools": [],
        "tool_choice": "none",
        "temperature": 0.7,
        "max_response_output_tokens": 200,
        "speed": 1.1,
        "tracing": "auto"
    }
}


<!-- Page 215 -->
Returned when a conversation is created. Emitted right after session
creation.
Returned when a conversation item is created. There are several scenarios
### that produce this event:
conversation.created
The conversation resource.
## Show properties
conversation object
The unique ID of the server event.
event_id string
The event type, must be conversation.created .
type string
OBJECT conversation.created
{
    "event_id": "event_9101",
    "type": "conversation.created",
    "conversation": {
        "id": "conv_001",
        "object": "realtime.conversation"
    }
}

conversation.item.created
The server is generating a Response, which if successful will produce
either one or two Items, which will be of type message  (role
assistant ) or type function_call .
OBJECT conversation.item.created
 
{
    "event_id": "event_1920",
    "type": "conversation.item.created",
    "previous_item_id": "msg_002",
    "item": {
        "id": "msg_003",


<!-- Page 216 -->
## Returned when a conversation item is retrieved with
conversation.item.retrieve .
The input audio buffer has been committed, either by the client or the
server (in server_vad  mode). The server will take the content of the
input audio buffer and add it to a new user message Item.
The client has sent a conversation.item.create  event to add a new
Item to the Conversation.
The unique ID of the server event.
event_id string
The item to add to the conversation.
## Show properties
item object
The ID of the preceding item in the Conversation context, allows the client to
understand the order of the conversation. Can be null  if the item has no
predecessor.
previous_item_id string or null
The event type, must be conversation.item.created .
type string
 
        "object": "realtime.item",
        "type": "message",
        "status": "completed",
        "role": "user",
        "content": []
    }
}

conversation.item.retrieved
The unique ID of the server event.
event_id string
item object
OBJECT conversation.item.retrieved
 
{
    "event_id": "event_1920",
    "type": "conversation.item.created",
    "previous_item_id": "msg_002",
    "item": {
        "id": "msg_003",


<!-- Page 217 -->
## This event is the output of audio transcription for user audio written to the
user audio buffer. Transcription begins when the input audio buffer is
committed by the client or server (in server_vad  mode). Transcription runs
asynchronously with Response creation, so this event may come before or
after the Response events.
Realtime API models accept audio natively, and thus input transcription is a
separate process run on a separate ASR (Automatic Speech Recognition)
model. The transcript may diverge somewhat from the model's
interpretation, and should be treated as a rough guide.
The item to add to the conversation.
## Show properties
The event type, must be conversation.item.retrieved .
type string
 
        "object": "realtime.item",
        "type": "message",
        "status": "completed",
        "role": "user",
        "content": [
            {
                "type": "input_audio",
                "transcript": "hello how are yo
                "audio": "base64encodedaudio=="
            }
        ]
    }
}

conversation.item.input_audio_transcription.completed
The index of the content part containing the audio.
content_index integer
OBJECT conversation.item.input_audio_transcriptio...
 
{
    "event_id": "event_2122",
    "type": "conversation.item.input_audio_tran
    "item_id": "msg_003",
    "content_index": 0,
    "transcript": "Hello, how are you?",
    "usage": {
      "type": "tokens",
      "total_tokens": 48,
      "input_tokens": 38,
      "input_token_details": {
        "text_tokens": 10,
        "audio_tokens": 28,
      },


<!-- Page 218 -->
## Returned when the text value of an input audio transcription content part is
updated.
The unique ID of the server event.
event_id string
The ID of the user message item containing the audio.
item_id string
The log probabilities of the transcription.
## Show properties
logprobs array or null
The transcribed text.
transcript string
The event type, must be conversation.item.input_audio_transcription.completed
.
type string
Usage statistics for the transcription.
## Show possible types
usage object
 
      "output_tokens": 10,
    }
}

conversation.item.input_audio_transcription.delta
The index of the content part in the item's content array.
content_index integer
OBJECT conversation.item.input_audio_transcriptio...
 
{
  "type": "conversation.item.input_audio_transcr
  "event_id": "event_001",
  "item_id": "item_001",
  "content_index": 0,


<!-- Page 219 -->
Returned when input audio transcription is configured, and a transcription
request for a user message failed. These events are separate from other
error  events so that the client can identify the related Item.
The text delta.
delta string
The unique ID of the server event.
event_id string
The ID of the item.
item_id string
The log probabilities of the transcription.
## Show properties
logprobs array or null
The event type, must be conversation.item.input_audio_transcription.delta .
type string
 
  "delta": "Hello"
}

conversation.item.input_audio_transcription.failed
The index of the content part containing the audio.
content_index integer
Details of the transcription error.
## Show properties
error object
OBJECT conversation.item.input_audio_transcriptio...
 
{
    "event_id": "event_2324",
    "type": "conversation.item.input_audio_tran
    "item_id": "msg_003",
    "content_index": 0,
    "error": {
        "type": "transcription_error",
        "code": "audio_unintelligible",
        "message": "The audio could not be tran
        "param": null


<!-- Page 220 -->
## Returned when an earlier assistant audio message item is truncated by the
client with a conversation.item.truncate  event. This event is used to
synchronize the server's understanding of the audio with the client's
playback.
## This action will truncate the audio and remove the server-side text
transcript to ensure there is no text in the context that hasn't been heard by
the user.
The unique ID of the server event.
event_id string
The ID of the user message item.
item_id string
The event type, must be conversation.item.input_audio_transcription.failed .
type string
 
    }

conversation.item.truncated
The duration up to which the audio was truncated, in milliseconds.
audio_end_ms integer
The index of the content part that was truncated.
content_index integer
The unique ID of the server event.
event_id string
OBJECT conversation.item.truncated
{
    "event_id": "event_2526",
    "type": "conversation.item.truncated",
    "item_id": "msg_004",
    "content_index": 0,
    "audio_end_ms": 1500
}


<!-- Page 221 -->
## Returned when an item in the conversation is deleted by the client with a
conversation.item.delete  event. This event is used to synchronize the
server's understanding of the conversation history with the client's view.
The ID of the assistant message item that was truncated.
item_id string
The event type, must be conversation.item.truncated .
type string
conversation.item.deleted
The unique ID of the server event.
event_id string
The ID of the item that was deleted.
item_id string
The event type, must be conversation.item.deleted .
type string
OBJECT conversation.item.deleted
{
    "event_id": "event_2728",
    "type": "conversation.item.deleted",
    "item_id": "msg_005"
}

input_audio_buffer.committed

<!-- Page 222 -->
Returned when an input audio buffer is committed, either by the client or
automatically in server VAD mode. The item_id  property is the ID of the
user message item that will be created, thus a conversation.item.created
event will also be sent to the client.
The unique ID of the server event.
event_id string
The ID of the user message item that will be created.
item_id string
The ID of the preceding item after which the new item will be inserted. Can be null  if
the item has no predecessor.
previous_item_id string or null
The event type, must be input_audio_buffer.committed .
type string
OBJECT input_audio_buffer.committed
{
    "event_id": "event_1121",
    "type": "input_audio_buffer.committed",
    "previous_item_id": "msg_001",
    "item_id": "msg_002"
}

input_audio_buffer.cleared

<!-- Page 223 -->
## Returned when the input audio buffer is cleared by the client with a
input_audio_buffer.clear  event.
Sent by the server when in server_vad  mode to indicate that speech has
been detected in the audio buffer. This can happen any time audio is added
to the buffer (unless speech is already detected). The client may want to use
this event to interrupt audio playback or provide visual feedback to the user.
The client should expect to receive a input_audio_buffer.speech_stopped
event when speech stops. The item_id  property is the ID of the user
message item that will be created when speech stops and will also be
included in the input_audio_buffer.speech_stopped  event (unless the client
manually commits the audio buffer during VAD activation).
The unique ID of the server event.
event_id string
The event type, must be input_audio_buffer.cleared .
type string
OBJECT input_audio_buffer.cleared
{
    "event_id": "event_1314",
    "type": "input_audio_buffer.cleared"
}

input_audio_buffer.speech_started
## Milliseconds from the start of all audio written to the buffer during the session when
speech was first detected. This will correspond to the beginning of audio sent to the
model, and thus includes the prefix_padding_ms  configured in the Session.
audio_start_ms integer
The unique ID of the server event.
event_id string
OBJECT input_audio_buffer.speech_started
 
 
{
    "event_id": "event_1516",
    "type": "input_audio_buffer.speech_started",
    "audio_start_ms": 1000,
    "item_id": "msg_003"
}


<!-- Page 224 -->
Returned in server_vad  mode when the server detects the end of speech
in the audio buffer. The server will also send an conversation.item.created
event with the user message item that is created from the audio buffer.
The ID of the user message item that will be created when speech stops.
item_id string
The event type, must be input_audio_buffer.speech_started .
type string
input_audio_buffer.speech_stopped
Milliseconds since the session started when speech stopped. This will correspond to
the end of audio sent to the model, and thus includes the min_silence_duration_ms
configured in the Session.
audio_end_ms integer
The unique ID of the server event.
event_id string
The ID of the user message item that will be created.
item_id string
The event type, must be input_audio_buffer.speech_stopped .
type string
OBJECT input_audio_buffer.speech_stopped
 
 
{
    "event_id": "event_1718",
    "type": "input_audio_buffer.speech_stopped",
    "audio_end_ms": 2000,
    "item_id": "msg_003"
}


<!-- Page 225 -->
Returned when a new Response is created. The first event of response
creation, where the response is in an initial state of in_progress .
Returned when a Response is done streaming. Always emitted, no matter
the final state. The Response object included in the response.done  event
will include all output Items in the Response but will omit the raw audio
data.
response.created
The unique ID of the server event.
event_id string
The response resource.
## Show properties
response object
The event type, must be response.created .
type string
OBJECT response.created
{
    "event_id": "event_2930",
    "type": "response.created",
    "response": {
        "id": "resp_001",
        "object": "realtime.response",
        "status": "in_progress",
        "status_details": null,
        "output": [],
        "usage": null
    }
}

response.done
The unique ID of the server event.
event_id string
response object
OBJECT response.done
 
{
    "event_id": "event_3132",
    "type": "response.done",
    "response": {
        "id": "resp_001",
        "object": "realtime.response",
        "status": "completed",
        "status_details": null,
        "output": [


<!-- Page 226 -->
Returned when a new Item is created during Response generation.
The response resource.
## Show properties
The event type, must be response.done .
type string
 
            {
                "id": "msg_006",
                "object": "realtime.item",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Sure, how can 
                    }
                ]
            }
        ],
        "usage": {
            "total_tokens":275,
            "input_tokens":127,
            "output_tokens":148,
            "input_token_details": {
                "cached_tokens":384,
                "text_tokens":119,
                "audio_tokens":8,
                "cached_tokens_details": {
                    "text_tokens": 128,
                    "audio_tokens": 256
                }
            },
            "output_token_details": {
              "text_tokens":36,
              "audio_tokens":112
            }
        }
    }
}

response.output_item.added
The unique ID of the server event.
event_id string
OBJECT response.output_item.added
 
{
    "event_id": "event_3334",
    "type": "response.output_item.added",


<!-- Page 227 -->
Returned when an Item is done streaming. Also emitted when a Response is
interrupted, incomplete, or cancelled.
The item to add to the conversation.
## Show properties
item object
The index of the output item in the Response.
output_index integer
The ID of the Response to which the item belongs.
response_id string
The event type, must be response.output_item.added .
type string
 
    "response_id": "resp_001",
    "output_index": 0,
    "item": {
        "id": "msg_007",
        "object": "realtime.item",
        "type": "message",
        "status": "in_progress",
        "role": "assistant",
        "content": []
    }
}

response.output_item.done
The unique ID of the server event.
event_id string
The item to add to the conversation.
## Show properties
item object
The index of the output item in the Response.
output_index integer
response_id string
OBJECT response.output_item.done
 
{
    "event_id": "event_3536",
    "type": "response.output_item.done",
    "response_id": "resp_001",
    "output_index": 0,
    "item": {
        "id": "msg_007",
        "object": "realtime.item",
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "Sure, I can help with 


<!-- Page 228 -->
## Returned when a new content part is added to an assistant message item
during response generation.
The ID of the Response to which the item belongs.
The event type, must be response.output_item.done .
type string
 
            }
        ]
    }
}

response.content_part.added
The index of the content part in the item's content array.
content_index integer
The unique ID of the server event.
event_id string
The ID of the item to which the content part was added.
item_id string
The index of the output item in the response.
output_index integer
The content part that was added.
## Show properties
part object
response_id string
OBJECT response.content_part.added
{
    "event_id": "event_3738",
    "type": "response.content_part.added",
    "response_id": "resp_001",
    "item_id": "msg_007",
    "output_index": 0,
    "content_index": 0,
    "part": {
        "type": "text",
        "text": ""
    }
}


<!-- Page 229 -->
## Returned when a content part is done streaming in an assistant message
item. Also emitted when a Response is interrupted, incomplete, or
cancelled.
The ID of the response.
The event type, must be response.content_part.added .
type string
response.content_part.done
The index of the content part in the item's content array.
content_index integer
The unique ID of the server event.
event_id string
The ID of the item.
item_id string
The index of the output item in the response.
output_index integer
The content part that is done.
## Show properties
part object
The ID of the response.
response_id string
OBJECT response.content_part.done
{
    "event_id": "event_3940",
    "type": "response.content_part.done",
    "response_id": "resp_001",
    "item_id": "msg_007",
    "output_index": 0,
    "content_index": 0,
    "part": {
        "type": "text",
        "text": "Sure, I can help with that."
    }
}


<!-- Page 230 -->
Returned when the text value of a "text" content part is updated.
The event type, must be response.content_part.done .
type string
response.text.delta
The index of the content part in the item's content array.
content_index integer
The text delta.
delta string
The unique ID of the server event.
event_id string
The ID of the item.
item_id string
The index of the output item in the response.
output_index integer
The ID of the response.
response_id string
type string
OBJECT response.text.delta
{
    "event_id": "event_4142",
    "type": "response.text.delta",
    "response_id": "resp_001",
    "item_id": "msg_007",
    "output_index": 0,
    "content_index": 0,
    "delta": "Sure, I can h"
}


<!-- Page 231 -->
Returned when the text value of a "text" content part is done streaming.
Also emitted when a Response is interrupted, incomplete, or cancelled.
The event type, must be response.text.delta .
response.text.done
The index of the content part in the item's content array.
content_index integer
The unique ID of the server event.
event_id string
The ID of the item.
item_id string
The index of the output item in the response.
output_index integer
The ID of the response.
response_id string
The final text content.
text string
The event type, must be response.text.done .
type string
OBJECT response.text.done
{
    "event_id": "event_4344",
    "type": "response.text.done",
    "response_id": "resp_001",
    "item_id": "msg_007",
    "output_index": 0,
    "content_index": 0,
    "text": "Sure, I can help with that."
}


<!-- Page 232 -->
## Returned when the model-generated transcription of audio output is
updated.
response.audio_transcript.delta
The index of the content part in the item's content array.
content_index integer
The transcript delta.
delta string
The unique ID of the server event.
event_id string
The ID of the item.
item_id string
The index of the output item in the response.
output_index integer
The ID of the response.
response_id string
The event type, must be response.audio_transcript.delta .
type string
OBJECT response.audio_transcript.delta
{
    "event_id": "event_4546",
    "type": "response.audio_transcript.delta",
    "response_id": "resp_001",
    "item_id": "msg_008",
    "output_index": 0,
    "content_index": 0,
    "delta": "Hello, how can I a"
}

response.audio_transcript.done

<!-- Page 233 -->
## Returned when the model-generated transcription of audio output is done
streaming. Also emitted when a Response is interrupted, incomplete, or
cancelled.
The index of the content part in the item's content array.
content_index integer
The unique ID of the server event.
event_id string
The ID of the item.
item_id string
The index of the output item in the response.
output_index integer
The ID of the response.
response_id string
The final transcript of the audio.
transcript string
The event type, must be response.audio_transcript.done .
type string
OBJECT response.audio_transcript.done
 
 
{
    "event_id": "event_4748",
    "type": "response.audio_transcript.done",
    "response_id": "resp_001",
    "item_id": "msg_008",
    "output_index": 0,
    "content_index": 0,
    "transcript": "Hello, how can I assist you to
}

response.audio.delta

<!-- Page 234 -->
Returned when the model-generated audio is updated.
Returned when the model-generated audio is done. Also emitted when a
Response is interrupted, incomplete, or cancelled.
The index of the content part in the item's content array.
content_index integer
Base64-encoded audio data delta.
delta string
The unique ID of the server event.
event_id string
The ID of the item.
item_id string
The index of the output item in the response.
output_index integer
The ID of the response.
response_id string
The event type, must be response.audio.delta .
type string
OBJECT response.audio.delta
{
    "event_id": "event_4950",
    "type": "response.audio.delta",
    "response_id": "resp_001",
    "item_id": "msg_008",
    "output_index": 0,
    "content_index": 0,
    "delta": "Base64EncodedAudioDelta"
}

response.audio.done
content_index integer
OBJECT response.audio.done
 
{
    "event_id": "event_5152",
    "type": "response.audio.done",


<!-- Page 235 -->
Returned when the model-generated function call arguments are updated.
The index of the content part in the item's content array.
The unique ID of the server event.
event_id string
The ID of the item.
item_id string
The index of the output item in the response.
output_index integer
The ID of the response.
response_id string
The event type, must be response.audio.done .
type string
 
    "response_id": "resp_001",
    "item_id": "msg_008",
    "output_index": 0,
    "content_index": 0
}

response.function_call_arguments.delta
The ID of the function call.
call_id string
The arguments delta as a JSON string.
delta string
OBJECT response.function_call_arguments.delta
 
{
    "event_id": "event_5354",
    "type": "response.function_call_arguments.de
    "response_id": "resp_002",
    "item_id": "fc_001",
    "output_index": 0,
    "call_id": "call_001",


<!-- Page 236 -->
## Returned when the model-generated function call arguments are done
streaming. Also emitted when a Response is interrupted, incomplete, or
cancelled.
The unique ID of the server event.
event_id string
The ID of the function call item.
item_id string
The index of the output item in the response.
output_index integer
The ID of the response.
response_id string
The event type, must be response.function_call_arguments.delta .
type string
 
    "delta": "{\"location\": \"San\""
}

response.function_call_arguments.done
The final arguments as a JSON string.
arguments string
The ID of the function call.
call_id string
The unique ID of the server event.
event_id string
OBJECT response.function_call_arguments.done
 
 
{
    "event_id": "event_5556",
    "type": "response.function_call_arguments.don
    "response_id": "resp_002",
    "item_id": "fc_001",
    "output_index": 0,
    "call_id": "call_001",
    "arguments": "{\"location\": \"San Francisco\
}


<!-- Page 237 -->
## Returned when a transcription session is updated with a
transcription_session.update  event, unless there is an error.
The ID of the function call item.
item_id string
The index of the output item in the response.
output_index integer
The ID of the response.
response_id string
The event type, must be response.function_call_arguments.done .
type string
transcription_session.updated
The unique ID of the server event.
event_id string
A new Realtime transcription session configuration.
When a session is created on the server via REST API, the session object also contains
an ephemeral key. Default TTL for keys is 10 minutes. This property is not present when
a session is updated via the WebSocket API.
## Show properties
session object
OBJECT transcription_session.updated
 
{
  "event_id": "event_5678",
  "type": "transcription_session.updated",
  "session": {
    "id": "sess_001",
    "object": "realtime.transcription_session",
    "input_audio_format": "pcm16",
    "input_audio_transcription": {
      "model": "gpt-4o-transcribe",
      "prompt": "",
      "language": ""
    },
    "turn_detection": {


<!-- Page 238 -->
Emitted at the beginning of a Response to indicate the updated rate limits.
When a Response is created some tokens will be "reserved" for the output
tokens, the rate limits shown here reflect that reservation, which is then
adjusted accordingly once the Response is completed.
The event type, must be transcription_session.updated .
type string
 
      "type": "server_vad",
      "threshold": 0.5,
      "prefix_padding_ms": 300,
      "silence_duration_ms": 500,
      "create_response": true,
      // "interrupt_response": false  -- this w
    },
    "input_audio_noise_reduction": {
      "type": "near_field"
    },
    "include": [
      "item.input_audio_transcription.avg_logpr
    ],
  }
}

rate_limits.updated
The unique ID of the server event.
event_id string
List of rate limit information.
## Show properties
rate_limits array
OBJECT rate_limits.updated
 
{
    "event_id": "event_5758",
    "type": "rate_limits.updated",
    "rate_limits": [
        {
            "name": "requests",
            "limit": 1000,
            "remaining": 999,
            "reset_seconds": 60
        },
        {
            "name": "tokens",


<!-- Page 239 -->
WebRTC Only: Emitted when the server begins streaming audio to the
client. This event is emitted after an audio content part has been added (
response.content_part.added ) to the response. Learn more.
The event type, must be rate_limits.updated .
type string
 
            "limit": 50000,
            "remaining": 49950,
            "reset_seconds": 60
        }
    ]
}

output_audio_buffer.started
The unique ID of the server event.
event_id string
The unique ID of the response that produced the audio.
response_id string
The event type, must be output_audio_buffer.started .
type string
OBJECT output_audio_buffer.started
{
    "event_id": "event_abc123",
    "type": "output_audio_buffer.started",
    "response_id": "resp_abc123"
}

output_audio_buffer.stopped

<!-- Page 240 -->
WebRTC Only: Emitted when the output audio buffer has been completely
drained on the server, and no more audio is forthcoming. This event is
emitted after the full response data has been sent to the client (
response.done ). Learn more.
WebRTC Only: Emitted when the output audio buffer is cleared. This
happens either in VAD mode when the user has interrupted (
input_audio_buffer.speech_started ), or when the client has emitted the
output_audio_buffer.clear  event to manually cut off the current audio
response. Learn more.
The unique ID of the server event.
event_id string
The unique ID of the response that produced the audio.
response_id string
The event type, must be output_audio_buffer.stopped .
type string
OBJECT output_audio_buffer.stopped
{
    "event_id": "event_abc123",
    "type": "output_audio_buffer.stopped",
    "response_id": "resp_abc123"
}

output_audio_buffer.cleared
The unique ID of the server event.
event_id string
The unique ID of the response that produced the audio.
response_id string
type string
OBJECT output_audio_buffer.cleared
{
    "event_id": "event_abc123",
    "type": "output_audio_buffer.cleared",
    "response_id": "resp_abc123"
}


<!-- Page 241 -->
## The Chat Completions API endpoint will generate a model response from a list of messages comprising a
conversation.
### Related guides:
Starting a new project? We recommend trying Responses to take advantage of the latest OpenAI platform
features. Compare Chat Completions with Responses.
POST https://api.openai.com/v1/chat/completions
Starting a new project? We recommend trying Responses to take
advantage of the latest OpenAI platform features. Compare
The event type, must be output_audio_buffer.cleared .
## Chat Completions
Quickstart
Text inputs and outputs
Image inputs
Audio inputs and outputs
Structured Outputs
Function calling
Conversation state
Create chat completion
Default
Image input
Streaming
Functions
Logp
Example request
gpt-5
python

<!-- Page 242 -->
Chat Completions with Responses.
Creates a model response for the given chat conversation. Learn more in
the text generation, vision, and audio guides.
## Parameter support can differ depending on the model used to generate the
response, particularly for newer reasoning models. Parameters that are only
supported for reasoning models are noted below. For the current state of
unsupported parameters in reasoning models, refer to the reasoning guide.
## Request body
A list of messages comprising the conversation so far. Depending on the model you
use, different message types (modalities) are supported, like text, images, and audio.
## Show possible types
messages array
Required
Model ID used to generate the response, like gpt-4o  or o3 . OpenAI offers a wide
range of models with different capabilities, performance characteristics, and price
points. Refer to the model guide to browse and compare available models.
model string
## Required
Parameters for audio output. Required when audio output is requested with
modalities: ["audio"] . Learn more.
## Show properties
audio object or null
Optional
Number between -2.0 and 2.0. Positive values penalize new tokens based on their
existing frequency in the text so far, decreasing the model's likelihood to repeat the
same line verbatim.
frequency_penalty number or null
## Optional
Defaults to 0
Deprecated in favor of tool_choice .
function_call
## Deprecated string or object
Optional
from openai import OpenAI
client = OpenAI()
completion = client.chat.completions.create(
  model="gpt-5",
  messages=[
    {"role": "developer", "content": "You are a 
    {"role": "user", "content": "Hello!"}
  ]
)
print(completion.choices[0].message)

## Response
 
{
  "id": "chatcmpl-B9MBs8CjcvOU2jLn4n570S5qMJKcT
  "object": "chat.completion",
  "created": 1741569952,
  "model": "gpt-4.1-2025-04-14",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I assist you
        "refusal": null,
        "annotations": []
      },
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 19,
    "completion_tokens": 10,
    "total_tokens": 29,
    "prompt_tokens_details": {
      "cached_tokens": 0,
      "audio_tokens": 0


<!-- Page 243 -->
Controls which (if any) function is called by the model.
none  means the model will not call a function and instead generates a message.
auto  means the model can pick between generating a message or calling a function.
Specifying a particular function via {"name": "my_function"}  forces the model to
call that function.
none  is the default when no functions are present. auto  is the default if functions
are present.
## Show possible types
Deprecated in favor of tools .
A list of functions the model may generate JSON inputs for.
## Show properties
functions
Deprecated array
Optional
Modify the likelihood of specified tokens appearing in the completion.
Accepts a JSON object that maps tokens (specified by their token ID in the tokenizer)
to an associated bias value from -100 to 100. Mathematically, the bias is added to the
logits generated by the model prior to sampling. The exact effect will vary per model,
but values between -1 and 1 should decrease or increase likelihood of selection; values
like -100 or 100 should result in a ban or exclusive selection of the relevant token.
logit_bias
map
## Optional
Defaults to null
Whether to return log probabilities of the output tokens or not. If true, returns the log
probabilities of each output token returned in the content  of message .
logprobs boolean or null
## Optional
Defaults to false
An upper bound for the number of tokens that can be generated for a completion,
including visible output tokens and reasoning tokens.
max_completion_tokens integer or null
## Optional
max_tokens
## Deprecated integer or null
Optional
 
    },
    "completion_tokens_details": {
      "reasoning_tokens": 0,
      "audio_tokens": 0,
      "accepted_prediction_tokens": 0,
      "rejected_prediction_tokens": 0
    }
  },
  "service_tier": "default"
}


<!-- Page 244 -->
The maximum number of tokens that can be generated in the chat completion. This
value can be used to control costs for text generated via API.
This value is now deprecated in favor of max_completion_tokens , and is not
compatible with o-series models.
Set of 16 key-value pairs that can be attached to an object. This can be useful for
storing additional information about the object in a structured format, and querying for objects via API or the dashboard.
Keys are strings with a maximum length of 64 characters. Values are strings with a
maximum length of 512 characters.
metadata
map
## Optional
Output types that you would like the model to generate. Most models are capable of
generating text, which is the default:
["text"]
The gpt-4o-audio-preview  model can also be used to generate audio. To request
that this model generate both text and audio responses, you can use:
["text", "audio"]
modalities array or null
## Optional
How many chat completion choices to generate for each input message. Note that you
will be charged based on the number of generated tokens across all of the choices.
Keep n  as 1  to minimize costs.
n integer or null
## Optional
Defaults to 1
Whether to enable parallel function calling during tool use.
parallel_tool_calls boolean
## Optional
Defaults to true
Configuration for a Predicted Output, which can greatly improve response times when
large parts of the model response are known ahead of time. This is most common
when you are regenerating a file with only minor changes to most of the content.
prediction object
## Optional

<!-- Page 245 -->
## Show possible types
Number between -2.0 and 2.0. Positive values penalize new tokens based on whether
they appear in the text so far, increasing the model's likelihood to talk about new topics.
presence_penalty number or null
## Optional
Defaults to 0
## Used by OpenAI to cache responses for similar requests to optimize your cache hit
rates. Replaces the user  field. Learn more.
prompt_cache_key string
## Optional
Constrains effort on reasoning for reasoning models. Currently supported values are
minimal , low , medium , and high . Reducing reasoning effort can result in faster
responses and fewer tokens used on reasoning in a response.
reasoning_effort string or null
## Optional
Defaults to medium
An object specifying the format that the model must output.
Setting to { "type": "json_schema", "json_schema": {...} }  enables Structured
Outputs which ensures the model will match your supplied JSON schema. Learn more
in the Structured Outputs guide.
Setting to { "type": "json_object" }  enables the older JSON mode, which ensures
the message the model generates is valid JSON. Using json_schema  is preferred for
models that support it.
## Show possible types
response_format object
## Optional
A stable identifier used to help detect users of your application that may be violating
OpenAI's usage policies. The IDs should be a string that uniquely identifies each user.
We recommend hashing their username or email address, in order to avoid sending us
any identifying information. Learn more.
safety_identifier string
## Optional
seed integer or null
Optional

<!-- Page 246 -->
This feature is in Beta. If specified, our system will make a best effort to sample
deterministically, such that repeated requests with the same seed  and parameters
should return the same result. Determinism is not guaranteed, and you should refer to
the system_fingerprint  response parameter to monitor changes in the backend.
Specifies the processing type used for serving the request.
When the service_tier  parameter is set, the response body will include the
service_tier  value based on the processing mode actually used to serve the
request. This response value may be different from the value set in the parameter.
service_tier string or null
## Optional
Defaults to auto
If set to 'auto', then the request will be processed with the service tier configured
in the Project settings. Unless otherwise configured, the Project will use 'default'.
If set to 'default', then the request will be processed with the standard pricing and
performance for the selected model.
If set to 'flex' or 'priority', then the request will be processed with the
corresponding service tier. Contact sales to learn more about Priority processing.
When not set, the default behavior is 'auto'.
Not supported with latest reasoning models o3  and o4-mini .
Up to 4 sequences where the API will stop generating further tokens. The returned text
will not contain the stop sequence.
stop string / array / null
## Optional
Defaults to null
Whether or not to store the output of this chat completion request for use in our
model distillation or evals products.
Supports text and image inputs. Note: image inputs over 10MB will be dropped.
store boolean or null
## Optional
Defaults to false
If set to true, the model response data will be streamed to the client as it is generated
using server-sent events. See the Streaming section below for more information, along
stream boolean or null
## Optional
Defaults to false

<!-- Page 247 -->
with the streaming responses guide for more information on how to handle the
streaming events.
Options for streaming response. Only set this when you set stream: true .
## Show properties
stream_options object or null
## Optional
Defaults to null
What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make
the output more random, while lower values like 0.2 will make it more focused and
deterministic. We generally recommend altering this or top_p  but not both.
temperature number or null
## Optional
Defaults to 1
Controls which (if any) tool is called by the model. none  means the model will not call
any tool and instead generates a message. auto  means the model can pick between
generating a message or calling one or more tools. required  means the model must
call one or more tools. Specifying a particular tool via
{"type": "function", "function": {"name": "my_function"}}  forces the model to
call that tool.
none  is the default when no tools are present. auto  is the default if tools are
present.
## Show possible types
tool_choice string or object
## Optional
A list of tools the model may call. You can provide either custom tools or function tools.
## Show possible types
tools array
Optional
An integer between 0 and 20 specifying the number of most likely tokens to return at
each token position, each with an associated log probability. logprobs  must be set to
true  if this parameter is used.
top_logprobs integer or null
## Optional
top_p number or null
## Optional
Defaults to 1

<!-- Page 248 -->
## Returns
An alternative to sampling with temperature, called nucleus sampling, where the model
considers the results of the tokens with top_p probability mass. So 0.1 means only the
tokens comprising the top 10% probability mass are considered.
We generally recommend altering this or temperature  but not both.
This field is being replaced by safety_identifier  and prompt_cache_key . Use
prompt_cache_key  instead to maintain caching optimizations. A stable identifier for
your end-users. Used to boost cache hit rates by better bucketing similar requests and
to help OpenAI detect and prevent abuse. Learn more.
user
## Deprecated string
Optional
Constrains the verbosity of the model's response. Lower values will result in more
concise responses, while higher values will result in more verbose responses. Currently
supported values are low , medium , and high .
verbosity string or null
## Optional
Defaults to medium
This tool searches the web for relevant results to use in a response. Learn more about
the web search tool.
## Show properties
web_search_options object
## Optional
Returns a chat completion object, or a streamed sequence of chat completion chunk objects if the request is streamed.
## Get chat completion

<!-- Page 249 -->
GET https://api.openai.com/v1/chat/completions/{completion_id}
Get a stored chat completion. Only Chat Completions that have been
created with the store  parameter set to true  will be returned.
## Path parameters
Returns
The ID of the chat completion to retrieve.
completion_id string
## Required
The ChatCompletion object matching the specified ID.
## Example request
python
 
 
from openai import OpenAI
client = OpenAI()
completions = client.chat.completions.list()
first_id = completions[0].id
first_completion = client.chat.completions.retrie
print(first_completion)

## Response
{
  "object": "chat.completion",
  "id": "chatcmpl-abc123",
  "model": "gpt-4o-2024-08-06",
  "created": 1738960610,
  "request_id": "req_ded8ab984ec4bf840f37566c101
  "tool_choice": null,
  "usage": {
    "total_tokens": 31,
    "completion_tokens": 18,
    "prompt_tokens": 13
  },
  "seed": 4944116822809979520,
  "top_p": 1.0,
  "temperature": 1.0,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "system_fingerprint": "fp_50cad350e4",
  "input_user": null,
  "service_tier": "default",
  "tools": null,
  "metadata": {},
  "choices": [
    {
      "index": 0,
      "message": {
        "content": "Mind of circuits hum,  \nLea


<!-- Page 250 -->
GET https://api.openai.com/v1/chat/completions/{completion_id}/messages
Get the messages in a stored chat completion. Only Chat Completions that
have been created with the store  parameter set to true  will be returned.
## Path parameters
Query parameters
Returns
        "role": "assistant",
        "tool_calls": null,
        "function_call": null
      },
      "finish_reason": "stop",
      "logprobs": null
    }
  ],
  "response_format": null
}

## Get chat messages
The ID of the chat completion to retrieve messages from.
completion_id string
## Required
Identifier for the last message from the previous pagination request.
after string
## Optional
Number of messages to retrieve.
limit integer
## Optional
Defaults to 20
Sort order for messages by timestamp. Use asc  for ascending order or desc  for
descending order. Defaults to asc .
order string
## Optional
Defaults to asc
A list of messages for the specified chat completion.
## Example request
python
 
 
from openai import OpenAI
client = OpenAI()
completions = client.chat.completions.list()
first_id = completions[0].id
first_completion = client.chat.completions.retrie
messages = client.chat.completions.messages.list(
print(messages)

## Response
 
 
{
  "object": "list",
  "data": [
    {
      "id": "chatcmpl-AyPNinnUqUDYo9SAdA52NobMfl
      "role": "user",
      "content": "write a haiku about ai",
      "name": null,
      "content_parts": null
    }
  ],
  "first_id": "chatcmpl-AyPNinnUqUDYo9SAdA52NobM
  "last_id": "chatcmpl-AyPNinnUqUDYo9SAdA52NobMf
  "has_more": false
}


<!-- Page 251 -->
GET https://api.openai.com/v1/chat/completions
List stored Chat Completions. Only Chat Completions that have been
stored with the store  parameter set to true  will be returned.
## Query parameters
Returns
List Chat Completions
Identifier for the last chat completion from the previous pagination request.
after string
## Optional
Number of Chat Completions to retrieve.
limit integer
## Optional
Defaults to 20
A list of metadata keys to filter the Chat Completions by. Example:
metadata[key1]=value1&metadata[key2]=value2
metadata
map
## Optional
The model used to generate the Chat Completions.
model string
## Optional
Sort order for Chat Completions by timestamp. Use asc  for ascending order or
desc  for descending order. Defaults to asc .
order string
## Optional
Defaults to asc
A list of Chat Completions matching the specified filters.
## Example request
python
from openai import OpenAI
client = OpenAI()
completions = client.chat.completions.list()
print(completions)

## Response
 
{
  "object": "list",
  "data": [
    {
      "object": "chat.completion",
      "id": "chatcmpl-AyPNinnUqUDYo9SAdA52NobMf
      "model": "gpt-4.1-2025-04-14",
      "created": 1738960610,
      "request_id": "req_ded8ab984ec4bf840f3756
      "tool_choice": null,
      "usage": {
        "total_tokens": 31,
        "completion_tokens": 18,
        "prompt_tokens": 13
      },
      "seed": 4944116822809979520,
      "top_p": 1.0,
      "temperature": 1.0,
      "presence_penalty": 0.0,
      "frequency_penalty": 0.0,
      "system_fingerprint": "fp_50cad350e4",
      "input_user": null,
      "service_tier": "default",
      "tools": null,
      "metadata": {},
      "choices": [


<!-- Page 252 -->
POST https://api.openai.com/v1/chat/completions/{completion_id}
Modify a stored chat completion. Only Chat Completions that have been
created with the store  parameter set to true  can be modified. Currently,
the only supported modification is to update the metadata  field.
## Path parameters
Request body
Returns
 
        {
          "index": 0,
          "message": {
            "content": "Mind of circuits hum,  
            "role": "assistant",
            "tool_calls": null,
            "function_call": null
          },
          "finish_reason": "stop",
          "logprobs": null
        }
      ],
      "response_format": null
    }
  ],
  "first_id": "chatcmpl-AyPNinnUqUDYo9SAdA52Nob
  "last_id": "chatcmpl-AyPNinnUqUDYo9SAdA52NobM
  "has_more": false
}

## Update chat completion
The ID of the chat completion to update.
completion_id string
## Required
Set of 16 key-value pairs that can be attached to an object. This can be useful for
storing additional information about the object in a structured format, and querying for objects via API or the dashboard.
Keys are strings with a maximum length of 64 characters. Values are strings with a
maximum length of 512 characters.
metadata
map
## Required
The ChatCompletion object matching the specified ID.
## Example request
python
 
 
from openai import OpenAI
client = OpenAI()
completions = client.chat.completions.list()
first_id = completions[0].id
updated_completion = client.chat.completions.upda
print(updated_completion)

## Response
 
{
  "object": "chat.completion",
  "id": "chatcmpl-AyPNinnUqUDYo9SAdA52NobMflmj2
  "model": "gpt-4o-2024-08-06",
  "created": 1738960610,
  "request_id": "req_ded8ab984ec4bf840f37566c10
  "tool_choice": null,
  "usage": {
    "total_tokens": 31,
    "completion_tokens": 18,
    "prompt_tokens": 13
  },
  "seed": 4944116822809979520,
  "top_p": 1.0,
  "temperature": 1.0,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "system_fingerprint": "fp_50cad350e4",
  "input_user": null,
  "service_tier": "default",


<!-- Page 253 -->
DELETE https://api.openai.com/v1/chat/completions/{completion_id}
Delete a stored chat completion. Only Chat Completions that have been
created with the store  parameter set to true  can be deleted.
## Path parameters
Returns
Represents a chat completion response returned by model, based on the
provided input.
 
  "tools": null,
  "metadata": {
    "foo": "bar"
  },
  "choices": [
    {
      "index": 0,
      "message": {
        "content": "Mind of circuits hum,  \nLe
        "role": "assistant",
        "tool_calls": null,
        "function_call": null
      },
      "finish_reason": "stop",
      "logprobs": null
    }
  ],
  "response_format": null
}

## Delete chat completion
The ID of the chat completion to delete.
completion_id string
## Required
A deletion confirmation object.
## Example request
python
 
 
from openai import OpenAI
client = OpenAI()
completions = client.chat.completions.list()
first_id = completions[0].id
delete_response = client.chat.completions.delete(
print(delete_response)

## Response
 
 
{
  "object": "chat.completion.deleted",
  "id": "chatcmpl-AyPNinnUqUDYo9SAdA52NobMflmj2",
  "deleted": true
}

## The chat completion object
OBJECT The chat completion object
 
{
  "id": "chatcmpl-B9MHDbslfkBeAs8l4bebGdFOJ6PeG


<!-- Page 254 -->
A list of chat completion choices. Can be more than one if n  is greater than 1.
## Show properties
choices array
The Unix timestamp (in seconds) of when the chat completion was created.
created integer
A unique identifier for the chat completion.
id string
The model used for the chat completion.
model string
The object type, which is always chat.completion .
object string
Specifies the processing type used for serving the request.
When the service_tier  parameter is set, the response body will include the
service_tier  value based on the processing mode actually used to serve the
request. This response value may be different from the value set in the parameter.
service_tier string or null
If set to 'auto', then the request will be processed with the service tier configured
in the Project settings. Unless otherwise configured, the Project will use 'default'.
If set to 'default', then the request will be processed with the standard pricing and
performance for the selected model.
If set to 'flex' or 'priority', then the request will be processed with the
corresponding service tier. Contact sales to learn more about Priority processing.
When not set, the default behavior is 'auto'.
This fingerprint represents the backend configuration that the model runs with.
system_fingerprint string
 
  "object": "chat.completion",
  "created": 1741570283,
  "model": "gpt-4o-2024-08-06",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The image shows a wooden bo
        "refusal": null,
        "annotations": []
      },
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 1117,
    "completion_tokens": 46,
    "total_tokens": 1163,
    "prompt_tokens_details": {
      "cached_tokens": 0,
      "audio_tokens": 0
    },
    "completion_tokens_details": {
      "reasoning_tokens": 0,
      "audio_tokens": 0,
      "accepted_prediction_tokens": 0,
      "rejected_prediction_tokens": 0
    }
  },
  "service_tier": "default",
  "system_fingerprint": "fp_fc9f1d7035"
}


<!-- Page 255 -->
An object representing a list of Chat Completions.
## Can be used in conjunction with the seed  request parameter to understand when
backend changes have been made that might impact determinism.
Usage statistics for the completion request.
## Show properties
usage object
The chat completion list object
An array of chat completion objects.
## Show properties
data array
The identifier of the first chat completion in the data array.
first_id string
Indicates whether there are more Chat Completions available.
has_more boolean
The identifier of the last chat completion in the data array.
last_id string
The type of this object. It is always set to "list".
object string
## OBJECT The chat completion list object
 
{
  "object": "list",
  "data": [
    {
      "object": "chat.completion",
      "id": "chatcmpl-AyPNinnUqUDYo9SAdA52NobMf
      "model": "gpt-4o-2024-08-06",
      "created": 1738960610,
      "request_id": "req_ded8ab984ec4bf840f3756
      "tool_choice": null,
      "usage": {
        "total_tokens": 31,
        "completion_tokens": 18,
        "prompt_tokens": 13
      },
      "seed": 4944116822809979520,
      "top_p": 1.0,
      "temperature": 1.0,
      "presence_penalty": 0.0,
      "frequency_penalty": 0.0,
      "system_fingerprint": "fp_50cad350e4",
      "input_user": null,
      "service_tier": "default",


<!-- Page 256 -->
 
      "tools": null,
      "metadata": {},
      "choices": [
        {
          "index": 0,
          "message": {
            "content": "Mind of circuits hum,  
            "role": "assistant",
            "tool_calls": null,
            "function_call": null
          },
          "finish_reason": "stop",
          "logprobs": null
        }
      ],
      "response_format": null
    }
  ],
  "first_id": "chatcmpl-AyPNinnUqUDYo9SAdA52Nob
  "last_id": "chatcmpl-AyPNinnUqUDYo9SAdA52NobM
  "has_more": false
}

## The chat completion message list object

<!-- Page 257 -->
An object representing a list of chat completion messages.
Stream Chat Completions in real time. Receive chunks of completions returned from the model using
server-sent events. Learn more.
## Represents a streamed chunk of a chat completion response returned by
the model, based on the provided input. Learn more.
An array of chat completion message objects.
## Show properties
data array
The identifier of the first chat message in the data array.
first_id string
Indicates whether there are more chat messages available.
has_more boolean
The identifier of the last chat message in the data array.
last_id string
The type of this object. It is always set to "list".
object string
## OBJECT The chat completion message list object
 
 
{
  "object": "list",
  "data": [
    {
      "id": "chatcmpl-AyPNinnUqUDYo9SAdA52NobMfl
      "role": "user",
      "content": "write a haiku about ai",
      "name": null,
      "content_parts": null
    }
  ],
  "first_id": "chatcmpl-AyPNinnUqUDYo9SAdA52NobM
  "last_id": "chatcmpl-AyPNinnUqUDYo9SAdA52NobMf
  "has_more": false
}

## Streaming
The chat completion chunk object
OBJECT The chat completion chunk object

<!-- Page 258 -->
A list of chat completion choices. Can contain more than one elements if n  is greater
than 1. Can also be empty for the last chunk if you set
stream_options: {"include_usage": true} .
## Show properties
choices array
The Unix timestamp (in seconds) of when the chat completion was created. Each
chunk has the same timestamp.
created integer
A unique identifier for the chat completion. Each chunk has the same ID.
id string
The model to generate the completion.
model string
The object type, which is always chat.completion.chunk .
object string
Specifies the processing type used for serving the request.
When the service_tier  parameter is set, the response body will include the
service_tier  value based on the processing mode actually used to serve the
request. This response value may be different from the value set in the parameter.
service_tier string or null
If set to 'auto', then the request will be processed with the service tier configured
in the Project settings. Unless otherwise configured, the Project will use 'default'.
If set to 'default', then the request will be processed with the standard pricing and
performance for the selected model.
If set to 'flex' or 'priority', then the request will be processed with the
corresponding service tier. Contact sales to learn more about Priority processing.
When not set, the default behavior is 'auto'.
{"id":"chatcmpl-123","object":"chat.completion.ch
{"id":"chatcmpl-123","object":"chat.completion.ch
....
{"id":"chatcmpl-123","object":"chat.completion.ch


<!-- Page 259 -->
Build assistants that can call models and use tools to perform tasks.
## Get started with the Assistants API
POST https://api.openai.com/v1/assistants
Create an assistant with a model and instructions.
## Request body
This fingerprint represents the backend configuration that the model runs with. Can be
used in conjunction with the seed  request parameter to understand when backend
changes have been made that might impact determinism.
system_fingerprint string
Usage statistics for the completion request.
## Show properties
usage object or null
Assistants
Beta
Create assistant
Beta
ID of the model to use. You can use the List models API to see all of your available
models, or see our Model overview for descriptions of them.
model string
## Required
Code Interpreter
Files
Example request
python
 
from openai import OpenAI
client = OpenAI()
my_assistant = client.beta.assistants.create(
    instructions="You are a personal math tutor
    name="Math Tutor",
    tools=[{"type": "code_interpreter"}],


<!-- Page 260 -->
The description of the assistant. The maximum length is 512 characters.
description string or null
## Optional
The system instructions that the assistant uses. The maximum length is 256,000
characters.
instructions string or null
## Optional
Set of 16 key-value pairs that can be attached to an object. This can be useful for
storing additional information about the object in a structured format, and querying for objects via API or the dashboard.
Keys are strings with a maximum length of 64 characters. Values are strings with a
maximum length of 512 characters.
metadata
map
## Optional
The name of the assistant. The maximum length is 256 characters.
name string or null
## Optional
Constrains effort on reasoning for reasoning models. Currently supported values are
minimal , low , medium , and high . Reducing reasoning effort can result in faster
responses and fewer tokens used on reasoning in a response.
reasoning_effort string or null
## Optional
Defaults to medium
Specifies the format that the model must output. Compatible with GPT-4o,
GPT-4 Turbo, and all GPT-3.5 Turbo models since gpt-3.5-turbo-1106 .
Setting to { "type": "json_schema", "json_schema": {...} }  enables Structured
Outputs which ensures the model will match your supplied JSON schema. Learn more
in the Structured Outputs guide.
Setting to { "type": "json_object" }  enables JSON mode, which ensures the
message the model generates is valid JSON.
Important: when using JSON mode, you must also instruct the model to produce JSON
yourself via a system or user message. Without this, the model may generate an
unending stream of whitespace until the generation reaches the token limit, resulting in
response_format
"auto" or object
## Optional
 
    model="gpt-4o",
)
print(my_assistant)

## Response
 
 
{
  "id": "asst_abc123",
  "object": "assistant",
  "created_at": 1698984975,
  "name": "Math Tutor",
  "description": null,
  "model": "gpt-4o",
  "instructions": "You are a personal math tutor
  "tools": [
    {
      "type": "code_interpreter"
    }
  ],
  "metadata": {},
  "top_p": 1.0,
  "temperature": 1.0,
  "response_format": "auto"
}


<!-- Page 261 -->
## Returns
a long-running and seemingly "stuck" request. Also note that the message content may
be partially cut off if finish_reason="length" , which indicates the generation
exceeded max_tokens  or the conversation exceeded the max context length.
## Show possible types
What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make
the output more random, while lower values like 0.2 will make it more focused and
deterministic.
temperature number or null
## Optional
Defaults to 1
A set of resources that are used by the assistant's tools. The resources are specific to
the type of tool. For example, the code_interpreter  tool requires a list of file IDs,
while the file_search  tool requires a list of vector store IDs.
## Show properties
tool_resources object or null
## Optional
A list of tool enabled on the assistant. There can be a maximum of 128 tools per
assistant. Tools can be of types code_interpreter , file_search , or function .
## Show possible types
tools array
Optional
Defaults to []
An alternative to sampling with temperature, called nucleus sampling, where the model
considers the results of the tokens with top_p probability mass. So 0.1 means only the
tokens comprising the top 10% probability mass are considered.
We generally recommend altering this or temperature but not both.
top_p number or null
## Optional
Defaults to 1
An assistant object.

<!-- Page 262 -->
GET https://api.openai.com/v1/assistants
Returns a list of assistants.
## Query parameters
Returns
List assistants
Beta
A cursor for use in pagination. after  is an object ID that defines your place in the list.
For instance, if you make a list request and receive 100 objects, ending with obj_foo,
your subsequent call can include after=obj_foo in order to fetch the next page of the
list.
after string
## Optional
A cursor for use in pagination. before  is an object ID that defines your place in the
list. For instance, if you make a list request and receive 100 objects, starting with
obj_foo, your subsequent call can include before=obj_foo in order to fetch the previous
page of the list.
before string
## Optional
A limit on the number of objects to be returned. Limit can range between 1 and 100,
and the default is 20.
limit integer
## Optional
Defaults to 20
Sort order by the created_at  timestamp of the objects. asc  for ascending order
and desc  for descending order.
order string
## Optional
Defaults to desc
Example request
python
from openai import OpenAI
client = OpenAI()
my_assistants = client.beta.assistants.list(
    order="desc",
    limit="20",
)
print(my_assistants.data)

## Response
 
{
  "object": "list",
  "data": [
    {
      "id": "asst_abc123",
      "object": "assistant",
      "created_at": 1698982736,
      "name": "Coding Tutor",
      "description": null,
      "model": "gpt-4o",
      "instructions": "You are a helpful assist
      "tools": [],
      "tool_resources": {},
      "metadata": {},
      "top_p": 1.0,
      "temperature": 1.0,
      "response_format": "auto"
    },
    {


<!-- Page 263 -->
GET https://api.openai.com/v1/assistants/{assistant_id}
Retrieves an assistant.
## Path parameters
Returns
A list of assistant objects.
 
      "id": "asst_abc456",
      "object": "assistant",
      "created_at": 1698982718,
      "name": "My Assistant",
      "description": null,
      "model": "gpt-4o",
      "instructions": "You are a helpful assist
      "tools": [],
      "tool_resources": {},
      "metadata": {},
      "top_p": 1.0,
      "temperature": 1.0,
      "response_format": "auto"
    },
    {
      "id": "asst_abc789",
      "object": "assistant",
      "created_at": 1698982643,
      "name": null,
      "description": null,
      "model": "gpt-4o",
      "instructions": null,
      "tools": [],
      "tool_resources": {},
      "metadata": {},
      "top_p": 1.0,
      "temperature": 1.0,
      "response_format": "auto"
    }
  ],
  "first_id": "asst_abc123",
  "last_id": "asst_abc789",
  "has_more": false
}

## Retrieve assistant
Beta
The ID of the assistant to retrieve.
assistant_id string
## Required
The assistant object matching the specified ID.
## Example request
python
 
 
from openai import OpenAI
client = OpenAI()
my_assistant = client.beta.assistants.retrieve("a
print(my_assistant)

## Response
 
 
{
  "id": "asst_abc123",
  "object": "assistant",
  "created_at": 1699009709,
  "name": "HR Helper",
  "description": null,
  "model": "gpt-4o",
  "instructions": "You are an HR bot, and you h
  "tools": [
    {
      "type": "file_search"
    }
  ],
  "metadata": {},
  "top_p": 1.0,
  "temperature": 1.0,


<!-- Page 264 -->
POST https://api.openai.com/v1/assistants/{assistant_id}
Modifies an assistant.
## Path parameters
Request body
 
Modify assistant
Beta
The ID of the assistant to modify.
assistant_id string
## Required
The description of the assistant. The maximum length is 512 characters.
description string or null
## Optional
The system instructions that the assistant uses. The maximum length is 256,000
characters.
instructions string or null
## Optional
Set of 16 key-value pairs that can be attached to an object. This can be useful for
storing additional information about the object in a structured format, and querying for objects via API or the dashboard.
Keys are strings with a maximum length of 64 characters. Values are strings with a
maximum length of 512 characters.
metadata
map
## Optional
model string
Optional
Example request
python
 
 
from openai import OpenAI
client = OpenAI()
my_updated_assistant = client.beta.assistants.up
  "asst_abc123",
  instructions="You are an HR bot, and you have 
  name="HR Helper",
  tools=[{"type": "file_search"}],
  model="gpt-4o"
)
print(my_updated_assistant)

## Response
 
{
  "id": "asst_123",
  "object": "assistant",
  "created_at": 1699009709,
  "name": "HR Helper",
  "description": null,
  "model": "gpt-4o",
  "instructions": "You are an HR bot, and you h
  "tools": [
    {
      "type": "file_search"
    }
  ],
  "tool_resources": {


<!-- Page 265 -->
ID of the model to use. You can use the List models API to see all of your available
models, or see our Model overview for descriptions of them.
The name of the assistant. The maximum length is 256 characters.
name string or null
## Optional
Constrains effort on reasoning for reasoning models. Currently supported values are
minimal , low , medium , and high . Reducing reasoning effort can result in faster
responses and fewer tokens used on reasoning in a response.
reasoning_effort string or null
## Optional
Defaults to medium
Specifies the format that the model must output. Compatible with GPT-4o,
GPT-4 Turbo, and all GPT-3.5 Turbo models since gpt-3.5-turbo-1106 .
Setting to { "type": "json_schema", "json_schema": {...} }  enables Structured
Outputs which ensures the model will match your supplied JSON schema. Learn more
in the Structured Outputs guide.
Setting to { "type": "json_object" }  enables JSON mode, which ensures the
message the model generates is valid JSON.
Important: when using JSON mode, you must also instruct the model to produce JSON
yourself via a system or user message. Without this, the model may generate an
unending stream of whitespace until the generation reaches the token limit, resulting in
a long-running and seemingly "stuck" request. Also note that the message content may
be partially cut off if finish_reason="length" , which indicates the generation
exceeded max_tokens  or the conversation exceeded the max context length.
## Show possible types
response_format
"auto" or object
## Optional
What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make
the output more random, while lower values like 0.2 will make it more focused and
deterministic.
temperature number or null
## Optional
Defaults to 1
tool_resources object or null
## Optional
 
    "file_search": {
      "vector_store_ids": []
    }
  },
  "metadata": {},
  "top_p": 1.0,
  "temperature": 1.0,
  "response_format": "auto"
}


<!-- Page 266 -->
## Returns
DELETE https://api.openai.com/v1/assistants/{assistant_id}
Delete an assistant.
## Path parameters
A set of resources that are used by the assistant's tools. The resources are specific to
the type of tool. For example, the code_interpreter  tool requires a list of file IDs,
while the file_search  tool requires a list of vector store IDs.
## Show properties
A list of tool enabled on the assistant. There can be a maximum of 128 tools per
assistant. Tools can be of types code_interpreter , file_search , or function .
## Show possible types
tools array
Optional
Defaults to []
An alternative to sampling with temperature, called nucleus sampling, where the model
considers the results of the tokens with top_p probability mass. So 0.1 means only the
tokens comprising the top 10% probability mass are considered.
We generally recommend altering this or temperature but not both.
top_p number or null
## Optional
Defaults to 1
The modified assistant object.
## Delete assistant
Beta
Example request
python
 
from openai import OpenAI
client = OpenAI()


<!-- Page 267 -->
## Returns
Represents an assistant  that can call the model and use tools.
The ID of the assistant to delete.
assistant_id string
## Required
Deletion status
 
response = client beta assistants delete("asst abc
## Response
{
  "id": "asst_abc123",
  "object": "assistant.deleted",
  "deleted": true
}

## The assistant object
Beta
The Unix timestamp (in seconds) for when the assistant was created.
created_at integer
The description of the assistant. The maximum length is 512 characters.
description string or null
The identifier, which can be referenced in API endpoints.
id string
The system instructions that the assistant uses. The maximum length is 256,000
characters.
instructions string or null
Set of 16 key-value pairs that can be attached to an object. This can be useful for
storing additional information about the object in a structured format, and querying for objects via API or the dashboard.
metadata
map
## OBJECT The assistant object
 
 
{
  "id": "asst_abc123",
  "object": "assistant",
  "created_at": 1698984975,
  "name": "Math Tutor",
  "description": null,
  "model": "gpt-4o",
  "instructions": "You are a personal math tutor
  "tools": [
    {
      "type": "code_interpreter"
    }
  ],
  "metadata": {},
  "top_p": 1.0,
  "temperature": 1.0,
  "response_format": "auto"
}


<!-- Page 268 -->
Keys are strings with a maximum length of 64 characters. Values are strings with a
maximum length of 512 characters.
ID of the model to use. You can use the List models API to see all of your available
models, or see our Model overview for descriptions of them.
model string
The name of the assistant. The maximum length is 256 characters.
name string or null
The object type, which is always assistant .
object string
Specifies the format that the model must output. Compatible with GPT-4o,
GPT-4 Turbo, and all GPT-3.5 Turbo models since gpt-3.5-turbo-1106 .
Setting to { "type": "json_schema", "json_schema": {...} }  enables Structured
Outputs which ensures the model will match your supplied JSON schema. Learn more
in the Structured Outputs guide.
Setting to { "type": "json_object" }  enables JSON mode, which ensures the
message the model generates is valid JSON.
Important: when using JSON mode, you must also instruct the model to produce JSON
yourself via a system or user message. Without this, the model may generate an
unending stream of whitespace until the generation reaches the token limit, resulting in
a long-running and seemingly "stuck" request. Also note that the message content may
be partially cut off if finish_reason="length" , which indicates the generation
exceeded max_tokens  or the conversation exceeded the max context length.
## Show possible types
response_format
"auto" or object
What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make
the output more random, while lower values like 0.2 will make it more focused and
deterministic.
temperature number or null

<!-- Page 269 -->
Create threads that assistants can interact with.
Related guide: Assistants
POST https://api.openai.com/v1/threads
A set of resources that are used by the assistant's tools. The resources are specific to
the type of tool. For example, the code_interpreter  tool requires a list of file IDs,
while the file_search  tool requires a list of vector store IDs.
## Show properties
tool_resources object or null
A list of tool enabled on the assistant. There can be a maximum of 128 tools per
assistant. Tools can be of types code_interpreter , file_search , or function .
## Show possible types
tools array
An alternative to sampling with temperature, called nucleus sampling, where the model
considers the results of the tokens with top_p probability mass. So 0.1 means only the
tokens comprising the top 10% probability mass are considered.
We generally recommend altering this or temperature but not both.
top_p number or null
## Threads
Beta
Create thread
Beta
Empty
Messages

<!-- Page 270 -->
Create a thread.
## Request body
Returns
GET https://api.openai.com/v1/threads/{thread_id}
A list of messages to start the thread with.
## Show properties
messages array
Optional
Set of 16 key-value pairs that can be attached to an object. This can be useful for
storing additional information about the object in a structured format, and querying for objects via API or the dashboard.
Keys are strings with a maximum length of 64 characters. Values are strings with a
maximum length of 512 characters.
metadata
map
## Optional
A set of resources that are made available to the assistant's tools in this thread. The
resources are specific to the type of tool. For example, the code_interpreter  tool
requires a list of file IDs, while the file_search  tool requires a list of vector store IDs.
## Show properties
tool_resources object or null
## Optional
A thread object.
## Example request
python
from openai import OpenAI
client = OpenAI()
empty_thread = client.beta.threads.create()
print(empty_thread)

## Response
{
  "id": "thread_abc123",
  "object": "thread",
  "created_at": 1699012949,
  "metadata": {},
  "tool_resources": {}
}

## Retrieve thread
Beta

<!-- Page 271 -->
Retrieves a thread.
## Path parameters
Returns
POST https://api.openai.com/v1/threads/{thread_id}
Modifies a thread.
## Path parameters
The ID of the thread to retrieve.
thread_id string
## Required
The thread object matching the specified ID.
## Example request
python
from openai import OpenAI
client = OpenAI()
my_thread = client.beta.threads.retrieve("thread_
print(my thread)

## Response
{
  "id": "thread_abc123",
  "object": "thread",
  "created_at": 1699014083,
  "metadata": {},
  "tool_resources": {
    "code_interpreter": {
      "file_ids": []
    }
  }
}

## Modify thread
Beta
The ID of the thread to modify. Only the metadata  can be modified.
thread_id string
## Required
Example request
python
 
from openai import OpenAI
client = OpenAI()
my_updated_thread = client.beta.threads.update(
  "thread_abc123",
  metadata={
    "modified": "true",
    "user": "abc123"
  }


<!-- Page 272 -->
## Request body
Returns
DELETE https://api.openai.com/v1/threads/{thread_id}
Delete a thread.
## Path parameters
Set of 16 key-value pairs that can be attached to an object. This can be useful for
storing additional information about the object in a structured format, and querying for objects via API or the dashboard.
Keys are strings with a maximum length of 64 characters. Values are strings with a
maximum length of 512 characters.
metadata
map
## Optional
A set of resources that are made available to the assistant's tools in this thread. The
resources are specific to the type of tool. For example, the code_interpreter  tool
requires a list of file IDs, while the file_search  tool requires a list of vector store IDs.
## Show properties
tool_resources object or null
## Optional
The modified thread object matching the specified ID.
 
)
print(my_updated_thread)

## Response
{
  "id": "thread_abc123",
  "object": "thread",
  "created_at": 1699014083,
  "metadata": {
    "modified": "true",
    "user": "abc123"
  },
  "tool_resources": {}
}

## Delete thread
Beta
thread_id string
## Required
Example request
python
 
 
from openai import OpenAI
client = OpenAI()
response = client.beta.threads.delete("thread_abc
print(response)


<!-- Page 273 -->
## Returns
Represents a thread that contains messages.
The ID of the thread to delete.
## Deletion status
Response
{
  "id": "thread_abc123",
  "object": "thread.deleted",
  "deleted": true
}

## The thread object
Beta
The Unix timestamp (in seconds) for when the thread was created.
created_at integer
The identifier, which can be referenced in API endpoints.
id string
Set of 16 key-value pairs that can be attached to an object. This can be useful for
storing additional information about the object in a structured format, and querying for objects via API or the dashboard.
Keys are strings with a maximum length of 64 characters. Values are strings with a
maximum length of 512 characters.
metadata
map
The object type, which is always thread .
object string
tool_resources object or null
## OBJECT The thread object
{
  "id": "thread_abc123",
  "object": "thread",
  "created_at": 1698107661,
  "metadata": {}
}


<!-- Page 274 -->
## Create messages within threads
Related guide: Assistants
POST https://api.openai.com/v1/threads/{thread_id}/messages
Create a message.
## Path parameters
Request body
A set of resources that are made available to the assistant's tools in this thread. The
resources are specific to the type of tool. For example, the code_interpreter  tool
requires a list of file IDs, while the file_search  tool requires a list of vector store IDs.
## Show properties
Messages
Beta
Create message
Beta
The ID of the thread to create a message for.
thread_id string
## Required
content string or array
Required
Example request
python
 
 
from openai import OpenAI
client = OpenAI()
thread_message = client.beta.threads.messages.cre
  "thread_abc123",
  role="user",
  content="How does AI work? Explain it in simple
)
print(thread_message)

## Response
 
{
  "id": "msg_abc123",


<!-- Page 275 -->
## Returns
GET https://api.openai.com/v1/threads/{thread_id}/messages
## Show possible types
The role of the entity that is creating the message. Allowed values include:
role string
## Required
user : Indicates the message is sent by an actual user and should be used in
most cases to represent user-generated messages.
assistant : Indicates the message is generated by the assistant. Use this value
to insert messages from the assistant into the conversation.
A list of files attached to the message, and the tools they should be added to.
## Show properties
attachments array or null
Optional
Set of 16 key-value pairs that can be attached to an object. This can be useful for
storing additional information about the object in a structured format, and querying for objects via API or the dashboard.
Keys are strings with a maximum length of 64 characters. Values are strings with a
maximum length of 512 characters.
metadata
map
## Optional
A message object.
 
  "object": "thread.message",
  "created_at": 1713226573,
  "assistant_id": null,
  "thread_id": "thread_abc123",
  "run_id": null,
  "role": "user",
  "content": [
    {
      "type": "text",
      "text": {
        "value": "How does AI work? Explain it 
        "annotations": []
      }
    }
  ],
  "attachments": [],
  "metadata": {}
}

## List messages
Beta
Example request
python

<!-- Page 276 -->
Returns a list of messages for a given thread.
## Path parameters
Query parameters
The ID of the thread the messages belong to.
thread_id string
## Required
A cursor for use in pagination. after  is an object ID that defines your place in the list.
For instance, if you make a list request and receive 100 objects, ending with obj_foo,
your subsequent call can include after=obj_foo in order to fetch the next page of the
list.
after string
## Optional
A cursor for use in pagination. before  is an object ID that defines your place in the
list. For instance, if you make a list request and receive 100 objects, starting with
obj_foo, your subsequent call can include before=obj_foo in order to fetch the previous
page of the list.
before string
## Optional
A limit on the number of objects to be returned. Limit can range between 1 and 100,
and the default is 20.
limit integer
## Optional
Defaults to 20
Sort order by the created_at  timestamp of the objects. asc  for ascending order
and desc  for descending order.
order string
## Optional
Defaults to desc
Filter messages by the run ID that generated them.
run_id string
## Optional
 
 
from openai import OpenAI
client = OpenAI()
thread_messages = client.beta.threads.messages.li
print(thread_messages.data)

## Response
 
{
  "object": "list",
  "data": [
    {
      "id": "msg_abc123",
      "object": "thread.message",
      "created_at": 1699016383,
      "assistant_id": null,
      "thread_id": "thread_abc123",
      "run_id": null,
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": {
            "value": "How does AI work? Explain
            "annotations": []
          }
        }
      ],
      "attachments": [],
      "metadata": {}
    },
    {
      "id": "msg_abc456",
      "object": "thread.message",
      "created_at": 1699016383,
      "assistant_id": null,
      "thread_id": "thread_abc123",
      "run_id": null,
      "role": "user",


<!-- Page 277 -->
## Returns
GET https://api.openai.com/v1/threads/{thread_id}/messages/{message_id}
Retrieve a message.
## Path parameters
Returns
A list of message objects.
 
      "content": [
        {
          "type": "text",
          "text": {
            "value": "Hello, what is AI?",
            "annotations": []
          }
        }
      ],
      "attachments": [],
      "metadata": {}
    }
  ],
  "first_id": "msg_abc123",
  "last_id": "msg_abc456",
  "has_more": false
}

## Retrieve message
Beta
The ID of the message to retrieve.
message_id string
## Required
The ID of the thread to which this message belongs.
thread_id string
## Required
The message object matching the specified ID.
## Example request
python
 
 
from openai import OpenAI
client = OpenAI()
message = client.beta.threads.messages.retrieve(
  message_id="msg_abc123",
  thread_id="thread_abc123",
)
print(message)

## Response
 
{
  "id": "msg_abc123",
  "object": "thread.message",
  "created_at": 1699017614,
  "assistant_id": null,
  "thread_id": "thread_abc123",
  "run_id": null,
  "role": "user",
  "content": [
    {
      "type": "text",
      "text": {
        "value": "How does AI work? Explain it 


<!-- Page 278 -->
POST https://api.openai.com/v1/threads/{thread_id}/messages/{message_id}
Modifies a message.
## Path parameters
Request body
 
        "annotations": []
      }
    }
  ],
  "attachments": [],
  "metadata": {}
}

## Modify message
Beta
The ID of the message to modify.
message_id string
## Required
The ID of the thread to which this message belongs.
thread_id string
## Required
Set of 16 key-value pairs that can be attached to an object. This can be useful for
storing additional information about the object in a structured format, and querying for objects via API or the dashboard.
Keys are strings with a maximum length of 64 characters. Values are strings with a
maximum length of 512 characters.
metadata
map
## Optional
Example request
python
 
 
from openai import OpenAI
client = OpenAI()
message = client.beta.threads.messages.update(
  message_id="msg_abc12",
  thread_id="thread_abc123",
  metadata={
    "modified": "true",
    "user": "abc123",
  },
)
print(message)

## Response
 
{
  "id": "msg_abc123",
  "object": "thread.message",
  "created_at": 1699017614,
  "assistant_id": null,
  "thread_id": "thread_abc123",
  "run_id": null,


<!-- Page 279 -->
## Returns
DELETE https://api.openai.com/v1/threads/{thread_id}/messages/{message_id}
Deletes a message.
## Path parameters
Returns
The modified message object.
 
  "role": "user",
  "content": [
    {
      "type": "text",
      "text": {
        "value": "How does AI work? Explain it 
        "annotations": []
      }
    }
  ],
  "file_ids": [],
  "metadata": {
    "modified": "true",
    "user": "abc123"
  }
}

## Delete message
Beta
The ID of the message to delete.
message_id string
## Required
The ID of the thread to which this message belongs.
thread_id string
## Required
Deletion status
Example request
python
 
 
from openai import OpenAI
client = OpenAI()
deleted_message = client.beta.threads.messages.de
  message_id="msg_abc12",
  thread_id="thread_abc123",
)
print(deleted_message)

## Response
{
  "id": "msg_abc123",
  "object": "thread.message.deleted",
  "deleted": true
}


<!-- Page 280 -->
Represents a message within a thread.
## The message object
Beta
If applicable, the ID of the assistant that authored this message.
assistant_id string or null
A list of files attached to the message, and the tools they were added to.
## Show properties
attachments array or null
The Unix timestamp (in seconds) for when the message was completed.
completed_at integer or null
The content of the message in array of text and/or images.
## Show possible types
content array
The Unix timestamp (in seconds) for when the message was created.
created_at integer
The identifier, which can be referenced in API endpoints.
id string
The Unix timestamp (in seconds) for when the message was marked as incomplete.
incomplete_at integer or null
On an incomplete message, details about why the message is incomplete.
## Show properties
incomplete_details object or null
metadata
map
## OBJECT The message object
 
 
{
  "id": "msg_abc123",
  "object": "thread.message",
  "created_at": 1698983503,
  "thread_id": "thread_abc123",
  "role": "assistant",
  "content": [
    {
      "type": "text",
      "text": {
        "value": "Hi! How can I help you today?"
        "annotations": []
      }
    }
  ],
  "assistant_id": "asst_abc123",
  "run_id": "run_abc123",
  "attachments": [],
  "metadata": {}
}


<!-- Page 281 -->
Represents an execution run on a thread.
Related guide: Assistants
Set of 16 key-value pairs that can be attached to an object. This can be useful for
storing additional information about the object in a structured format, and querying for objects via API or the dashboard.
Keys are strings with a maximum length of 64 characters. Values are strings with a
maximum length of 512 characters.
The object type, which is always thread.message .
object string
The entity that produced the message. One of user  or assistant .
role string
The ID of the run associated with the creation of this message. Value is null  when
messages are created manually using the create message or create thread endpoints.
run_id string or null
The status of the message, which can be either in_progress , incomplete , or
completed .
status string
The thread ID that this message belongs to.
thread_id string
## Runs
Beta

<!-- Page 282 -->
POST https://api.openai.com/v1/threads/{thread_id}/runs
Create a run.
## Path parameters
Query parameters
Request body
Create run
Beta
The ID of the thread to run.
thread_id string
## Required
A list of additional fields to include in the response. Currently the only supported value
is step_details.tool_calls[*].file_search.results[*].content  to fetch the file
search result content.
See the file search tool documentation for more information.
include[]
array
## Optional
The ID of the assistant to use to execute this run.
assistant_id string
## Required
Appends additional instructions at the end of the instructions for the run. This is useful
for modifying the behavior on a per-run basis without overriding other instructions.
additional_instructions string or null
## Optional
Adds additional messages to the thread before creating the run.
additional_messages array or null
## Optional
Default
Streaming
Streaming with Functions
Example request
python
from openai import OpenAI
client = OpenAI()
run = client.beta.threads.runs.create(
  thread_id="thread_abc123",
  assistant_id="asst_abc123"
)
print(run)

## Response
 
{
  "id": "run_abc123",
  "object": "thread.run",
  "created_at": 1699063290,
  "assistant_id": "asst_abc123",
  "thread_id": "thread_abc123",
  "status": "queued",
  "started_at": 1699063290,
  "expires_at": null,
  "cancelled_at": null,
  "failed_at": null,
  "completed_at": 1699063291,
  "last_error": null,
  "model": "gpt-4o",
  "instructions": null,
  "incomplete_details": null,
  "tools": [
    {
      "type": "code_interpreter"
    }


<!-- Page 283 -->
## Show properties
Overrides the instructions of the assistant. This is useful for modifying the behavior on
a per-run basis.
instructions string or null
## Optional
The maximum number of completion tokens that may be used over the course of the
run. The run will make a best effort to use only the number of completion tokens
specified, across multiple turns of the run. If the run exceeds the number of completion
tokens specified, the run will end with status incomplete . See incomplete_details
for more info.
max_completion_tokens integer or null
## Optional
The maximum number of prompt tokens that may be used over the course of the run.
The run will make a best effort to use only the number of prompt tokens specified,
across multiple turns of the run. If the run exceeds the number of prompt tokens
specified, the run will end with status incomplete . See incomplete_details  for
more info.
max_prompt_tokens integer or null
## Optional
Set of 16 key-value pairs that can be attached to an object. This can be useful for
storing additional information about the object in a structured format, and querying for objects via API or the dashboard.
Keys are strings with a maximum length of 64 characters. Values are strings with a
maximum length of 512 characters.
metadata
map
## Optional
The ID of the Model to be used to execute this run. If a value is provided here, it will
override the model associated with the assistant. If not, the model associated with the
assistant will be used.
model string
## Optional
Whether to enable parallel function calling during tool use.
parallel_tool_calls boolean
## Optional
Defaults to true
 
  ],
  "metadata": {},
  "usage": null,
  "temperature": 1.0,
  "top_p": 1.0,
  "max_prompt_tokens": 1000,
  "max_completion_tokens": 1000,
  "truncation_strategy": {
    "type": "auto",
    "last_messages": null
  },
  "response_format": "auto",
  "tool_choice": "auto",
  "parallel_tool_calls": true
}


<!-- Page 284 -->
Constrains effort on reasoning for reasoning models. Currently supported values are
minimal , low , medium , and high . Reducing reasoning effort can result in faster
responses and fewer tokens used on reasoning in a response.
reasoning_effort string or null
## Optional
Defaults to medium
Specifies the format that the model must output. Compatible with GPT-4o,
GPT-4 Turbo, and all GPT-3.5 Turbo models since gpt-3.5-turbo-1106 .
Setting to { "type": "json_schema", "json_schema": {...} }  enables Structured
Outputs which ensures the model will match your supplied JSON schema. Learn more
in the Structured Outputs guide.
Setting to { "type": "json_object" }  enables JSON mode, which ensures the
message the model generates is valid JSON.
Important: when using JSON mode, you must also instruct the model to produce JSON
yourself via a system or user message. Without this, the model may generate an
unending stream of whitespace until the generation reaches the token limit, resulting in
a long-running and seemingly "stuck" request. Also note that the message content may
be partially cut off if finish_reason="length" , which indicates the generation
exceeded max_tokens  or the conversation exceeded the max context length.
## Show possible types
response_format
"auto" or object
## Optional
If true , returns a stream of events that happen during the Run as server-sent events,
terminating when the Run enters a terminal state with a data: [DONE]  message.
stream boolean or null
## Optional
What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make
the output more random, while lower values like 0.2 will make it more focused and
deterministic.
temperature number or null
## Optional
Defaults to 1
Controls which (if any) tool is called by the model. none  means the model will not call
any tools and instead generates a message. auto  is the default value and means the
tool_choice string or object
## Optional

<!-- Page 285 -->
## Returns
model can pick between generating a message or calling one or more tools.
required  means the model must call one or more tools before responding to the
user. Specifying a particular tool like {"type": "file_search"}  or
{"type": "function", "function": {"name": "my_function"}}  forces the model to
call that tool.
## Show possible types
Override the tools the assistant can use for this run. This is useful for modifying the
behavior on a per-run basis.
## Show possible types
tools array or null
Optional
An alternative to sampling with temperature, called nucleus sampling, where the model
considers the results of the tokens with top_p probability mass. So 0.1 means only the
tokens comprising the top 10% probability mass are considered.
We generally recommend altering this or temperature but not both.
top_p number or null
## Optional
Defaults to 1
Controls for how a thread will be truncated prior to the run. Use this to control the intial
context window of the run.
## Show properties
truncation_strategy object or null
## Optional
A run object.
## Create thread and run
Beta

<!-- Page 286 -->
POST https://api.openai.com/v1/threads/runs
Create a thread and run it in one request.
## Request body
The ID of the assistant to use to execute this run.
assistant_id string
## Required
Override the default system message of the assistant. This is useful for modifying the
behavior on a per-run basis.
instructions string or null
## Optional
The maximum number of completion tokens that may be used over the course of the
run. The run will make a best effort to use only the number of completion tokens
specified, across multiple turns of the run. If the run exceeds the number of completion
tokens specified, the run will end with status incomplete . See incomplete_details
for more info.
max_completion_tokens integer or null
## Optional
The maximum number of prompt tokens that may be used over the course of the run.
The run will make a best effort to use only the number of prompt tokens specified,
across multiple turns of the run. If the run exceeds the number of prompt tokens
specified, the run will end with status incomplete . See incomplete_details  for
more info.
max_prompt_tokens integer or null
## Optional
Set of 16 key-value pairs that can be attached to an object. This can be useful for
storing additional information about the object in a structured format, and querying for objects via API or the dashboard.
Keys are strings with a maximum length of 64 characters. Values are strings with a
maximum length of 512 characters.
metadata
map
## Optional
Default
Streaming
Streaming with Functions
Example request
python
 
 
from openai import OpenAI
client = OpenAI()
run = client.beta.threads.create_and_run(
  assistant_id="asst_abc123",
  thread={
    "messages": [
      {"role": "user", "content": "Explain deep 
    ]
  }
)
print(run)

## Response
 
{
  "id": "run_abc123",
  "object": "thread.run",
  "created_at": 1699076792,
  "assistant_id": "asst_abc123",
  "thread_id": "thread_abc123",
  "status": "queued",
  "started_at": null,
  "expires_at": 1699077392,
  "cancelled_at": null,
  "failed_at": null,
  "completed_at": null,
  "required_action": null,
  "last_error": null,
  "model": "gpt-4o",
  "instructions": "You are a helpful assistant.
  "tools": [],
  "tool_resources": {},


<!-- Page 287 -->
The ID of the Model to be used to execute this run. If a value is provided here, it will
override the model associated with the assistant. If not, the model associated with the
assistant will be used.
model string
## Optional
Whether to enable parallel function calling during tool use.
parallel_tool_calls boolean
## Optional
Defaults to true
Specifies the format that the model must output. Compatible with GPT-4o,
GPT-4 Turbo, and all GPT-3.5 Turbo models since gpt-3.5-turbo-1106 .
Setting to { "type": "json_schema", "json_schema": {...} }  enables Structured
Outputs which ensures the model will match your supplied JSON schema. Learn more
in the Structured Outputs guide.
Setting to { "type": "json_object" }  enables JSON mode, which ensures the
message the model generates is valid JSON.
Important: when using JSON mode, you must also instruct the model to produce JSON
yourself via a system or user message. Without this, the model may generate an
unending stream of whitespace until the generation reaches the token limit, resulting in
a long-running and seemingly "stuck" request. Also note that the message content may
be partially cut off if finish_reason="length" , which indicates the generation
exceeded max_tokens  or the conversation exceeded the max context length.
## Show possible types
response_format
"auto" or object
## Optional
If true , returns a stream of events that happen during the Run as server-sent events,
terminating when the Run enters a terminal state with a data: [DONE]  message.
stream boolean or null
## Optional
What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make
the output more random, while lower values like 0.2 will make it more focused and
deterministic.
temperature number or null
## Optional
Defaults to 1
 
  "metadata": {},
  "temperature": 1.0,
  "top_p": 1.0,
  "max_completion_tokens": null,
  "max_prompt_tokens": null,
  "truncation_strategy": {
    "type": "auto",
    "last_messages": null
  },
  "incomplete_details": null,
  "usage": null,
  "response_format": "auto",
  "tool_choice": "auto",
  "parallel_tool_calls": true
}


<!-- Page 288 -->
Options to create a new thread. If no thread is provided when running a request, an
empty thread will be created.
## Show properties
thread object
Optional
Controls which (if any) tool is called by the model. none  means the model will not call
any tools and instead generates a message. auto  is the default value and means the
model can pick between generating a message or calling one or more tools.
required  means the model must call one or more tools before responding to the
user. Specifying a particular tool like {"type": "file_search"}  or
{"type": "function", "function": {"name": "my_function"}}  forces the model to
call that tool.
## Show possible types
tool_choice string or object
## Optional
A set of resources that are used by the assistant's tools. The resources are specific to
the type of tool. For example, the code_interpreter  tool requires a list of file IDs,
while the file_search  tool requires a list of vector store IDs.
## Show properties
tool_resources object or null
## Optional
Override the tools the assistant can use for this run. This is useful for modifying the
behavior on a per-run basis.
## Show possible types
tools array or null
Optional
An alternative to sampling with temperature, called nucleus sampling, where the model
considers the results of the tokens with top_p probability mass. So 0.1 means only the
tokens comprising the top 10% probability mass are considered.
We generally recommend altering this or temperature but not both.
top_p number or null
## Optional
Defaults to 1
truncation_strategy object or null
## Optional

<!-- Page 289 -->
## Returns
GET https://api.openai.com/v1/threads/{thread_id}/runs
Returns a list of runs belonging to a thread.
## Path parameters
Query parameters
Controls for how a thread will be truncated prior to the run. Use this to control the intial
context window of the run.
## Show properties
A run object.
## List runs
Beta
The ID of the thread the run belongs to.
thread_id string
## Required
A cursor for use in pagination. after  is an object ID that defines your place in the list.
For instance, if you make a list request and receive 100 objects, ending with obj_foo,
your subsequent call can include after=obj_foo in order to fetch the next page of the
list.
after string
## Optional
Example request
python
from openai import OpenAI
client = OpenAI()
runs = client.beta.threads.runs.list(
  "thread_abc123"
)
print(runs)

## Response
 
{
  "object": "list",
  "data": [
    {
      "id": "run_abc123",
      "object": "thread.run",
      "created_at": 1699075072,
      "assistant_id": "asst_abc123",


<!-- Page 290 -->
## Returns
GET https://api.openai.com/v1/threads/{thread_id}/runs/{run_id}
Retrieves a run.
## Path parameters
A cursor for use in pagination. before  is an object ID that defines your place in the
list. For instance, if you make a list request and receive 100 objects, starting with
obj_foo, your subsequent call can include before=obj_foo in order to fetch the previous
page of the list.
before string
## Optional
A limit on the number of objects to be returned. Limit can range between 1 and 100,
and the default is 20.
limit integer
## Optional
Defaults to 20
Sort order by the created_at  timestamp of the objects. asc  for ascending order
and desc  for descending order.
order string
## Optional
Defaults to desc
A list of run objects.
 
      "thread_id": "thread_abc123",
      "status": "completed",
      "started_at": 1699075072,
      "expires_at": null,
      "cancelled_at": null,
      "failed_at": null,
      "completed_at": 1699075073,
      "last_error": null,
      "model": "gpt-4o",
      "instructions": null,
      "incomplete_details": null,
      "tools": [
        {
          "type": "code_interpreter"
        }
      ],
      "tool_resources": {
        "code_interpreter": {
          "file_ids": [
            "file-abc123",
            "file-abc456"
          ]
        }
      },
      "metadata": {},
      "usage": {
        "prompt_tokens": 123,
        "completion_tokens": 456,
        "total_tokens": 579
      },
      "temperature": 1.0,
      "top_p": 1.0,
      "max_prompt_tokens": 1000,
      "max_completion_tokens": 1000,
      "truncation_strategy": {
        "type": "auto",
        "last_messages": null
      },
      "response_format": "auto",
      "tool_choice": "auto",

## Retrieve run
Beta
run_id string
## Required
Example request
python
 
from openai import OpenAI
client = OpenAI()
run = client.beta.threads.runs.retrieve(
  thread_id="thread_abc123",
  run_id="run_abc123"
)


<!-- Page 291 -->
## Returns
POST https://api.openai.com/v1/threads/{thread_id}/runs/{run_id}
Modifies a run.
## Path parameters
 
      "parallel_tool_calls": true
    },
    {
      "id": "run_abc456",
      "object": "thread.run",
      "created_at": 1699063290,
      "assistant_id": "asst_abc123",
      "thread_id": "thread_abc123",
      "status": "completed",
      "started_at": 1699063290,
      "expires_at": null,
      "cancelled_at": null,
      "failed_at": null,
      "completed_at": 1699063291,
      "last_error": null,
      "model": "gpt-4o",
      "instructions": null,
      "incomplete_details": null,
      "tools": [
        {
          "type": "code_interpreter"
        }
      ],
      "tool_resources": {
        "code_interpreter": {
          "file_ids": [
            "file-abc123",
            "file-abc456"
          ]
        }
      },
      "metadata": {},
      "usage": {
        "prompt_tokens": 123,
        "completion_tokens": 456,
        "total_tokens": 579
      },
      "temperature": 1.0,
      "top_p": 1.0,
      "max_prompt_tokens": 1000,

The ID of the run to retrieve.
The ID of the thread that was run.
thread_id string
## Required
The run object matching the specified ID.
 
print(run)

## Response
 
{
  "id": "run_abc123",
  "object": "thread.run",
  "created_at": 1699075072,
  "assistant_id": "asst_abc123",
  "thread_id": "thread_abc123",
  "status": "completed",
  "started_at": 1699075072,
  "expires_at": null,
  "cancelled_at": null,
  "failed_at": null,
  "completed_at": 1699075073,
  "last_error": null,
  "model": "gpt-4o",
  "instructions": null,
  "incomplete_details": null,
  "tools": [
    {
      "type": "code_interpreter"
    }
  ],
  "metadata": {},
  "usage": {
    "prompt_tokens": 123,
    "completion_tokens": 456,
    "total_tokens": 579
  },
  "temperature": 1.0,
  "top_p": 1.0,
  "max_prompt_tokens": 1000,
  "max_completion_tokens": 1000,
  "truncation_strategy": {
    "type": "auto",
    "last_messages": null
  },
  "response_format": "auto",

## Modify run
Beta
Example request
python
 
from openai import OpenAI
client = OpenAI()
run = client.beta.threads.runs.update(
  thread_id="thread_abc123",


<!-- Page 292 -->
## Request body
Returns
POST https://api.openai.com/v1/threads/{thread_id}/runs/{run_id}/submit_to
ol_outputs
 
      "max_completion_tokens": 1000,
      "truncation_strategy": {
        "type": "auto",
        "last_messages": null
      },
      "response_format": "auto",
      "tool_choice": "auto",
      "parallel_tool_calls": true
    }
  ],
  "first_id": "run_abc123",
  "last_id": "run_abc456",
"has more": false

  "tool_choice": "auto",
  "parallel_tool_calls": true
}

The ID of the run to modify.
run_id string
## Required
The ID of the thread that was run.
thread_id string
## Required
Set of 16 key-value pairs that can be attached to an object. This can be useful for
storing additional information about the object in a structured format, and querying for objects via API or the dashboard.
Keys are strings with a maximum length of 64 characters. Values are strings with a
maximum length of 512 characters.
metadata
map
## Optional
The modified run object matching the specified ID.
 
  run_id="run_abc123",
  metadata={"user_id": "user_abc123"},
)
print(run)

## Response
 
{
  "id": "run_abc123",
  "object": "thread.run",
  "created_at": 1699075072,
  "assistant_id": "asst_abc123",
  "thread_id": "thread_abc123",
  "status": "completed",
  "started_at": 1699075072,
  "expires_at": null,
  "cancelled_at": null,
  "failed_at": null,
  "completed_at": 1699075073,
  "last_error": null,
  "model": "gpt-4o",
  "instructions": null,
  "incomplete_details": null,
  "tools": [
    {
      "type": "code_interpreter"
    }
  ],
  "tool_resources": {
    "code_interpreter": {
      "file_ids": [
        "file-abc123",
        "file-abc456"
      ]
    }
  },
  "metadata": {
    "user_id": "user_abc123"
  },

## Submit tool outputs to run
Beta
Default
Streaming
Example request
python

<!-- Page 293 -->
When a run has the status: "requires_action"  and required_action.type  is
submit_tool_outputs , this endpoint can be used to submit the outputs from
the tool calls once they're all completed. All outputs must be submitted in a
single request.
## Path parameters
Request body
Returns
 
  "usage": {
    "prompt_tokens": 123,
    "completion_tokens": 456,
    "total_tokens": 579
  },
  "temperature": 1.0,
  "top_p": 1.0,
  "max_prompt_tokens": 1000,
  "max_completion_tokens": 1000,
  "truncation_strategy": {
    "type": "auto",
    "last_messages": null
  },
  "response_format": "auto",
  "tool_choice": "auto",
  "parallel_tool_calls": true
}

The ID of the run that requires the tool output submission.
run_id string
## Required
The ID of the thread to which this run belongs.
thread_id string
## Required
A list of tools for which the outputs are being submitted.
## Show properties
tool_outputs array
## Required
If true , returns a stream of events that happen during the Run as server-sent events,
terminating when the Run enters a terminal state with a data: [DONE]  message.
stream boolean or null
## Optional
The modified run object matching the specified ID.
from openai import OpenAI
client = OpenAI()
run = client.beta.threads.runs.submit_tool_outpu
  thread_id="thread_123",
  run_id="run_123",
  tool_outputs=[
    {
      "tool_call_id": "call_001",
      "output": "70 degrees and sunny."
    }
  ]
)
print(run)

## Response
 
{
  "id": "run_123",
  "object": "thread.run",
  "created_at": 1699075592,
  "assistant_id": "asst_123",
  "thread_id": "thread_123",
  "status": "queued",
  "started_at": 1699075592,
  "expires_at": 1699076192,
  "cancelled_at": null,
  "failed_at": null,
  "completed_at": null,
  "last_error": null,
  "model": "gpt-4o",
  "instructions": null,
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_current_weather",
        "description": "Get the current weather
        "parameters": {


<!-- Page 294 -->
POST https://api.openai.com/v1/threads/{thread_id}/runs/{run_id}/cancel
Cancels a run that is in_progress .
## Path parameters
Returns
 
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "The city and stat
            },
            "unit": {
              "type": "string",
              "enum": ["celsius", "fahrenheit"]
            }
          },
          "required": ["location"]
        }
      }
    }
  ],
  "metadata": {},
  "usage": null,
  "temperature": 1.0,
  "top_p": 1.0,
  "max_prompt_tokens": 1000,
  "max_completion_tokens": 1000,
  "truncation_strategy": {
    "type": "auto",
    "last_messages": null
  },
  "response_format": "auto",
  "tool_choice": "auto",
  "parallel_tool_calls": true
}

## Cancel a run
Beta
The ID of the run to cancel.
run_id string
## Required
The ID of the thread to which this run belongs.
thread_id string
## Required
The modified run object matching the specified ID.
## Example request
python
from openai import OpenAI
client = OpenAI()
run = client.beta.threads.runs.cancel(
  thread_id="thread_abc123",
  run_id="run_abc123"
)
print(run)

## Response
 
{
  "id": "run_abc123",
  "object": "thread.run",
  "created_at": 1699076126,
  "assistant_id": "asst_abc123",
  "thread_id": "thread_abc123",
  "status": "cancelling",
  "started_at": 1699076126,
  "expires_at": 1699076726,
  "cancelled_at": null,
  "failed_at": null,
  "completed_at": null,
  "last_error": null,
  "model": "gpt-4o",
  "instructions": "You summarize books.",
  "tools": [
    {
      "type": "file_search"
    }
  ],
  "tool_resources": {
    "file_search": {


<!-- Page 295 -->
Represents an execution run on a thread.
 
      "vector_store_ids": ["vs_123"]
    }
  },
  "metadata": {},
  "usage": null,
  "temperature": 1.0,
  "top_p": 1.0,
  "response_format": "auto",
  "tool_choice": "auto",
  "parallel_tool_calls": true
}

## The run object
Beta
The ID of the assistant used for execution of this run.
assistant_id string
The Unix timestamp (in seconds) for when the run was cancelled.
cancelled_at integer or null
The Unix timestamp (in seconds) for when the run was completed.
completed_at integer or null
The Unix timestamp (in seconds) for when the run was created.
created_at integer
The Unix timestamp (in seconds) for when the run will expire.
expires_at integer or null
The Unix timestamp (in seconds) for when the run failed.
failed_at integer or null
The identifier, which can be referenced in API endpoints.
id string
Details on why the run is incomplete. Will be null  if the run is not incomplete.
## Show properties
incomplete_details object or null
instructions string
## OBJECT The run object
 
{
  "id": "run_abc123",
  "object": "thread.run",
  "created_at": 1698107661,
  "assistant_id": "asst_abc123",
  "thread_id": "thread_abc123",
  "status": "completed",
  "started_at": 1699073476,
  "expires_at": null,
  "cancelled_at": null,
  "failed_at": null,
  "completed_at": 1699073498,
  "last_error": null,
  "model": "gpt-4o",
  "instructions": null,
  "tools": [{"type": "file_search"}, {"type": "
  "metadata": {},
  "incomplete_details": null,
  "usage": {
    "prompt_tokens": 123,
    "completion_tokens": 456,
    "total_tokens": 579
  },
  "temperature": 1.0,
  "top_p": 1.0,
  "max_prompt_tokens": 1000,
  "max_completion_tokens": 1000,
  "truncation_strategy": {
    "type": "auto",
    "last_messages": null
  },


<!-- Page 296 -->
The instructions that the assistant used for this run.
The last error associated with this run. Will be null  if there are no errors.
## Show properties
last_error object or null
## The maximum number of completion tokens specified to have been used over the
course of the run.
max_completion_tokens integer or null
## The maximum number of prompt tokens specified to have been used over the course
of the run.
max_prompt_tokens integer or null
Set of 16 key-value pairs that can be attached to an object. This can be useful for
storing additional information about the object in a structured format, and querying for objects via API or the dashboard.
Keys are strings with a maximum length of 64 characters. Values are strings with a
maximum length of 512 characters.
metadata
map
The model that the assistant used for this run.
model string
The object type, which is always thread.run .
object string
Whether to enable parallel function calling during tool use.
parallel_tool_calls boolean
Details on the action required to continue the run. Will be null  if no action is
required.
required_action object or null
 
  "response_format": "auto",
  "tool_choice": "auto",
  "parallel_tool_calls": true
}


<!-- Page 297 -->
## Show properties
Specifies the format that the model must output. Compatible with GPT-4o,
GPT-4 Turbo, and all GPT-3.5 Turbo models since gpt-3.5-turbo-1106 .
Setting to { "type": "json_schema", "json_schema": {...} }  enables Structured
Outputs which ensures the model will match your supplied JSON schema. Learn more
in the Structured Outputs guide.
Setting to { "type": "json_object" }  enables JSON mode, which ensures the
message the model generates is valid JSON.
Important: when using JSON mode, you must also instruct the model to produce JSON
yourself via a system or user message. Without this, the model may generate an
unending stream of whitespace until the generation reaches the token limit, resulting in
a long-running and seemingly "stuck" request. Also note that the message content may
be partially cut off if finish_reason="length" , which indicates the generation
exceeded max_tokens  or the conversation exceeded the max context length.
## Show possible types
response_format
"auto" or object
The Unix timestamp (in seconds) for when the run was started.
started_at integer or null
The status of the run, which can be either queued , in_progress ,
requires_action , cancelling , cancelled , failed , completed , incomplete
, or expired .
status string
The sampling temperature used for this run. If not set, defaults to 1.
temperature number or null
The ID of the thread that was executed on as a part of this run.
thread_id string
tool_choice string or object

<!-- Page 298 -->
Represents the steps (model and tool calls) taken during the run.
Controls which (if any) tool is called by the model. none  means the model will not call
any tools and instead generates a message. auto  is the default value and means the
model can pick between generating a message or calling one or more tools.
required  means the model must call one or more tools before responding to the
user. Specifying a particular tool like {"type": "file_search"}  or
{"type": "function", "function": {"name": "my_function"}}  forces the model to
call that tool.
## Show possible types
The list of tools that the assistant used for this run.
## Show possible types
tools array
The nucleus sampling value used for this run. If not set, defaults to 1.
top_p number or null
Controls for how a thread will be truncated prior to the run. Use this to control the intial
context window of the run.
## Show properties
truncation_strategy object or null
Usage statistics related to the run. This value will be null  if the run is not in a
terminal state (i.e. in_progress , queued , etc.).
## Show properties
usage object or null
Run steps
Beta

<!-- Page 299 -->
Related guide: Assistants
GET https://api.openai.com/v1/threads/{thread_id}/runs/{run_id}/steps
Returns a list of run steps belonging to a run.
## Path parameters
Query parameters
List run steps
Beta
The ID of the run the run steps belong to.
run_id string
## Required
The ID of the thread the run and run steps belong to.
thread_id string
## Required
A cursor for use in pagination. after  is an object ID that defines your place in the list.
For instance, if you make a list request and receive 100 objects, ending with obj_foo,
your subsequent call can include after=obj_foo in order to fetch the next page of the
list.
after string
## Optional
A cursor for use in pagination. before  is an object ID that defines your place in the
list. For instance, if you make a list request and receive 100 objects, starting with
obj_foo, your subsequent call can include before=obj_foo in order to fetch the previous
page of the list.
before string
## Optional
Example request
python
 
 
from openai import OpenAI
client = OpenAI()
run_steps = client.beta.threads.runs.steps.list(
    thread_id="thread_abc123",
    run_id="run_abc123"
)
print(run_steps)

## Response
 
{
  "object": "list",
  "data": [
    {
      "id": "step_abc123",
      "object": "thread.run.step",
      "created_at": 1699063291,
      "run_id": "run_abc123",
      "assistant_id": "asst_abc123",
      "thread_id": "thread_abc123",
      "type": "message_creation",
      "status": "completed",
      "cancelled_at": null,
      "completed_at": 1699063291,
      "expired_at": null,
      "failed_at": null,


<!-- Page 300 -->
## Returns
GET https://api.openai.com/v1/threads/{thread_id}/runs/{run_id}/steps/{ste
p_id}
Retrieves a run step.
## Path parameters
A list of additional fields to include in the response. Currently the only supported value
is step_details.tool_calls[*].file_search.results[*].content  to fetch the file
search result content.
See the file search tool documentation for more information.
include[]
array
## Optional
A limit on the number of objects to be returned. Limit can range between 1 and 100,
and the default is 20.
limit integer
## Optional
Defaults to 20
Sort order by the created_at  timestamp of the objects. asc  for ascending order
and desc  for descending order.
order string
## Optional
Defaults to desc
A list of run step objects.
 
      "last_error": null,
      "step_details": {
        "type": "message_creation",
        "message_creation": {
          "message_id": "msg_abc123"
        }
      },
      "usage": {
        "prompt_tokens": 123,
        "completion_tokens": 456,
        "total_tokens": 579
      }
    }
  ],
  "first_id": "step_abc123",
  "last_id": "step_abc456",
  "has_more": false
}

## Retrieve run step
Beta
run_id string
## Required
Example request
python
 
from openai import OpenAI
client = OpenAI()
run_step = client.beta.threads.runs.steps.retri
    thread_id="thread_abc123",
    run_id="run_abc123",
    step_id="step_abc123"


<!-- Page 301 -->
## Query parameters
Returns
Represents a step in execution of a run.
The ID of the run to which the run step belongs.
The ID of the run step to retrieve.
step_id string
## Required
The ID of the thread to which the run and run step belongs.
thread_id string
## Required
A list of additional fields to include in the response. Currently the only supported value
is step_details.tool_calls[*].file_search.results[*].content  to fetch the file
search result content.
See the file search tool documentation for more information.
include[]
array
## Optional
The run step object matching the specified ID.
 
)
print(run_step)

## Response
{
  "id": "step_abc123",
  "object": "thread.run.step",
  "created_at": 1699063291,
  "run_id": "run_abc123",
  "assistant_id": "asst_abc123",
  "thread_id": "thread_abc123",
  "type": "message_creation",
  "status": "completed",
  "cancelled_at": null,
  "completed_at": 1699063291,
  "expired_at": null,
  "failed_at": null,
  "last_error": null,
  "step_details": {
    "type": "message_creation",
    "message_creation": {
      "message_id": "msg_abc123"
    }
  },
  "usage": {
    "prompt_tokens": 123,
    "completion_tokens": 456,
    "total_tokens": 579
  }
}

## The run step object
Beta
The ID of the assistant associated with the run step.
assistant_id string
## OBJECT The run step object
 
{
  "id": "step_abc123",
  "object": "thread.run.step",
  "created_at": 1699063291,


<!-- Page 302 -->
The Unix timestamp (in seconds) for when the run step was cancelled.
cancelled_at integer or null
The Unix timestamp (in seconds) for when the run step completed.
completed_at integer or null
The Unix timestamp (in seconds) for when the run step was created.
created_at integer
The Unix timestamp (in seconds) for when the run step expired. A step is considered
expired if the parent run is expired.
expired_at integer or null
The Unix timestamp (in seconds) for when the run step failed.
failed_at integer or null
The identifier of the run step, which can be referenced in API endpoints.
id string
The last error associated with this run step. Will be null  if there are no errors.
## Show properties
last_error object or null
Set of 16 key-value pairs that can be attached to an object. This can be useful for
storing additional information about the object in a structured format, and querying for objects via API or the dashboard.
Keys are strings with a maximum length of 64 characters. Values are strings with a
maximum length of 512 characters.
metadata
map
The object type, which is always thread.run.step .
object string
 
  "run_id": "run_abc123",
  "assistant_id": "asst_abc123",
  "thread_id": "thread_abc123",
  "type": "message_creation",
  "status": "completed",
  "cancelled_at": null,
  "completed_at": 1699063291,
  "expired_at": null,
  "failed_at": null,
  "last_error": null,
  "step_details": {
    "type": "message_creation",
    "message_creation": {
      "message_id": "msg_abc123"
    }
  },
  "usage": {
    "prompt_tokens": 123,
    "completion_tokens": 456,
    "total_tokens": 579
  }
}


<!-- Page 303 -->
Stream the result of executing a Run or resuming a Run after submitting tool outputs. You can stream
events from the Create Thread and Run, Create Run, and Submit Tool Outputs endpoints by passing
The ID of the run that this run step is a part of.
run_id string
The status of the run step, which can be either in_progress , cancelled , failed ,
completed , or expired .
status string
The details of the run step.
## Show possible types
step_details object
The ID of the thread that was run.
thread_id string
The type of run step, which can be either message_creation  or tool_calls .
type string
Usage statistics related to the run step. This value will be null  while the run step's
status is in_progress .
## Show properties
usage object or null
Streaming
Beta

<!-- Page 304 -->
"stream": true . The response will be a Server-Sent events stream. Our Node and Python SDKs provide
helpful utilities to make streaming easy. Reference the Assistants API quickstart to learn more.
Represents a message delta i.e. any changed fields on a message during
streaming.
Represents a run step delta i.e. any changed fields on a run step during
streaming.
## The message delta object
Beta
The delta containing the fields that have changed on the Message.
## Show properties
delta object
The identifier of the message, which can be referenced in API endpoints.
id string
The object type, which is always thread.message.delta .
object string
## OBJECT The message delta object
 
 
{
  "id": "msg_123",
  "object": "thread.message.delta",
  "delta": {
    "content": [
      {
        "index": 0,
        "type": "text",
        "text": { "value": "Hello", "annotations
      }
    ]
  }
}

## The run step delta object
Beta
The delta containing the fields that have changed on the run step.
delta object
## OBJECT The run step delta object
 
{
  "id": "step_123",
  "object": "thread.run.step.delta",
  "delta": {
    "step_details": {


<!-- Page 305 -->
Represents an event emitted when streaming a Run.
## Each event in a server-sent events stream has an event  and data
### property:
We emit events whenever a new object is created, transitions to a new
state, or is being streamed in parts (deltas). For example, we emit
thread.run.created  when a new run is created, thread.run.completed  when
a run completes, and so on. When an Assistant chooses to create a
message during a run, we emit a thread.message.created event , a
thread.message.in_progress  event, many thread.message.delta  events, and
finally a thread.message.completed  event.
## Show properties
The identifier of the run step, which can be referenced in API endpoints.
id string
The object type, which is always thread.run.step.delta .
object string
 
      "type": "tool_calls",
      "tool_calls": [
        {
          "index": 0,
          "id": "call_123",
          "type": "code_interpreter",
          "code_interpreter": { "input": "", "o
        }
      ]
    }
  }
}

## Assistant stream events
Beta
event: thread.created
data: {"id": "thread_123", "object": "thread", ...}

<!-- Page 306 -->
We may add additional events over time, so we recommend handling
unknown events gracefully in your code. See the Assistants API quickstart
to learn how to integrate the Assistants API with streaming.
Occurs when a stream ends.
done
data  is [DONE]
Occurs when an error occurs. This can happen due to an internal server error or a
timeout.
error
data  is an error
Occurs when a new thread is created.
thread.created
data  is a thread
Occurs when a message is completed.
thread.message.completed
data  is a message
Occurs when a message is created.
thread.message.created
data  is a message
Occurs when parts of a Message are being streamed.
thread.message.delta
data  is a message delta
Occurs when a message moves to an in_progress  state.
thread.message.in_progress
data  is a message
Occurs when a message ends before it is completed.
thread.message.incomplete
data  is a message
thread.run.cancelled
data  is a run

<!-- Page 307 -->
Occurs when a run is cancelled.
Occurs when a run moves to a cancelling  status.
thread.run.cancelling
data  is a run
Occurs when a run is completed.
thread.run.completed
data  is a run
Occurs when a new run is created.
thread.run.created
data  is a run
Occurs when a run expires.
thread.run.expired
data  is a run
Occurs when a run fails.
thread.run.failed
data  is a run
Occurs when a run moves to an in_progress  status.
thread.run.in_progress
data  is a run
Occurs when a run ends with status incomplete .
thread.run.incomplete
data  is a run
Occurs when a run moves to a queued  status.
thread.run.queued
data  is a run
Occurs when a run moves to a requires_action  status.
thread.run.requires_action
data  is a run
thread.run.step.cancelled
data  is a run step

<!-- Page 308 -->
Programmatically manage your organization. The Audit Logs endpoint provides a log of all actions taken in
the organization for security and monitoring purposes. To access these endpoints please generate an
Admin API Key through the API Platform Organization overview. Admin API keys cannot be used for non-
administration endpoints. For best practices on setting up your organization, please refer to this guide
Occurs when a run step is cancelled.
Occurs when a run step is completed.
thread.run.step.completed
data  is a run step
Occurs when a run step is created.
thread.run.step.created
data  is a run step
Occurs when parts of a run step are being streamed.
thread.run.step.delta
data  is a run step delta
Occurs when a run step expires.
thread.run.step.expired
data  is a run step
Occurs when a run step fails.
thread.run.step.failed
data  is a run step
Occurs when a run step moves to an in_progress  state.
thread.run.step.in_progress
data  is a run step
## Administration

<!-- Page 309 -->
## Admin API keys enable Organization Owners to programmatically manage various aspects of their
organization, including users, projects, and API keys. These keys provide administrative capabilities, such
as creating, updating, and deleting users; managing projects; and overseeing API key lifecycles.
### Key Features of Admin API Keys:
Only Organization Owners have the authority to create and utilize Admin API keys. To manage these keys,
Organization Owners can navigate to the Admin Keys section of their API Platform dashboard.
For direct access to the Admin Keys management page, Organization Owners can use the following link:
https://platform.openai.com/settings/organization/admin-keys
It's crucial to handle Admin API keys with care due to their elevated permissions. Adhering to best
practices, such as regular key rotation and assigning appropriate permissions, enhances security and
ensures proper governance within the organization.
GET https://api.openai.com/v1/organization/admin_api_keys
## List organization API keys
Query parameters
Admin API Keys
User Management: Invite new users, update roles, and remove users from the organization.
Project Management: Create, update, archive projects, and manage user assignments within projects.
API Key Oversight: List, retrieve, and delete API keys associated with projects.
List all organization and project API keys.
## Example request
curl
 
 
curl https://api.openai.com/v1/organization/admin
  -H "Authorization: Bearer $OPENAI_ADMIN_KEY" \
  -H "Content-Type: application/json"


<!-- Page 310 -->
## Returns
POST https://api.openai.com/v1/organization/admin_api_keys
## Create an organization admin API key
Request body
after string or null
Optional
limit integer
Optional
Defaults to 20
order string
## Optional
Defaults to asc
A list of admin and project API key objects.
## Response
{
  "object": "list",
  "data": [
    {
      "object": "organization.admin_api_key",
      "id": "key_abc",
      "name": "Main Admin Key",
      "redacted_value": "sk-admin...def",
      "created_at": 1711471533,
      "last_used_at": 1711471534,
      "owner": {
        "type": "service_account",
        "object": "organization.service_account"
        "id": "sa_456",
        "name": "My Service Account",
        "created_at": 1711471533,
        "role": "member"
      }
    }
  ],
  "first_id": "key_abc",
  "last_id": "key_abc",
  "has_more": false
}

## Create admin API key
Example request
curl
 
curl -X POST https://api.openai.com/v1/organizat
  -H "Authorization: Bearer $OPENAI_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{


<!-- Page 311 -->
## Returns
GET https://api.openai.com/v1/organization/admin_api_keys/{key_id}
## Retrieve a single organization API key
Path parameters
name string
Required
The created AdminApiKey object.
 
      "name": "New Admin Key"
  }'

## Response
{
  "object": "organization.admin_api_key",
  "id": "key_xyz",
  "name": "New Admin Key",
  "redacted_value": "sk-admin...xyz",
  "created_at": 1711471533,
  "last_used_at": 1711471534,
  "owner": {
    "type": "user",
    "object": "organization.user",
    "id": "user_123",
    "name": "John Doe",
    "created_at": 1711471533,
    "role": "owner"
  },
  "value": "sk-admin-1234abcd"
}

## Retrieve admin API key
key_id string
## Required
Example request
curl
 
 
curl https://api.openai.com/v1/organization/admin
  -H "Authorization: Bearer $OPENAI_ADMIN_KEY" \
  -H "Content-Type: application/json"

## Response

<!-- Page 312 -->
## Returns
The requested AdminApiKey object.
{
  "object": "organization.admin_api_key",
  "id": "key_abc",
  "name": "Main Admin Key",
  "redacted_value": "sk-admin...xyz",
  "created_at": 1711471533,
  "last_used_at": 1711471534,
  "owner": {
    "type": "user",
    "object": "organization.user",
    "id": "user_123",
    "name": "John Doe",
    "created_at": 1711471533,
    "role": "owner"
  }
}

## Delete admin API key

<!-- Page 313 -->
DELETE https://api.openai.com/v1/organization/admin_api_keys/{key_id}
## Delete an organization admin API key
Path parameters
Returns
Represents an individual Admin API key in an org.
key_id string
## Required
A confirmation object indicating the key was deleted.
## Example request
curl
 
 
curl -X DELETE https://api.openai.com/v1/organiza
  -H "Authorization: Bearer $OPENAI_ADMIN_KEY" \
  -H "Content-Type: application/json"

## Response
 
 
{
  "id": "key_abc",
  "object": "organization.admin_api_key.deleted",
  "deleted": true
}

## The admin API key object
The Unix timestamp (in seconds) of when the API key was created
created_at integer
The identifier, which can be referenced in API endpoints
id string
The Unix timestamp (in seconds) of when the API key was last used
last_used_at integer or null
## The name of the API key
name string
OBJECT The admin API key object
 
{
  "object": "organization.admin_api_key",
  "id": "key_abc",
  "name": "Main Admin Key",
  "redacted_value": "sk-admin...xyz",
  "created_at": 1711471533,
  "last_used_at": 1711471534,
  "owner": {
    "type": "user",
    "object": "organization.user",
    "id": "user_123",
    "name": "John Doe",
    "created_at": 1711471533,
    "role": "owner"


<!-- Page 314 -->
Invite and manage invitations for an organization.
GET https://api.openai.com/v1/organization/invites
Returns a list of invites in the organization.
## Query parameters
The object type, which is always organization.admin_api_key object
string
## Show properties
owner object
The redacted value of the API key
redacted_value string
The value of the API key. Only shown on create.
value string
 
  }
}

## Invites
List invites
after string
Optional
Example request
curl
 
 
curl https://api.openai.com/v1/organization/invit
  -H "Authorization: Bearer $OPENAI_ADMIN_KEY" \
  -H "Content-Type: application/json"

## Response

<!-- Page 315 -->
## Returns
POST https://api.openai.com/v1/organization/invites
Create an invite for a user to the organization. The invite must be accepted
by the user before they have access to the organization.
## Request body
A cursor for use in pagination. after  is an object ID that defines your place in the list.
For instance, if you make a list request and receive 100 objects, ending with obj_foo,
your subsequent call can include after=obj_foo in order to fetch the next page of the
list.
A limit on the number of objects to be returned. Limit can range between 1 and 100,
and the default is 20.
limit integer
## Optional
Defaults to 20
A list of Invite objects.
{
  "object": "list",
  "data": [
    {
      "object": "organization.invite",
      "id": "invite-abc",
      "email": "user@example.com",
      "role": "owner",
      "status": "accepted",
      "invited_at": 1711471533,
      "expires_at": 1711471533,
      "accepted_at": 1711471533
    }
  ],
  "first_id": "invite-abc",
  "last_id": "invite-abc",
  "has_more": false

## Create invite
Send an email to this address
email string
Required
owner  or reader
role string
Required
Example request
curl
 
curl -X POST https://api.openai.com/v1/organiza
  -H "Authorization: Bearer $OPENAI_ADMIN_KEY" 
  -H "Content-Type: application/json" \
  -d '{
      "email": "anotheruser@example.com",
      "role": "reader",
      "projects": [
        {
          "id": "project-xyz",
          "role": "member"
        },
        {
          "id": "project-abc",


<!-- Page 316 -->
## Returns
GET https://api.openai.com/v1/organization/invites/{invite_id}
Retrieves an invite.
## Path parameters
Returns
An array of projects to which membership is granted at the same time the org invite is
accepted. If omitted, the user will be invited to the default project for compatibility with
legacy behavior.
## Show properties
projects array
Optional
The created Invite object.
 
          "role": "owner"
        }
      ]
  }'

## Response
{
  "object": "organization.invite",
  "id": "invite-def",
  "email": "anotheruser@example.com",
  "role": "reader",
  "status": "pending",
  "invited_at": 1711471533,
  "expires_at": 1711471533,
  "accepted_at": null,
  "projects": [
    {
      "id": "project-xyz",
      "role": "member"
    },
    {
      "id": "project-abc",
      "role": "owner"
    }
  ]
}

## Retrieve invite
The ID of the invite to retrieve.
invite_id string
## Required
Example request
curl
 
 
curl https://api.openai.com/v1/organization/invit
  -H "Authorization: Bearer $OPENAI_ADMIN_KEY" \
  -H "Content-Type: application/json"

## Response
 
{
    "object": "organization.invite",
    "id": "invite-abc",
    "email": "user@example.com",


<!-- Page 317 -->
DELETE https://api.openai.com/v1/organization/invites/{invite_id}
Delete an invite. If the invite has already been accepted, it cannot be
deleted.
## Path parameters
Returns
Represents an individual invite  to the organization.
The Invite object matching the specified ID.
 
    "role": "owner",
    "status": "accepted",
    "invited_at": 1711471533,
    "expires_at": 1711471533,
    "accepted_at": 1711471533
}

## Delete invite
The ID of the invite to delete.
invite_id string
## Required
Confirmation that the invite has been deleted
Example request
curl
 
 
curl -X DELETE https://api.openai.com/v1/organiza
  -H "Authorization: Bearer $OPENAI_ADMIN_KEY" \
  -H "Content-Type: application/json"

## Response
{
    "object": "organization.invite.deleted",
    "id": "invite-abc",
    "deleted": true
}

## The invite object
OBJECT The invite object

<!-- Page 318 -->
The Unix timestamp (in seconds) of when the invite was accepted.
accepted_at integer
## The email address of the individual to whom the invite was sent
email string
The Unix timestamp (in seconds) of when the invite expires.
expires_at integer
The identifier, which can be referenced in API endpoints
id string
The Unix timestamp (in seconds) of when the invite was sent.
invited_at integer
The object type, which is always organization.invite object
string
The projects that were granted membership upon acceptance of the invite.
## Show properties
projects array
owner  or reader
role string
accepted , expired , or pending
status string
{
  "object": "organization.invite",
  "id": "invite-abc",
  "email": "user@example.com",
  "role": "owner",
  "status": "accepted",
  "invited_at": 1711471533,
  "expires_at": 1711471533,
  "accepted_at": 1711471533,
  "projects": [
    {
      "id": "project-xyz",
      "role": "member"
    }
  ]
}

## Users

<!-- Page 319 -->
Manage users and their role in an organization.
GET https://api.openai.com/v1/organization/users
Lists all of the users in the organization.
## Query parameters
Returns
List users
A cursor for use in pagination. after  is an object ID that defines your place in the list.
For instance, if you make a list request and receive 100 objects, ending with obj_foo,
your subsequent call can include after=obj_foo in order to fetch the next page of the
list.
after string
## Optional
Filter by the email address of users.
emails array
## Optional
A limit on the number of objects to be returned. Limit can range between 1 and 100,
and the default is 20.
limit integer
## Optional
Defaults to 20
A list of User objects.
## Example request
curl
 
 
curl https://api.openai.com/v1/organization/users
  -H "Authorization: Bearer $OPENAI_ADMIN_KEY" \
  -H "Content-Type: application/json"

## Response
{
    "object": "list",
    "data": [
        {
            "object": "organization.user",
            "id": "user_abc",
            "name": "First Last",
            "email": "user@example.com",
            "role": "owner",
            "added_at": 1711471533
        }
    ],
    "first_id": "user-abc",
    "last_id": "user-xyz",
    "has_more": false
}


<!-- Page 320 -->
POST https://api.openai.com/v1/organization/users/{user_id}
Modifies a user's role in the organization.
## Path parameters
Request body
Returns
GET https://api.openai.com/v1/organization/users/{user_id}
Retrieves a user by their identifier.
## Path parameters
Modify user
The ID of the user.
user_id string
## Required
owner  or reader
role string
Required
The updated User object.
## Example request
curl
 
 
curl -X POST https://api.openai.com/v1/organizati
  -H "Authorization: Bearer $OPENAI_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
      "role": "owner"
  }'

## Response
{
    "object": "organization.user",
    "id": "user_abc",
    "name": "First Last",
    "email": "user@example.com",
    "role": "owner",
    "added_at": 1711471533
}

## Retrieve user
Example request
curl
 
 
curl https://api.openai.com/v1/organization/user
  -H "Authorization: Bearer $OPENAI_ADMIN_KEY" \


<!-- Page 321 -->
## Returns
DELETE https://api.openai.com/v1/organization/users/{user_id}
Deletes a user from the organization.
## Path parameters
Returns
The ID of the user.
user_id string
## Required
The User object matching the specified ID.
## Response
{
    "object": "organization.user",
    "id": "user_abc",
    "name": "First Last",
    "email": "user@example.com",
    "role": "owner",
    "added_at": 1711471533
}

## Delete user
The ID of the user.
user_id string
## Required
Confirmation of the deleted user
Example request
curl
 
 
curl -X DELETE https://api.openai.com/v1/organiza
  -H "Authorization: Bearer $OPENAI_ADMIN_KEY" \
  -H "Content-Type: application/json"

## Response
{
    "object": "organization.user.deleted",
    "id": "user_abc",
    "deleted": true
}


<!-- Page 322 -->
Represents an individual user  within an organization.
Manage the projects within an orgnanization includes creation, updating, and archiving or projects. The
Default project cannot be archived.
## The user object
The Unix timestamp (in seconds) of when the user was added.
added_at integer
## The email address of the user
email string
The identifier, which can be referenced in API endpoints
id string
## The name of the user
name string
The object type, which is always organization.user object
string
owner  or reader
role string
## OBJECT The user object
{
    "object": "organization.user",
    "id": "user_abc",
    "name": "First Last",
    "email": "user@example.com",
    "role": "owner",
    "added_at": 1711471533
}

## Projects

<!-- Page 323 -->
GET https://api.openai.com/v1/organization/projects
Returns a list of projects.
## Query parameters
Returns
List projects
A cursor for use in pagination. after  is an object ID that defines your place in the list.
For instance, if you make a list request and receive 100 objects, ending with obj_foo,
your subsequent call can include after=obj_foo in order to fetch the next page of the
list.
after string
## Optional
If true  returns all projects including those that have been archived . Archived
projects are not included by default.
include_archived boolean
## Optional
Defaults to false
A limit on the number of objects to be returned. Limit can range between 1 and 100,
and the default is 20.
limit integer
## Optional
Defaults to 20
A list of Project objects.
## Example request
curl
 
 
curl https://api.openai.com/v1/organization/proje
  -H "Authorization: Bearer $OPENAI_ADMIN_KEY" \
  -H "Content-Type: application/json"

## Response
{
    "object": "list",
    "data": [
        {
            "id": "proj_abc",
            "object": "organization.project",
            "name": "Project example",
            "created_at": 1711471533,
            "archived_at": null,
            "status": "active"
        }
    ],
    "first_id": "proj-abc",
    "last_id": "proj-xyz",
    "has_more": false
}

## Create project

<!-- Page 324 -->
POST https://api.openai.com/v1/organization/projects
Create a new project in the organization. Projects can be created and
archived, but cannot be deleted.
## Request body
Returns
GET https://api.openai.com/v1/organization/projects/{project_id}
Retrieves a project.
## Path parameters
The friendly name of the project, this name appears in reports.
name string
## Required
The created Project object.
## Example request
curl
 
 
curl -X POST https://api.openai.com/v1/organizati
  -H "Authorization: Bearer $OPENAI_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
      "name": "Project ABC"
  }'

## Response
{
    "id": "proj_abc",
    "object": "organization.project",
    "name": "Project ABC",
    "created_at": 1711471533,
    "archived_at": null,
    "status": "active"
}

## Retrieve project
The ID of the project.
project_id string
## Required
Example request
curl
 
 
curl https://api.openai.com/v1/organization/proje
  -H "Authorization: Bearer $OPENAI_ADMIN_KEY" \
  -H "Content-Type: application/json"

## Response
 
{
    "id": "proj_abc",


<!-- Page 325 -->
## Returns
POST https://api.openai.com/v1/organization/projects/{project_id}
Modifies a project in the organization.
## Path parameters
Request body
Returns
The Project object matching the specified ID.
 
    "object": "organization.project",
    "name": "Project example",
    "created_at": 1711471533,
    "archived_at": null,
    "status": "active"
}

## Modify project
The ID of the project.
project_id string
## Required
The updated name of the project, this name appears in reports.
name string
## Required
The updated Project object.
## Example request
curl
 
 
curl -X POST https://api.openai.com/v1/organizati
  -H "Authorization: Bearer $OPENAI_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
      "name": "Project DEF"
  }'


<!-- Page 326 -->
POST https://api.openai.com/v1/organization/projects/{project_id}/archive
Archives a project in the organization. Archived projects cannot be used or
updated.
## Path parameters
Returns
Represents an individual project.
## Archive project
The ID of the project.
project_id string
## Required
The archived Project object.
## Example request
curl
 
 
curl -X POST https://api.openai.com/v1/organizati
  -H "Authorization: Bearer $OPENAI_ADMIN_KEY" \
  -H "Content-Type: application/json"

## Response
{
    "id": "proj_abc",
    "object": "organization.project",
    "name": "Project DEF",
    "created_at": 1711471533,
    "archived_at": 1711471533,
    "status": "archived"
}

## The project object
The Unix timestamp (in seconds) of when the project was archived or null .
archived_at integer or null
The Unix timestamp (in seconds) of when the project was created.
created_at integer
id string
## OBJECT The project object
{
    "id": "proj_abc",
    "object": "organization.project",
    "name": "Project example",
    "created_at": 1711471533,
    "archived_at": null,
    "status": "active"
}


<!-- Page 327 -->
Manage users within a project, including adding, updating roles, and removing users.
GET https://api.openai.com/v1/organization/projects/{project_id}/users
Returns a list of users in the project.
## Path parameters
The identifier, which can be referenced in API endpoints
The name of the project. This appears in reporting.
name string
The object type, which is always organization.project object
string
active  or archived
status string
## Project users
List project users
The ID of the project.
project_id string
## Required
Example request
curl
 
 
curl https://api.openai.com/v1/organization/proje
  -H "Authorization: Bearer $OPENAI_ADMIN_KEY" \
  -H "Content-Type: application/json"

## Response
{
    "object": "list",
    "data": [


<!-- Page 328 -->
## Query parameters
Returns
POST https://api.openai.com/v1/organization/projects/{project_id}/users
Adds a user to the project. Users must already be members of the
organization to be added to a project.
## Path parameters
A cursor for use in pagination. after  is an object ID that defines your place in the list.
For instance, if you make a list request and receive 100 objects, ending with obj_foo,
your subsequent call can include after=obj_foo in order to fetch the next page of the
list.
after string
## Optional
A limit on the number of objects to be returned. Limit can range between 1 and 100,
and the default is 20.
limit integer
## Optional
Defaults to 20
A list of ProjectUser objects.
        {
            "object": "organization.project.user
            "id": "user_abc",
            "name": "First Last",
            "email": "user@example.com",
            "role": "owner",
            "added_at": 1711471533
        }
    ],
    "first_id": "user-abc",
    "last_id": "user-xyz",
    "has_more": false
}

## Create project user
The ID of the project.
project_id string
## Required
Example request
curl
 
 
curl -X POST https://api.openai.com/v1/organizati
  -H "Authorization: Bearer $OPENAI_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
      "user_id": "user_abc",
      "role": "member"
  }'

## Response

<!-- Page 329 -->
## Request body
Returns
GET https://api.openai.com/v1/organization/projects/{project_id}/users/{us
er_id}
Retrieves a user in the project.
## Path parameters
owner  or member
role string
Required
The ID of the user.
user_id string
## Required
The created ProjectUser object.
{
    "object": "organization.project.user",
    "id": "user_abc",
    "email": "user@example.com",
    "role": "owner",
    "added_at": 1711471533
}

## Retrieve project user
The ID of the project.
project_id string
## Required
The ID of the user.
user_id string
## Required
Example request
curl
 
 
curl https://api.openai.com/v1/organization/proje
  -H "Authorization: Bearer $OPENAI_ADMIN_KEY" \
  -H "Content-Type: application/json"

## Response
 
 
{
    "object": "organization.project.user",
    "id": "user_abc",
    "name": "First Last",
    "email": "user@example.com",
    "role": "owner",


<!-- Page 330 -->
## Returns
POST https://api.openai.com/v1/organization/projects/{project_id}/users/{u
ser_id}
Modifies a user's role in the project.
## Path parameters
Request body
Returns
The ProjectUser object matching the specified ID.
 
" dd d
t"
1711471533
## Modify project user
The ID of the project.
project_id string
## Required
The ID of the user.
user_id string
## Required
owner  or member
role string
Required
Example request
curl
 
 
curl -X POST https://api.openai.com/v1/organizati
  -H "Authorization: Bearer $OPENAI_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
      "role": "owner"
  }'

## Response
{
    "object": "organization.project.user",
    "id": "user_abc",
    "name": "First Last",
    "email": "user@example.com",
    "role": "owner",
    "added_at": 1711471533
}


<!-- Page 331 -->
DELETE https://api.openai.com/v1/organization/projects/{project_id}/users/
{user_id}
Deletes a user from the project.
## Path parameters
Returns
The updated ProjectUser object.
## Delete project user
The ID of the project.
project_id string
## Required
The ID of the user.
user_id string
## Required
Confirmation that project has been deleted or an error in case of an archived project,
which has no users
## Example request
curl
 
 
curl -X DELETE https://api.openai.com/v1/organiza
  -H "Authorization: Bearer $OPENAI_ADMIN_KEY" \
  -H "Content-Type: application/json"

## Response
 
 
{
    "object": "organization.project.user.deleted"
    "id": "user_abc",
    "deleted": true
}

## The project user object

<!-- Page 332 -->
Represents an individual user in a project.
Manage service accounts within a project. A service account is a bot user that is not associated with a user.
If a user leaves an organization, their keys and membership in projects will no longer work. Service
accounts do not have this limitation. However, service accounts can also be deleted from a project.
The Unix timestamp (in seconds) of when the project was added.
added_at integer
## The email address of the user
email string
The identifier, which can be referenced in API endpoints
id string
## The name of the user
name string
The object type, which is always organization.project.user object
string
owner  or member
role string
## OBJECT The project user object
{
    "object": "organization.project.user",
    "id": "user_abc",
    "name": "First Last",
    "email": "user@example.com",
    "role": "owner",
    "added_at": 1711471533
}

## Project service accounts
List project service accounts

<!-- Page 333 -->
GET https://api.openai.com/v1/organization/projects/{project_id}/service_a
ccounts
Returns a list of service accounts in the project.
## Path parameters
Query parameters
Returns
The ID of the project.
project_id string
## Required
A cursor for use in pagination. after  is an object ID that defines your place in the list.
For instance, if you make a list request and receive 100 objects, ending with obj_foo,
your subsequent call can include after=obj_foo in order to fetch the next page of the
list.
after string
## Optional
A limit on the number of objects to be returned. Limit can range between 1 and 100,
and the default is 20.
limit integer
## Optional
Defaults to 20
A list of ProjectServiceAccount objects.
## Example request
curl
 
 
curl https://api.openai.com/v1/organization/proje
  -H "Authorization: Bearer $OPENAI_ADMIN_KEY" \
  -H "Content-Type: application/json"

## Response
 
 
{
    "object": "list",
    "data": [
        {
            "object": "organization.project.serv
            "id": "svc_acct_abc",
            "name": "Service Account",
            "role": "owner",
            "created_at": 1711471533
        }
    ],
    "first_id": "svc_acct_abc",
    "last_id": "svc_acct_xyz",
    "has_more": false
}

## Create project service account

<!-- Page 334 -->
POST https://api.openai.com/v1/organization/projects/{project_id}/service_
accounts
Creates a new service account in the project. This also returns an
unredacted API key for the service account.
## Path parameters
Request body
Returns
GET https://api.openai.com/v1/organization/projects/{project_id}/service_a
ccounts/{service_account_id}
Retrieves a service account in the project.
The ID of the project.
project_id string
## Required
The name of the service account being created.
name string
## Required
The created ProjectServiceAccount object.
## Example request
curl
 
 
curl -X POST https://api.openai.com/v1/organizati
  -H "Authorization: Bearer $OPENAI_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
      "name": "Production App"
  }'

## Response
 
 
{
    "object": "organization.project.service_acco
    "id": "svc_acct_abc",
    "name": "Production App",
    "role": "member",
    "created_at": 1711471533,
    "api_key": {
        "object": "organization.project.service_
        "value": "sk-abcdefghijklmnop123",
        "name": "Secret Key",
        "created_at": 1711471533,
        "id": "key_abc"
    }
}

## Retrieve project service account
Example request
curl
 
curl https://api.openai.com/v1/organization/proj
  -H "Authorization: Bearer $OPENAI_ADMIN_KEY" \


<!-- Page 335 -->
## Path parameters
Returns
DELETE https://api.openai.com/v1/organization/projects/{project_id}/servic
e_accounts/{service_account_id}
Deletes a service account from the project.
## Path parameters
The ID of the project.
project_id string
## Required
The ID of the service account.
service_account_id string
## Required
The ProjectServiceAccount object matching the specified ID.
 
-H "Content-Type: application/json"
## Response
 
 
{
    "object": "organization.project.service_accou
    "id": "svc_acct_abc",
    "name": "Service Account",
    "role": "owner",
    "created_at": 1711471533
}

## Delete project service account
The ID of the project.
project_id string
## Required
The ID of the service account.
service_account_id string
## Required
Example request
curl
 
 
curl -X DELETE https://api.openai.com/v1/organiza
  -H "Authorization: Bearer $OPENAI_ADMIN_KEY" \
  -H "Content-Type: application/json"

## Response
 
 
{
    "object": "organization.project.service_accou
    "id": "svc_acct_abc",
    "deleted": true
}


<!-- Page 336 -->
## Returns
Represents an individual service account in a project.
Confirmation of service account being deleted, or an error in case of an archived
project, which has no service accounts
## The project service account object
The Unix timestamp (in seconds) of when the service account was created
created_at integer
The identifier, which can be referenced in API endpoints
id string
## The name of the service account
name string
The object type, which is always organization.project.service_account object
string
owner  or member
role string
## OBJECT The project service account object
 
 
{
    "object": "organization.project.service_accou
    "id": "svc_acct_abc",
    "name": "Service Account",
    "role": "owner",
    "created_at": 1711471533
}

## Project API keys

<!-- Page 337 -->
Manage API keys for a given project. Supports listing and deleting keys for users. This API does not allow
issuing keys for users, as users need to authorize themselves to generate keys.
GET https://api.openai.com/v1/organization/projects/{project_id}/api_keys
Returns a list of API keys in the project.
## Path parameters
Query parameters
Returns
List project API keys
The ID of the project.
project_id string
## Required
A cursor for use in pagination. after  is an object ID that defines your place in the list.
For instance, if you make a list request and receive 100 objects, ending with obj_foo,
your subsequent call can include after=obj_foo in order to fetch the next page of the
list.
after string
## Optional
A limit on the number of objects to be returned. Limit can range between 1 and 100,
and the default is 20.
limit integer
## Optional
Defaults to 20
## Example request
curl
 
 
curl https://api.openai.com/v1/organization/proje
  -H "Authorization: Bearer $OPENAI_ADMIN_KEY" \
  -H "Content-Type: application/json"

## Response
 
{
    "object": "list",
    "data": [
        {
            "object": "organization.project.api
            "redacted_value": "sk-abc...def",
            "name": "My API Key",
            "created_at": 1711471533,
            "last_used_at": 1711471534,
            "id": "key_abc",
            "owner": {
                "type": "user",
                "user": {
                    "object": "organization.pro
                    "id": "user_abc",
                    "name": "First Last",
                    "email": "user@example.com"
                    "role": "owner",
                    "added_at": 1711471533
                }
            }


<!-- Page 338 -->
GET https://api.openai.com/v1/organization/projects/{project_id}/api_keys/
{key_id}
Retrieves an API key in the project.
## Path parameters
Returns
A list of ProjectApiKey objects.
 
        }
    ],
    "first_id": "key_abc",
    "last_id": "key_xyz",
    "has_more": false
}

## Retrieve project API key
The ID of the API key.
key_id string
## Required
The ID of the project.
project_id string
## Required
The ProjectApiKey object matching the specified ID.
## Example request
curl
 
 
curl https://api.openai.com/v1/organization/proje
  -H "Authorization: Bearer $OPENAI_ADMIN_KEY" \
  -H "Content-Type: application/json"

## Response
 
{
    "object": "organization.project.api_key",
    "redacted_value": "sk-abc...def",
    "name": "My API Key",
    "created_at": 1711471533,
    "last_used_at": 1711471534,
    "id": "key_abc",
    "owner": {
        "type": "user",
        "user": {
            "object": "organization.project.use
            "id": "user_abc",
            "name": "First Last",
            "email": "user@example.com",
            "role": "owner",
            "added_at": 1711471533
        }


<!-- Page 339 -->
DELETE https://api.openai.com/v1/organization/projects/{project_id}/api_ke
ys/{key_id}
Deletes an API key from the project.
## Path parameters
Returns
Represents an individual API key in a project.
 
}
## Delete project API key
The ID of the API key.
key_id string
## Required
The ID of the project.
project_id string
## Required
Confirmation of the key's deletion or an error if the key belonged to a service account
## Example request
curl
 
 
curl -X DELETE https://api.openai.com/v1/organiza
  -H "Authorization: Bearer $OPENAI_ADMIN_KEY" \
  -H "Content-Type: application/json"

## Response
 
 
{
    "object": "organization.project.api_key.delet
    "id": "key_abc",
    "deleted": true
}

## The project API key object
created_at integer
## OBJECT The project API key object

<!-- Page 340 -->
Manage rate limits per model for projects. Rate limits may be configured to be equal to or lower than the
organization's rate limits.
The Unix timestamp (in seconds) of when the API key was created
The identifier, which can be referenced in API endpoints
id string
The Unix timestamp (in seconds) of when the API key was last used.
last_used_at integer
## The name of the API key
name string
The object type, which is always organization.project.api_key object
string
## Show properties
owner object
The redacted value of the API key
redacted_value string
{
    "object": "organization.project.api_key",
    "redacted_value": "sk-abc...def",
    "name": "My API Key",
    "created_at": 1711471533,
    "last_used_at": 1711471534,
    "id": "key_abc",
    "owner": {
        "type": "user",
        "user": {
            "object": "organization.project.user
            "id": "user_abc",
            "name": "First Last",
            "email": "user@example.com",
            "role": "owner",
            "created_at": 1711471533
        }
    }

## Project rate limits
List project rate limits

<!-- Page 341 -->
GET https://api.openai.com/v1/organization/projects/{project_id}/rate_limi
ts
Returns the rate limits per model for a project.
## Path parameters
Query parameters
Returns
The ID of the project.
project_id string
## Required
A cursor for use in pagination. after  is an object ID that defines your place in the list.
For instance, if you make a list request and receive 100 objects, ending with obj_foo,
your subsequent call can include after=obj_foo in order to fetch the next page of the
list.
after string
## Optional
A cursor for use in pagination. before  is an object ID that defines your place in the
list. For instance, if you make a list request and receive 100 objects, beginning with
obj_foo, your subsequent call can include before=obj_foo in order to fetch the previous
page of the list.
before string
## Optional
A limit on the number of objects to be returned. The default is 100.
limit integer
## Optional
Defaults to 100
A list of ProjectRateLimit objects.
## Example request
curl
 
 
curl https://api.openai.com/v1/organization/proje
  -H "Authorization: Bearer $OPENAI_ADMIN_KEY" \
  -H "Content-Type: application/json"

## Response
{
    "object": "list",
    "data": [
        {
          "object": "project.rate_limit",
          "id": "rl-ada",
          "model": "ada",
          "max_requests_per_1_minute": 600,
          "max_tokens_per_1_minute": 150000,
          "max_images_per_1_minute": 10
        }
    ],
    "first_id": "rl-ada",
    "last_id": "rl-ada",
    "has_more": false
}


<!-- Page 342 -->
POST https://api.openai.com/v1/organization/projects/{project_id}/rate_lim
its/{rate_limit_id}
Updates a project rate limit.
## Path parameters
Request body
Modify project rate limit
The ID of the project.
project_id string
## Required
The ID of the rate limit.
rate_limit_id string
## Required
The maximum batch input tokens per day. Only relevant for certain models.
batch_1_day_max_input_tokens integer
## Optional
The maximum audio megabytes per minute. Only relevant for certain models.
max_audio_megabytes_per_1_minute integer
## Optional
The maximum images per minute. Only relevant for certain models.
max_images_per_1_minute integer
## Optional
The maximum requests per day. Only relevant for certain models.
max_requests_per_1_day integer
## Optional
Example request
curl
 
 
curl -X POST https://api.openai.com/v1/organizati
  -H "Authorization: Bearer $OPENAI_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
      "max_requests_per_1_minute": 500
  }'

## Response
{
    "object": "project.rate_limit",
    "id": "rl-ada",
    "model": "ada",
    "max_requests_per_1_minute": 600,
    "max_tokens_per_1_minute": 150000,
    "max_images_per_1_minute": 10
  }


<!-- Page 343 -->
## Returns
Represents a project rate limit config.
The maximum requests per minute.
max_requests_per_1_minute integer
## Optional
The maximum tokens per minute.
max_tokens_per_1_minute integer
## Optional
The updated ProjectRateLimit object.
## The project rate limit object
The maximum batch input tokens per day. Only present for relevant models.
batch_1_day_max_input_tokens integer
The identifier, which can be referenced in API endpoints.
id string
The maximum audio megabytes per minute. Only present for relevant models.
max_audio_megabytes_per_1_minute integer
The maximum images per minute. Only present for relevant models.
max_images_per_1_minute integer
The maximum requests per day. Only present for relevant models.
max_requests_per_1_day integer
## OBJECT The project rate limit object
{
    "object": "project.rate_limit",
    "id": "rl_ada",
    "model": "ada",
    "max_requests_per_1_minute": 600,
    "max_tokens_per_1_minute": 150000,
    "max_images_per_1_minute": 10
}


<!-- Page 344 -->
Logs of user actions and configuration changes within this organization. To log events, an Organization
Owner must activate logging in the Data Controls Settings. Once activated, for security reasons, logging
cannot be deactivated.
GET https://api.openai.com/v1/organization/audit_logs
List user actions and configuration changes within this organization.
## Query parameters
The maximum requests per minute.
max_requests_per_1_minute integer
The maximum tokens per minute.
max_tokens_per_1_minute integer
The model this rate limit applies to.
model string
The object type, which is always project.rate_limit object
string
## Audit logs
List audit logs
Example request
curl
 
 
curl https://api.openai.com/v1/organization/audit
-H "Authorization: Bearer $OPENAI_ADMIN_KEY" \
-H "Content-Type: application/json"


<!-- Page 345 -->
Return only events performed by users with these emails.
actor_emails[]
array
## Optional
Return only events performed by these actors. Can be a user ID, a service account ID,
or an api key tracking ID.
actor_ids[]
array
## Optional
A cursor for use in pagination. after  is an object ID that defines your place in the list.
For instance, if you make a list request and receive 100 objects, ending with obj_foo,
your subsequent call can include after=obj_foo in order to fetch the next page of the
list.
after string
## Optional
A cursor for use in pagination. before  is an object ID that defines your place in the
list. For instance, if you make a list request and receive 100 objects, starting with
obj_foo, your subsequent call can include before=obj_foo in order to fetch the previous
page of the list.
before string
## Optional
Return only events whose effective_at  (Unix seconds) is in this range.
## Show properties
effective_at object
## Optional
Return only events with a type  in one of these values. For example,
project.created . For all options, see the documentation for the audit log object.
event_types[]
array
## Optional
A limit on the number of objects to be returned. Limit can range between 1 and 100,
and the default is 20.
limit integer
## Optional
Defaults to 20
Return only events for these projects.
project_ids[]
array
## Optional
Response
 
{
    "object": "list",
    "data": [
        {
            "id": "audit_log-xxx_yyyymmdd",
            "type": "project.archived",
            "effective_at": 1722461446,
            "actor": {
                "type": "api_key",
                "api_key": {
                    "type": "user",
                    "user": {
                        "id": "user-xxx",
                        "email": "user@example.
                    }
                }
            },
            "project.archived": {
                "id": "proj_abc"
            },
        },
        {
            "id": "audit_log-yyy__20240101",
            "type": "api_key.updated",
            "effective_at": 1720804190,
            "actor": {
                "type": "session",
                "session": {
                    "user": {
                        "id": "user-xxx",
                        "email": "user@example.
                    },
                    "ip_address": "127.0.0.1",
                    "user_agent": "Mozilla/5.0 
                    "ja3": "a497151ce4338a12c44
                    "ja4": "q13d0313h3_55b375c5
                    "ip_address_details": {
                      "country": "US",


<!-- Page 346 -->
## Returns
A log of a user action or configuration change within this organization.
Return only events performed on these targets. For example, a project ID updated.
resource_ids[]
array
## Optional
A list of paginated Audit Log objects.
 
                      "city": "San Francisco",
                      "region": "California",
                      "region_code": "CA",
                      "asn": "1234",
                      "latitude": "37.77490",
                      "longitude": "-122.41940"
                    }
                }
            },
            "api_key.updated": {
                "id": "key_xxxx",
                "data": {
                    "scopes": ["resource_2.oper
                }
            },
        }
    ],
    "first_id": "audit_log-xxx__20240101",
    "last_id": "audit_log_yyy__20240101",
    "has_more": true
}

## The audit log object
The actor who performed the audit logged action.
## Show properties
actor object
The details for events with this type .
## Show properties
api_key.created object
The details for events with this type .
## Show properties
api_key.deleted object
The details for events with this type .
## Show properties
api_key.updated object
## OBJECT The audit log object
 
{
    "id": "req_xxx_20240101",
    "type": "api_key.created",
    "effective_at": 1720804090,
    "actor": {
        "type": "session",
        "session": {
            "user": {
                "id": "user-xxx",
                "email": "user@example.com"
            },
            "ip_address": "127.0.0.1",
            "user_agent": "Mozilla/5.0 (Windows
        }
    },
    "api_key.created": {
        "id": "key_xxxx",
        "data": {
            "scopes": ["resource.operation"]
        }


<!-- Page 347 -->
The details for events with this type .
## Show properties
certificate.created object
The details for events with this type .
## Show properties
certificate.deleted object
The details for events with this type .
## Show properties
certificate.updated object
The details for events with this type .
## Show properties
certificates.activated object
The details for events with this type .
## Show properties
certificates.deactivated object
## The project and fine-tuned model checkpoint that the checkpoint permission was
created for.
## Show properties
checkpoint_permission.created object
The details for events with this type .
## Show properties
checkpoint_permission.deleted object
The Unix timestamp (in seconds) of the event.
effective_at integer
 
    }
}


<!-- Page 348 -->
The ID of this log.
id string
The details for events with this type .
## Show properties
invite.accepted object
The details for events with this type .
## Show properties
invite.deleted object
The details for events with this type .
## Show properties
invite.sent object
The details for events with this type .
## Show properties
login.failed object
The details for events with this type .
## Show properties
logout.failed object
The details for events with this type .
## Show properties
organization.updated object
The project that the action was scoped to. Absent for actions not scoped to projects.
## Note that any admin actions taken via Admin API keys are associated with the default
project.
project object

<!-- Page 349 -->
## Show properties
The details for events with this type .
## Show properties
project.archived object
The details for events with this type .
## Show properties
project.created object
The details for events with this type .
## Show properties
project.updated object
The details for events with this type .
## Show properties
rate_limit.deleted object
The details for events with this type .
## Show properties
rate_limit.updated object
The details for events with this type .
## Show properties
service_account.created object
The details for events with this type .
## Show properties
service_account.deleted object
The details for events with this type .
service_account.updated object

<!-- Page 350 -->
The Usage API provides detailed insights into your activity across the OpenAI API. It also includes a
separate Costs endpoint, which offers visibility into your spend, breaking down consumption by invoice line
items and project IDs.
While the Usage API delivers granular usage data, it may not always reconcile perfectly with the Costs due
to minor differences in how usage and spend are recorded. For financial purposes, we recommend using
the Costs endpoint or the Costs tab in the Usage Dashboard, which will reconcile back to your billing
invoice.
## Show properties
The event type.
type string
The details for events with this type .
## Show properties
user.added object
The details for events with this type .
## Show properties
user.deleted object
The details for events with this type .
## Show properties
user.updated object
## Usage

<!-- Page 351 -->
GET https://api.openai.com/v1/organization/usage/completions
Get completions usage details for the organization.
## Query parameters
Completions
Start time (Unix seconds) of the query time range, inclusive.
start_time integer
## Required
Return only usage for these API keys.
api_key_ids array
## Optional
If true , return batch jobs only. If false , return non-batch jobs only. By default,
return both.
batch boolean
## Optional
Width of each time bucket in response. Currently 1m , 1h  and 1d  are supported,
default to 1d .
bucket_width string
## Optional
Defaults to 1d
End time (Unix seconds) of the query time range, exclusive.
end_time integer
## Optional
Group the usage data by the specified fields. Support fields include project_id ,
user_id , api_key_id , model , batch  or any combination of them.
group_by array
## Optional
Specifies the number of buckets to return.
limit integer
## Optional
bucket_width=1d : default: 7, max: 31
## Example request
curl
 
 
curl "https://api.openai.com/v1/organization/usag
-H "Authorization: Bearer $OPENAI_ADMIN_KEY" \
-H "Content-Type: application/json"

## Response
 
 
{
    "object": "page",
    "data": [
        {
            "object": "bucket",
            "start_time": 1730419200,
            "end_time": 1730505600,
            "results": [
                {
                    "object": "organization.usa
                    "input_tokens": 1000,
                    "output_tokens": 500,
                    "input_cached_tokens": 800,
                    "input_audio_tokens": 0,
                    "output_audio_tokens": 0,
                    "num_model_requests": 5,
                    "project_id": null,
                    "user_id": null,
                    "api_key_id": null,
                    "model": null,
                    "batch": null
                }
            ]
        }
    ],
    "has_more": true,


<!-- Page 352 -->
## Returns
The aggregated completions usage details of the specific time bucket.
bucket_width=1h : default: 24, max: 168
bucket_width=1m : default: 60, max: 1440
Return only usage for these models.
models array
## Optional
A cursor for use in pagination. Corresponding to the next_page  field from the
previous response.
page string
## Optional
Return only usage for these projects.
project_ids array
## Optional
Return only usage for these users.
user_ids array
## Optional
A list of paginated, time bucketed Completions usage objects.
 
"
t
"
"
## AAAAAGdG dEiJdKOAAAAAG
YA
Completions usage object
When group_by=api_key_id , this field provides the API key ID of the grouped usage
result.
api_key_id string or null
## OBJECT Completions usage object
 
{
    "object": "organization.usage.completions.r
    "input_tokens": 5000,
    "output_tokens": 1000,
    "input_cached_tokens": 4000,


<!-- Page 353 -->
When group_by=batch , this field tells whether the grouped usage result is batch or
not.
batch boolean or null
The aggregated number of audio input tokens used, including cached tokens.
input_audio_tokens integer
## The aggregated number of text input tokens that has been cached from previous
requests. For customers subscribe to scale tier, this includes scale tier tokens.
input_cached_tokens integer
The aggregated number of text input tokens used, including cached tokens. For
customers subscribe to scale tier, this includes scale tier tokens.
input_tokens integer
When group_by=model , this field provides the model name of the grouped usage
result.
model string or null
The count of requests made to the model.
num_model_requests integer
object string
The aggregated number of audio output tokens used.
output_audio_tokens integer
The aggregated number of text output tokens used. For customers subscribe to scale
tier, this includes scale tier tokens.
output_tokens integer
project_id string or null
 
    "input_audio_tokens": 300,
    "output_audio_tokens": 200,
    "num_model_requests": 5,
    "project_id": "proj_abc",
    "user_id": "user-abc",
    "api_key_id": "key_abc",
    "model": "gpt-4o-mini-2024-07-18",
    "batch": false
}


<!-- Page 354 -->
GET https://api.openai.com/v1/organization/usage/embeddings
Get embeddings usage details for the organization.
## Query parameters
When group_by=project_id , this field provides the project ID of the grouped usage
result.
When group_by=user_id , this field provides the user ID of the grouped usage result.
user_id string or null
## Embeddings
Start time (Unix seconds) of the query time range, inclusive.
start_time integer
## Required
Return only usage for these API keys.
api_key_ids array
## Optional
Width of each time bucket in response. Currently 1m , 1h  and 1d  are supported,
default to 1d .
bucket_width string
## Optional
Defaults to 1d
End time (Unix seconds) of the query time range, exclusive.
end_time integer
## Optional
group_by array
## Optional
Example request
curl
 
 
curl "https://api.openai.com/v1/organization/usag
-H "Authorization: Bearer $OPENAI_ADMIN_KEY" \
-H "Content-Type: application/json"

## Response
 
{
    "object": "page",
    "data": [
        {
            "object": "bucket",
            "start_time": 1730419200,
            "end_time": 1730505600,
            "results": [
                {
                    "object": "organization.usa
                    "input_tokens": 16,
                    "num_model_requests": 2,
                    "project_id": null,
                    "user_id": null,
                    "api_key_id": null,
                    "model": null
                }


<!-- Page 355 -->
## Returns
Group the usage data by the specified fields. Support fields include project_id ,
user_id , api_key_id , model  or any combination of them.
Specifies the number of buckets to return.
limit integer
## Optional
bucket_width=1d : default: 7, max: 31
bucket_width=1h : default: 24, max: 168
bucket_width=1m : default: 60, max: 1440
Return only usage for these models.
models array
## Optional
A cursor for use in pagination. Corresponding to the next_page  field from the
previous response.
page string
## Optional
Return only usage for these projects.
project_ids array
## Optional
Return only usage for these users.
user_ids array
## Optional
A list of paginated, time bucketed Embeddings usage objects.
 
            ]
        }
    ],
    "has_more": false,
    "next_page": null
}

## Embeddings usage object

<!-- Page 356 -->
The aggregated embeddings usage details of the specific time bucket.
GET https://api.openai.com/v1/organization/usage/moderations
Get moderations usage details for the organization.
When group_by=api_key_id , this field provides the API key ID of the grouped usage
result.
api_key_id string or null
The aggregated number of input tokens used.
input_tokens integer
When group_by=model , this field provides the model name of the grouped usage
result.
model string or null
The count of requests made to the model.
num_model_requests integer
object string
When group_by=project_id , this field provides the project ID of the grouped usage
result.
project_id string or null
When group_by=user_id , this field provides the user ID of the grouped usage result.
user_id string or null
## OBJECT Embeddings usage object
 
 
{
    "object": "organization.usage.embeddings.resu
    "input_tokens": 20,
    "num_model_requests": 2,
    "project_id": "proj_abc",
    "user_id": "user-abc",
    "api_key_id": "key_abc",
    "model": "text-embedding-ada-002-v2"
}

## Moderations
Example request
curl

<!-- Page 357 -->
## Query parameters
Start time (Unix seconds) of the query time range, inclusive.
start_time integer
## Required
Return only usage for these API keys.
api_key_ids array
## Optional
Width of each time bucket in response. Currently 1m , 1h  and 1d  are supported,
default to 1d .
bucket_width string
## Optional
Defaults to 1d
End time (Unix seconds) of the query time range, exclusive.
end_time integer
## Optional
Group the usage data by the specified fields. Support fields include project_id ,
user_id , api_key_id , model  or any combination of them.
group_by array
## Optional
Specifies the number of buckets to return.
limit integer
## Optional
bucket_width=1d : default: 7, max: 31
bucket_width=1h : default: 24, max: 168
bucket_width=1m : default: 60, max: 1440
Return only usage for these models.
models array
## Optional
A cursor for use in pagination. Corresponding to the next_page  field from the
previous response.
page string
## Optional
curl "https://api.openai.com/v1/organization/usag
-H "Authorization: Bearer $OPENAI_ADMIN_KEY" \

## Response
 
 
{
    "object": "page",
    "data": [
        {
            "object": "bucket",
            "start_time": 1730419200,
            "end_time": 1730505600,
            "results": [
                {
                    "object": "organization.usag
                    "input_tokens": 16,
                    "num_model_requests": 2,
                    "project_id": null,
                    "user_id": null,
                    "api_key_id": null,
                    "model": null
                }
            ]
        }
    ],
    "has_more": false,
    "next_page": null
}


<!-- Page 358 -->
## Returns
The aggregated moderations usage details of the specific time bucket.
Return only usage for these projects.
project_ids array
## Optional
Return only usage for these users.
user_ids array
## Optional
A list of paginated, time bucketed Moderations usage objects.
## Moderations usage object
When group_by=api_key_id , this field provides the API key ID of the grouped usage
result.
api_key_id string or null
The aggregated number of input tokens used.
input_tokens integer
When group_by=model , this field provides the model name of the grouped usage
result.
model string or null
The count of requests made to the model.
num_model_requests integer
## OBJECT Moderations usage object
 
 
{
    "object": "organization.usage.moderations.res
    "input_tokens": 20,
    "num_model_requests": 2,
    "project_id": "proj_abc",
    "user_id": "user-abc",
    "api_key_id": "key_abc",
    "model": "text-moderation"
}


<!-- Page 359 -->
GET https://api.openai.com/v1/organization/usage/images
Get images usage details for the organization.
## Query parameters object
string
When group_by=project_id , this field provides the project ID of the grouped usage
result.
project_id string or null
When group_by=user_id , this field provides the user ID of the grouped usage result.
user_id string or null
## Images
Start time (Unix seconds) of the query time range, inclusive.
start_time integer
## Required
Return only usage for these API keys.
api_key_ids array
## Optional
Width of each time bucket in response. Currently 1m , 1h  and 1d  are supported,
default to 1d .
bucket_width string
## Optional
Defaults to 1d
End time (Unix seconds) of the query time range, exclusive.
end_time integer
## Optional
Example request
curl
 
 
curl "https://api.openai.com/v1/organization/usag
-H "Authorization: Bearer $OPENAI_ADMIN_KEY" \
-H "Content-Type: application/json"

## Response
 
{
    "object": "page",
    "data": [
        {
            "object": "bucket",
            "start_time": 1730419200,
            "end_time": 1730505600,
            "results": [
                {
                    "object": "organization.usa
                    "images": 2,
                    "num_model_requests": 2,


<!-- Page 360 -->
Group the usage data by the specified fields. Support fields include project_id ,
user_id , api_key_id , model , size , source  or any combination of them.
group_by array
## Optional
Specifies the number of buckets to return.
limit integer
## Optional
bucket_width=1d : default: 7, max: 31
bucket_width=1h : default: 24, max: 168
bucket_width=1m : default: 60, max: 1440
Return only usage for these models.
models array
## Optional
A cursor for use in pagination. Corresponding to the next_page  field from the
previous response.
page string
## Optional
Return only usage for these projects.
project_ids array
## Optional
Return only usages for these image sizes. Possible values are 256x256 , 512x512 ,
1024x1024 , 1792x1792 , 1024x1792  or any combination of them.
sizes array
## Optional
Return only usages for these sources. Possible values are image.generation ,
image.edit , image.variation  or any combination of them.
sources array
## Optional
Return only usage for these users.
user_ids array
## Optional
 
                    "size": null,
                    "source": null,
                    "project_id": null,
                    "user_id": null,
                    "api_key_id": null,
                    "model": null
                }
            ]
        }
    ],
    "has_more": false,
    "next_page": null
}


<!-- Page 361 -->
## Returns
The aggregated images usage details of the specific time bucket.
A list of paginated, time bucketed Images usage objects.
## Images usage object
When group_by=api_key_id , this field provides the API key ID of the grouped usage
result.
api_key_id string or null
The number of images processed.
images integer
When group_by=model , this field provides the model name of the grouped usage
result.
model string or null
The count of requests made to the model.
num_model_requests integer
object string
When group_by=project_id , this field provides the project ID of the grouped usage
result.
project_id string or null
size string or null
## OBJECT Images usage object
 
 
{
    "object": "organization.usage.images.result"
    "images": 2,
    "num_model_requests": 2,
    "size": "1024x1024",
    "source": "image.generation",
    "project_id": "proj_abc",
    "user_id": "user-abc",
    "api_key_id": "key_abc",
    "model": "dall-e-3"
}


<!-- Page 362 -->
GET https://api.openai.com/v1/organization/usage/audio_speeches
Get audio speeches usage details for the organization.
## Query parameters
When group_by=size , this field provides the image size of the grouped usage result.
When group_by=source , this field provides the source of the grouped usage result,
possible values are image.generation , image.edit , image.variation .
source string or null
When group_by=user_id , this field provides the user ID of the grouped usage result.
user_id string or null
## Audio speeches
Start time (Unix seconds) of the query time range, inclusive.
start_time integer
## Required
Return only usage for these API keys.
api_key_ids array
## Optional
Width of each time bucket in response. Currently 1m , 1h  and 1d  are supported,
default to 1d .
bucket_width string
## Optional
Defaults to 1d
End time (Unix seconds) of the query time range, exclusive.
end_time integer
## Optional
Example request
curl
 
 
curl "https://api.openai.com/v1/organization/usag
-H "Authorization: Bearer $OPENAI_ADMIN_KEY" \
-H "Content-Type: application/json"

## Response
 
{
    "object": "page",
    "data": [
        {
            "object": "bucket",
            "start_time": 1730419200,
            "end_time": 1730505600,
            "results": [
                {
                    "object": "organization.usa
                    "characters": 45,
                    "num_model_requests": 1,
                    "project_id": null,


<!-- Page 363 -->
## Returns
Group the usage data by the specified fields. Support fields include project_id ,
user_id , api_key_id , model  or any combination of them.
group_by array
## Optional
Specifies the number of buckets to return.
limit integer
## Optional
bucket_width=1d : default: 7, max: 31
bucket_width=1h : default: 24, max: 168
bucket_width=1m : default: 60, max: 1440
Return only usage for these models.
models array
## Optional
A cursor for use in pagination. Corresponding to the next_page  field from the
previous response.
page string
## Optional
Return only usage for these projects.
project_ids array
## Optional
Return only usage for these users.
user_ids array
## Optional
A list of paginated, time bucketed Audio speeches usage objects.
 
                    "user_id": null,
                    "api_key_id": null,
                    "model": null
                }
            ]
        }
    ],
    "has_more": false,
    "next_page": null
}


<!-- Page 364 -->
The aggregated audio speeches usage details of the specific time bucket.
## Audio speeches usage object
When group_by=api_key_id , this field provides the API key ID of the grouped usage
result.
api_key_id string or null
The number of characters processed.
characters integer
When group_by=model , this field provides the model name of the grouped usage
result.
model string or null
The count of requests made to the model.
num_model_requests integer
object string
When group_by=project_id , this field provides the project ID of the grouped usage
result.
project_id string or null
When group_by=user_id , this field provides the user ID of the grouped usage result.
user_id string or null
## OBJECT Audio speeches usage object
 
 
{
    "object": "organization.usage.audio_speeches.
    "characters": 45,
    "num_model_requests": 1,
    "project_id": "proj_abc",
    "user_id": "user-abc",
    "api_key_id": "key_abc",
    "model": "tts-1"
}

## Audio transcriptions

<!-- Page 365 -->
GET https://api.openai.com/v1/organization/usage/audio_transcriptions
Get audio transcriptions usage details for the organization.
## Query parameters
Start time (Unix seconds) of the query time range, inclusive.
start_time integer
## Required
Return only usage for these API keys.
api_key_ids array
## Optional
Width of each time bucket in response. Currently 1m , 1h  and 1d  are supported,
default to 1d .
bucket_width string
## Optional
Defaults to 1d
End time (Unix seconds) of the query time range, exclusive.
end_time integer
## Optional
Group the usage data by the specified fields. Support fields include project_id ,
user_id , api_key_id , model  or any combination of them.
group_by array
## Optional
Specifies the number of buckets to return.
limit integer
## Optional
bucket_width=1d : default: 7, max: 31
bucket_width=1h : default: 24, max: 168
bucket_width=1m : default: 60, max: 1440
Return only usage for these models.
models array
## Optional
Example request
curl
 
 
curl "https://api.openai.com/v1/organization/usag
-H "Authorization: Bearer $OPENAI_ADMIN_KEY" \
-H "Content-Type: application/json"

## Response
 
 
{
    "object": "page",
    "data": [
        {
            "object": "bucket",
            "start_time": 1730419200,
            "end_time": 1730505600,
            "results": [
                {
                    "object": "organization.usag
                    "seconds": 20,
                    "num_model_requests": 1,
                    "project_id": null,
                    "user_id": null,
                    "api_key_id": null,
                    "model": null
                }
            ]
        }
    ],
    "has_more": false,
    "next_page": null
}


<!-- Page 366 -->
## Returns
The aggregated audio transcriptions usage details of the specific time
bucket.
A cursor for use in pagination. Corresponding to the next_page  field from the
previous response.
page string
## Optional
Return only usage for these projects.
project_ids array
## Optional
Return only usage for these users.
user_ids array
## Optional
A list of paginated, time bucketed Audio transcriptions usage objects.
## Audio transcriptions usage object
When group_by=api_key_id , this field provides the API key ID of the grouped usage
result.
api_key_id string or null
When group_by=model , this field provides the model name of the grouped usage
result.
model string or null
num_model_requests integer
## OBJECT Audio transcriptions usage object
 
 
{
    "object": "organization.usage.audio_transcrip
    "seconds": 10,
    "num_model_requests": 1,
    "project_id": "proj_abc",
    "user_id": "user-abc",
    "api_key_id": "key_abc",
    "model": "tts-1"
}


<!-- Page 367 -->
GET https://api.openai.com/v1/organization/usage/vector_stores
Get vector stores usage details for the organization.
## Query parameters
The count of requests made to the model.
object string
When group_by=project_id , this field provides the project ID of the grouped usage
result.
project_id string or null
The number of seconds processed.
seconds integer
When group_by=user_id , this field provides the user ID of the grouped usage result.
user_id string or null
## Vector stores
Start time (Unix seconds) of the query time range, inclusive.
start_time integer
## Required
Width of each time bucket in response. Currently 1m , 1h  and 1d  are supported,
default to 1d .
bucket_width string
## Optional
Defaults to 1d
end_time integer
## Optional
Example request
curl
 
 
curl "https://api.openai.com/v1/organization/usag
-H "Authorization: Bearer $OPENAI_ADMIN_KEY" \
-H "Content-Type: application/json"

## Response
 
{
    "object": "page",
    "data": [
        {
            "object": "bucket",
            "start_time": 1730419200,
            "end_time": 1730505600,


<!-- Page 368 -->
## Returns
The aggregated vector stores usage details of the specific time bucket.
End time (Unix seconds) of the query time range, exclusive.
Group the usage data by the specified fields. Support fields include project_id .
group_by array
## Optional
Specifies the number of buckets to return.
limit integer
## Optional
bucket_width=1d : default: 7, max: 31
bucket_width=1h : default: 24, max: 168
bucket_width=1m : default: 60, max: 1440
A cursor for use in pagination. Corresponding to the next_page  field from the
previous response.
page string
## Optional
Return only usage for these projects.
project_ids array
## Optional
A list of paginated, time bucketed Vector stores usage objects.
 
            "results": [
                {
                    "object": "organization.usa
                    "usage_bytes": 1024,
                    "project_id": null
                }
            ]
        }
    ],
    "has_more": false,
    "next_page": null
}

## Vector stores usage object
OBJECT Vector stores usage object

<!-- Page 369 -->
GET https://api.openai.com/v1/organization/usage/code_interpreter_sessions
Get code interpreter sessions usage details for the organization.
## Query parameters object
string
When group_by=project_id , this field provides the project ID of the grouped usage
result.
project_id string or null
The vector stores usage in bytes.
usage_bytes integer
{
    "object": "organization.usage.vector_stores.r
    "usage_bytes": 1024,
    "project id": "proj abc"

## Code interpreter sessions
Start time (Unix seconds) of the query time range, inclusive.
start_time integer
## Required
Width of each time bucket in response. Currently 1m , 1h  and 1d  are supported,
default to 1d .
bucket_width string
## Optional
Defaults to 1d
End time (Unix seconds) of the query time range, exclusive.
end_time integer
## Optional
Group the usage data by the specified fields. Support fields include project_id .
group_by array
## Optional
Example request
curl
 
 
curl "https://api.openai.com/v1/organization/usag
-H "Authorization: Bearer $OPENAI_ADMIN_KEY" \
-H "Content-Type: application/json"

## Response
 
{
    "object": "page",
    "data": [
        {
            "object": "bucket",
            "start_time": 1730419200,
            "end_time": 1730505600,
            "results": [
                {
                    "object": "organization.usa
                    "num_sessions": 1,
                    "project_id": null


<!-- Page 370 -->
## Returns
The aggregated code interpreter sessions usage details of the specific time
bucket.
Specifies the number of buckets to return.
limit integer
## Optional
bucket_width=1d : default: 7, max: 31
bucket_width=1h : default: 24, max: 168
bucket_width=1m : default: 60, max: 1440
A cursor for use in pagination. Corresponding to the next_page  field from the
previous response.
page string
## Optional
Return only usage for these projects.
project_ids array
## Optional
A list of paginated, time bucketed Code interpreter sessions usage objects.
 
                }
            ]
        }
    ],
    "has_more": false,
    "next_page": null
}

## Code interpreter sessions usage object
The number of code interpreter sessions.
num_sessions integer
object string
## OBJECT Code interpreter sessions usage object
 
 
{
    "object": "organization.usage.code_interprete
    "num_sessions": 1,
    "project_id": "proj_abc"
}


<!-- Page 371 -->
GET https://api.openai.com/v1/organization/costs
Get costs details for the organization.
## Query parameters
When group_by=project_id , this field provides the project ID of the grouped usage
result.
project_id string or null
## Costs
Start time (Unix seconds) of the query time range, inclusive.
start_time integer
## Required
Width of each time bucket in response. Currently only 1d  is supported, default to
1d .
bucket_width string
## Optional
Defaults to 1d
End time (Unix seconds) of the query time range, exclusive.
end_time integer
## Optional
Group the costs by the specified fields. Support fields include project_id ,
line_item  and any combination of them.
group_by array
## Optional
A limit on the number of buckets to be returned. Limit can range between 1 and 180,
and the default is 7.
limit integer
## Optional
Defaults to 7
## Example request
curl
 
 
curl "https://api.openai.com/v1/organization/cost
-H "Authorization: Bearer $OPENAI_ADMIN_KEY" \
-H "Content-Type: application/json"

## Response
 
{
    "object": "page",
    "data": [
        {
            "object": "bucket",
            "start_time": 1730419200,
            "end_time": 1730505600,
            "results": [
                {
                    "object": "organization.cos
                    "amount": {
                        "value": 0.06,
                        "currency": "usd"
                    },
                    "line_item": null,
                    "project_id": null
                }
            ]
        }


<!-- Page 372 -->
## Returns
The aggregated costs details of the specific time bucket.
A cursor for use in pagination. Corresponding to the next_page  field from the
previous response.
page string
## Optional
Return only costs for these projects.
project_ids array
## Optional
A list of paginated, time bucketed Costs objects.
 
    ],
    "has_more": false,
    "next_page": null
}

## Costs object
The monetary value in its associated currency.
## Show properties
amount object
When group_by=line_item , this field provides the line item of the grouped costs
result.
line_item string or null object
string
project_id string or null
## OBJECT Costs object
{
    "object": "organization.costs.result",
    "amount": {
      "value": 0.06,
      "currency": "usd"
    },
    "line_item": "Image models",
    "project_id": "proj_abc"
}


<!-- Page 373 -->
Manage Mutual TLS certificates across your organization and projects.
Learn more about Mutual TLS.
POST https://api.openai.com/v1/organization/certificates
Upload a certificate to the organization. This does not automatically
activate the certificate.
Organizations can upload up to 50 certificates.
## Request body
Returns
When group_by=project_id , this field provides the project ID of the grouped costs
result.
## Certificates
Beta
Upload certificate
The certificate content in PEM format
content string
Required
An optional name for the certificate
name string
Optional
Example request
curl
 
 
curl -X POST https://api.openai.com/v1/organizati
-H "Authorization: Bearer $OPENAI_ADMIN_KEY" \
-H "Content-Type: application/json" \
-d '{
  "name": "My Example Certificate",
  "certificate": "-----BEGIN CERTIFICATE-----\\nM
}'

## Response
 
{
  "object": "certificate",
  "id": "cert_abc",
  "name": "My Example Certificate",
  "created_at": 1234567,
  "certificate_details": {


<!-- Page 374 -->
GET https://api.openai.com/v1/organization/certificates/{certificate_id}
Get a certificate that has been uploaded to the organization.
You can get a certificate regardless of whether it is active or not.
## Path parameters
Query parameters
Returns
A single Certificate object.
 
    "valid_at": 12345667,
    "expires_at": 12345678
  }
}

## Get certificate
Unique ID of the certificate to retrieve.
certificate_id string
## Required
A list of additional fields to include in the response. Currently the only supported value
is content  to fetch the PEM content of the certificate.
include array
## Optional
A single Certificate object.
## Example request
curl
 
 
curl "https://api.openai.com/v1/organization/cert
-H "Authorization: Bearer $OPENAI_ADMIN_KEY"

## Response
 
 
{
  "object": "certificate",
  "id": "cert_abc",
  "name": "My Example Certificate",
  "created_at": 1234567,
  "certificate_details": {
    "valid_at": 1234567,
    "expires_at": 12345678,
    "content": "-----BEGIN CERTIFICATE-----MIIDe
  }
}


<!-- Page 375 -->
POST https://api.openai.com/v1/organization/certificates/{certificate_id}
Modify a certificate. Note that only the name can be modified.
## Request body
Returns
DELETE https://api.openai.com/v1/organization/certificates/{certificate_i
d}
Delete a certificate from the organization.
The certificate must be inactive for the organization and all projects.
## Modify certificate
The updated name for the certificate
name string
Required
The updated Certificate object.
## Example request
curl
 
 
curl -X POST https://api.openai.com/v1/organizati
-H "Authorization: Bearer $OPENAI_ADMIN_KEY" \
-H "Content-Type: application/json" \
-d '{
  "name": "Renamed Certificate"
}'

## Response
{
  "object": "certificate",
  "id": "cert_abc",
  "name": "Renamed Certificate",
  "created_at": 1234567,
  "certificate_details": {
    "valid_at": 12345667,
    "expires_at": 12345678
  }
}

## Delete certificate
Example request
curl
 
 
curl -X DELETE https://api.openai.com/v1/organiza
-H "Authorization: Bearer $OPENAI_ADMIN_KEY"


<!-- Page 376 -->
## Returns
GET https://api.openai.com/v1/organization/certificates
List uploaded certificates for this organization.
## Query parameters
Returns
A confirmation object indicating the certificate was deleted.
## Response
{
  "object": "certificate.deleted",
  "id": "cert_abc"

## List organization certificates
A cursor for use in pagination. after  is an object ID that defines your place in the list.
For instance, if you make a list request and receive 100 objects, ending with obj_foo,
your subsequent call can include after=obj_foo in order to fetch the next page of the
list.
after string
## Optional
A limit on the number of objects to be returned. Limit can range between 1 and 100,
and the default is 20.
limit integer
## Optional
Defaults to 20
Sort order by the created_at  timestamp of the objects. asc  for ascending order
and desc  for descending order.
order string
## Optional
Defaults to desc
Example request
curl
 
 
curl https://api.openai.com/v1/organization/certi
-H "Authorization: Bearer $OPENAI_ADMIN_KEY"

## Response
 
 
{
  "object": "list",
  "data": [
    {
      "object": "organization.certificate",
      "id": "cert_abc",
      "name": "My Example Certificate",
      "active": true,
      "created_at": 1234567,
      "certificate_details": {
        "valid_at": 12345667,
        "expires_at": 12345678
      }
    },
  ],
  "first_id": "cert_abc",
  "last_id": "cert_abc",


<!-- Page 377 -->
GET https://api.openai.com/v1/organization/projects/{project_id}/certifica
tes
List certificates for this project.
## Path parameters
Query parameters
A list of Certificate objects.
## List project certificates
The ID of the project.
project_id string
## Required
A cursor for use in pagination. after  is an object ID that defines your place in the list.
For instance, if you make a list request and receive 100 objects, ending with obj_foo,
your subsequent call can include after=obj_foo in order to fetch the next page of the
list.
after string
## Optional
A limit on the number of objects to be returned. Limit can range between 1 and 100,
and the default is 20.
limit integer
## Optional
Defaults to 20
order string
## Optional
Defaults to desc
Example request
curl
 
 
curl https://api.openai.com/v1/organization/proje
-H "Authorization: Bearer $OPENAI_ADMIN_KEY"

## Response
 
 
{
  "object": "list",
  "data": [
    {
      "object": "organization.project.certificat
      "id": "cert_abc",
      "name": "My Example Certificate",
      "active": true,
      "created_at": 1234567,
      "certificate_details": {
        "valid_at": 12345667,
        "expires_at": 12345678
      }
    },
  ],
  "first_id": "cert_abc",
  "last_id": "cert_abc",
  "has_more": false
}


<!-- Page 378 -->
## Returns
POST https://api.openai.com/v1/organization/certificates/activate
Activate certificates at the organization level.
You can atomically and idempotently activate up to 10 certificates at a time.
## Request body
Returns
Sort order by the created_at  timestamp of the objects. asc  for ascending order
and desc  for descending order.
A list of Certificate objects.
## Activate certificates for organization
certificate_ids array
## Required
A list of Certificate objects that were activated.
## Example request
curl
 
 
curl https://api.openai.com/v1/organization/certi
-H "Authorization: Bearer $OPENAI_ADMIN_KEY" \
-H "Content-Type: application/json" \
-d '{
  "data": ["cert_abc", "cert_def"]
}'

## Response
 
{
  "object": "organization.certificate.activatio
  "data": [
    {
      "object": "organization.certificate",
      "id": "cert_abc",
      "name": "My Example Certificate",
      "active": true,
      "created_at": 1234567,
      "certificate_details": {
        "valid_at": 12345667,


<!-- Page 379 -->
POST https://api.openai.com/v1/organization/certificates/deactivate
Deactivate certificates at the organization level.
You can atomically and idempotently deactivate up to 10 certificates at a
time.
## Request body
Returns
 
        "expires_at": 12345678
      }
    },
    {
      "object": "organization.certificate",
      "id": "cert_def",
      "name": "My Example Certificate 2",
      "active": true,
      "created_at": 1234567,
      "certificate_details": {
        "valid_at": 12345667,
        "expires_at": 12345678
      }
    },
  ],
}

## Deactivate certificates for organization
certificate_ids array
## Required
A list of Certificate objects that were deactivated.
## Example request
curl
 
 
curl https://api.openai.com/v1/organization/certi
-H "Authorization: Bearer $OPENAI_ADMIN_KEY" \
-H "Content-Type: application/json" \
-d '{
  "data": ["cert_abc", "cert_def"]
}'

## Response
 
{
  "object": "organization.certificate.deactivat
  "data": [
    {
      "object": "organization.certificate",
      "id": "cert_abc",
      "name": "My Example Certificate",
      "active": false,


<!-- Page 380 -->
POST https://api.openai.com/v1/organization/projects/{project_id}/certific
ates/activate
Activate certificates at the project level.
You can atomically and idempotently activate up to 10 certificates at a time.
## Path parameters
Request body
 
      "created_at": 1234567,
      "certificate_details": {
        "valid_at": 12345667,
        "expires_at": 12345678
      }
    },
    {
      "object": "organization.certificate",
      "id": "cert_def",
      "name": "My Example Certificate 2",
      "active": false,
      "created_at": 1234567,
      "certificate_details": {
        "valid_at": 12345667,
        "expires_at": 12345678
      }
    },
  ],
}

## Activate certificates for project
The ID of the project.
project_id string
## Required
certificate_ids array
## Required
Example request
curl
 
 
curl https://api.openai.com/v1/organization/proje
-H "Authorization: Bearer $OPENAI_ADMIN_KEY" \
-H "Content-Type: application/json" \
-d '{
  "data": ["cert_abc", "cert_def"]
}'

## Response
 
{
  "object": "organization.project.certificate.a
  "data": [
    {
      "object": "organization.project.certifica
      "id": "cert_abc",


<!-- Page 381 -->
## Returns
POST https://api.openai.com/v1/organization/projects/{project_id}/certific
ates/deactivate
Deactivate certificates at the project level. You can atomically and
idempotently deactivate up to 10 certificates at a time.
## Path parameters
Request body
A list of Certificate objects that were activated.
 
      "name": "My Example Certificate",
      "active": true,
      "created_at": 1234567,
      "certificate_details": {
        "valid_at": 12345667,
        "expires_at": 12345678
      }
    },
    {
      "object": "organization.project.certifica
      "id": "cert_def",
      "name": "My Example Certificate 2",
      "active": true,
      "created_at": 1234567,
      "certificate_details": {
        "valid_at": 12345667,
        "expires_at": 12345678
      }
    },
  ],
}

## Deactivate certificates for project
The ID of the project.
project_id string
## Required
Example request
curl
 
 
curl https://api.openai.com/v1/organization/proje
-H "Authorization: Bearer $OPENAI_ADMIN_KEY" \
-H "Content-Type: application/json" \
-d '{
  "data": ["cert_abc", "cert_def"]
}'

## Response
 
{
  "object": "organization.project.certificate.d
  "data": [


<!-- Page 382 -->
## Returns
Represents an individual certificate  uploaded to the organization.
certificate_ids array
## Required
A list of Certificate objects that were deactivated.
 
    {
      "object": "organization.project.certifica
      "id": "cert_abc",
      "name": "My Example Certificate",
      "active": false,
      "created_at": 1234567,
      "certificate_details": {
        "valid_at": 12345667,
        "expires_at": 12345678
      }
    },
    {
      "object": "organization.project.certifica
      "id": "cert_def",
      "name": "My Example Certificate 2",
      "active": false,
      "created_at": 1234567,
      "certificate_details": {
        "valid_at": 12345667,
        "expires_at": 12345678
      }
    },
  ],
}

## The certificate object
Whether the certificate is currently active at the specified scope. Not returned when
getting details for a specific certificate.
active boolean
## Show properties
certificate_details object
created_at integer
## OBJECT The certificate object
 
 
{
  "object": "certificate",
  "id": "cert_abc",
  "name": "My Certificate",
  "created_at": 1234567,
  "certificate_details": {
    "valid_at": 1234567,
    "expires_at": 12345678,
    "content": "-----BEGIN CERTIFICATE----- MII


<!-- Page 383 -->
Given a prompt, the model will return one or more predicted completions along with the probabilities of
alternative tokens at each position. Most developer should use our Chat Completions API to leverage our
best and newest models.
POST https://api.openai.com/v1/completions
The Unix timestamp (in seconds) of when the certificate was uploaded.
The identifier, which can be referenced in API endpoints
id string
The name of the certificate.
name string
The object type.
object string
If creating, updating, or getting a specific certificate, the object type is
certificate .
If listing, activating, or deactivating certificates for the organization, the object
type is organization.certificate .
If listing, activating, or deactivating certificates for a project, the object type is
organization.project.certificate .
 
## Completions
Legacy
Create completion
Legacy
No streaming
Streaming

<!-- Page 384 -->
Creates a completion for the provided prompt and parameters.
## Request body
ID of the model to use. You can use the List models API to see all of your available
models, or see our Model overview for descriptions of them.
model string
## Required
The prompt(s) to generate completions for, encoded as a string, array of strings, array
of tokens, or array of token arrays.
Note that <|endoftext|> is the document separator that the model sees during training,
so if a prompt is not specified the model will generate as if from the beginning of a new
document.
prompt string or array
## Required
Generates best_of  completions server-side and returns the "best" (the one with the
highest log probability per token). Results cannot be streamed.
When used with n , best_of  controls the number of candidate completions and
n  specifies how many to return  best_of  must be greater than n .
Note: Because this parameter generates many completions, it can quickly consume
your token quota. Use carefully and ensure that you have reasonable settings for
max_tokens  and stop .
best_of integer or null
## Optional
Defaults to 1
## Echo back the prompt in addition to the completion
echo boolean or null
Optional
Defaults to false
Number between -2.0 and 2.0. Positive values penalize new tokens based on their
existing frequency in the text so far, decreasing the model's likelihood to repeat the
same line verbatim.
See more information about frequency and presence penalties.
frequency_penalty number or null
## Optional
Defaults to 0
Example r...
gpt-3.5-turbo-instruct
python
from openai import OpenAI
client = OpenAI()
client.completions.create(
  model="gpt-3.5-turbo-instruct",
  prompt="Say this is a test",
  max_tokens=7,
  temperature=0
)

## Response
{
  "id": "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7",
  "object": "text_completion",
  "created": 1589478378,
  "model": "gpt-3.5-turbo-instruct",
  "system_fingerprint": "fp_44709d6fcb",
  "choices": [
    {
      "text": "\n\nThis is indeed a test",
      "index": 0,
      "logprobs": null,
      "finish_reason": "length"
    }
  ],
  "usage": {
    "prompt_tokens": 5,
    "completion_tokens": 7,
    "total_tokens": 12
  }
}


<!-- Page 385 -->
Modify the likelihood of specified tokens appearing in the completion.
Accepts a JSON object that maps tokens (specified by their token ID in the GPT
tokenizer) to an associated bias value from -100 to 100. You can use this tokenizer tool
to convert text to token IDs. Mathematically, the bias is added to the logits generated
by the model prior to sampling. The exact effect will vary per model, but values
between -1 and 1 should decrease or increase likelihood of selection; values like -100 or
100 should result in a ban or exclusive selection of the relevant token.
As an example, you can pass {"50256": -100}  to prevent the <|endoftext|> token
from being generated.
logit_bias
map
## Optional
Defaults to null
Include the log probabilities on the logprobs  most likely output tokens, as well the
chosen tokens. For example, if logprobs  is 5, the API will return a list of the 5 most
likely tokens. The API will always return the logprob  of the sampled token, so there
may be up to logprobs+1  elements in the response.
The maximum value for logprobs  is 5.
logprobs integer or null
## Optional
Defaults to null
The maximum number of tokens that can be generated in the completion.
The token count of your prompt plus max_tokens  cannot exceed the model's context
length. Example Python code for counting tokens.
max_tokens integer or null
## Optional
Defaults to 16
How many completions to generate for each prompt.
Note: Because this parameter generates many completions, it can quickly consume
your token quota. Use carefully and ensure that you have reasonable settings for
max_tokens  and stop .
n integer or null
## Optional
Defaults to 1
Number between -2.0 and 2.0. Positive values penalize new tokens based on whether
they appear in the text so far, increasing the model's likelihood to talk about new topics.
presence_penalty number or null
## Optional
Defaults to 0

<!-- Page 386 -->
See more information about frequency and presence penalties.
If specified, our system will make a best effort to sample deterministically, such that
repeated requests with the same seed  and parameters should return the same
result.
Determinism is not guaranteed, and you should refer to the system_fingerprint
response parameter to monitor changes in the backend.
seed integer or null
## Optional
Not supported with latest reasoning models o3  and o4-mini .
Up to 4 sequences where the API will stop generating further tokens. The returned text
will not contain the stop sequence.
stop string / array / null
## Optional
Defaults to null
Whether to stream back partial progress. If set, tokens will be sent as data-only
server-sent events as they become available, with the stream terminated by a
data: [DONE]  message. Example Python code.
stream boolean or null
## Optional
Defaults to false
Options for streaming response. Only set this when you set stream: true .
## Show properties
stream_options object or null
## Optional
Defaults to null
The suffix that comes after a completion of inserted text.
This parameter is only supported for gpt-3.5-turbo-instruct .
suffix string or null
## Optional
Defaults to null
What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make
the output more random, while lower values like 0.2 will make it more focused and
deterministic.
We generally recommend altering this or top_p  but not both.
temperature number or null
## Optional
Defaults to 1

<!-- Page 387 -->
## Returns
Represents a completion response from the API. Note: both the streamed
and non-streamed response objects share the same shape (unlike the chat
endpoint).
An alternative to sampling with temperature, called nucleus sampling, where the model
considers the results of the tokens with top_p probability mass. So 0.1 means only the
tokens comprising the top 10% probability mass are considered.
We generally recommend altering this or temperature  but not both.
top_p number or null
## Optional
Defaults to 1
A unique identifier representing your end-user, which can help OpenAI to monitor and
detect abuse. Learn more.
user string
## Optional
Returns a completion object, or a sequence of completion objects if the request is
streamed.
## The completion object
Legacy
The list of completion choices the model generated for the input prompt.
## Show properties
choices array
The Unix timestamp (in seconds) of when the completion was created.
created integer
## OBJECT The completion object
 
{
  "id": "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7",
  "object": "text_completion",
  "created": 1589478378,
  "model": "gpt-4-turbo",
  "choices": [
    {
      "text": "\n\nThis is indeed a test",
      "index": 0,
      "logprobs": null,
      "finish_reason": "length"


<!-- Page 388 -->
A unique identifier for the completion.
id string
The model used for completion.
model string
The object type, which is always "text_completion"
object string
This fingerprint represents the backend configuration that the model runs with.
## Can be used in conjunction with the seed  request parameter to understand when
backend changes have been made that might impact determinism.
system_fingerprint string
Usage statistics for the completion request.
## Show properties
usage object
 
    }
  ],
  "usage": {
    "prompt_tokens": 5,
    "completion_tokens": 7,
    "total_tokens": 12
  }
}

