"""
AWS Bedrock Provider Implementation

Supports multiple authentication methods:
- IAM credentials (access key + secret key + region)
- AWS profile-based authentication
- Cross-region inference
- Global inference (inference profiles)

Supported Bedrock services:
- Bedrock Runtime (InvokeModel)
- Bedrock Converse API
- Cross-region inference
- Inference profiles
"""

from typing import Dict, Any, Optional, List, Callable
from models import ChatCompletionRequest, Tool
from .base import BaseProvider
import json
import logging

logger = logging.getLogger(__name__)

try:
    import boto3
    from botocore.auth import SigV4Auth
    from botocore.awsrequest import AWSRequest
    from botocore.exceptions import ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    logger.warning(
        "boto3 is not installed. Install it with: pip install boto3>=1.28.0"
    )


class BedrockProvider(BaseProvider):
    """
    Provider implementation for AWS Bedrock.

    Supports multiple authentication and inference modes:
    - IAM credentials (access_key, secret_key, region)
    - AWS profile authentication
    - Cross-region inference
    - Global inference (inference profiles)
    - API key authentication (via custom Bedrock endpoints)
    """

    # Bedrock model ID mappings for common models
    MODEL_IDS = {
        # Anthropic Claude models
        "claude-3-5-sonnet": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "claude-3-5-haiku": "anthropic.claude-3-5-haiku-20241022-v1:0",
        "claude-3-opus": "anthropic.claude-3-opus-20240229-v1:0",
        "claude-3-sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
        "claude-3-haiku": "anthropic.claude-3-haiku-20240307-v1:0",
        "claude-instant": "anthropic.claude-instant-v1",
        "claude-2.1": "anthropic.claude-v2:1",
        "claude-2": "anthropic.claude-v2",

        # Meta Llama models
        "llama3-2-1b": "us.meta.llama3-2-1b-instruct-v1:0",
        "llama3-2-3b": "us.meta.llama3-2-3b-instruct-v1:0",
        "llama3-2-11b": "us.meta.llama3-2-11b-instruct-v1:0",
        "llama3-2-90b": "us.meta.llama3-2-90b-instruct-v1:0",
        "llama3-1-8b": "meta.llama3-1-8b-instruct-v1:0",
        "llama3-1-70b": "meta.llama3-1-70b-instruct-v1:0",
        "llama3-1-405b": "meta.llama3-1-405b-instruct-v1:0",
        "llama3-8b": "meta.llama3-8b-instruct-v1:0",
        "llama3-70b": "meta.llama3-70b-instruct-v1:0",
        "llama2-13b": "meta.llama2-13b-chat-v1",
        "llama2-70b": "meta.llama2-70b-chat-v1",

        # Amazon Titan models
        "titan-text-express": "amazon.titan-text-express-v1",
        "titan-text-lite": "amazon.titan-text-lite-v1",
        "titan-text-premier": "amazon.titan-text-premier-v1:0",
        "titan-embed-text": "amazon.titan-embed-text-v1",
        "titan-embed-text-v2": "amazon.titan-embed-text-v2:0",
        "titan-embed-image": "amazon.titan-embed-image-v1",
        "titan-image-generator": "amazon.titan-image-generator-v1",

        # Mistral AI models
        "mistral-7b": "mistral.mistral-7b-instruct-v0:2",
        "mixtral-8x7b": "mistral.mixtral-8x7b-instruct-v0:1",
        "mistral-large": "mistral.mistral-large-2402-v1:0",
        "mistral-large-2407": "mistral.mistral-large-2407-v1:0",
        "mistral-small": "mistral.mistral-small-2402-v1:0",

        # Cohere models
        "cohere-command-text": "cohere.command-text-v14",
        "cohere-command-light-text": "cohere.command-light-text-v14",
        "cohere-command-r": "cohere.command-r-v1:0",
        "cohere-command-r-plus": "cohere.command-r-plus-v1:0",
        "cohere-embed-english": "cohere.embed-english-v3",
        "cohere-embed-multilingual": "cohere.embed-multilingual-v3",

        # AI21 Labs models
        "j2-ultra": "ai21.j2-ultra-v1",
        "j2-mid": "ai21.j2-mid-v1",
        "jamba-instruct": "ai21.jamba-instruct-v1:0",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        region: str = "us-east-1",
        profile_name: Optional[str] = None,
        use_cross_region: bool = False,
        use_global_inference: bool = False,
        inference_profile_id: Optional[str] = None,
        session_token: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the Bedrock provider.

        Authentication methods (in order of precedence):
        1. API key (for custom Bedrock endpoints)
        2. Explicit IAM credentials (access_key + secret_key)
        3. AWS profile
        4. Default AWS credential chain (environment, IAM role, etc.)

        Args:
            api_key: API key for custom Bedrock endpoints (optional)
            access_key: AWS access key ID (optional)
            secret_key: AWS secret access key (optional)
            region: AWS region (default: us-east-1)
            profile_name: AWS profile name from ~/.aws/credentials (optional)
            use_cross_region: Enable cross-region inference (default: False)
            use_global_inference: Use global inference profiles (default: False)
            inference_profile_id: Specific inference profile ID (optional)
            session_token: AWS session token for temporary credentials (optional)
            endpoint_url: Custom endpoint URL (optional)
            **kwargs: Additional boto3 client parameters
        """
        if not BOTO3_AVAILABLE:
            raise ImportError(
                "boto3 is required for Bedrock provider. "
                "Install it with: pip install boto3>=1.28.0"
            )

        super().__init__(api_key or "bedrock")  # BaseProvider requires api_key

        self.region = region
        self.use_cross_region = use_cross_region
        self.use_global_inference = use_global_inference
        self.inference_profile_id = inference_profile_id
        self.endpoint_url = endpoint_url
        self.api_key_mode = api_key is not None

        # Initialize boto3 session and client
        try:
            # Create session based on authentication method
            if profile_name:
                logger.info(f"Using AWS profile: {profile_name}")
                self.session = boto3.Session(
                    profile_name=profile_name,
                    region_name=region
                )
            elif access_key and secret_key:
                logger.info(f"Using IAM credentials for region: {region}")
                self.session = boto3.Session(
                    aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key,
                    aws_session_token=session_token,
                    region_name=region
                )
            else:
                logger.info(f"Using default AWS credential chain for region: {region}")
                self.session = boto3.Session(region_name=region)

            # Create Bedrock Runtime client
            client_kwargs = {
                'region_name': region,
                **kwargs
            }

            if endpoint_url:
                client_kwargs['endpoint_url'] = endpoint_url

            self.bedrock_runtime = self.session.client(
                'bedrock-runtime',
                **client_kwargs
            )

            # Create standard Bedrock client for listing models, etc.
            self.bedrock = self.session.client('bedrock', **client_kwargs)

            logger.info("Bedrock provider initialized successfully")

        except NoCredentialsError:
            logger.error("No AWS credentials found")
            raise ValueError(
                "AWS credentials not found. Please provide access_key/secret_key, "
                "set up AWS profile, or configure AWS credentials."
            )
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock provider: {e}")
            raise

        # Set up headers for API key mode (custom endpoints)
        self.headers = {
            "Content-Type": "application/json"
        }

        if self.api_key_mode and api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def _get_model_id(self, model: str) -> str:
        """
        Convert friendly model name to Bedrock model ID.

        Args:
            model: Model name (friendly name or full model ID)

        Returns:
            Full Bedrock model ID
        """
        # If it's already a full model ID (contains dots), return as-is
        if '.' in model or ':' in model:
            return model

        # Otherwise, try to map from friendly name
        return self.MODEL_IDS.get(model.lower(), model)

    def _format_messages_for_converse(
        self,
        messages: List[Dict[str, Any]]
    ) -> tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Format messages for Bedrock Converse API.

        Args:
            messages: List of message dictionaries

        Returns:
            Tuple of (formatted_messages, system_prompt)
        """
        system_prompt = None
        formatted_messages = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Extract system messages
            if role == "system":
                if system_prompt is None:
                    system_prompt = content
                else:
                    system_prompt += f"\n\n{content}"
            else:
                # Convert role to Bedrock format
                bedrock_role = "user" if role in ["user", "system"] else "assistant"
                formatted_messages.append({
                    "role": bedrock_role,
                    "content": [{"text": content}]
                })

        return formatted_messages, system_prompt

    def _format_tools_for_converse(
        self,
        tools: Optional[List[Dict[str, Any]]]
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Format tools for Bedrock Converse API.

        Args:
            tools: List of tool dictionaries

        Returns:
            Formatted tools for Bedrock
        """
        if not tools:
            return None

        formatted_tools = []
        for tool in tools:
            formatted_tool = {
                "toolSpec": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "inputSchema": {
                        "json": tool.get("parameters", {})
                    }
                }
            }
            formatted_tools.append(formatted_tool)

        return formatted_tools

    def prepare_request(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        """
        Prepare the request payload for Bedrock API.

        Args:
            request: ChatCompletionRequest object

        Returns:
            Prepared request payload
        """
        payload = request.model_dump(exclude_none=True)

        # Get the full model ID
        model_id = self._get_model_id(payload.get("model", "claude-3-5-sonnet"))

        # Format messages and extract system prompt
        messages, system_prompt = self._format_messages_for_converse(
            payload.get("messages", [])
        )

        # Format tools if present
        tools = self._format_tools_for_converse(payload.get("tools"))

        # Build inference configuration
        inference_config = {}
        if "temperature" in payload:
            inference_config["temperature"] = payload["temperature"]
        if "max_tokens" in payload:
            inference_config["maxTokens"] = payload["max_tokens"]
        if "top_p" in payload:
            inference_config["topP"] = payload["top_p"]

        # Build the Converse API payload
        converse_payload = {
            "modelId": model_id,
            "messages": messages,
        }

        if system_prompt:
            converse_payload["system"] = [{"text": system_prompt}]

        if tools:
            converse_payload["toolConfig"] = {"tools": tools}

        if inference_config:
            converse_payload["inferenceConfig"] = inference_config

        # Add additional model-specific parameters
        if "stop" in payload or "stop_sequences" in payload:
            stop_sequences = payload.get("stop") or payload.get("stop_sequences", [])
            if isinstance(stop_sequences, str):
                stop_sequences = [stop_sequences]
            if stop_sequences:
                converse_payload["additionalModelRequestFields"] = {
                    "stop_sequences": stop_sequences
                }

        return converse_payload

    def chat_completion(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        """
        Create a chat completion using Bedrock Converse API.

        Args:
            request: ChatCompletionRequest object

        Returns:
            Response from Bedrock
        """
        payload = self.prepare_request(request)

        try:
            # Use cross-region inference if enabled
            if self.use_cross_region:
                logger.info("Using cross-region inference")
                response = self.bedrock_runtime.converse_stream(**payload)
            # Use inference profile if specified
            elif self.use_global_inference or self.inference_profile_id:
                profile_id = self.inference_profile_id or "default"
                logger.info(f"Using inference profile: {profile_id}")
                payload["modelId"] = profile_id
                response = self.bedrock_runtime.converse(**payload)
            else:
                # Standard Converse API call
                response = self.bedrock_runtime.converse(**payload)

            return response

        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            logger.error(f"Bedrock API error ({error_code}): {error_message}")
            raise RuntimeError(f"Bedrock error: {error_message}") from e
        except Exception as e:
            logger.error(f"Unexpected error calling Bedrock: {e}")
            raise

    def invoke_model(
        self,
        model_id: str,
        body: Dict[str, Any],
        accept: str = "application/json",
        content_type: str = "application/json"
    ) -> Dict[str, Any]:
        """
        Invoke a Bedrock model directly using InvokeModel API.

        This provides lower-level access for models that don't support Converse API.

        Args:
            model_id: The Bedrock model ID
            body: Request body (model-specific format)
            accept: Accept header (default: application/json)
            content_type: Content-Type header (default: application/json)

        Returns:
            Response from the model
        """
        try:
            response = self.bedrock_runtime.invoke_model(
                modelId=model_id,
                body=json.dumps(body),
                accept=accept,
                contentType=content_type
            )

            response_body = json.loads(response['body'].read())
            return response_body

        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            logger.error(f"Bedrock InvokeModel error ({error_code}): {error_message}")
            raise RuntimeError(f"Bedrock error: {error_message}") from e

    def list_foundation_models(
        self,
        by_provider: Optional[str] = None,
        by_output_modality: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List available foundation models in Bedrock.

        Args:
            by_provider: Filter by provider (e.g., "Anthropic", "Meta", "Amazon")
            by_output_modality: Filter by output modality (e.g., "TEXT", "IMAGE")

        Returns:
            List of model summaries
        """
        try:
            params = {}
            if by_provider:
                params['byProvider'] = by_provider
            if by_output_modality:
                params['byOutputModality'] = by_output_modality

            response = self.bedrock.list_foundation_models(**params)
            return response.get('modelSummaries', [])

        except ClientError as e:
            logger.error(f"Error listing models: {e}")
            return []

    def get_inference_profiles(self) -> List[Dict[str, Any]]:
        """
        List available inference profiles for global/cross-region inference.

        Returns:
            List of inference profile summaries
        """
        try:
            response = self.bedrock.list_inference_profiles()
            return response.get('inferenceProfileSummaries', [])
        except ClientError as e:
            logger.error(f"Error listing inference profiles: {e}")
            return []

    def get_chat_completion_endpoint(self) -> str:
        """
        Get the chat completion endpoint.

        Note: This is primarily for compatibility with the base provider pattern.
        Bedrock uses boto3 SDK calls rather than direct HTTP requests.
        """
        if self.endpoint_url:
            return self.endpoint_url
        return f"https://bedrock-runtime.{self.region}.amazonaws.com"

    def process_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the response from Bedrock Converse API.

        Args:
            response: Raw response from Bedrock

        Returns:
            Processed response in standardized format
        """
        try:
            # Extract the output from Converse API response
            if "output" in response:
                output = response["output"]
                message = output.get("message", {})

                # Build standardized response
                processed = {
                    "choices": [{
                        "message": {
                            "role": message.get("role", "assistant"),
                            "content": ""
                        },
                        "finish_reason": response.get("stopReason", "stop")
                    }]
                }

                # Extract text content
                content_blocks = message.get("content", [])
                text_parts = []
                tool_calls = []

                for block in content_blocks:
                    if "text" in block:
                        text_parts.append(block["text"])
                    elif "toolUse" in block:
                        tool_use = block["toolUse"]
                        tool_calls.append({
                            "id": tool_use.get("toolUseId", ""),
                            "type": "function",
                            "function": {
                                "name": tool_use.get("name", ""),
                                "arguments": json.dumps(tool_use.get("input", {}))
                            }
                        })

                # Set content
                processed["choices"][0]["message"]["content"] = "\n".join(text_parts)

                # Add tool calls if present
                if tool_calls:
                    processed["choices"][0]["message"]["tool_calls"] = tool_calls

                # Add usage information
                if "usage" in response:
                    usage = response["usage"]
                    processed["usage"] = {
                        "prompt_tokens": usage.get("inputTokens", 0),
                        "completion_tokens": usage.get("outputTokens", 0),
                        "total_tokens": usage.get("totalTokens", 0)
                    }

                # Add metadata
                if "metrics" in response:
                    processed["metrics"] = response["metrics"]

                return processed

            # If not Converse API response, return as-is
            return response

        except Exception as e:
            logger.error(f"Error processing Bedrock response: {e}", exc_info=True)
            return response
