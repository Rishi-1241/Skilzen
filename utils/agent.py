import uuid
from google.cloud.dialogflowcx_v3 import AgentsClient, SessionsClient
from google.cloud.dialogflowcx_v3.types import session

def run_sample():
    # Replace these values with your project details
    project_id = "certain-math-447716-d1"  # Your Google Cloud project ID
    location_id = "global"  # Your agent's location, e.g., "global"
    agent_id = "ab039e5f-d9ce-4feb-90ad-4184f23f01e5"  # Your Dialogflow CX agent ID
    flow_id = "dd90ab06-761a-410d-bb04-f60368c323ac"

    # Construct the agent path
    agent = f"projects/{project_id}/locations/{location_id}/agents/{agent_id}"

    # Use a unique session ID for the interaction
    session_id = uuid.uuid4()

    # Texts for testing
    texts = ["yaha ke placement stats kaise hai"]

    # Language code for interaction
    language_code = "en-us"

    print(f"Starting with project_id: {project_id}, location_id: {location_id}, agent_id: {agent_id}, flow_id: {flow_id}")
    detect_intent_texts(agent, session_id, texts, language_code, flow_id)
from google.cloud.dialogflowcx_v3 import QueryParameters

def detect_intent_texts(agent, session_id, texts, language_code, flow_id):
    """Returns the result of detect intent with texts as inputs.

    Using the same `session_id` between requests allows continuation
    of the conversation."""
    # Construct the session path with the flow ID
    environment_id = "draft"  # Or use "production" if appropriate

    # Construct the session path with the environment ID
    session_path = f"{agent}/environments/{environment_id}/sessions/{session_id}?flow={flow_id}"

    print(f"Session path: {session_path}\n")

    # Configure API endpoint based on location
    client_options = None
    agent_components = AgentsClient.parse_agent_path(agent)
    location_id = agent_components["location"]
    if location_id != "global":
        api_endpoint = f"{location_id}-dialogflow.googleapis.com:443"
        print(f"API Endpoint: {api_endpoint}\n")
        client_options = {"api_endpoint": api_endpoint}

    # Create a session client
    session_client = SessionsClient(client_options=client_options)

    # Iterate through the input texts
    for text in texts:
        # Prepare text input
        text_input = session.TextInput(text=text)
        query_input = session.QueryInput(text=text_input, language_code=language_code)

        # Create a detect intent request
        request = session.DetectIntentRequest(
    session=session_path,
    query_input=query_input,
        )

        # Call the API
        print(f"Sending request for text: {text}")
        response = session_client.detect_intent(request=request)

        # Display query and response
        print("=" * 20)
        print(f"Query text: {response.query_result.text}")
        response_messages = [
            " ".join(msg.text.text) for msg in response.query_result.response_messages
        ]
        print(f"Response text: {' '.join(response_messages)}\n")

# Run the sample
if __name__ == "__main__":
    run_sample()