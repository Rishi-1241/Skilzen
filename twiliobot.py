from twilio.rest import Client

# Twilio credentials from your account
account_sid = 'AC92cc50e265ccc9edccae3ded8efb79c7'  # Replace with your Twilio Account SID
auth_token = '2ee65ec31a441922752ac141c82d32bc'  # Replace with your Twilio Auth Token

# Your Twilio phone number
twilio_number = '+16207027520'  # Replace with your Twilio phone number

# Recipient phone number
to_number = '+916232158146'  # Replace with the recipient's phone number

# Create a Twilio client
client = Client(account_sid, auth_token)

# Make the call
call = client.calls.create(
    to=to_number,
    from_=twilio_number,
    url='https://studio.twilio.com/v2/Flows/FWca2f2ce2e174c88508ddce9e115174f7/Executions' 
)

print(f"Call initiated: {call.sid}")
