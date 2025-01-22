import time
import requests
from bs4 import BeautifulSoup

def initiate_call():
    """
    Initiates a call using the DeepCall API and returns the callId.
    """
    url = (
        "https://s-ct3.sarv.com/v2/clickToCall/para"
        "?&user_id=77080605&token=dQboxDwaaUqy4LJgXrjA&from=6232158146&to=9303748115"
    )
    headers = {"cache-control": "no-cache"}
    response = requests.post(url, headers=headers)

    print(f"Response status: {response.status_code}")
    print(f"Response body: {response.text}")

    if response.status_code == 200:
        response_data = response.json()
        if response_data.get("status") == "success":
            print("Call initiated successfully.")
            return response_data.get("callId")  # Return the callId
        else:
            print("Failed to initiate call: API returned failure.")
    else:
        print(f"Failed to initiate call: {response.status_code} - {response.text}")
    
    return None  # Return None if call initiation fails

def check_call_connected(call_id, to_number):
    """Checks if the call has been answered by the TO_number."""
    url = "https://ctv1.sarv.com/telephony/0/liveCall/analysis"
    response = requests.get(url)
    print(response.text)
    if response.status_code == 200:
        print("Live call analysis retrieved successfully.")
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the live call list
        live_call_list = soup.find('ul', class_='livecallList')
        if live_call_list:
            # Find the dynamic call block using the class name which matches call_id
            print("List found")
            call_block = live_call_list.find('li', class_=call_id)  # The class name is call_id

            if call_block:
                print("call id matched")
                # Look for the customer number div inside this call block
                customer_number_div = call_block.find('div', class_='cutmr_number')
                if customer_number_div:
                    customer_number = customer_number_div.get_text(strip=True)
                    # If the customer number matches the TO_number, return True
                    if customer_number == to_number:
                        print(f"Call connected with {to_number}")
                        return True

        # If no match found, print a message and return False
        print(f"Call not connected with {to_number} yet.")
        return False

    else:
        print(f"Failed to retrieve live call analysis. Status code: {response.status_code}")
        return False

# Main Script
call_id = initiate_call()  # Get the callId
if call_id:
    to_number = "9303748115"  # Replace with the actual TO number

    # Poll for call connection status
    while not check_call_connected(call_id, to_number):
        time.sleep(5)  # Retry every 5 seconds
else:
    print("Call initiation failed. Exiting.")
