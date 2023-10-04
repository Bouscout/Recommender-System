import json
import requests
import secrets
import os
from dotenv import load_dotenv

load_dotenv()

# After signing up on myanimelist.com you should be able to get a client ID and secret code
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")


# 1. Generate a new Code Verifier / Code Challenge.
def get_new_code_verifier() -> str:
    token = secrets.token_urlsafe(100)
    return token[:128]


# 2. Print the URL needed to authorise your application.
def print_new_authorisation_url(code_challenge: str):
    global CLIENT_ID

    url = f'https://myanimelist.net/v1/oauth2/authorize?response_type=code&client_id={CLIENT_ID}&code_challenge={code_challenge}'
    print(f'Authorise your application by clicking here: {url}\n')

#def502004cca3984af832008fd38ac7fc90ac88312e355cca89f41e77f635cba25173486ad3bc69c55f928b095322c5388988e91f5ab6a062f5816c8c8b56ce044d76cbc4c3b8b7ce2eb1fa47ef178f8fb3c7475dca6180312825425db04b494c6e4dbe5ac00ccf43e0ed1ece7a65a52c59be693e7553e0c401fbbb0683fac80abdff622094c6fe484b4330e02f7f6ebde1feba7e358dbae556c3de9e7436b615fccb926c6d172fcff3b492a4abf796c58fcbfa0e101252550d9ba8051085206007aa25a6d5873a6bb69377274f3289c0de5be836b7fdc34ffc93b29b7a82ac3743ccefcca0ec8a3aabdf242eaa7d4baaf0d60eacc26bd980184ef7b7220529d100e4b342d011e515c884da4d417a1613a9b15466253bcb8c5404d127ef47d90ea2ecd91d4ed8ced4be68cc6ca078c9a632a343fb59e0f5edc40339e640a4226a77baece93f96bfa0317e2180d8d3fa188657a7f3d5f7a0011c2d13d3dd4e56e4a90a3d352a6f91c1d9a37a5a2587525af187bae3b5c0464c1196ca7fc3bd4b21b61e263766c338816fdaec9628e9fd10d7db27937cfd9e2f11ad88de993f6790ef82a9cb8d0d6cd336229c99d86433e0a339b81a437106d1a9691cd4f0b2c66dddc35282c9695f7ac1beb9b930c2794c869629ea7f390ef7f43c98e9227c510c36338db026a7307b852
# 3. Once you've authorised your application, you will be redirected to the webpage you've
#    specified in the API panel. The URL will contain a parameter named "code" (the Authorisation
#    Code). You need to feed that code to the application.
def generate_new_token(authorisation_code: str, code_verifier: str) -> dict:
    global CLIENT_ID, CLIENT_SECRET

    url = 'https://myanimelist.net/v1/oauth2/token'
    data = {
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'code': authorisation_code,
        'code_verifier': code_verifier,
        'grant_type': 'authorization_code'
    }

    response = requests.post(url, data)
    response.raise_for_status()  # Check whether the request contains errors

    token = response.json()
    response.close()
    print('Token generated successfully!')

    with open('my_anime_list_parser/token.json', 'w') as file:
        json.dump(token, file, indent = 4)
        print('Token saved in "token.json"')

    return token


# 4. Test the API by requesting your profile information
def print_user_info(access_token: str):
    url = 'https://api.myanimelist.net/v2/users/@me'
    response = requests.get(url, headers = {
        'Authorization': f'Bearer {access_token}'
        })
    
    response.raise_for_status()
    user = response.json()
    response.close()

    print(f"\n>>> Greetings {user['name']}! <<<")

def parse_code(raw_code:str):
    url, code = raw_code.split("?")
    _, trimmed_code = code.split("=")
    return trimmed_code

if __name__ == '__main__':
    code_verifier = code_challenge = get_new_code_verifier()
    print_new_authorisation_url(code_challenge)

    authorisation_code = input('Copy-paste the Authorisation Code: ').strip()

    full_code = parse_code(authorisation_code)

    print("===========================")
    print("the code is : ", full_code)
    input("press enter to continue")
    authorisation_code = full_code

    token = generate_new_token(authorisation_code, code_verifier)

    print_user_info(token['access_token'])
    print("==========================")
    print("token refreshed")
