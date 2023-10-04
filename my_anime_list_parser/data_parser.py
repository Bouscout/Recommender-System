import requests
import json

access_token = None
with open("my_anime_list_parser/token.json", 'r') as file :
    data = json.load(file)

access_token = data['access_token']

def parse_data(response):
    anime = json.loads(response.text)
    return anime

def make_request(lien):
    response = requests.get(lien, headers={
        "Authorization" : f"Bearer {access_token}"
    })

    return response
# url = "https://api.myanimelist.net/v2/anime/44511?fields=id,title,main_picture,alternative_titles,start_date,end_date,synopsis,mean,rank,popularity,num_list_users,num_scoring_users,nsfw,created_at,updated_at,media_type,status,genres,my_list_status,num_episodes,start_season,broadcast,source,average_episode_duration,rating,pictures,background,related_anime,related_manga,recommendations,studios,statistics"
url = "https://api.myanimelist.net/v2/anime?q=one&limit=4"
response = make_request(url)

if response.status_code == 200 :
    anime = parse_data(response)

    print("result : ", anime)
    print("=====================================")
    for key in anime :
        # print(f"{key} : {anime[key]}")
        print(key)

    while True :
        query = input("enter a key to check :  ")
        if query == "0" :
            break

        else :
            try :
                print(f"{query} : {anime[query]}")
            except :
                pass


else :
    print("the response status is : ", response.status_code)


