"""
Access the api of myanimelist and fetch info regarding certain shows
"""
import json
import requests

class Anime_Fetcher():
    def __init__(self, path:str=None, attributes=None) -> None:
        token_path = path if path else "my_anime_list_parser/token.json"

        self.base_url = "https://api.myanimelist.net/v2/anime"

        # list of the informations that we want
        default = [
            "id",
            "title",
            "alternative_titles",
            "start_date",
            "end_date",
            "synopsis",
            "mean",
            "rank",
            "popularity",
            "num_list_users",
            "num_scoring_users",
            "nsfw",
            "created_at",
            "updated_at",
            "media_type",
            "status",
            "genres",
            "num_episodes",
            "start_season",
            "broadcast",
            "source",
            "average_episode_duration",
            "rating",
            "recommendations",
            "studios", # too many so just filter the major ones and store the rest as others
            "statistics"
        ]

        self.attributes = attributes if hasattr(attributes, "__len__") else default
        self.create_request_query(self.attributes, default=True)

        self.init_token(token_path)

    def create_request_query(self, features:list, default=False):
        """
        Create a format for making a search query\n
        set default to True to replace the default search query to this one
        """
        query = ",".join(features)

        if default :
            self.query = query
        return query
    
    def init_token(self, path):
        with open(path, 'r') as file :
            data = json.load(file)
        
        self.access_token = data['access_token']


    def request(self, link):
        response = requests.get(link, headers={
            "Authorization" : f"Bearer {self.access_token}"
        })

        if response.status_code == 200 :
            anime = json.loads(response.text)
            return anime
        
        else :
            raise ValueError(f"there has been a problem, status code : {response.status_code}")

    def request_details(self, id:int, query=None):
        """request specific detail for an anime"""
        feature_query = query if query else self.query

        request_url = self.base_url + f"/{id}?fields=" + feature_query
        response = self.request(request_url)
        return response
    

    def request_search(self, search:str, limit:int=10):
        """for searching animes based on the title, provide limit to trim the number of result"""
        request_url = self.base_url + f"?q={search}&limit={limit}"
        response = self.request(request_url)
        return response
    
    def request_from_rank(self, rank_type:str, limit:int=50, offset:int=0):
        """request a list of anime sorted by a ranking system of the type provided in the parameter"""
        request_url = self.base_url + f"/ranking?ranking_type={rank_type}&limit={limit}" 
        if offset :
            request_url += f"&offset={offset}"

        response = self.request(request_url)
        return response
    
    def request_seasonal(self, season:str, year:int=2023, * ,limit=50, offset=0, fields=None, sort=None):
        """Request a list of the anime airing at the requested season"""
        # https://api.myanimelist.net/v2/anime/season/2017/summer?limit=4
        request_url = self.base_url + f"/season/{year}/{season}?limit={limit}" 

        if offset :
            request_url += f"&offset={offset}"

        if fields :
            query = self.create_request_query(fields)
            request_url += f"&fields={query}"

        if sort :
            request_url += f"&sort={sort}"

        response = self.request(request_url)
        return response
    
