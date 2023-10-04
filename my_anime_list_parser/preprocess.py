# for preprocessing the information from my anime list in correct shape before passing it to the database
import datetime

# conclued relevant simple informations after some experimenting
simple_attributes = [
 'title',
 'synopsis',
 'mean',
 'rank',
 'popularity',
 'num_list_users',
 'num_scoring_users',
 'nsfw',
 'media_type',
 'status',
 'genres',
 'num_episodes',
 'start_season',
 'source',
 'average_episode_duration',
 'rating',
 'recommendations',
 'studios']

def parse_info_dict(serie:dict):
    # simple key combinaison first
    info_dict = {}
    for key in simple_attributes :
        if key in serie :
            info_dict[key] = serie[key]
    
    # parse alternative title
    if "alternative_titles" in serie :
        all_titles = []
        titles = serie["alternative_titles"]
        if "en" in titles :
            all_titles.append(titles["en"])
        if "synonyms" in titles :
            for value in titles["synonyms"] :
                all_titles.append(value)

        if len(all_titles) > 0 :
            info_dict["alternative_titles"] = "|".join(all_titles)
        
    
    # parsing the genre
    genres_raw = serie["genres"]
    genre_str = '|'.join([genre["name"] for genre in genres_raw])
    info_dict["genres"] = genre_str

    # parsing the airing season
    if "start_season" in serie :
        raw_season = serie["start_season"]
        season_str = "|".join([str(raw_season["year"]), raw_season["season"]])
        info_dict["start_season"] = season_str

    # parse the broadcast day and time
    # not always present
    if "broadcast" in serie :
        raw_broadcast = serie["broadcast"]
        
        start_time = raw_broadcast["start_time"] if "start_time" in raw_broadcast else "00:00"

        broadcast_str = "|".join([raw_broadcast["day_of_the_week"], start_time])  
        info_dict["broadcast"] = broadcast_str
    

    # parsing the start and end date
    if "end_date" in serie :
        end_date = serie["end_date"]
        end_str = list(map(int, end_date.split("-")))
        info_dict["end_date"] = datetime.date(*end_str)
    
    if "start_date" in serie :
        start_date = serie["start_date"]
        start_str = list(map(int, start_date.split("-")))
        info_dict["start_date"] = datetime.date(*start_str)


    # parsing the studio
    studios = serie["studios"]
    studio_str = "|".join([stod["name"] for stod in studios])
    info_dict["studios"] = studio_str

    # parsing the statistics
    stat = serie["statistics"]["status"]
    for elem, valeur in stat.items() :
        info_dict[elem] = int(valeur)

    # parsing the recommendation ids
    recommend_str = [] 
    for suggested in serie["recommendations"] :
        num_recommend = suggested["num_recommendations"]
        suggested = suggested["node"]
        values = f"{suggested['id']},{suggested['title']},{num_recommend}"

        recommend_str.append(values)

    all_recommendations = "|".join(recommend_str)
    info_dict["recommendations"] = all_recommendations

    # shaping the id
    info_dict["myid"] = serie["id"]

    return info_dict

def parse_id_from_node(serie) :
    serie = serie["node"]
    id, name = serie["id"], serie["title"]
    return (id, name)

def parse_boolean(raw_str, dico:dict):
    copy_dico = dict(dico)
    options = str(raw_str).split("|")

    for cas in options :
        if cas in copy_dico :
            copy_dico[cas] = 1

        elif "other" in copy_dico :
            copy_dico["other"] = 1

    return copy_dico
# all relevant attributes

# attributes = {
# 'title': 'TEXT',
# "alternative_titles",
#  'start_date': 'DATE',
#  'end_date': 'DATE',
#  'synopsis': 'TEXT',
#  'mean': 'DOUBLE',
#  'rank': 'NUMBER',
#  'popularity': 'NUMBER',
#  'num_list_users': 'NUMBER',
#  'num_scoring_users': 'NUMBER',
#  'nsfw': 'TEXT',
#  'media_type': 'TEXT',
#  'status': 'TEXT',
#  'genres': 'TEXT',
#  'num_episodes': 'NUMBER',
#  'start_season': 'TEXT',
#  'broadcast': 'TEXT',
#  'source': 'TEXT',
#  'average_episode_duration': 'NUMBER',
#  'rating': 'TEXT',
#  'recommendations': 'TEXT',
#  'studios': 'TEXT',
#  'watching': 'NUMBER',
#  'completed': 'NUMBER',
#  'on_hold': 'NUMBER',
#  'dropped': 'NUMBER',
#  'plan_to_watch': 'NUMBER',
#  'myid': 'NUMBER'}