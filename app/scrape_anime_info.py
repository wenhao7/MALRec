import os
from bs4 import BeautifulSoup
import requests
import time
import pandas as pd
import random
import re
import csv

# Variables
site_url = 'https://myanimelist.net'
top_anime_url = site_url + '/topanime.php?limit='
BASE_PATH = "data"
HTML_PATH = BASE_PATH + "/html"
req_head = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0'}


def sleep(t=3):
    """
    Implements a randomized sleep time to circumvent scrape detection when using fixed delays

    Parameters
    ----------
    t : int
        minimum sleep time, sleep time follows a normal distribution with mean of 1.5*parameter.

    Returns
    -------
    None.

    """
    rand_t = random.random() * (t) + 0.5
    time.sleep(rand_t)
    print(f"Sleeping for {rand_t}s")

def parse_episodes(content):
    """
    Cleans and formats string extracted from scraped html by removing extra whitespace and formatting the scraped data into python list

    Parameters
    ----------
    content : List[str]
        List of strings scraped from target HTML.

    Returns
    -------
    result : List[str]
        List of strings with extra whitespaces removed.

    """
    result = []
    for i in content:
        r = i.strip()
        result.append(r)
    return result

def return_numeric(string):
    """
    Return only numeric characters from a string

    Parameters
    ----------
    string : str
        String that should contain only numeric characters.

    Returns
    -------
    text : str
        String with non-numeric characters removed.

    """
    try:
        # Regex to find all numeric characters
        text = re.findall("\d+", string)[0]
    except IndexError:
        text = '?'
    return text
    
def write_csv(items, path):
    """
    Save our prepared dictionary of scraped data to a .csv file.
    Used when scraping high level information from "Top Anime" pages.

    Parameters
    ----------
    items : Dict
        Dictionary containing the relevant scraped categories and information.
    path : str
        File path or file name of the .csv file to save to.

    Returns
    -------
    None.

    """
    # Open the file in write mode
    with open(path, 'w', encoding='utf-8') as f:
        # Return if there's nothing to write
        if len(items) == 0:
            return
        
        # Write the headers in the first line
        headers = list(items[0].keys())
        f.write(','.join(headers) + '\n')
        
        # Write one item per line
        for item in items:
            values = []
            for header in headers:
                values.append(str(item.get(header, "")).replace(',',' '))
            f.write(','.join(values) + "\n")
            
### Extract high level information from row_contents
def extract_info(top_anime, row_contents):
    """
    Extract high level information when scraping "Top Anime" pages

    Parameters
    ----------
    top_anime : List[Dict]
        List of dictionaries containing information of top anime titles scraped thus far .
    row_contents : bs4.element.ResultSet
        bs4 ResultSet of top anime titles found from scraped HTML.

    Returns
    -------
    top_anime : List[Dict]
        Updated list of dictionaries containing information of top anime titles scraped thus far .
    stop : Bool
        Returns True when non-scored anime title is found to trigger early stop condition.

    """
    stop = False
    for i in range(len(row_contents)):
        episode = parse_episodes(row_contents[i].find('div', class_ = "information di-ib mt4").text.strip().split('\n'))
        id_str = row_contents[i].find('td', class_='title al va-t word-break').find('a')['id']
        ranking = {
            'Id' : return_numeric(id_str),
            'Rank' : row_contents[i].find('td', class_ = "rank ac").find('span').text,
            'Title': row_contents[i].find('div', class_="di-ib clearfix").find('a').text,
            'Rating': row_contents[i].find('td', class_="score ac fs14").find('span').text,
            'Image_URL': row_contents[i].find('td', class_ ='title al va-t word-break').find('img')['data-src'],
            'Type' : episode[0].split('(')[0].strip(),
            'Episodes': return_numeric(episode[0].split('(')[1]),
            'Dates': episode[1],
            'Members': return_numeric(episode[2])
        }
        top_anime.append(ranking)
        if ranking['Rating']=='N/A':
            stop = True
    return top_anime, stop

def scrape_top_anime(file_name='scrape_top_anime.csv', t=3):
    """
    Loop to scrape top anime pages, stop when non-scored title is found.

    Parameters
    ----------
    file_name : str, optional
        File path or file name of .csv file to write to. The default is 'scrape_top_anime.csv'.
    t : int, optional
        Minimum time to wait between requests in seconds. The default is 3.

    Returns
    -------
    None.

    """
    top_anime = []
    stop = False
    counts = 0
    while not stop:
        sleep(3)
        response = requests.get(top_anime_url + str(counts))
        print(f"Current counts: {counts}, Request Status: {response.status_code}")
        while response.status_code != 200:
            sleep()
            response = requests.get(top_anime_url + str(counts))
        doc = BeautifulSoup(response.text)
        row_contents = doc.find_all('tr', {'class':'ranking-list'})
        top_anime, stop = extract_info(top_anime, row_contents)
        counts += 50
    
    write_csv(top_anime, file_name)
    
def get_link_by_text(soup, anime_id, text):
    """
    Retrieve webpage url of specific pages for a title

    Parameters
    ----------
    soup : bs4.BeautifulSoup
        bs4 object of scraped HTML from target URL.
    anime_id : int
        Anime title ID on the website.
    text : str
        The detailed page to scrape for additional information.

    Returns
    -------
    str
        url string of the target webpage.

    """
    urls = list(filter(lambda x: str(anime_id) in x["href"], soup.find_all("a", text=text)))
    return urls[0]["href"]

def get_request(link, req_head, anime_id):
    """
    Helper function to try get request; if fail 3 times log the title id in .csv file

    Parameters
    ----------
    link : str
        Target url to be send GET request.
    req_head : Dict
        Request header for our sent request.
    anime_id : int
        Anime title ID on the website.

    Returns
    -------
    data : requests.models.Response
        Request response from the scraped link.

    """
    for _ in range(3):
        try:
            data = requests.get(link, headers=req_head)
            if data.status_code !=200:
                sleep()
                continue
            else:
                return data
        except:
            buffer_t = random.random() * (40) + 100
            time.sleep(buffer_t)
            continue
    print(f"Error with Title Id {anime_id}")
    if not 'log_id.csv' in os.listdir():
        with open('log_id.csv','w', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='|',lineterminator='\n')
            #headers = ['MAL_Id', 'URL']
            writer.writerow([anime_id, link])
    with open('log_id.csv','a', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='|',lineterminator='\n')
        writer.writerow([anime_id, link])


def get_review_tags(soup_tags, soup_reviews, anime_id):
    """
    Format scrapped reviews + review tags data

    Parameters
    ----------
    soup_tags : bs4.BeautifulSoup
        bs4 object of scraped HTML of tags page.
    soup_reviews : bs4.BeautifulSoup
        bs4 object of scraped HTML of reviews page.
    anime_id : int
        Anime title ID on the website.

    Returns
    -------
    output : List
        List containing anime id, a single review entry, a list of associated tags.

    """
    extra_tags = ['Funny','Informative','Well-written','Creative','Preliminary']
    review_tags = []
    output = []
    soup_reviews = [r.get_text() for r in soup_reviews]
    for soup_tag in soup_tags:
        curr_tags = []
        tags = soup_tag.text
        #tags = re.findall('[A-Z][^A-Z]*', tags)
        if 'Not' in tags:
            curr_tags.append("Not-Recommended")
        elif "Mixed" in tags:
            curr_tags.append("Mixed-Feelings")
        else:
            curr_tags.append("Recommended")
        for tag in extra_tags:
            if tag in tags:
                curr_tags.append(tag)
        review_tags.append(curr_tags)
    rt =  list(zip(soup_reviews, review_tags))    
    for row in rt:
        r, t = row
        output.append([anime_id, r, t])
    return output


def write_new_reviews(file_name, l):
    """
    Helper function to write reviews/tags to csv file

    Parameters
    ----------
    file_name : str
        File path / File name of .csv file to write to.
    l : List
        List of review entries.

    Returns
    -------
    None.

    """
    if not l:
        return
    if not file_name in os.listdir():
        with open(file_name,'w', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='|',lineterminator='\n')
            headers = ['MAL_Id','Review','Tags']
            writer.writerow(headers)
    with open(file_name,'a', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='|',lineterminator='\n')
        for row in l:
            writer.writerow(row)
        
def get_reviews(link, anime_id, n=1):
    """
    Extract nth page of reviews

    Parameters
    ----------
    link : str
        Target url to scrape.
    anime_id : int
        Anime title ID on the website.

    Returns
    -------
    bs4.element.ResultSet
        bs4 ResultSet of scrapped review tags.
    bs4.element.ResultSet
        bs4 ResultSet of scrapped review entries.

    """
    sleep()
    review_link = f"{link}?p=" + str(n)
    #data = requests.get(review_link, header=req_head)
    data = get_request(review_link, req_head, anime_id)
    if data is None:
        return ['Error'],['Error']
    soup = BeautifulSoup(data.text, "html.parser")
    tags = soup.find_all("div", class_ = "tags")
    reviews = soup.find_all("div", class_="text")
    return tags, reviews
    
def get_recs(link_recommendations, anime_id):
    """
    Extract recommended anime title and number of recommendations

    Parameters
    ----------
    link_recommendations : str
        url of target webpage to scrape.
    anime_id : int
        Anime title ID on the website.

    Returns
    -------
    List
        List of recommended anime ID for the target title.
    List
        List of counts each recommended anime ID has been voted.

    """
    sleep()
    #data = requests.get(link_recomendations, header=req_head)
    data = get_request(link_recommendations, req_head, anime_id)
    if data is None:
        return ['Error'],['Error']
    soup = BeautifulSoup(data.text, "html.parser")
    soup.script.decompose()
    rec_ids = []
    rec_counts = []
    soup_ids = soup.find_all('div', {'class':'hoverinfo'})
    soup_rec_counts = soup.find_all('a', {'class':'js-similar-recommendations-button'})
    for i in range(len(soup_ids)):
        rec_id = return_numeric(soup_ids[i]['rel'])
        rec_ids.append(rec_id)
        if i < len(soup_rec_counts):
            rec_counts.append(soup_rec_counts[i].find('strong').text)
        else:
            rec_counts.append('1')
    return rec_ids, rec_counts

def scrape_anime_info(link_stats, anime_id, anime_info):
    """
    Extract title details and statistics

    Parameters
    ----------
    link_stats : str
        url of target webpage to scrape.
    anime_id : int
        Anime title ID on the website.
    anime_info : Dict
        Dict where keys are the relevant information that we are looking to scrape.

    Returns
    -------
    anime_info : Dict
        Dict storing the updated scraped detailed anime information.

    """
    # Get webpage
    #data = requests.get(link_stats, header=req_head)
    data = get_request(link_stats, req_head, anime_id)
    if data is None:
        return anime_info
    soup = BeautifulSoup(data.text, "html.parser")
    soup.script.decompose()
    
    # Scrape and store information in dict
    anime_info["MAL_Id"] = anime_id
    anime_info["Name"] = soup.find("h1", {"class": "title-name h1_bold_none"}).text.strip()

    score = soup.find("span", {"itemprop": "ratingValue"})
    if score is None:
        score = '?'
    try:
        anime_info['Score'] = score.text.strip()
    except:
        print('Empty Score')
        
    anime_info['Genres'] = [x.text.strip() for x in soup.findAll("span", {"itemprop": "genre"})]
    try:
        anime_info['Demographic'] = anime_info['Genres'][-1]
    except:
        print('Empty Genre')

    for s in soup.findAll("span", {"class": "dark_text"}):
        info = [x.strip().replace(" ", " ") for x in s.parent.text.split(":")]
        cat, v = info[0], ":".join(info[1:])
        v.replace("\t", "")
        
        if cat in ['Synonyms','Japanese','English']:
            cat += '_Name'
            v = v.replace(',', '')
            anime_info[cat] = v
            continue
        if cat in ['Broadcast','Genres','Demographic','Score'] or cat not in anime_info.keys():
            continue
        elif cat in ['Producers','Licensors','Studios']:
            v = [x.strip() for x in v.split(",")]
        elif cat in ['Ranked','Popularity']:
            v = v.replace('#',"")
            v = v.replace(',', '')
        elif cat in ['Members','Favorites','Watching','Completed','On-Hold','Dropped','Plan to Watch','Total']:
            v = v.replace(',','')
            
        anime_info[cat] = v

    # Scrape scoring stats
    for s in soup.find("div", {"id": "horiznav_nav"}).parent.findAll(
        "div", {"class": "updatesBar"}):
        cat = f"Score-{s.parent.parent.parent.find('td', class_='score-label').text}"
        v = ([x.strip() for x in s.parent.text.split("%")][-1].strip("(votes)"))
        anime_info[cat] = str(v).strip()
    return anime_info

def write_new_row(file_name, d):
    """
    Helper function to write dict to csv as a new row

    Parameters
    ----------
    file_name : str
        File path or file name of .csv file to write to.
    d : Dict
        Dict storing the updated scraped detailed anime information.

    Returns
    -------
    None.

    """
    if not file_name in os.listdir():
        with open(file_name,'w', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='|',lineterminator='\n')
            headers = list(d.keys())
            writer.writerow(headers)
    with open(file_name,'a', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='|',lineterminator='\n')
        values = []
        for k, v in d.items():
            values.append(str(v))
        writer.writerow(values)

# Scrape various information from the anime title through the links to its webpages
def scrape_anime(anime_id):
    """
    For a given anime ID, prepare the relevant urls to be scraped, and the Dict that will store the required information before calling scrape_anime_info() to scrape this information.

    Parameters
    ----------
    anime_id : int
        Anime title ID on the website.

    Returns
    -------
    None.

    """
    #path = f"{HTML_PATH}/{anime_id}"
    #if f"{anime_id}.zip" in os.listdir(f'{HTML_PATH}'):
    #    return
    
    #os.makedirs(path, exist_ok=True)
    sleep()
    #data = requests.get(f"https://myanimelist.net/anime/{anime_id}", header=req_head)
    data = get_request(f"https://myanimelist.net/anime/{anime_id}", req_head, anime_id)
    if data is None:
        return
    
    soup = BeautifulSoup(data.text, "html.parser")
    soup.script.decompose()
    va = []
    for s in soup.find_all('td', class_='va-t ar pl4 pr4'):
        va.append(s.a.text)
    #save(f"{HTML_PATH}/{anime_id}/details.html", soup.prettify())
    
    # Get urls to detailed webpages
    link_review = get_link_by_text(soup, anime_id, "Reviews")
    link_recommendations = get_link_by_text(soup, anime_id, "Recommendations")
    link_stats = get_link_by_text(soup, anime_id, "Stats")
    #link_staff = get_link_by_text(soup, anime_id, "Characters & Staff")
    
    # Dict to store information
    key_list = ['MAL_Id','Name','Synonyms_Name','Japanese_Name','English_Name','Type','Episodes','Status','Aired','Premiered','Producers','Licensors','Studios','Source','Genres','Demographic','Duration','Rating','Score','Ranked','Popularity','Members','Favorites','Watching','Completed','On-Hold','Dropped','Plan to Watch','Total','Score-10','Score-9', 'Score-8', 'Score-7', 'Score-6', 'Score-5', 'Score-4','Score-3', 'Score-2', 'Score-1','Synopsis','Voice_Actors','Recommended_Ids','Recommended_Counts']
    anime_info = {key:'?' for key in key_list}
    
    # Scrape relevant information from the urls
    anime_info = scrape_anime_info(link_stats, anime_id, anime_info)
    anime_info['Synopsis'] = soup.find('p', {'itemprop':'description'}).text.replace('\r','').replace('\n','').replace('\t','')    
    anime_info['Voice_Actors'] = va
    rec_ids, rec_counts = get_recs(link_recommendations, anime_id)
    anime_info['Recommended_Ids'] = rec_ids
    anime_info['Recommended_Counts'] = rec_counts
    write_new_row('anime_info.csv', anime_info)
    
    soup_tags, soup_reviews = get_reviews(link_review, anime_id)
    if len(soup_tags) > 0 and len(soup_reviews) > 0:
        review_data = get_review_tags(soup_tags, soup_reviews, anime_id)
        write_new_reviews('anime_reviews.csv', review_data)
         
def scrape_all_anime_info(anime_list_file_name, i=0):
    """
    Function to scrape all titles found within a given .csv file

    Parameters
    ----------
    anime_list_file_name : str
        File path / file name of the .csv file containing the anime titles to scrape.
    i : int, optional
        Position in the file to start scraping from. The default is 0.

    Returns
    -------
    None.

    """
    df = pd.read_csv(anime_list_file_name)
    for aid in df.Id[i:]:
        scrape_anime(aid)
        i+=1
        print(f'Latest Title: {aid}, Title Completed: {i}/13300')
        if not i%20:
            print(time.asctime())