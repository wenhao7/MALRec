import os
from bs4 import BeautifulSoup
import requests
import time
import pandas as pd
import random
import re
import csv
import logging
import tqdm
from multiprocessing.pool import ThreadPool as Pool

# Instantiate variables
start_year, end_year = 1917, 2024
seasons = ['winter','spring','summer','fall']
req_head = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0',
           'X-MAL-CLIENT-ID':'e09c24c7eb88c3f399d9bd1355b4e015'}
seasonal_anime_filename = 'seasonal_anime.csv'

logging.basicConfig(filename='seasonal_anime.log', filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')


# Process Number of Episodes and Episode Duration into their own dictionary
def process_prodsrc(row_content):
    """
    Process number of episodes and episode duration from bs4 object into their own dictionary

    Parameters
    ----------
    row_content : bs4.element.Tag
        Scraped HTML parsed by bs4

    Returns
    -------
    content_dict : Dict
        Dictionary containing parsed Episodes and Duration.

    """
    content = [x.replace(' ','') for x in row_content.find('div', {'class':'prodsrc'}).text.split('\n') if x.replace(' ','') != '']
    content_dict = {'Episodes':0,'Duration':0}
    for c in content:
        if ('ep' in c or 'eps' in c) and 'Sep' not in c:
            content_dict['Episodes'] = c.replace('eps', '').replace('ep','').replace(',','')
        elif 'min' in c:
            content_dict['Duration'] = c.replace('min', '')
    return content_dict

# Process Studio, Source, Themes, Demographic into their own dictionary
def process_properties(row_content):
    """
    Process studio, source, themes, and demographic information from bs4 object into their own dictionary

    Parameters
    ----------
    row_content : bs4.element.Tag
        Scraped HTML parsed by bs4.

    Returns
    -------
    content_dict : Dict
        Dictionary containing parsed Studio, Source, Themes, Demographic.

    """
    content = [x for x in row_content.find('div', {'class':'properties'}).text.split('\n') if x != '']
    content_dict = {'Studio':'', 'Source': '', 'Theme': '', 'Demographic': ''}
    for c in content:
        for k in content_dict.keys():
            if k in c:
                content_dict[k] = c.replace(k, '')
    return content_dict

# Clean Synopsis text
def clean_text(text):
    """
    Clean text information, removing escape sequence and additional whitespaces

    Parameters
    ----------
    text : str
        String text.

    Returns
    -------
    text : str
        Cleaned string text.

    """
    text = text.replace('\t','').replace('\n',' ').replace('\r',' ')
    text = re.sub(' +', ' ',text).rstrip('\\').strip()
    return text

# Function to create a dictionary containing all the above information
def extract_info(row_contents):
    """
    Create dictionary containing processed scraped information

    Parameters
    ----------
    row_contents : bs4.element.ResultSet
        Scraped HTML parsed by bs4.

    Returns
    -------
    seasonal_contents : List
        List of dictionaries containing 1 scraped Title per dict.

    """
    seasonal_contents = []
    for i in range(len(row_contents)):
        prodsrc = process_prodsrc(row_contents[i])
        properties = process_properties(row_contents[i])
        id_ = row_contents[i].find('div', {'class':'genres'})
        title_ = row_contents[i].find('span', {'class':'js-title'})
        score_ = row_contents[i].find('span', {'class':'js-score'})
        members_ = row_contents[i].find('span', {'class':'js-members'})
        start_date_ = row_contents[i].find('span', {'class':'js-start_date'})
        image_ = row_contents[i].find('img')
        
        contents = {
            'MAL_Id': id_.get('id', -1) if id_ else '',
            'Name': title_.text if title_ else '',
            'Image': image_.get('src','') or image_.get('data-src','') if image_ else '',
            'Score': score_.text if score_ else '',
            'Members': members_.text if members_ else '',
            'Start_Date': start_date_.text if start_date_ else '',
            'Episodes': prodsrc['Episodes'],
            'Duration': prodsrc['Duration'],
            'Genres': [x.text.strip() for x in row_contents[i].findAll('span', {'class': 'genre'})],
            'Studio': properties['Studio'],
            'Source': properties['Source'],
            'Themes': re.findall('[A-Z][^A-Z]*', properties['Theme']),
            'Demographic': re.findall('[A-Z][^A-Z]*', properties['Demographic']),
            'Synopsis': clean_text(row_contents[i].find('div', {'class':'synopsis'}).text)
        }
        seasonal_contents.append(contents)
    return seasonal_contents

# Helper functions
### Implement randomized sleep time in between requests to reduce chance of being blocked from site
def sleep(t=3):
    """
    Randomized sleep with a standard distribution with mean time around the input + 0.5

    Parameters
    ----------
    t : int, optional
        Desired approximate mean sleep time. The default is 3.

    Returns
    -------
    None.

    """
    rand_t = random.random() * (t) +0.5
    time.sleep(rand_t)
    
    
def write_seasonal_csv(items, path):
    """
    Save processed scraped data to a file

    Parameters
    ----------
    items : List
        List of dictionaries containing scraped data for a Title per dict.
    path : str
        File path or file name of the file to write to.

    Returns
    -------
    None.

    """
    written_id = set()
    
    # Assign header names with handling of seasons with no new release in certain media types
    for i in range(len(items)):
        if items[i]:
            headers = list(items[i][0].keys())
            break
    
    # In case no new titles released
    if headers:
        # Open the file in write mode
        if not path in os.listdir():
            with open(path, 'w', encoding='utf-8') as f:
                # Return if there's nothing to write
                if len(items) == 0:
                    return
                # Write the headers in the first line
                f.write('|'.join(headers) + '\n')
                
        with open(path, 'a', encoding='utf-8') as f:
            # Write one item per line
            for i in range(len(items)):
                for item in items[i]:
                    values = []
                    # Check if title has already been added to prevent duplicated entries, some shows span multiple seasons
                    if item.get('MAL_Id') in written_id:
                        continue
                    for header in headers:
                        values.append(str(item.get(header, "")).replace('|',' '))
                    f.write('|'.join(values) + "\n")          
                    written_id.add(item.get('Id'))

### Send request to website
def get_response(url):   
    """
    Handle sending of requests, taking care of error handling, logging, and retry/delay mechanics in the event of rate limiting from target.

    Parameters
    ----------
    url : str
        Target URL to send request.

    Returns
    -------
    row_contents : bs4.element.ResultSet
        Scraped HTML information parsed by bs4.

    """
    # Try for up to 3 times per URL
    for _ in range(3):
        try:
            sleep(3)
            response = requests.get(url, headers=req_head)
            # If response is good we return the BS object for further processing
            if response.status_code == 200:
                doc = BeautifulSoup(response.text)
                row_contents = doc.find_all('div', {'class':'js-anime-category-producer'})
                if row_contents is None:
                    logging.warning(f'row_contents is None for {url}')
                    print(f'----------- row_contents is None for {url} ------------')
                return row_contents
            
            # If response suggests we are rate limited, make this thread sleep for ~3 minutes before continuing on next loop
            elif response.status_code == 429 or response.status_code == 405:
                logging.warning(f'{response.status_code} for {url}')
                print(f'----------- {response.status_code} occured for {url} ------------')
                buffer_t = random.random() * (40) + 160
                sleep(buffer_t)
                continue
            
            # Any other unexpected response
            else:
                logging.warning(f'{response.status_code} for {url}')
                print(f'----------- {response.status_code} occured for {url} ------------')
                sleep(5)
                continue
        
        # Any unexpected issues with sending request
        except:
            logging.error('Error trying to send request')
            buffer_t = random.random() * (40) + 100
            sleep(buffer_t)
            continue            
    print("-----------------------------Error sending request-----------------------------")
    print(time.asctime())

    
# Scrape all URLs between start and end years for specified seasons, 
# multiprocessing available, option to override with a specified list of url available as well
def scrape(file_name, start_year=1917, end_year=2024, seasons=['winter','spring','summer','fall'], req=req_head, nprocesses=4, url_list=None):
    """
    Given a start year, end year, and seasons to scrape, function will automatically generate the relevant URLs and process scraped data.
    If user-defined list of URLs is provided the above behaviour will be skipped.
    Multiprocessing is available and enabled.

    Parameters
    ----------
    file_name : str
        File path or file name of file to write to.
    start_year : int, optional
        Start year to scrape, inclusive. The default is 1917.
    end_year : int, optional
        End year to scrape, inclusive. The default is 2024.
    seasons : List, optional
        List of seasons to scrape. The default is ['winter','spring','summer','fall'].
    req : Dict, optional
        Header to include in request. The default is req_head.
    nprocesses : int, optional
        Number of concurrent processors. The default is 4.
    url_list : List, optional
        List of user defined URLs if seasonal scraping behaviour is not desired. The default is None.

    Returns
    -------
    anime_list : List
        List of dict containing processed scraped information of a Title per dict.

    """
    top_anime_url = 'https://myanimelist.net/anime/season/'
    
    # If specific URLs are not provided, a list of URLs will be generated based on start/end years and seasons provided.
    if not url_list:
        url_list = [top_anime_url + str(year) + '/' + str(season) for year in range(start_year,end_year+1) for season in seasons]
    
    anime_list = []
    # nprocesses number of threads processing URL list in sequence parallelly
    with Pool(processes=nprocesses) as pool:
        for type_contents in tqdm.tqdm(pool.imap(get_response, url_list), total=len(url_list)):
            if type_contents is None:
                continue
            for i in range(len(type_contents)):
                row_contents = type_contents[i].find_all('div', {'class':'js-anime-category-producer'})
                mediatype = type_contents[i].find('div', {'class':'anime-header'}).text
                seasonal_contents = extract_info(row_contents, mediatype)
                anime_list.append(seasonal_contents)
            sleep(5) # a few seconds sleep before next request is sent to avoid rate limit by site
    
    # Write scraped data to disk
    write_seasonal_csv(anime_list, file_name)
    return anime_list