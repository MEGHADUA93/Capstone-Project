#!/usr/bin/env python
# coding: utf-8

# Segmenting and Clustering Neighborhoods in Toronto IBM

# Part I: scrape Data from the Wikipedia page

# In[1]:


from bs4 import BeautifulSoup
import requests
import pandas as pd


# In[2]:


url = "https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M"
source = requests.get(url).text
Torontodata = BeautifulSoup(source, 'lxml')


# In[3]:


column_names = ['Postalcode','Borough','Neighborhood']
T_Postalcode = pd.DataFrame(columns = column_names)

content = Torontodata.find('div', class_='mw-parser-output')
table = content.table.tbody
postcode = 0
borough = 0
neighborhood = 0

for tr in table.find_all('tr'):
    i = 0
    for td in tr.find_all('td'):
        if i == 0:
            postcode = td.text
            i = i + 1
        elif i == 1:
            borough = td.text
            i = i + 1
        elif i == 2: 
            neighborhood = td.text.strip('\n').replace(']','')
    T_Postalcode = T_Postalcode.append({'Postalcode': postcode,'Borough': borough,'Neighborhood': neighborhood},ignore_index=True)


# In[4]:


T_Postalcode = T_Postalcode[T_Postalcode.Borough!='Not assigned']
T_Postalcode = T_Postalcode[T_Postalcode.Borough!= 0]
T_Postalcode.reset_index(drop = True, inplace = True)
i = 0
for i in range(0,T_Postalcode.shape[0]):
    if T_Postalcode.iloc[i][2] == 'Not assigned':
        T_Postalcode.iloc[i][2] = T_Postalcode.iloc[i][1]
        i = i+1
                                 
data1 = T_Postalcode.groupby(['Postalcode','Borough'])['Neighborhood'].apply(', '.join).reset_index()
data1.head()


# In[5]:


data1 = data1.dropna()
empty = 'Not assigned'
data1 = data1[(data1.Postalcode != empty ) & (data1.Borough != empty) & (data1.Neighborhood != empty)]


# In[6]:


data1.head()


# In[10]:


def neighborhood_list(grouped):    
    return ', '.join(sorted(grouped['Neighborhood'].tolist()))
                    
grp = data1.groupby(['Postalcode', 'Borough'])
data2 = grp.apply(neighborhood_list).reset_index(name='Neighborhood')


# In[11]:


print(data2.shape)
data2.head()


# In[12]:


data2


# Part II: Add the latitude and longitude

# In[14]:


import numpy as np
import geocoder


# In[15]:


def get_latlng(postal_code):
    # initialize your variable to None
    lat_lng_coords = None
    # loop until you get the coordinates
    while(lat_lng_coords is None):
        g = geocoder.arcgis('{}, Toronto, Ontario'.format(postal_code))
        lat_lng_coords = g.latlng
    return lat_lng_coords
    
get_latlng('M6S')


# In[16]:


postal_codes = data2['Postalcode']    
coordins = [ get_latlng(postal_code) for postal_code in postal_codes.tolist() ]


# In[17]:


data3 = pd.DataFrame(coordins, columns=['Latitude', 'Longitude'])
data2['Latitude'] = data3['Latitude']
data2['Longitude'] = data3['Longitude']


# In[18]:


data2[data2.Postalcode == 'M6S']


# In[19]:


data2


# In[20]:


import requests
import re
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import json
from geopy.geocoders import Nominatim 
from pandas.io.json import json_normalize 
import matplotlib.cm as cm
import matplotlib.colors as colors
from sklearn.cluster import KMeans 
import folium


# In[21]:


address = 'Toronto'

geolocator = Nominatim()
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Toronto are {}, {}.'.format(latitude, longitude))

# create map of Toronto using latitude and longitude values
map_toronto = folium.Map(location=[latitude, longitude], zoom_start=10)

# add markers to map
for lat, lng, borough, neighborhood in zip(data2['Latitude'], data2['Longitude'], data2['Borough'], data2['Neighborhood']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='red',
        fill=True,
        fill_color='#080807',
        fill_opacity=0.7,
        parse_html=False).add_to(map_toronto)  
    
map_toronto


# In[22]:


North_York_data= data2[data2['Borough'] == 'North York'].reset_index(drop=True)
address1 = 'North York,Toronto'

geolocator1 = Nominatim()
location1 = geolocator1.geocode(address1)
latitude1 = location1.latitude
longitude1 = location1.longitude
print('The geograpical coordinate of North York are {}, {}.'.format(latitude1, longitude1))


# In[23]:


map_N_York = folium.Map(location=[latitude1, longitude1], zoom_start=11)

# add markers to map
for lat, lng, label in zip(North_York_data['Latitude'], North_York_data['Longitude'], North_York_data['Neighborhood']):
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='red',
        fill=True,
        fill_color='#080807',
        fill_opacity=0.7,
        parse_html=False).add_to(map_N_York)  
    
map_N_York


# In[24]:


neighborhood_latitude = North_York_data.loc[0, 'Latitude']
neighborhood_longitude = North_York_data.loc[0, 'Longitude']

neighborhood_name = North_York_data.loc[0, 'Neighborhood']

print('Latitude and longitude values of {} are {}, {}.'.format(neighborhood_name, 
                                                               neighborhood_latitude, 
                                                               neighborhood_longitude))

VERSION = 20200229
LIMIT = 100
radius = 500
url = 'https://api.foursquare.com/v2/venues/explore?client_id={}&client_secret={}&ll={},{}&v={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET, latitude1, longitude1, VERSION, radius, LIMIT)


# In[25]:


def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']


# In[26]:


results = requests.get(url).json()
venues = results['response']['groups'][0]['items']
    
nearby_venues = json_normalize(venues) # flatten JSON

# filter columns
filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
nearby_venues =nearby_venues.loc[:, filtered_columns]

# filter the category for each row
nearby_venues['venue.categories'] = nearby_venues.apply(get_category_type, axis=1)

# clean columns
nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]

nearby_venues.head(5)


# In[ ]:


def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# In[27]:


North_York_venues = getNearbyVenues(names=North_York_data['Neighborhood'],
                                   latitudes=North_York_data['Latitude'],
                                   longitudes=North_York_data['Longitude']
                                  )


# In[28]:


North_York_venues.head(3)
print(North_York_venues.groupby('Neighborhood').count()[:4])


# In[29]:


# one hot encoding
N_York_onehot = pd.get_dummies(North_York_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
N_York_onehot['Neighborhood'] = North_York_venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [N_York_onehot.columns[-1]] + list(N_York_onehot.columns[:-1])
N_York_onehot = N_York_onehot[fixed_columns]

N_York_grouped = N_York_onehot.groupby('Neighborhood').mean().reset_index()


# In[31]:



def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[32]:


num_top_venues = 15

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = N_York_grouped['Neighborhood']

for ind in np.arange(N_York_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(N_York_grouped.iloc[ind, :], num_top_venues)


# In[33]:


North_York_data = North_York_data.drop(16)
# set number of clusters
kclusters = 5

N_York_grouped_clustering = N_York_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(N_York_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10]


# In[34]:


N_York_merged = North_York_data

# add clustering labels
N_York_merged['Cluster Labels'] = kmeans.labels_

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
N_York_merged = N_York_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')


# In[35]:


map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i+x+(i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(N_York_merged['Latitude'], N_York_merged['Longitude'], N_York_merged['Neighborhood'], N_York_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# In[ ]:





# In[ ]:




