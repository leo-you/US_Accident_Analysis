#!/usr/bin/env python
# coding: utf-8

# # US Accident Anaysis
# 
# ## Introduction
# The data from this analysis is from [kaggle US Accidents: A Countrywide Traffic Accident Dataset (2016 - 2019)](https://www.kaggle.com/sobhanmoosavi/us-accidents), which contains accident data are collected from February 2016 to December 2019 in 49 US states, more documentation about this data can be found [here](https://smoosavi.org/datasets/us_accidents).
# 
# The purpose of this analysis is to analyze the data and find out what are the key variables that impact the **severity** of the traffic accidents that happened in US and ultimately predict the severity of the accidents based on given variables through **data visulization** and **regression analysis** using Python.
# 
# ## Loading data
# 
# The analysis starts with loading data and necessary library.

# In[32]:


# Load necessary library
import os
import pandas as pd
import numpy as np
import math
import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import matplotlib.ticker as mtick
import seaborn as sns
import folium
import branca.colormap as cm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import geopandas as gpd
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn')
get_ipython().run_line_magic('matplotlib', 'inline')

plt.rcParams["figure.figsize"] = (15,8)


# Previewing the data and some summary statistics

# In[33]:


# Load and preview data 
accident = pd.read_csv("/Users/leo/Personal/Python/US Accident Analysis/US_Accidents_Dec19.csv")
accident.head()


# In[34]:


# Summary Statistics
accident.describe()


# Checking Columns for nas

# In[35]:


# Check each column for nas
accident.isnull().sum()


# Cleanning the data by removing unnecessary columns that won't be used in the analysis. Then review the cleaned dataset

# In[36]:


# Exclude unnecessary columns
exclude = ["TMC","End_Lat","End_Lng","Description","Number","Street","Timezone",
           "Airport_Code","Weather_Timestamp","Civil_Twilight","Nautical_Twilight","Astronomical_Twilight"]
accident_clean = accident.drop(exclude,axis=1)
accident_clean.head()


# In[37]:


# Check nas after excluding unnecessary columns
accident_clean.isnull().sum()


# To prepare the dataset for further analysis, some additional columns are added
# 
# - Time_Diff: Time difference between start time and end time of the accident
# - Year: Year of start time
# - Month: Month of start time
# - Day: Day of start time
# - Hour: Hour of start time
# 
# During this step, also excluded records in 2015 and 2020 where there aren't enough data in these two years. This should give us a relatively clean dataset to start the analysis.

# In[38]:


# Adding calculation of time difference of start and end time in minutes
accident_clean.Start_Time = pd.to_datetime(accident_clean.Start_Time)
accident_clean.End_Time = pd.to_datetime(accident_clean.End_Time)
accident_clean["Time_Diff"] = (accident_clean.End_Time - accident_clean.Start_Time).astype('timedelta64[m]')

accident_clean["Start_Date"] = accident_clean["Start_Time"].dt.date
accident_clean["End_Date"] = accident_clean["End_Time"].dt.date
accident_clean["Year"] = accident_clean["Start_Time"].dt.year
accident_clean["Month"] = accident_clean["Start_Time"].dt.month
accident_clean["Day"] = accident_clean["Start_Time"].dt.day
accident_clean["Hour"] = accident_clean["Start_Time"].dt.hour

# Excluding accidents in 2015 and 2020 where there's not enough data
accident_clean = accident_clean[(accident_clean["Year"] > 2015) & (accident_clean["Year"] < 2020)]
group = accident_clean.groupby(["Year"]).agg(Count = ('ID','count'))

# Verify data
accident_clean.head()


# ## Data Visualization
# 
# First group the data by **year** and **Severity** and use `size()` to count the records in each group, then use `unstack()` to pivot the result.
# 
# Most of the accidents falls under **Severity 2 and 3** and the number of accidents are increasing year over year.

# In[39]:


# Examine data
accident_clean.groupby(["Year","Severity"]).size().unstack()


# Plot the data to visualize year over year traffic accident trend

# In[40]:


# accident_clean.groupby(["Start_Date","Severity"])["ID"].count()

# Group by year and Group by year and severity
group_year = accident_clean.groupby(["Year"]).agg(Count = ('ID','count'))
group_year_sev = accident_clean.groupby(["Year","Severity"]).size().unstack()

# YoY Total Accident Count
# fig = plt.figure(figsize=(15,8))

# plt.subplot(1, 2, 1)
plt.plot(group_year.index, group_year["Count"])
plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
plt.xticks(np.arange(2016, 2020, 1.0))

plt.show


# Then plot the data by **Severity** and by **Year** to show the trend of accidents by year and by accident severity.
# 
# From the plot, we can see that:
# 
# - **Severity 2** accidents are *increasing* year over year in a rapid speed
# - **Severity 3** accidents have seen a *decrease* in 2019 compared to 2018
# - **Severity 1 and 4** are relatively flat year over year

# In[41]:


# YoY trend by severity, more in 2, 1 and 4 looks like flat, need to see in a bar plot
# fig = plt.figure(figsize=(15,8))
group_year_sev2 = accident_clean.groupby(["Year","Severity"]).agg(Count = ('ID','count')).reset_index()
sns.lineplot(x='Year',y='Count',hue="Severity",data=group_year_sev2,palette="Set1")
plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
plt.xticks(np.arange(2016, 2020, 1.0))
plt.show


# The following bar plots are also showing the similiar information then observed previously

# In[42]:


# YoY Severity Count
# group_year_sev = accident_clean.groupby(["Year","Severity"]).agg(Count = ('ID','count'))
# group_year_sev
accident_clean.groupby(["Year","Severity"]).size().unstack().plot(kind='bar',stacked=True)


# The 100% stacked bar plot shows the % breakdown of each severity by year.

# In[43]:


# Makes more sense to show stacked 100%, a different view
accident_clean.groupby(["Year","Severity"]).size().groupby(level=0).apply(
    lambda x: 100 * x / x.sum()
).unstack().plot(kind='bar',stacked=True)

plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
plt.legend(loc = 'upper right',title = 'Severity')
plt.show()


# The dataset also contains some variables regarding the weather condition when the accidents happened. We also want to examine those variables to see if any of the weather related variables have impact on the severity of the accidents.
# 
# We want to use boxplot from `seaborn` library to see how the weather related variable changed in different type of accidents.
# 
# Below is a boxplot of **Temperature** and **Severity**, we can see that there are almost no difference in median temparature in Severity 1,2 and 3, while lower medium temperature in severity 4, which might indicate that lower temperature might result to more severe accidents.

# In[44]:


# Boxplot to show if temperature has impact on the severity of the accident, 
# looks like the more severe accident has lower temperature
sns.boxplot(x="Severity", y="Temperature(F)", data=accident_clean, palette="Set1")


# Below is a boxplot of **Humidity** and **Severity**, similiarly, we can see that higher humidity might lead to more severe accidents.

# In[45]:


# Boxplot to show if temperature has impact on the severity of the accident, 
# looks like the more severe accident has lower temperature
sns.boxplot(x="Severity", y="Humidity(%)", data=accident_clean, palette="Set1")


# Below is a boxplot of **Wind Chill** and **Severity**, similiarly, we can see that lower Wind Chill might lead to more severe accidents.

# In[46]:


# Examine wind chill and accident severity, lower wind chill cause more severe accidents
sns.boxplot(x="Severity", y="Wind_Chill(F)", data=accident_clean, palette="Set1")


# There are some other categorical variables that might have impact on the severity of the accidents, we are using `crosstab()` function to get a count of the accidents in each group

# In[47]:


# Count of Severity by Sunrise_Sunset to see if more severe accidents happened at night
pd.crosstab(accident_clean["Severity"], accident_clean["Sunrise_Sunset"], 
            rownames=['Severity'], colnames=['Sunrise_Sunset'])


# The 100% stacked barplot below indicates that more severe accidents are happening during the night then in the day time.

# In[48]:


# Severity 1 and two has same % between day and night while 3 and 4 has more accidents % at nights
accident_clean.groupby(["Severity","Sunrise_Sunset"]).size().groupby(level=0).apply(
    lambda x: 100 * x / x.sum()
).unstack().plot(kind='bar',stacked=True)

plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
plt.legend(loc = 'upper right',title = 'Sunrise/Sunset')
plt.show()


# The plot below shows that most accidents are happening on the **right** side of the road

# In[49]:


# Most accidents happened on the right side of the road
# Severity 3 has more on right then the left side of the road
accident_clean[accident_clean.Side != " "].groupby(["Severity","Side"]).size().groupby(level=0).apply(
    lambda x: 100 * x / x.sum()
).unstack().plot(kind='bar',stacked=True)
plt.legend(loc = 'upper right')
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
plt.show()


# The plor below shows by month, the break down of the accidents by severity, more severe accidents (Severity 3 and 4) happened in June and July.

# In[50]:


# Examining Severity by month, most severe accidents (3 and 4) happened in June and July
accident_clean.groupby(["Month","Severity"]).size().groupby(level=0).apply(
    lambda x: 100 * x / x.sum()
).unstack().plot(kind='bar',stacked=True,figsize = (15,8))
plt.legend(loc = 'upper right')
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
plt.show()


# For numeric variables that are severely skewed, boxplot won't be a good interpretation visually, for those variables, we are using `groupby()` to get the mean by **Severity**

# In[51]:


sns.distplot(accident_clean['Distance(mi)'])


# In[52]:


accident_clean.groupby('Severity')['Distance(mi)'].mean()
# df[(df['year'] > 2012) & (df['reports'] < 30)]
# sns.boxplot(x="Severity", y="Wind_Speed(mph)", 
#             data=accident_clean[accident_clean["Wind_Speed(mph)"] <= 50], palette="Set1")


# In[53]:


accident_clean.groupby('Severity')['Time_Diff'].median()
# sns.boxplot(x="Severity", y="Time_Diff", data=accident_clean, palette="Set1")


# As expected, more severe accidents will affect longer distances and last longer time.
# 
# There are another set of variables that describes the condition of the road, wuch as whether there's a bump in the road, or whether there's traffic signal.
# 
# The following analysis will focus on these variables.
# 
# First select these variables and Severity into a new dataset called `accident road`.

# In[54]:


accident_road = accident[['Severity','Amenity', 'Bump','Crossing','Give_Way',
                         'Junction','No_Exit','Railway','Roundabout','Station',
                         'Stop','Traffic_Calming','Traffic_Signal','Turning_Loop']]
accident_road.head()


# In order to plot these variables in the same grid, we used `melt()` and then `groupby()` to reshape the dataset.

# In[55]:


accident_road_melt = pd.melt(accident_road,id_vars =['Severity'],value_vars=['Amenity', 'Bump','Crossing','Give_Way',
                         'Junction','No_Exit','Railway','Roundabout','Station',
                         'Stop','Traffic_Calming','Traffic_Signal','Turning_Loop'])
group_road = accident_road_melt.groupby(["Severity","variable","value"]).agg(Count = ('value','count')).reset_index()
group_road.head()
# pd.pivot_table(data=accident_road_melt,index='Severity',columns=['value'],aggfunc='count')


# Following plot shows for each variable break down by accident severity, the count of True and False.
# 
# From the plot, **Corssing**, **Junction**, **Tracffic Signal** have some impact.

# In[56]:


g = sns.catplot(x="Severity", y="Count",
            hue="value", col="variable",
            col_wrap=3, data=group_road, kind="bar",
            height=4, aspect=.7)
g.fig.set_figwidth(15)
g.fig.set_figheight(8)


# Similiarly, the output below summarized the Trues and Falses by each variables grouped by Severity.

# In[57]:


# Count of True and False of each road condition and group by Severity
(accident_road.set_index('Severity')
 .groupby(level='Severity')
# to do the count of columns nj, wd, wpt against the column ptype using 
# groupby + value_counts
 .apply(lambda g: g.apply(pd.value_counts))
 .unstack(level=1)
 .fillna(0))


# Using the `heatmap()` function from `seaborn` library, we want to know for each month, which day of the month are more likely to have more accidents.
# 
# From the heatmap below, we can see that more accidents are happening in the second half of the year and it looked like high accidents days are spead randomly and varied in each month.

# In[58]:


# More accidents are happening in the second half of the year
group_day = accident_clean.groupby(["Month","Day"]).size().unstack()
ax = sns.heatmap(group_day, cmap="YlGnBu",linewidths=0.1)


# Similiarly, we want to see in what time of the day, more accidents will happen using a heatmap.
# 
# As expected, Most accidents happened between 7 and 8, which is the morning rush hour. Morning rush hour have much more accidents then the afternoon rush hour, which is 4 to 6 in the afternoon. This trend is observed consistently throughout each day of the month.

# In[59]:


# Most accidents happened between 7 and 8, which is the morning rush hour
# morning rush hour have much more accidents then the afternoon rush hour, which is 4 to 6 in the afternoon

group_hour = accident_clean.groupby(["Day","Hour"]).size().unstack()
ax = sns.heatmap(group_hour, cmap="YlGnBu",linewidths=0.1)


# After wrapping up the data visualization of numeric and categorical variables, we also want to visualiza geo data using a map.
# 
# The static map below are plot using `Basemap` from `mpl_toolkits`. The scatter plot is a good way to start.

# In[60]:


from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt

# A basic map
# m=Basemap(llcrnrlon=-160, llcrnrlat=-75,urcrnrlon=160,urcrnrlat=80)
m = Basemap(llcrnrlat=20,urcrnrlat=50,llcrnrlon=-130,urcrnrlon=-60)

# m = Basemap(projection='lcc', resolution='h', lat_0=37.5, lon_0=-119,
#             width=1E6, height=1.2E6,
#            llcrnrlat=20,urcrnrlat=50,llcrnrlon=-130,urcrnrlon=-60)

m.shadedrelief()
m.drawcoastlines(color='gray') 
m.drawcountries(color='gray') 
m.drawstates(color='gray')

lat = accident_clean.Start_Lat.tolist()
lon = accident_clean.Start_Lng.tolist()

x,y = m(lon,lat)
m.plot(x,y,'bo',alpha = 0.2)


# The next step is the plot a Choropleth map using `folium` library.
# 
# First download the US shape file from [US Census](https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html). The shapefile used in this analysis are [state shape file](https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_500k.zip) and [zip code shape file](https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_zcta510_500k.zip).
# 
# We are using `geopandas` to load the shapefile, then `groupby()` state or zipcode and `merge()` to get the final dataset for map plotting.

# In[61]:


# US Shape file from https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_500k.zip
shapefile_state = '/Users/leo/Personal/Python/cb_2018_us_state_500k/cb_2018_us_state_500k.shp'

#Read shapefile using Geopandas
gdf_state = gpd.read_file(shapefile_state)
gdf_state.head()


# In[62]:


group_state = accident_clean.groupby(["State"]).agg(Count = ('ID','count'))
group_state.reset_index(level=0, inplace=True)
group_state[:5]


# In[63]:


# Merge shape file with accident data
state_map = gdf_state.merge(group_state, left_on = 'STUSPS', right_on = 'State')
state_map.head()


# In[64]:


# group_state
m = folium.Map(location=[37, -102], zoom_start=4)

folium.TileLayer('CartoDB positron',name="Light Map",control=False).add_to(m)

# myscale = (state_map['Count'].quantile((0,0.2,0.4,0.6,0.8,1))).tolist()

m.choropleth(
    geo_data=state_map,
    name='Choropleth',
    data=state_map,
    columns=['State','Count'],
    key_on="feature.properties.State",
    fill_color='YlGnBu',
#     threshold_scale=myscale,
    fill_opacity=1,
    line_opacity=0.2,
    legend_name='Count of Accidents',
    smooth_factor=0
)


style_function = lambda x: {'fillColor': '#ffffff', 
                            'color':'#000000', 
                            'fillOpacity': 0.1, 
                            'weight': 0.1}
highlight_function = lambda x: {'fillColor': '#000000', 
                                'color':'#000000', 
                                'fillOpacity': 0.50, 
                                'weight': 0.1}

toolkit = folium.features.GeoJson(
    state_map,
    style_function=style_function, 
    control=False,
    highlight_function=highlight_function, 
    tooltip=folium.features.GeoJsonTooltip(
        fields=['State','Count'],
        aliases=['State: ','# of Accidents: '],
        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;") 
    )
)

m.add_child(toolkit)
m.keep_in_front(toolkit)
folium.LayerControl().add_to(m)

m


# Hover over each state will pop a toolkit that show the state name and the count of accidents. Most of the accidents in this dataset are in **California**, followed by **Texas** and **Florida**.
# 
# Then we want to focus on the State of California to see which area of the state have the most accidents. using the similar code that produced US State Choropleth map. the following code produced Calfornia State zip code Choropleth map.

# In[65]:


shapefile_zip = '/Users/leo/Personal/Python/cb_2018_us_zcta510_500k/cb_2018_us_zcta510_500k.shp'

#Read shapefile using Geopandas
gdf_zip = gpd.read_file(shapefile_zip)
gdf_zip.head()


# In[66]:


accident_clean['Zipcode'] = accident_clean['Zipcode'].str[:5]
# accident_clean['Zipcode']
group_ca = accident_clean[accident_clean["State"] == 'CA'].groupby(["Zipcode","City"]).agg(Count = ('ID','count'))
group_ca.reset_index(inplace=True)
group_ca[:5]


# In[67]:


ca_map = gdf_zip.merge(group_ca, left_on = 'ZCTA5CE10', right_on = 'Zipcode')
ca_map.head()


# In[68]:


# Get center of the map
x_map=ca_map.centroid.x.mean()
y_map=ca_map.centroid.y.mean()
print(x_map,y_map)


# In[69]:


m2 = folium.Map(location=[y_map, x_map], zoom_start=6)

folium.TileLayer('CartoDB positron',name="Light Map",control=False).add_to(m2)

# myscale = (state_map['Count'].quantile((0,0.2,0.4,0.6,0.8,1))).tolist()

m2.choropleth(
    geo_data=ca_map,
    name='Choropleth',
    data=ca_map,
    columns=['Zipcode','Count'],
    key_on="feature.properties.Zipcode",
    fill_color='YlGnBu',
#     threshold_scale=myscale,
    fill_opacity=1,
    line_opacity=0.2,
    legend_name='Count of Accidents',
    smooth_factor=0
)


style_function = lambda x: {'fillColor': '#ffffff', 
                            'color':'#000000', 
                            'fillOpacity': 0.1, 
                            'weight': 0.1}
highlight_function = lambda x: {'fillColor': '#000000', 
                                'color':'#000000', 
                                'fillOpacity': 0.50, 
                                'weight': 0.1}

toolkit = folium.features.GeoJson(
    ca_map,
    style_function=style_function, 
    control=False,
    highlight_function=highlight_function, 
    tooltip=folium.features.GeoJsonTooltip(
        fields=['Zipcode','City','Count'],
        aliases=['Zipcode: ','City','# of Accidents: '],
        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;") 
    )
)

m2.add_child(toolkit)
m2.keep_in_front(toolkit)
folium.LayerControl().add_to(m2)

m2


# As expected, most accidents are happened in metropolitans such as **San Francisco Bay area** and **Los Angeles Area**. However, they are not necessarily the downtown area, but its surrounding cities.
# 
# ## Regression Analysis
# 
# After data visualization, we have a brief understanding of the dataset as well as the variables to choose for the regression analysis.
# 
# Since the dependent variable is **Severity**, which is a categorical variable. Logestic Regression model is chosen to perform the analysis.
# 
# The independent variables are selected based on the previous analysis:
# 
# - Distance(mi)
# - Time_Diff
# - Temperature(F)
# - Wind_Chill(F)
# - Humidity(%)
# - Pressure(in)
# - Visibility(mi)
# - Wind_Speed(mph)
# - Precipitation(in)
# 
# Using `sklearn` library, first separate the dependent variable **y** and independent variable **x** and store them in separate objects. The use `scale()` from `preprocessing` module to properly scale the **x** variable before the regression.
# 
# Next step will be using `train_test_split` from `sklearn.model_selection` module to split test and train dataset. Finally fit the logestic regression model.

# In[70]:


variable = ["Severity","Distance(mi)","Time_Diff","Temperature(F)","Wind_Chill(F)","Humidity(%)",
           "Pressure(in)","Visibility(mi)","Wind_Speed(mph)","Precipitation(in)"]
accident_model = accident_clean[variable]
accident_model = accident_model.dropna()
# accident_model['Severity'] = np.where(accident_model['Severity']<=2, 0, 1)
accident_model.head()


# In[71]:


Y = accident_model.loc[:,'Severity'].values
X = accident_model.loc[:,'Distance(mi)':'Precipitation(in)'].values

standardized_X = preprocessing.scale(X)
train_x, test_x, train_y, test_y = train_test_split(standardized_X,Y , test_size=0.3, random_state=0)

model = LogisticRegression(solver='lbfgs',multi_class='multinomial',max_iter=1000)
model.fit(train_x, train_y)


# Use `score()` to evaluate the model, the model gets 0.72, which is decent for the first try.

# In[72]:


model.score(test_x, test_y)


# Finally use `confusion_matrix` to see how many of the prediction are correct and how many are incorrect.
# 
# We can see that **Severity 2 and 3** are the most accidents. The model did a good job in identifing **Severity 2** but not **Severity 3**.

# In[73]:


model_y = model.predict(test_x)

mat = confusion_matrix(test_y,model_y)
sns.heatmap(mat, square=True, annot=True, cbar=False) 
plt.xlabel('predicted value')
plt.ylabel('true value')

