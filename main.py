import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.cluster import KMeans

with open('time_series_classification_data_20.json', 'r') as file:
    data = json.load(file)

records = []
for user, visits in data.items():
    for visit in visits:
        visit['user'] = user
        visit['ts'] = datetime.strptime(visit['ts'], "%Y-%m-%d %H:%M:%S.%f")
        records.append(visit)

df = pd.DataFrame(records)
grouped = df.groupby([df['ts'].dt.date, 'venue_type']).size().reset_index(name='counts')


# STEP 1
plt.figure(figsize=(14, 7))
for venue_type in grouped['venue_type'].unique():
    subset = grouped[grouped['venue_type'] == venue_type]
    plt.plot(subset['ts'], subset['counts'], marker='o', label=venue_type)

plt.xlabel('Date')
plt.ylabel('Number of Visits')
plt.title('Number of Visits Over Time by Venue Type')
plt.legend(title='Venue Type')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


venue_counts = df['venue_type'].value_counts()
plt.figure(figsize=(10, 6))
plt.bar(venue_counts.index, venue_counts.values)
plt.xlabel('Venue Type')
plt.ylabel('Total Number of Visits')
plt.title('Total Number of Visits per Venue Type')
plt.show()

# STEP 2
df['day_of_month'] = df['ts'].dt.day
df['hour_of_day'] = df['ts'].dt.hour


# get the count of visits to each venue type for each user
user_venue_counts = df.pivot_table(index='user', columns='venue_type', aggfunc='size', fill_value=0)

# get the count of visits for each day of the month for each user
user_day_counts = df.pivot_table(index='user', columns='day_of_month', aggfunc='size', fill_value=0)

# get the count of visits for each hour of the day for each user
user_hour_counts = df.pivot_table(index='user', columns='hour_of_day', aggfunc='size', fill_value=0)

user_features = pd.concat([user_venue_counts, user_day_counts, user_hour_counts], axis=1)


# STEP 3
# how many days have passed 
df['days_since_start'] = (df['ts'] - datetime(2023, 11, 16)).dt.days

# how many full weeks have passed
df['weight'] = 1 / (1 + df['days_since_start'] // 7)

# first week have a weight: 1/1 = 1. Second week has a weight of 1/2 = 0.5, ...

weighted_counts = df.groupby(['user', 'venue_type'])['weight'].sum().reset_index()
user_venue_weights = weighted_counts.pivot_table(index='user', columns='venue_type', values='weight', fill_value=0)

kmeans = KMeans(n_clusters=4, random_state=42)
user_venue_weights['cluster'] = kmeans.fit_predict(user_venue_weights)


# STEP 4: User Classification
df['hour'] = df['ts'].dt.hour
df['day_of_week'] = df['ts'].dt.dayofweek

user_preferences = {}

for user, group in df.groupby('user'):
    sun_lover = group['hour'].between(6, 18).mean() > 0.5
    moon_lover = not sun_lover
    weekend_lover = group['day_of_week'].isin([5, 6]).mean() > 0.5
    weekday_lover = not weekend_lover
    venue_type = group['venue_type'].mode()[0]

    if sun_lover:
        day_pref = 'Sun lover'
    else:
        day_pref = 'Moon lover'
        
    if weekend_lover:
        week_pref = 'Weekend lover'
    else:
        week_pref = 'Weekday lover'
    
    venue_pref = f"{venue_type.capitalize()} lover"
    
    user_preferences[user] = {
        "Day Preference": day_pref,
        "Week Preference": week_pref,
        "Venue Preference": venue_pref
    }

for user, prefs in user_preferences.items():
    print(f"{user}: {prefs}")
