import streamlit as st
import pandas as pd
import preprocessor, helper
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

import streamlit as st
from recommendation import recommend_sport
from clustering import preprocess_for_clustering, perform_kmeans_clustering



st.set_page_config(
    page_title="Olympic Data Analysis",  # Tab ka title
    page_icon="🏅",                      # Tab me dikhne wala emoji icon
    layout="wide",                       # Page layout (can be 'centered' or 'wide')
    initial_sidebar_state="expanded"     # Sidebar open by default
)


import streamlit as st

def set_bg_and_sidebar_color():
    st.markdown(
        """
        <style>
            /* Main background - light orange */
            .stApp {
                background-color: #CBC3E3;
                color: #1a1a1a;
            }

            /* Sidebar - light green */
            section[data-testid="stSidebar"] {
                background-color: #e6ffe6;
                color: #1a1a1a;
            }

            /* Taskbar/Header - light grey */
            header[data-testid="stHeader"] {
                background-color: #006400;
            }

            /* Optional: Top bar text color (if needed) */
            header[data-testid="stHeader"] h1 {
                color: #333333;
            }

            /* Force dark, readable text everywhere in main app and sidebar */
            .stApp, .stApp p, .stApp span, .stApp label,
            .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5,
            section[data-testid="stSidebar"] p,
            section[data-testid="stSidebar"] span,
            section[data-testid="stSidebar"] label,
            section[data-testid="stSidebar"] h1,
            section[data-testid="stSidebar"] div {
                color: #1a1a1a !important;
            }

            /* Table / dataframe text */
            .stApp table, .stApp th, .stApp td {
                color: #1a1a1a !important;
            }

            /* Radio button labels in sidebar */
            div[role="radiogroup"] label p {
                color: #1a1a1a !important;
            }

            /* Number inputs, select boxes, text inputs - light silver background */
            .stNumberInput input,
            .stTextInput input,
            div[data-baseweb="select"] > div,
            div[data-baseweb="select"] div {
                background-color: #D3D3D3 !important;
                color: #1a1a1a !important;
            }

            /* Plus/minus step buttons on number inputs */
            .stNumberInput button {
                background-color: #D3D3D3 !important;
                color: #1a1a1a !important;
            }

            /* Dropdown menu (opened list) background */
            ul[role="listbox"] {
                background-color: #D3D3D3 !important;
            }
            ul[role="listbox"] li {
                color: #1a1a1a !important;
            }

            /* Buttons (e.g. Predict Medal) */
            .stButton button {
                background-color: #D3D3D3 !important;
                color: #1a1a1a !important;
                border: 1px solid #a9a9a9 !important;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

# Place this right after imports
set_bg_and_sidebar_color()


# Call the function to apply the styles
set_bg_and_sidebar_color()

df = pd.read_csv('athlete_events.csv')
region_df = pd.read_csv('noc_regions.csv')
df = preprocessor.preprocess(df, region_df)

# Train model if not already saved
if not os.path.exists('medal_predictor.pkl'):
    model_df = df[['Age', 'Height', 'Weight', 'Sex', 'Sport', 'region', 'Medal']].dropna()
    model_df = model_df[model_df['Medal'].notna()]

    # Encode categorical features
    le_sex = LabelEncoder()
    le_sport = LabelEncoder()
    le_region = LabelEncoder()
    le_medal = LabelEncoder()

    model_df['Sex'] = le_sex.fit_transform(model_df['Sex'])
    model_df['Sport'] = le_sport.fit_transform(model_df['Sport'])
    model_df['region'] = le_region.fit_transform(model_df['region'])
    model_df['Medal'] = le_medal.fit_transform(model_df['Medal'])

    X = model_df[['Age', 'Height', 'Weight', 'Sex', 'Sport', 'region']]
    y = model_df['Medal']

    # Impute any missing numerical values
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, 'medal_predictor.pkl')
    joblib.dump((le_sex, le_sport, le_region, le_medal), 'label_encoders.pkl')

# Train performance prediction model if not already saved
if not os.path.exists('performance_predictor.pkl'):
    # Build performance dataset: improvement between consecutive Olympics
    perf_records = []
    for name, grp in df.sort_values(['Name','Year']).groupby('Name'):
        grp = grp.dropna(subset=['Medal','Age'])
        if grp.shape[0] < 2: continue
        years = sorted(grp['Year'].unique())
        for i in range(len(years)-1):
            now = grp[grp['Year']==years[i]]
            nxt = grp[grp['Year']==years[i+1]]
            if now.empty or nxt.empty: continue
            perf_records.append({
                'AgeNow': now['Age'].values[0],
                'AgeNext': nxt['Age'].values[0],
                'MedalsNow': now['Medal'].count(),
                'Sport': now['Sport'].values[0],
                'region': now['region'].values[0],
                'Improved': int(nxt['Medal'].count() > now['Medal'].count())
            })
    perf_df = pd.DataFrame(perf_records)

    # Encode categorical
    le_sport_perf = LabelEncoder()
    le_region_perf = LabelEncoder()
    perf_df['Sport'] = le_sport_perf.fit_transform(perf_df['Sport'].astype(str))
    perf_df['region'] = le_region_perf.fit_transform(perf_df['region'].astype(str))

    Xp = perf_df[['AgeNow','AgeNext','MedalsNow','Sport','region']]
    yp = perf_df['Improved']

    # Impute missing (if any)
    imp_perf = SimpleImputer(strategy='mean')
    Xp = pd.DataFrame(imp_perf.fit_transform(Xp), columns=Xp.columns)

    Xp_train, Xp_test, yp_train, yp_test = train_test_split(Xp, yp, test_size=0.2, random_state=42)
    perf_model = LogisticRegression(max_iter=1000)
    perf_model.fit(Xp_train, yp_train)

    joblib.dump(perf_model, 'performance_predictor.pkl')
    joblib.dump((le_sport_perf, le_region_perf), 'performance_encoders.pkl')

st.sidebar.title("Olympics Analysis")
st.sidebar.image('file_00000000247c61f7b7da8bed4cdc44be.png')
user_menu = st.sidebar.radio('Select an Option',(
    'Medal Tally','Overall Analysis','Country-wise Analysis',
    'Athlete wise Analysis','Medal Prediction','Performance Prediction', 'Sport Recommendation System','Map Visualization','Clustering Analysis'
))

if user_menu == 'Medal Tally':
    years, country = helper.country_year_list(df)
    selected_year = st.sidebar.selectbox("Select Year", years)
    selected_country = st.sidebar.selectbox("Select Country", country)

    medal_tally = helper.fetch_medal_tally(df, selected_year, selected_country)

    if selected_year == 'Overall' and selected_country == 'Overall':
        st.title("Overall Tally")
    elif selected_year != 'Overall' and selected_country == 'Overall':
        st.title("Medal Tally in " + str(selected_year) + " Olympics")
    elif selected_year == 'Overall' and selected_country != 'Overall':
        st.title(selected_country + " overall performance")
    else:
        st.title(selected_country + " performance in " + str(selected_year) + " Olympics")

    st.table(medal_tally)

elif user_menu == 'Overall Analysis':
    editions = df['Year'].nunique() - 1
    cities = df['City'].nunique()
    sports = df['Sport'].nunique()
    events = df['Event'].nunique()
    athletes = df['Name'].nunique()
    nations = df['region'].nunique()

    st.title("Top Statistics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Editions", editions)
    col2.metric("Hosts", cities)
    col3.metric("Sports", sports)

    col1, col2, col3 = st.columns(3)
    col1.metric("Events", events)
    col2.metric("Nations", nations)
    col3.metric("Athletes", athletes)

    st.title("Participating Nations over the years")
    fig = px.line(helper.data_over_time(df, 'region'), x="Edition", y="Count")
    st.plotly_chart(fig)

    st.title("Events over the years")
    fig = px.line(helper.data_over_time(df, 'Event'), x="Edition", y="Count")
    st.plotly_chart(fig)

    st.title("Athletes over the years")
    fig = px.line(helper.data_over_time(df, 'Name'), x="Edition", y="Count")
    st.plotly_chart(fig)

    st.title("No. of Events over time(Every Sport)")
    fig, ax = plt.subplots(figsize=(20, 20))
    x = df.drop_duplicates(['Year', 'Sport', 'Event'])
    ax = sns.heatmap(x.pivot_table(index='Sport', columns='Year', values='Event', aggfunc='count').fillna(0).astype(int), annot=True)
    st.pyplot(fig)

    st.title("Most successful Athletes")
    sport_list = sorted(df['Sport'].unique().tolist())
    sport_list.insert(0, 'Overall')
    selected_sport = st.selectbox('Select a Sport', sport_list)
    x = helper.most_successful(df, selected_sport)
    st.table(x)

elif user_menu == 'Country-wise Analysis':
    st.sidebar.title('Country-wise Analysis')
    country_list = sorted(df['region'].dropna().unique().tolist())
    selected_country = st.sidebar.selectbox('Select a Country', country_list)

    country_df = helper.yearwise_medal_tally(df, selected_country)
    st.title(selected_country + " Medal Tally over the years")
    st.plotly_chart(px.line(country_df, x="Year", y="Medal"))

    st.title(selected_country + " excels in the following sports")
    pt = helper.country_event_heatmap(df, selected_country)
    fig, ax = plt.subplots(figsize=(20, 20))
    ax = sns.heatmap(pt, annot=True)
    st.pyplot(fig)

    st.title("Top 10 athletes of " + selected_country)
    top10_df = helper.most_successful_countrywise(df, selected_country)
    st.table(top10_df)

elif user_menu == 'Athlete wise Analysis':
    athlete_df = df.drop_duplicates(subset=['Name', 'region'])

    x1 = athlete_df['Age'].dropna()
    x2 = athlete_df[athlete_df['Medal'] == 'Gold']['Age'].dropna()
    x3 = athlete_df[athlete_df['Medal'] == 'Silver']['Age'].dropna()
    x4 = athlete_df[athlete_df['Medal'] == 'Bronze']['Age'].dropna()

    fig = ff.create_distplot([x1, x2, x3, x4], ['Overall Age', 'Gold Medalist', 'Silver Medalist', 'Bronze Medalist'], show_hist=False)
    st.title("Distribution of Age")
    st.plotly_chart(fig)

    x, name = [], []
    famous_sports = ['Basketball', 'Judo', 'Football', 'Tug-Of-War', 'Athletics', 'Swimming', 'Badminton', 'Sailing',
                     'Gymnastics', 'Art Competitions', 'Handball', 'Weightlifting', 'Wrestling', 'Water Polo', 'Hockey',
                     'Rowing', 'Fencing', 'Shooting', 'Boxing', 'Taekwondo', 'Cycling', 'Diving', 'Canoeing', 'Tennis',
                     'Golf', 'Softball', 'Archery', 'Volleyball', 'Synchronized Swimming', 'Table Tennis', 'Baseball',
                     'Rhythmic Gymnastics', 'Rugby Sevens', 'Beach Volleyball', 'Triathlon', 'Rugby', 'Polo', 'Ice Hockey']
    for sport in famous_sports:
        temp_df = athlete_df[athlete_df['Sport'] == sport]
        x.append(temp_df[temp_df['Medal'] == 'Gold']['Age'].dropna())
        name.append(sport)

    fig = ff.create_distplot(x, name, show_hist=False, show_rug=False)
    st.title("Distribution of Age wrt Sports (Gold Medalist)")
    st.plotly_chart(fig)

    sport_list = sorted(df['Sport'].unique().tolist())
    sport_list.insert(0, 'Overall')
    st.title('Height Vs Weight')
    selected_sport = st.selectbox('Select a Sport', sport_list)
    temp_df = helper.weight_v_height(df, selected_sport)
    fig, ax = plt.subplots()
    ax = sns.scatterplot(x='Weight', y='Height', hue='Medal', style='Sex', s=60, data=temp_df)
    st.pyplot(fig)

    st.title("Men Vs Women Participation Over the Years")
    final = helper.men_vs_women(df)
    fig = px.line(final, x="Year", y=["Male", "Female"])
    st.plotly_chart(fig)

elif user_menu == 'Medal Prediction':
    st.title("🎯 Medal Prediction for an Athlete")

    age = st.number_input("Age", min_value=10, max_value=60)
    height = st.number_input("Height (cm)", min_value=100, max_value=250)
    weight = st.number_input("Weight (kg)", min_value=30, max_value=200)
    sex = st.selectbox("Sex", ['M', 'F'])
    sport = st.selectbox("Sport", sorted(df['Sport'].dropna().unique()))
    region = st.selectbox("Country", sorted(df['region'].dropna().unique()))

    if st.button("Predict Medal"):
        model = joblib.load('medal_predictor.pkl')
        le_sex, le_sport, le_region, le_medal = joblib.load('label_encoders.pkl')

        input_data = pd.DataFrame([[
            age,
            height,
            weight,
            le_sex.transform([sex])[0],
            le_sport.transform([sport])[0],
            le_region.transform([region])[0]
        ]], columns=['Age', 'Height', 'Weight', 'Sex', 'Sport', 'region'])

        prediction = model.predict(input_data)[0]
        medal = le_medal.inverse_transform([prediction])[0]

        st.subheader(f"🏅 Predicted Medal: {medal}")


elif user_menu == 'Performance Prediction':
    st.title("\U0001F3C6 Will the Athlete Improve in the Next Olympics?")

    age_now = st.number_input("Current Age", min_value=10, max_value=60)
    age_next = st.number_input("Next Olympic Age", min_value=14, max_value=65)
    medals_now = st.number_input("Current Total Medals", min_value=0, max_value=10)
    sport = st.selectbox("Sport", sorted(df['Sport'].dropna().unique()))
    region = st.selectbox("Country", sorted(df['region'].dropna().unique()))

    if st.button("Predict Performance"):
        perf_model = joblib.load('performance_predictor.pkl')
        le_sport_perf, le_region_perf = joblib.load('performance_encoders.pkl')

        input_perf = pd.DataFrame([[
            age_now,
            age_next,
            medals_now,
            le_sport_perf.transform([sport])[0],
            le_region_perf.transform([region])[0]
        ]], columns=['AgeNow', 'AgeNext', 'MedalsNow', 'Sport', 'region'])

        prediction = perf_model.predict(input_perf)[0]
        if prediction:
            st.success("✅ Likely to improve in the next Olympics!")
        else:
            st.warning("⚠️ Unlikely to improve based on past patterns.")






if user_menu == 'Map Visualization':
    st.title("🌍 Olympic Medal Map - Filtered View")

    # Load and preprocess data
    df = pd.read_csv('athlete_events.csv')
    region_df = pd.read_csv('noc_regions.csv')
    df = preprocessor.preprocess(df, region_df)

    medal_df = df.dropna(subset=['Medal'])

    year_list = sorted(medal_df['Year'].unique().tolist())
    year_list.insert(0, 'Overall')

    sport_list = sorted(medal_df['Sport'].dropna().unique().tolist())
    sport_list.insert(0, 'Overall')

    selected_year = st.selectbox("Select Year", year_list)
    selected_sport = st.selectbox("Select Sport", sport_list)

    filtered_df = medal_df.copy()
    if selected_year != 'Overall':
        filtered_df = filtered_df[filtered_df['Year'] == selected_year]
    if selected_sport != 'Overall':
        filtered_df = filtered_df[filtered_df['Sport'] == selected_sport]

    country_medals = filtered_df.groupby('region').count()['Medal'].reset_index()
    country_medals.columns = ['Country', 'Medal_Count']

    fig = px.choropleth(country_medals,
                        locations='Country',
                        locationmode='country names',
                        color='Medal_Count',
                        color_continuous_scale='Turbo',
                        title='Medal Distribution')

    st.plotly_chart(fig, use_container_width=True)





elif user_menu == 'Sport Recommendation System':
    st.title("🎯 Sport Recommendation Based on Your Body Profile")
    age = st.number_input('Enter Age', 10, 60)
    height = st.number_input('Enter Height (cm)', 100, 250)
    weight = st.number_input('Enter Weight (kg)', 30, 200)
    gender = st.radio('Gender', ('M', 'F'))

    if st.button('Recommend Sport'):
        from recommendation import recommend_sport
        sport = recommend_sport(df, age, height, weight, gender)
        st.success(f"Recommended Sport: 🏅 **{sport}**")







elif user_menu == 'Clustering Analysis':
    st.title("🎯 Olympic Clustering Analysis")

    num_clusters = st.slider("Select number of clusters", 2, 10, step=1)

    cluster_data = preprocess_for_clustering(df)
    clustered = perform_kmeans_clustering(cluster_data, num_clusters)

    st.subheader("Clustered Country Data")
    st.dataframe(clustered.sort_values(by='Cluster'))

    cluster_selected = st.selectbox("Select a Cluster to Explore", sorted(clustered['Cluster'].unique()))
    filtered = clustered[clustered['Cluster'] == cluster_selected]
    st.write(f"Countries in Cluster {cluster_selected}:")
    st.dataframe(filtered[['Country', 'Medal_Count', 'Participation_Count']])

    # Optional: plot clusters
    import plotly.express as px
    fig = px.scatter(clustered, x='Participation_Count', y='Medal_Count', color='Cluster',
                     hover_data=['Country'], title='Country Clustering by Olympic Participation & Medals')
    st.plotly_chart(fig)