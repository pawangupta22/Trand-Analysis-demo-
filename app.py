import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ===============================
# 🎯 APP TITLE
# ===============================
st.title("Social Media Trend Analysis Dashboard")

# ===============================
# 📂 FILE UPLOAD
# ===============================
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("📌 Raw Data")
    st.dataframe(df.head())

    # ===============================
    # 📌 CLEAN DATA
    # ===============================
    df['Post_Date'] = pd.to_datetime(df['Post_Date'], errors='coerce')
    df = df.dropna()
    df = df.drop_duplicates()

    # ===============================
    # 📌 FEATURE ENGINEERING
    # ===============================
    df['Total_Engagement'] = df['Likes'] + df['Shares'] + df['Comments']

    df_original = df.copy()

    # ===============================
    # 📌 ENCODING
    # ===============================
    le_platform = LabelEncoder()
    le_content = LabelEncoder()
    le_region = LabelEncoder()

    df['Platform'] = le_platform.fit_transform(df['Platform'])
    df['Content_Type'] = le_content.fit_transform(df['Content_Type'])
    df['Region'] = le_region.fit_transform(df['Region'])

    # ===============================
    # 📌 K-MEANS
    # ===============================
    features = df[['Platform', 'Content_Type', 'Region',
                   'Views', 'Likes', 'Shares', 'Comments', 'Total_Engagement']]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=4, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # ===============================
    # 📌 TREND LABEL
    # ===============================
    cluster_engagement = df.groupby('Cluster')['Total_Engagement'].mean()
    trending_cluster = cluster_engagement.idxmax()

    df['Trending'] = df['Cluster'].apply(lambda x: 1 if x == trending_cluster else 0)

    # ===============================
    # 📌 RANDOM FOREST
    # ===============================
    X = df[['Platform', 'Content_Type', 'Region',
            'Views', 'Likes', 'Shares', 'Comments']]

    y = df['Trending']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # ===============================
    # 📊 MODEL PERFORMANCE
    # ===============================
    st.subheader("Model Performance")
    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.text(classification_report(y_test, y_pred))

    # ===============================
    # 🔥 TRENDING DATA
    # ===============================
    df['Platform'] = df_original['Platform']
    df['Content_Type'] = df_original['Content_Type']
    df['Hashtag'] = df_original['Hashtag']

    trending_df = df[df['Trending'] == 1]

    # ===============================
    # 🔥 TOP HASHTAGS
    # ===============================
    st.subheader("Top Trending Hashtags")

    top_hashtags = (
        trending_df.groupby('Hashtag')['Total_Engagement']
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    st.dataframe(top_hashtags.head(10))

    # ===============================
    # 📈 TREND SCORE
    # ===============================
    st.subheader("Trend Score")

    top_hashtags['Score'] = 100 * (
        (top_hashtags['Total_Engagement'] - top_hashtags['Total_Engagement'].min()) /
        (top_hashtags['Total_Engagement'].max() - top_hashtags['Total_Engagement'].min())
    )

    st.dataframe(top_hashtags[['Hashtag', 'Score']].head(10))

    # ===============================
    # 📱 PLATFORM-WISE TRENDS
    # ===============================
    st.subheader("Platform-wise Trends")

    platform_trend = (
        trending_df.groupby(['Platform', 'Hashtag'])['Total_Engagement']
        .sum()
        .reset_index()
    )

    platforms = platform_trend['Platform'].unique()

    for platform in platforms:
        st.markdown(f"### 🔹 {platform}")
        temp = platform_trend[platform_trend['Platform'] == platform] \
            .sort_values(by='Total_Engagement', ascending=False).head(3)
        st.dataframe(temp)

    # ===============================
    # 🎬 BEST CONTENT TYPE
    # ===============================
    st.subheader("Best Content Type per Hashtag")

    content_trend = (
        trending_df.groupby(['Hashtag', 'Content_Type'])['Total_Engagement']
        .sum()
        .reset_index()
    )

    content_trend = content_trend.sort_values(
        ['Hashtag', 'Total_Engagement'], ascending=[True, False]
    )

    best_content_list = []

    for tag in content_trend['Hashtag'].unique():
        top = content_trend[content_trend['Hashtag'] == tag].head(1)
        best_content_list.append(top)

    best_content_df = pd.concat(best_content_list)
    st.dataframe(best_content_df)

    # ===============================
    # 📉 TREND DIRECTION
    # ===============================
    st.subheader("Trend Direction")

    df['Month'] = df['Post_Date'].dt.to_period('M')

    trend = (
        df.groupby(['Hashtag', 'Month'])['Total_Engagement']
        .sum()
        .reset_index()
    )

    trend_direction = []

    for tag in trend['Hashtag'].unique():
        temp = trend[trend['Hashtag'] == tag].sort_values('Month')

        if len(temp) >= 2:
            first = temp.iloc[0]['Total_Engagement']
            last = temp.iloc[-1]['Total_Engagement']

            if last > first:
                status = "Rising"
            elif last < first:
                status = "Falling"
            else:
                status = "Stable"

            trend_direction.append((tag, status))

    trend_df = pd.DataFrame(trend_direction, columns=['Hashtag', 'Trend'])
    st.dataframe(trend_df.head(10))

    # ===============================
    # 🚀 SMART INSIGHTS
    # ===============================
    st.subheader("Smart Insights")

    for tag in top_hashtags['Hashtag'].head(5):

        best_platform = (
            platform_trend[platform_trend['Hashtag'] == tag]
            .sort_values(by='Total_Engagement', ascending=False)
            .iloc[0]['Platform']
        )

        best_content = (
            content_trend[content_trend['Hashtag'] == tag]
            .sort_values(by='Total_Engagement', ascending=False)
            .iloc[0]['Content_Type']
        )

        trend_status = trend_df[trend_df['Hashtag'] == tag]['Trend'].values

        if len(trend_status) > 0:
            trend_status = trend_status[0]
        else:
            trend_status = "Stable"

        st.write(f"Use **{tag}** on **{best_platform}** with **{best_content}** ({trend_status})")
