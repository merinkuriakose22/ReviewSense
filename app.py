import pandas as pd
import streamlit as st
import plotly.express as px

from auth import (
    login,
    logout,
    is_authenticated,
    get_role
)

from nlp_pipeline import (
    load_data,
    load_model,
    predict_sentiment,
    extract_keywords
)


# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="ReviewSense",
    page_icon="📊",
    layout="wide"
)


# ==================================================
# AUTHENTICATION
# ==================================================

if not is_authenticated():

    st.title(
        "ReviewSense"
    )

    st.subheader(
        "Customer Sentiment Intelligence Platform"
    )

    st.info(
        "Please login from the sidebar."
    )

    login()

    st.stop()


# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title(
    "ReviewSense"
)

st.sidebar.write(
    f"User: **{st.session_state['username']}**"
)

st.sidebar.write(
    f"Role: **{get_role()}**"
)

logout()


# ==================================================
# LOAD DATA
# ==================================================

@st.cache_data
def get_data():

    return load_data()


df = get_data()


# ==================================================
# HEADER
# ==================================================

st.title(
    "ReviewSense"
)

st.markdown(
    """
    ### Customer Sentiment Intelligence Dashboard

    Analyze customer feedback using a fine-tuned
    RoBERTa sentiment classification model.
    """
)


# ==================================================
# MODEL
# ==================================================

try:

    tokenizer, model, device = load_model()

    model_available = True

except Exception as error:

    model_available = False

    st.warning(
        "Fine-tuned model not found. "
        "Run `python train_model.py` first."
    )


# ==================================================
# SIDEBAR FILTERS
# ==================================================

st.sidebar.header(
    "Filters"
)


# Brand filter

brands = [
    "All"
] + sorted(
    df["Brand"]
    .dropna()
    .unique()
    .tolist()
)


selected_brand = st.sidebar.selectbox(
    "Brand",
    brands
)


# Category filter

categories = [
    "All"
] + sorted(
    df["Category"]
    .dropna()
    .unique()
    .tolist()
)


selected_category = (
    st.sidebar.selectbox(
        "Category",
        categories
    )
)


# Rating filter

rating_range = st.sidebar.slider(
    "Rating",
    min_value=1,
    max_value=5,
    value=(1, 5)
)


# Apply filters

filtered_df = df.copy()


if selected_brand != "All":

    filtered_df = filtered_df[
        filtered_df["Brand"]
        == selected_brand
    ]


if selected_category != "All":

    filtered_df = filtered_df[
        filtered_df["Category"]
        == selected_category
    ]


filtered_df = filtered_df[
    filtered_df["Rating"]
    .between(
        rating_range[0],
        rating_range[1]
    )
]


# ==================================================
# KPI CARDS
# ==================================================

st.header(
    "Overview"
)


total_reviews = len(
    filtered_df
)


average_rating = (
    filtered_df["Rating"]
    .mean()
    if total_reviews > 0
    else 0
)


positive_count = len(
    filtered_df[
        filtered_df[
            "initial_sentiment"
        ] == "Positive"
    ]
)


negative_count = len(
    filtered_df[
        filtered_df[
            "initial_sentiment"
        ] == "Negative"
    ]
)


neutral_count = len(
    filtered_df[
        filtered_df[
            "initial_sentiment"
        ] == "Neutral"
    ]
)


conflict_count = int(
    filtered_df["conflict"].sum()
)


col1, col2, col3, col4, col5 = (
    st.columns(5)
)


col1.metric(
    "Total Reviews",
    total_reviews
)

col2.metric(
    "Average Rating",
    f"{average_rating:.2f}"
)

col3.metric(
    "Positive",
    positive_count
)

col4.metric(
    "Negative",
    negative_count
)

col5.metric(
    "Conflicts",
    conflict_count
)


# ==================================================
# SENTIMENT DISTRIBUTION
# ==================================================

st.header(
    "Sentiment Distribution"
)


sentiment_counts = (
    filtered_df[
        "initial_sentiment"
    ]
    .value_counts()
    .reset_index()
)


sentiment_counts.columns = [
    "Sentiment",
    "Count"
]


fig = px.pie(
    sentiment_counts,
    names="Sentiment",
    values="Count",
    title="Customer Sentiment Distribution"
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# ==================================================
# PRODUCT PERFORMANCE
# ==================================================

st.header(
    "Product Performance"
)


product_stats = (
    filtered_df
    .groupby("Product")
    .agg(
        Reviews=("Review", "count"),
        Average_Rating=("Rating", "mean")
    )
    .reset_index()
    .sort_values(
        "Reviews",
        ascending=False
    )
)


fig = px.bar(
    product_stats.head(15),
    x="Product",
    y="Average_Rating",
    title="Average Rating by Product"
)


fig.update_layout(
    xaxis_tickangle=-45
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# ==================================================
# BRAND PERFORMANCE
# ==================================================

st.header(
    "Brand Performance"
)


brand_stats = (
    filtered_df
    .groupby("Brand")
    .agg(
        Reviews=("Review", "count"),
        Average_Rating=("Rating", "mean")
    )
    .reset_index()
    .sort_values(
        "Average_Rating",
        ascending=False
    )
)


st.dataframe(
    brand_stats,
    use_container_width=True
)


# ==================================================
# CATEGORY PERFORMANCE
# ==================================================

st.header(
    "Category Performance"
)


category_stats = (
    filtered_df
    .groupby("Category")
    .agg(
        Reviews=("Review", "count"),
        Average_Rating=("Rating", "mean")
    )
    .reset_index()
    .sort_values(
        "Average_Rating",
        ascending=False
    )
)


fig = px.bar(
    category_stats,
    x="Category",
    y="Average_Rating",
    title="Average Rating by Category"
)


fig.update_layout(
    xaxis_tickangle=-45
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# ==================================================
# SENTIMENT TREND
# ==================================================

st.header(
    "Sentiment Trend"
)


trend_df = filtered_df.copy()

trend_df["Date"] = pd.to_datetime(
    trend_df["Date"],
    errors="coerce"
)


trend_df = (
    trend_df
    .dropna(subset=["Date"])
    .groupby(
        pd.Grouper(
            key="Date",
            freq="ME"
        )
    )
    .agg(
        Average_Rating=(
            "Rating",
            "mean"
        )
    )
    .reset_index()
)


fig = px.line(
    trend_df,
    x="Date",
    y="Average_Rating",
    markers=True,
    title="Average Customer Rating Over Time"
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# ==================================================
# KEYWORD ANALYSIS
# ==================================================

st.header(
    "Keyword Analysis"
)


keywords = extract_keywords(
    filtered_df["Review"],
    top_n=20
)


keyword_df = pd.DataFrame(
    keywords,
    columns=[
        "Keyword",
        "Frequency"
    ]
)


fig = px.bar(
    keyword_df,
    x="Frequency",
    y="Keyword",
    orientation="h",
    title="Most Frequent Keywords"
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# ==================================================
# CONFLICT ANALYSIS
# ==================================================

st.header(
    "Rating/Text Conflict Analysis"
)


conflict_df = filtered_df[
    filtered_df["conflict"] == True
]


st.write(
    f"Potential conflicts detected: "
    f"**{len(conflict_df)}**"
)


if len(conflict_df) > 0:

    st.dataframe(
        conflict_df[
            [
                "Product",
                "Brand",
                "Rating",
                "Review",
                "initial_sentiment",
                "text_signal"
            ]
        ],
        use_container_width=True
    )


# ==================================================
# ROberta PREDICTION
# ==================================================

st.header(
    "Review Sentiment Prediction"
)


review_input = st.text_area(
    "Enter a customer review",
    placeholder=(
        "Example: The product works well "
        "and the setup was very easy."
    )
)


if st.button(
    "Analyze Review"
):

    if not review_input.strip():

        st.warning(
            "Please enter a review."
        )

    elif not model_available:

        st.error(
            "RoBERTa model is not available. "
            "Run train_model.py first."
        )

    else:

        result = predict_sentiment(
            [review_input],
            tokenizer,
            model,
            device
        )[0]

        sentiment = result[
            "sentiment"
        ]

        confidence = result[
            "confidence"
        ]

        st.success(
            f"Sentiment: {sentiment}"
        )

        st.metric(
            "Confidence",
            f"{confidence:.2%}"
        )


# ==================================================
# REVIEW EXPLORER
# ==================================================

st.header(
    "Review Explorer"
)


display_columns = [
    col
    for col in [
        "ID",
        "Product",
        "Brand",
        "Category",
        "Rating",
        "Review",
        "Date",
        "initial_sentiment"
    ]
    if col in filtered_df.columns
]


st.dataframe(
    filtered_df[
        display_columns
    ],
    use_container_width=True,
    height=500
)
