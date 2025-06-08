import streamlit as st
import pandas as pd
import plotly.express as px
import requests # Used for making HTTP requests to the Gemini API
import json # Used for parsing JSON responses from the API
import io # Used to handle in-memory file operations

# --- Configuration ---
# Set the page configuration for a wide layout and a fun emoji icon
st.set_page_config(
    page_title="MoodMelt's Interactive Media Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None, # Remove default Streamlit menu items
    page_icon="🍓"
)

# Define custom colors for consistency with the fruit theme
# These colors correspond to dark pink, yellow, green, orange, blue from the original request
CUSTOM_COLORS = ['#E91E63', '#FFEB3B', '#8BC34A', '#FF9800', '#2196F3', '#A155B9', '#6B5B95']

# Gemini API configuration (API key is automatically provided in Canvas environment)
GEMINI_API_KEY = "" # Leave this empty, Canvas will inject the API key
GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

# --- Helper Functions ---

@st.cache_data # Cache the data processing to avoid re-running on every interaction
def process_csv_data(uploaded_file):
    """
    Processes and cleans the uploaded CSV file data.
    - Converts 'Date' to datetime.
    - Fills missing 'Engagements' with 0.
    - Normalizes column names.
    Args:
        uploaded_file (streamlit.runtime.uploaded_file_manager.UploadedFile): The uploaded CSV file object.
    Returns:
        pandas.DataFrame: A cleaned DataFrame.
    Raises:
        ValueError: If required columns are missing or data is malformed.
    """
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(uploaded_file)

        # Normalize column names: lowercase and replace spaces with underscores
        df.columns = df.columns.str.strip().str.lower().str.replace(r'\s+', '_', regex=True)

        # Define required columns for validation
        required_columns = ['date', 'platform', 'sentiment', 'location', 'engagements', 'media_type']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            # Format missing column names for user-friendly error message
            formatted_missing = [col.replace('_', ' ').title() for col in missing_columns]
            raise ValueError(f"Missing required columns: {', '.join(formatted_missing)}. "
                             f"Please ensure your CSV has 'Date', 'Platform', 'Sentiment', 'Location', 'Engagements', 'Media Type' columns.")

        # Convert 'date' column to datetime objects
        # errors='coerce' will turn unparseable dates into NaT (Not a Time)
        df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=True) # Try dayfirst for DD/MM/YYYY or DD-MM-YYYY

        # Drop rows where 'date' could not be parsed (NaT values)
        initial_rows = len(df)
        df.dropna(subset=['date'], inplace=True)
        if len(df) < initial_rows:
            st.warning(f"Skipped {initial_rows - len(df)} rows due to invalid date formats.")

        # Fill missing 'engagements' with 0 and convert to integer
        df['engagements'] = pd.to_numeric(df['engagements'], errors='coerce').fillna(0).astype(int)

        if df.empty:
            raise ValueError("No valid data rows found after cleaning. Check your CSV content and date formats.")

        return df

    except Exception as e:
        st.error(f"Error processing CSV: {e}")
        raise

@st.cache_data(show_spinner=False) # Cache chart generation, no spinner as it's quick
def generate_plotly_charts(df):
    """
    Generates Plotly interactive charts from the processed DataFrame.
    Args:
        df (pandas.DataFrame): The cleaned DataFrame.
    Returns:
        dict: A dictionary containing Plotly figures for each chart.
    """
    charts = {}

    # 1. Sentiment Breakdown (Pie Chart)
    sentiment_counts = df['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    charts['sentiment_pie'] = px.pie(
        sentiment_counts,
        values='Count',
        names='Sentiment',
        title='<b style="font-size: 24px;">Sentiment Sweet Spot</b>',
        color_discrete_sequence=CUSTOM_COLORS,
        hole=0.3 # Adds a donut hole
    )
    charts['sentiment_pie'].update_traces(textposition='inside', textinfo='percent+label')
    charts['sentiment_pie'].update_layout(showlegend=True, margin=dict(t=50, b=50, l=0, r=0))

    # 2. Engagement Trend over time (Line Chart)
    # Aggregate engagements by date
    daily_engagements = df.groupby(df['date'].dt.date)['engagements'].sum().reset_index()
    daily_engagements.columns = ['Date', 'Total Engagements']
    charts['engagement_line'] = px.line(
        daily_engagements,
        x='Date',
        y='Total Engagements',
        title='<b style="font-size: 24px;">Engagement\'s Juicy Journey</b>',
        color_discrete_sequence=[CUSTOM_COLORS[3]] # Orange for line chart
    )
    charts['engagement_line'].update_traces(mode='lines+markers', marker=dict(size=8))
    charts['engagement_line'].update_layout(hovermode="x unified", margin=dict(t=50, b=50, l=50, r=50))

    # 3. Platform Engagements (Bar Chart)
    platform_engagements = df.groupby('platform')['engagements'].sum().reset_index()
    platform_engagements.columns = ['Platform', 'Total Engagements']
    platform_engagements = platform_engagements.sort_values('Total Engagements', ascending=False)
    charts['platform_bar'] = px.bar(
        platform_engagements,
        x='Platform',
        y='Total Engagements',
        title='<b style="font-size: 24px;">Platform Power Play</b>',
        color='Platform', # Color bars by platform
        color_discrete_sequence=CUSTOM_COLORS
    )
    charts['platform_bar'].update_layout(showlegend=False, margin=dict(t=50, b=50, l=50, r=50))

    # 4. Media Type Mix (Pie Chart)
    media_type_counts = df['media_type'].value_counts().reset_index()
    media_type_counts.columns = ['Media Type', 'Count']
    charts['media_pie'] = px.pie(
        media_type_counts,
        values='Count',
        names='Media Type',
        title='<b style="font-size: 24px;">Media Mix Magic</b>',
        color_discrete_sequence=CUSTOM_COLORS,
        hole=0.3
    )
    charts['media_pie'].update_traces(textposition='inside', textinfo='percent+label')
    charts['media_pie'].update_layout(showlegend=True, margin=dict(t=50, b=50, l=0, r=0))

    # 5. Top 5 Locations (Bar Chart)
    location_engagements = df.groupby('location')['engagements'].sum().reset_index()
    location_engagements.columns = ['Location', 'Total Engagements']
    location_engagements = location_engagements.sort_values('Total Engagements', ascending=False).head(5)
    charts['location_bar'] = px.bar(
        location_engagements,
        x='Location',
        y='Total Engagements',
        title='<b style="font-size: 24px;">Location Lowdown</b>',
        color='Location', # Color bars by location
        color_discrete_sequence=CUSTOM_COLORS
    )
    charts['location_bar'].update_layout(showlegend=False, margin=dict(t=50, b=50, l=50, r=50))

    return charts

@st.cache_data(show_spinner="Juicing up your insights...") # Cache insights and show a spinner
def fetch_gemini_insights(data_summary_dict):
    """
    Fetches insights from the Gemini API based on data summaries.
    Args:
        data_summary_dict (dict): A dictionary containing summarized data for LLM prompting.
    Returns:
        dict: A dictionary of insights for each category and overall recommendations.
    """
    all_insights = {
        "sentiment_insights": [],
        "engagement_insights": [],
        "platform_insights": [],
        "media_insights": [],
        "location_insights": [],
        "overall_recommendations": []
    }

    # Helper function to call the Gemini API
    def get_llm_response(prompt):
        try:
            headers = {"Content-Type": "application/json"}
            payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
            response = requests.post(GEMINI_API_URL, headers=headers, data=json.dumps(payload))
            response.raise_for_status() # Raise an exception for HTTP errors
            result = response.json()
            if result.get("candidates") and result["candidates"][0].get("content") and result["candidates"][0]["content"].get("parts"):
                return [line.strip() for line in result["candidates"][0]["content"]["parts"][0]["text"].split('\n') if line.strip()]
            return ["Could not generate insights (no valid response)."]
        except requests.exceptions.RequestException as e:
            st.error(f"Gemini API Error: {e}. Please check your network connection or API key.")
            return ["Failed to fetch insights (API error)."]
        except Exception as e:
            st.error(f"Error processing Gemini response: {e}")
            return ["Failed to process insights response."]

    # Generate prompts and fetch insights for each category
    sentiment_prompt = f"Analyze this sentiment data, focusing on distribution and dominant sentiments: {data_summary_dict['sentiment_summary']}. Provide 3 brief, concise, and easy-to-understand insights."
    all_insights["sentiment_insights"] = get_llm_response(sentiment_prompt)

    engagement_prompt = f"Analyze this daily engagement trend data: {data_summary_dict['engagement_summary']}. Provide 3 brief, concise, and easy-to-understand insights. Focus on peaks, dips, and overall trends."
    all_insights["engagement_insights"] = get_llm_response(engagement_prompt)

    platform_prompt = f"Analyze this platform engagement data: {data_summary_dict['platform_summary']}. Provide 3 brief, concise, and easy-to-understand insights. Highlight top platforms and significant differences in engagement."
    all_insights["platform_insights"] = get_llm_response(platform_prompt)

    media_prompt = f"Analyze this media type distribution data: {data_summary_dict['media_summary']}. Provide 3 brief, concise, and easy-to-understand insights. Focus on dominant media types and their share."
    all_insights["media_insights"] = get_llm_response(media_prompt)

    location_prompt = f"Analyze this top 5 locations by engagement data: {data_summary_dict['location_summary']}. Provide 3 brief, concise, and easy-to-understand insights. Point out the most engaging locations."
    all_insights["location_insights"] = get_llm_response(location_prompt)

    # Generate overall business recommendations
    overall_prompt = (
        f"Based on the following aggregated media intelligence data summaries, provide 3-5 brief, concise, "
        f"and easy-to-understand overall business recommendations and actionable steps. Focus on what the user should *do* next.\n"
        f"Sentiment: {data_summary_dict['sentiment_summary']}.\n"
        f"Engagement Trend: {data_summary_dict['engagement_summary']}.\n"
        f"Platform Engagements: {data_summary_dict['platform_summary']}.\n"
        f"Media Type Mix: {data_summary_dict['media_summary']}.\n"
        f"Top Locations: {data_summary_dict['location_summary']}."
    )
    all_insights["overall_recommendations"] = get_llm_response(overall_prompt)

    return all_insights

# --- Streamlit App Layout ---

st.title("MoodMelt's Interactive Media Intelligence Dashboard 🍓")
st.markdown(
    """
    <p style="font-size: 24px; color: #555; text-align: center; font-weight: 500;">
        Ready to slice and dice your data with a splash of fun?
    </p>
    """,
    unsafe_allow_html=True
)

st.write("---") # Separator

uploaded_file = st.file_uploader(
    "Upload Your Juicy CSV! 🍇",
    type="csv",
    help="Columns needed: 'Date', 'Platform', 'Sentiment', 'Location', 'Engagements', 'Media Type'."
)

# Initialize session state for processed data and charts
if 'df' not in st.session_state:
    st.session_state.df = None
if 'charts' not in st.session_state:
    st.session_state.charts = None
if 'insights' not in st.session_state:
    st.session_state.insights = None

if uploaded_file is not None:
    # Process the file only if a new one is uploaded or it's the first run
    if st.session_state.df is None or st.session_state.last_uploaded_file_id != uploaded_file.id:
        st.session_state.last_uploaded_file_id = uploaded_file.id
        with st.spinner("Juicing up your data..."):
            try:
                # Process Data
                st.session_state.df = process_csv_data(uploaded_file)

                # Generate Charts
                st.session_state.charts = generate_plotly_charts(st.session_state.df)

                # Prepare summaries for LLM
                data_summary = {
                    'sentiment_summary': st.session_state.df['sentiment'].value_counts().to_dict(),
                    'engagement_summary': st.session_state.df.groupby(st.session_state.df['date'].dt.date)['engagements'].sum().head(10).to_dict(),
                    'platform_summary': st.session_state.df.groupby('platform')['engagements'].sum().sort_values(ascending=False).to_dict(),
                    'media_summary': st.session_state.df['media_type'].value_counts().to_dict(),
                    'location_summary': st.session_state.df.groupby('location')['engagements'].sum().sort_values(ascending=False).head(5).to_dict(),
                }

                # Fetch Insights
                st.session_state.insights = fetch_gemini_insights(data_summary)

            except ValueError as e:
                st.error(f"Data Error: {e}")
                st.session_state.df = None
                st.session_state.charts = None
                st.session_state.insights = None
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                st.session_state.df = None
                st.session_state.charts = None
                st.session_state.insights = None
    else:
        # File is already processed, no need to re-process
        pass


if st.session_state.df is not None and st.session_state.charts is not None and st.session_state.insights is not None:
    st.success("Data processed and insights ready!")

    # --- Display Charts and Insights ---
    st.markdown("## 📊 Your Data in a Nutshell")

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(st.session_state.charts['sentiment_pie'], use_container_width=True)
        st.markdown("### Fresh Insights: Sentiment Sweet Spot")
        for i, insight in enumerate(st.session_state.insights['sentiment_insights']):
            st.markdown(f"**✨** {insight}")

    with col2:
        st.plotly_chart(st.session_state.charts['engagement_line'], use_container_width=True)
        st.markdown("### Fresh Insights: Engagement's Juicy Journey")
        for i, insight in enumerate(st.session_state.insights['engagement_insights']):
            st.markdown(f"**✨** {insight}")

    st.write("---") # Separator

    col3, col4 = st.columns(2)

    with col3:
        st.plotly_chart(st.session_state.charts['platform_bar'], use_container_width=True)
        st.markdown("### Fresh Insights: Platform Power Play")
        for i, insight in enumerate(st.session_state.insights['platform_insights']):
            st.markdown(f"**✨** {insight}")

    with col4:
        st.plotly_chart(st.session_state.charts['media_pie'], use_container_width=True)
        st.markdown("### Fresh Insights: Media Mix Magic")
        for i, insight in enumerate(st.session_state.insights['media_insights']):
            st.markdown(f"**✨** {insight}")

    st.write("---") # Separator

    # Location chart will take full width
    st.plotly_chart(st.session_state.charts['location_bar'], use_container_width=True)
    st.markdown("### Fresh Insights: Location Lowdown")
    for i, insight in enumerate(st.session_state.insights['location_insights']):
        st.markdown(f"**✨** {insight}")

    st.write("---") # Separator

    # --- Overall Business Recommendations ---
    st.markdown("## 🚀 Your Next Steps: Sweet Success!")
    st.markdown(
        """
        <p style="font-size: 20px; color: #4A4A4A; font-weight: 600;">
            Based on our analysis, here are some juicy recommendations to guide your next moves:
        </p>
        """,
        unsafe_allow_html=True
    )
    if st.session_state.insights['overall_recommendations']:
        for i, rec in enumerate(st.session_state.insights['overall_recommendations']):
            st.markdown(f"**✅** {rec}")
    else:
        st.info("No overall recommendations generated yet. Please ensure your data is comprehensive.")

    st.write("---") # Separator

    st.markdown(
        """
        <p style="font-size: 18px; color: #777; text-align: center;">
            That's all the delicious data for now! Keep melting moods! 🍍
        </p>
        """,
        unsafe_allow_html=True
    )
