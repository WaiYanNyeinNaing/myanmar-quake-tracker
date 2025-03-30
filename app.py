import requests
import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import time
import plotly.express as px

# Set page config as the first Streamlit command
st.set_page_config(page_title="Myanmar Earthquake Monitor", layout="wide")

# Geographical bounds for Myanmar
MIN_LATITUDE = 9.5
MAX_LATITUDE = 28.5
MIN_LONGITUDE = 92.2
MAX_LONGITUDE = 101.2

# Minimum magnitude to filter small earthquakes
MIN_MAGNITUDE = 2.5

# USGS API endpoint
API_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"

# Translations dictionary (English and Burmese)
translations = {
    'en': {
        'title': 'Myanmar Earthquake Monitor',
        'description': 'This application displays recent earthquakes in and around Myanmar.',
        'latest_earthquake': 'Latest Earthquake',
        'magnitude': 'Magnitude',
        'location': 'Location',
        'time': 'Time (UTC)',
        'alerts': 'Alerts',
        'no_earthquakes': 'No earthquakes found in the specified region and time period.',
        'hours_back': 'Hours to look back',
        'min_magnitude': 'Minimum magnitude',
        'refresh_data': 'Refresh Data Now',
        'last_updated': 'Last updated',
        'earthquake_map': 'Earthquake Map',
        'earthquake_analysis': 'Earthquake Analysis',
        'location_analysis': 'Location Analysis',
        'time_series': 'Time Series',
        'magnitude_distribution': 'Magnitude Distribution by Location',
        'top_strongest': 'Top 10 Strongest Earthquakes',
        'earthquake_timeline': 'Earthquake Timeline',
        'daily_count': 'Daily Earthquake Count',
        'activity_heatmap': 'Earthquake Activity by Hour and Day of Week',
        'earthquake_details': 'Earthquake Details',
        'download_csv': 'Download data as CSV',
        'found_earthquakes': 'Found {count} earthquakes in the specified region and time.',
        'real_time_monitoring': 'Real-time Monitoring',
        'enable_monitoring': 'Enable real-time monitoring',
        'refresh_interval': 'Refresh interval (seconds)',
        'monitoring_active': 'Real-time monitoring active. Refreshing every {interval} seconds.',
        'recent_alerts': 'Recent Alerts',
        'recent_activity': 'Recent Earthquake Activity',
        'advanced_settings': 'Advanced Settings',
        'language': 'Language',
        'about': 'About',
        'about_text': 'This application fetches real-time earthquake data from the USGS Earthquake API for the Myanmar region. The geographical bounds are:\n- Latitude: 9.5° to 28.5°\n- Longitude: 92.2° to 101.2°',
        'earthquakes_24h': 'Earthquakes (24h)',
        'max_magnitude_24h': 'Max Magnitude (24h)',
        'trend': 'Trend',
        'hours_ago': 'hours ago',
        'settings': 'Settings',
        'avg_magnitude': 'Average Magnitude by Location (Top 15)',
        'strongest_earthquakes': 'Strongest Earthquakes by Location',
        'time_presets': {
            "5 minutes": "5 minutes",
            "30 minutes": "30 minutes",
            "1 hour": "1 hour",
            "8 hours": "8 hours",
            "24 hours": "24 hours",
            "48 hours": "48 hours"
        },
        'depth': 'Depth'
    },
    'my': {
        'title': 'မြန်မာငလျင်စောင့်ကြည့်ရေး',
        'description': 'မြန်မာနိုင်ငံရှိ မကြာသေးမီ ငလျင်များနှင့် အချိန်နှင့်တပြေးညီ စောင့်ကြည့်မှုကို ကြည့်ပါ။',
        'latest_earthquake': 'နောက်ဆုံးငလျင်',
        'magnitude': 'အင်အား',
        'location': 'တည်နေရာ',
        'time': 'အချိန် (UTC)',
        'alerts': 'သတိပေးချက်များ',
        'no_earthquakes': 'သတ်မှတ်ထားသော ဒေသနှင့် အချိန်ကာလတွင် ငလျင်များ မတွေ့ရှိပါ။',
        'hours_back': 'နာရီများ ပြန်ကြည့်ရန်',
        'min_magnitude': 'အနည်းဆုံး အင်အား',
        'refresh_data': 'ဒေတာကို အသစ်တင်ပါ',
        'last_updated': 'နောက်ဆုံးအပ်ဒိတ်',
        'earthquake_map': 'ငလျင်မြေပုံ',
        'earthquake_analysis': 'ငလျင်ဆန်းစစ်ချက်',
        'location_analysis': 'တည်နေရာဆန်းစစ်ချက်',
        'time_series': 'အချိန်အလိုက်ဖြစ်စဉ်',
        'magnitude_distribution': 'တည်နေရာအလိုက် အင်အားဖြန့်ဝေမှု',
        'top_strongest': 'အင်အားအကြီးဆုံး ငလျင် ၁၀ ခု',
        'earthquake_timeline': 'ငလျင်အချိန်ဇယား',
        'daily_count': 'နေ့စဉ်ငလျင်အရေအတွက်',
        'activity_heatmap': 'နာရီနှင့် ရက်သတ္တပတ်အလိုက် ငလျင်လှုပ်ရှားမှု',
        'earthquake_details': 'ငလျင်အသေးစိတ်',
        'download_csv': 'ဒေတာကို CSV အဖြစ် ဒေါင်းလုဒ်လုပ်ပါ',
        'found_earthquakes': 'သတ်မှတ်ထားသော ဒေသနှင့် အချိန်တွင် ငလျင် {count} ခု တွေ့ရှိခဲ့သည်။',
        'real_time_monitoring': 'အချိန်နှင့်တပြေးညီ စောင့်ကြည့်ခြင်း',
        'enable_monitoring': 'အချိန်နှင့်တပြေးညီ စောင့်ကြည့်ခြင်းကို ဖွင့်ပါ',
        'refresh_interval': 'ပြန်လည်စတင်ကြားချိန် (စက္ကန့်)',
        'monitoring_active': 'အချိန်နှင့်တပြေးညီ စောင့်ကြည့်ခြင်း အသက်ဝင်နေသည်။ {interval} စက္ကန့်တိုင်း ပြန်လည်စတင်နေသည်။',
        'recent_alerts': 'မကြာသေးမီ သတိပေးချက်များ',
        'recent_activity': 'မကြာသေးမီ ငလျင်လှုပ်ရှားမှု',
        'advanced_settings': 'အဆင့်မြင့်ဆက်တင်များ',
        'language': 'ဘာသာစကား',
        'about': 'အကြောင်း',
        'about_text': 'ဤအပလီကေးရှင်းသည် မြန်မာဒေသအတွက် USGS ငလျင်ဒေတာကို အချိန်နှင့်တပြေးညီ ရယူပါသည်။ ဒေသဆိုင်ရာ နယ်နိမိတ်များမှာ:\n- လတ္တီတွဒ်: 9.5° မှ 28.5°\n- လောင်ဂျီတွဒ်: 92.2° မှ 101.2°',
        'earthquakes_24h': 'ငလျင်များ (၂၄နာရီ)',
        'max_magnitude_24h': 'အမြင့်ဆုံးအင်အား (၂၄နာရီ)',
        'trend': 'လမ်းကြောင်း',
        'hours_ago': 'နာရီက',
        'settings': 'ဆက်တင်များ',
        'avg_magnitude': 'တည်နေရာအလိုက် ပျမ်းမျှအင်အား (ထိပ်ဆုံး ၁၅ ခု)',
        'strongest_earthquakes': 'တည်နေရာအလိုက် အင်အားအကြီးဆုံး ငလျင်များ',
        'time_presets': {
            "5 minutes": "၅ မိနစ်",
            "30 minutes": "၃၀ မိနစ်",
            "1 hour": "၁ နာရီ",
            "8 hours": "၈ နာရီ",
            "24 hours": "၂၄ နာရီ",
            "48 hours": "၄၈ နာရီ"
        },
        'depth': 'အနက်'
    }
}

def get_time_range(hours_back=1):
    """Returns start and end times for the API request (UTC)."""
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=hours_back)
    return start_time.isoformat(), end_time.isoformat()

def fetch_earthquake_data(start_time, end_time, min_magnitude=MIN_MAGNITUDE):
    """Fetches earthquake data from USGS API for the specified time and region."""
    params = {
        "format": "geojson",
        "starttime": start_time,
        "endtime": end_time,
        "minlatitude": MIN_LATITUDE,
        "maxlatitude": MAX_LATITUDE,
        "minlongitude": MIN_LONGITUDE,
        "maxlongitude": MAX_LONGITUDE,
        "minmagnitude": min_magnitude
    }
    try:
        response = requests.get(API_URL, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error fetching data: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Network error: {e}")
        return None

def parse_earthquake_data(data):
    """Parses earthquake details from the API response and returns as a DataFrame."""
    if data and "features" in data:
        earthquakes = data["features"]
        if earthquakes:
            eq_data = []
            for eq in earthquakes:
                properties = eq["properties"]
                coordinates = eq["geometry"]["coordinates"]
                eq_data.append({
                    "Magnitude": properties["mag"],
                    "Location": properties["place"],
                    "Time (UTC)": datetime.fromtimestamp(properties["time"] / 1000).strftime("%Y-%m-%d %H:%M:%S"),
                    "Depth (km)": coordinates[2],
                    "Latitude": coordinates[1],
                    "Longitude": coordinates[0],
                    "Details URL": properties["url"]
                })
            return pd.DataFrame(eq_data)
        else:
            return pd.DataFrame()
    else:
        return pd.DataFrame()

def generate_alert(quake, lang_code):
    """Generate alert message for new earthquake in the selected language."""
    if lang_code == 'en':
        return f"New earthquake detected! {translations[lang_code]['magnitude']}: {quake['Magnitude']} at {quake['Location']} ({quake['Time (UTC)']})"
    else:
        return f"ငလျင်အသစ်တွေ့ရှိပါသည်! {translations[lang_code]['magnitude']}: {quake['Magnitude']} {quake['Location']} တွင် ({quake['Time (UTC)']})"

# Initialize session state for tracking new earthquakes
if 'previous_data' not in st.session_state:
    st.session_state.previous_data = None
if 'last_update_time' not in st.session_state:
    st.session_state.last_update_time = datetime.now()
if 'alert_history' not in st.session_state:
    st.session_state.alert_history = []

# Language selection in sidebar
st.sidebar.title(f"{translations['en']['language']} / {translations['my']['language']}")
lang_code = st.sidebar.radio("", ["en", "my"], format_func=lambda x: "English" if x == "en" else "မြန်မာဘာသာ")

st.title(translations[lang_code]['title'])
st.write(translations[lang_code]['description'])

# Sidebar for controls
st.sidebar.header(translations[lang_code]['settings'])

# Replace the hours_back slider with a selectbox of preset options
hours_options = {
    "5 minutes": 5/60,
    "30 minutes": 30/60,
    "1 hour": 1,
    "8 hours": 8,
    "24 hours": 24,
    "48 hours": 48
}

# Create a list of options with proper translations
time_options = list(hours_options.keys())
translated_options = [translations[lang_code]['time_presets'][option] for option in time_options]

# Display the selectbox with translated options
selected_time_option_index = st.sidebar.selectbox(
    translations[lang_code]['hours_back'],
    options=range(len(time_options)),
    format_func=lambda i: translated_options[i],
    index=3  # Default to 8 hours
)

# Get the selected hours value
selected_time_key = time_options[selected_time_option_index]
hours_back = hours_options[selected_time_key]

min_magnitude = st.sidebar.slider(translations[lang_code]['min_magnitude'], 2.0, 8.0, MIN_MAGNITUDE, 0.1)

# Real-time monitoring settings
st.sidebar.header(translations[lang_code]['real_time_monitoring'])
auto_refresh = st.sidebar.checkbox(translations[lang_code]['enable_monitoring'], value=False)
refresh_interval = st.sidebar.slider(translations[lang_code]['refresh_interval'], 
                                    min_value=30, 
                                    max_value=600, 
                                    value=300, 
                                    step=30,
                                    disabled=not auto_refresh)

# Main content
col1, col2 = st.columns([3, 1])

with col2:
    st.subheader(translations[lang_code]['about'])
    st.write(translations[lang_code]['about_text'])
    
    # Display last update time and manual refresh button
    st.write(f"{translations[lang_code]['last_updated']}: {st.session_state.last_update_time.strftime('%Y-%m-%d %H:%M:%S')}")
    if st.button(translations[lang_code]['refresh_data']):
        st.session_state.last_update_time = datetime.now()
        st.experimental_rerun()
    
    # Real-time monitoring status
    if auto_refresh:
        st.success(translations[lang_code]['monitoring_active'].format(interval=refresh_interval))
        time_since_update = (datetime.now() - st.session_state.last_update_time).total_seconds()
        progress = min(time_since_update / refresh_interval, 1.0)
        st.progress(progress)
    
    # Alert history section
    if st.session_state.alert_history:
        st.subheader(translations[lang_code]['recent_alerts'])
        for alert in st.session_state.alert_history[-5:]:  # Show last 5 alerts
            st.warning(alert)

with col1:
    start_time, end_time = get_time_range(hours_back=hours_back)
    
    with st.spinner("Fetching earthquake data..."):
        data = fetch_earthquake_data(start_time, end_time, min_magnitude)
        df = parse_earthquake_data(data)
    
    # Check for new earthquakes
    if st.session_state.previous_data is not None and not df.empty:
        previous_times = set(st.session_state.previous_data['Time (UTC)'])
        current_times = set(df['Time (UTC)'])
        new_times = current_times - previous_times
        
        if new_times:
            new_quakes = df[df['Time (UTC)'].isin(new_times)]
            for _, quake in new_quakes.iterrows():
                alert_msg = generate_alert(quake, lang_code)
                st.session_state.alert_history.append(alert_msg)
    
    # Update previous data
    if not df.empty:
        st.session_state.previous_data = df.copy()
    
    if not df.empty:
        st.success(translations[lang_code]['found_earthquakes'].format(count=len(df)))
        
        # Display a summary of the earthquakes found
        st.subheader(translations[lang_code]['earthquake_details'])
        
        # Sort by magnitude (descending) to show strongest earthquakes first
        sorted_df = df.sort_values(by='Magnitude', ascending=False)
        
        # Create a container for the earthquake summaries
        summary_container = st.container()
        
        # Display each earthquake in an easy-to-read format
        for i, quake in sorted_df.iterrows():
            with summary_container:
                st.info(f"""
                **{translations[lang_code]['magnitude']}:** {quake['Magnitude']:.1f}
                
                **{translations[lang_code]['location']}:** {quake['Location']}
                
                **{translations[lang_code]['time']}:** {quake['Time (UTC)']}
                
                **{translations[lang_code]['depth']}:** {quake['Depth (km)']:.1f} km
                """)
        
        # Display map
        st.subheader(translations[lang_code]['earthquake_map'])
        map_data = df[['Latitude', 'Longitude']].copy()
        # Rename columns to match what streamlit.map expects
        map_data = map_data.rename(columns={
            'Latitude': 'lat',
            'Longitude': 'lon'
        })
        st.map(map_data)
        
        # Alternative visualization with plotly
        fig = px.scatter_mapbox(df, 
                               lat="Latitude", 
                               lon="Longitude", 
                               size="Magnitude",
                               color="Magnitude",
                               hover_name="Location",
                               hover_data=["Time (UTC)", "Depth (km)"],
                               color_continuous_scale=px.colors.sequential.Reds,
                               size_max=15, 
                               zoom=5,
                               mapbox_style="open-street-map")
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional visualizations
        st.subheader(translations[lang_code]['earthquake_analysis'])
        
        # Convert time to datetime for time-based analysis
        df['Time'] = pd.to_datetime(df['Time (UTC)'])
        
        # Create tabs for different visualizations
        tab1, tab2 = st.tabs([translations[lang_code]['location_analysis'], translations[lang_code]['time_series']])
        
        with tab1:
            # Top 10 strongest earthquakes
            st.subheader(translations[lang_code]['top_strongest'])
            top_10 = df.sort_values(by="Magnitude", ascending=False).head(10)
            fig_top = px.bar(top_10, 
                            x="Magnitude", 
                            y="Location", 
                            orientation='h', 
                            color="Magnitude",
                            color_continuous_scale=px.colors.sequential.Reds,
                            hover_data=["Time (UTC)"],
                            title=translations[lang_code]['strongest_earthquakes'])
            fig_top.update_layout(yaxis_title="", xaxis_title=translations[lang_code]['magnitude'])
            st.plotly_chart(fig_top, use_container_width=True)
        
        with tab2:
            # Time series of earthquakes
            df_time = df.sort_values("Time")
            fig_time = px.scatter(df_time, 
                                 x="Time", 
                                 y="Magnitude", 
                                 size="Magnitude", 
                                 color="Magnitude",
                                 hover_name="Location",
                                 title=translations[lang_code]['earthquake_timeline'],
                                 color_continuous_scale=px.colors.sequential.Reds)
            fig_time.update_layout(xaxis_title="Date & Time", yaxis_title=translations[lang_code]['magnitude'])
            st.plotly_chart(fig_time, use_container_width=True)
            
            # Earthquakes by day
            df['Date'] = df['Time'].dt.date
            daily_counts = df.groupby('Date').size().reset_index(name='Count')
            daily_max = df.groupby('Date')['Magnitude'].max().reset_index(name='Max Magnitude')
            daily_data = daily_counts.merge(daily_max, on='Date')
            
            fig_daily = px.bar(daily_data, 
                              x='Date', 
                              y='Count',
                              color='Max Magnitude',
                              color_continuous_scale=px.colors.sequential.Reds,
                              title=translations[lang_code]['daily_count'])
            fig_daily.update_layout(xaxis_title="Date", yaxis_title="Number of Earthquakes")
            st.plotly_chart(fig_daily, use_container_width=True)
            
            # Heatmap of earthquake activity by hour and day of week
            df['Hour'] = df['Time'].dt.hour
            df['Day of Week'] = df['Time'].dt.day_name()
            
            # Order days of week properly
            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            # Create pivot table for heatmap
            heatmap_data = pd.pivot_table(
                df, 
                values='Magnitude', 
                index='Day of Week',
                columns='Hour',
                aggfunc='count'
            ).reindex(days_order)
            
            # Fill NaN values with 0
            heatmap_data = heatmap_data.fillna(0)
            
            # Make sure all hours (0-23) are present in the columns
            for hour in range(24):
                if hour not in heatmap_data.columns:
                    heatmap_data[hour] = 0
            
            # Sort columns to ensure hours are in order
            heatmap_data = heatmap_data.reindex(sorted(heatmap_data.columns), axis=1)
            
            fig_heatmap = px.imshow(
                heatmap_data,
                labels=dict(x="Hour of Day (UTC)", y="Day of Week", color="Earthquake Count"),
                color_continuous_scale="Reds",
                title=translations[lang_code]['activity_heatmap']
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Remove the duplicate Earthquake Details table section
        # Instead, just keep the download button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            translations[lang_code]['download_csv'],
            csv,
            f"myanmar_earthquakes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
            key='download-csv'
        )
    else:
        st.info(translations[lang_code]['no_earthquakes'])

# Auto-refresh logic
if auto_refresh:
    time_since_update = (datetime.now() - st.session_state.last_update_time).total_seconds()
    if time_since_update >= refresh_interval:
        st.session_state.last_update_time = datetime.now()
        st.experimental_rerun()
    else:
        # Wait for the remaining time and then refresh
        time.sleep(min(1, refresh_interval - time_since_update))  # Sleep at most 1 second to keep UI responsive
        st.experimental_rerun()