# Myanmar Quake Tracker

A real-time earthquake monitoring application for Myanmar and surrounding regions, providing up-to-date information on seismic activity in an easy-to-understand format.

![Myanmar Quake Tracker Screenshot](https://via.placeholder.com/800x450.png?text=Myanmar+Quake+Tracker)

## Features

- **Real-time Monitoring**: Live earthquake data from USGS Earthquake API
- **Bilingual Interface**: Full support for both English and Burmese (မြန်မာဘာသာ)
- **Interactive Visualizations**: Maps, time series analysis, magnitude distribution charts, and activity heatmaps
- **Customizable Filters**: Time range options, magnitude thresholds, and automatic refresh intervals
- **Alert System**: Notifications for new earthquakes
- **Data Export**: Download earthquake data in CSV format

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/WaiYanNyeinNaing/myanmar-quake-tracker.git
   ```

2. Navigate to the project directory:
   ```
   cd myanmar-quake-tracker
   ```

3. Create and activate a virtual environment:
   ```
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

5. Run the application:
   ```
   streamlit run app.py
   ```

## Usage

1. Select your preferred language (English or Burmese)
2. Choose the time range for earthquake data
3. Set minimum magnitude threshold
4. Enable real-time monitoring if desired
5. Explore the interactive visualizations and data

## Technologies

- Python
- Streamlit
- Plotly
- Pandas
- USGS Earthquake API

## Contact

For questions or feedback, please contact:

Dr. Wai Yan Nyein Naing  
GitHub: [@WaiYanNyeinNaing](https://github.com/WaiYanNyeinNaing)

## License

This project is licensed under the MIT License - see the LICENSE file for details.