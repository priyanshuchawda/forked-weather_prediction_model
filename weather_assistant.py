from datetime import datetime, timedelta
from combined_forecast import get_weather_forecast
from gemini_client import GeminiClient, WeatherQuery

def format_weather_response(weather_data, query_info: WeatherQuery):
    """Format weather data into a natural language response"""
    date_str = weather_data['date'].strftime('%Y-%m-%d')
    temp = weather_data['temperature']
    rain = weather_data['rainfall']
    wind = weather_data['wind_speed']
    
    # Format based on info type
    if query_info.info_type == 'temperature':
        response = (
            f"🌡️ Temperature on {date_str} in {query_info.city}: {temp}°C\n"
            f"{'🌞 Pleasant day!' if 20 <= temp <= 28 else '🔥 Quite warm!' if temp > 28 else '❄️ Bit cool!'}"
        )
    elif query_info.info_type == 'rain':
        response = (
            f"🌧️ Rainfall expected on {date_str} in {query_info.city}: {rain}mm\n"
            f"{'☔ Carry an umbrella!' if rain > 2 else '🌂 Light rain possible.' if rain > 0 else '☀️ No rain expected.'}"
        )
    elif query_info.info_type == 'wind':
        response = (
            f"💨 Wind speed on {date_str} in {query_info.city}: {wind} km/h\n"
            f"{'🌪️ Windy conditions!' if wind > 20 else '🍃 Gentle breeze.' if wind > 10 else '😊 Calm weather.'}"
        )
    else:
        response = (
            f"Weather Forecast for {query_info.city} - {date_str}:\n"
            f"🌡️ Temperature: {temp}°C\n"
            f"🌧️ Rainfall: {rain}mm\n"
            f"💨 Wind Speed: {wind} km/h\n\n"
            f"Summary: "
            f"{'🌞' if temp > 25 else '⛅'} "
            f"{'☔' if rain > 2 else '🌂' if rain > 0 else '☀️'} "
            f"{'🌪️' if wind > 20 else '🍃'}"
        )
    
    return response

def get_weather_forecast_for_query(query):
    """Main function to handle weather queries"""
    # Parse the query using Gemini
    gemini_client = GeminiClient()
    query_info = gemini_client.parse_weather_query(query)

    if not query_info:
        return "Sorry, I couldn't understand your query. Please try again."

    # Get current date and check if we need to update our model
    current_date = datetime.now()
    try:
        with open('last_update.txt', 'r') as f:
            last_update = datetime.strptime(f.read().strip(), '%Y-%m-%d')
        days_since_update = (current_date.date() - last_update.date()).days
    except:
        days_since_update = float('inf')
    
    # If data is more than 7 days old, notify user
    if days_since_update > 7:
        print("\nNote: Weather data needs updating. Predictions may be less accurate.")
    
    # Get combined forecast
    coordinates = None
    if query_info.city.lower() != "pune":
        print(f"Searching for coordinates for {query_info.city} using Google Search...")
        coordinates = gemini_client.get_coordinates_from_google_search(query_info.city)
        if not coordinates:
            return f"Sorry, I couldn't find coordinates for {query_info.city}."
        
    combined_forecast = get_weather_forecast(query_info.city, lat=coordinates['lat'] if coordinates else None, lon=coordinates['lon'] if coordinates else None)
    
    if not combined_forecast:
        return "Sorry, I couldn't get the weather forecast for that location."

    # Get forecast for requested date
    if query_info.date:
        if query_info.date.lower() == 'today':
            target_date = datetime.now().date()
        elif query_info.date.lower() == 'tomorrow':
            target_date = datetime.now().date() + timedelta(days=1)
        else:
            try:
                target_date = datetime.strptime(query_info.date, '%Y-%m-%d').date()
            except ValueError:
                return "Sorry, I couldn't understand the date in your query."
    else:
        target_date = datetime.now().date()

    forecast = None
    for day in combined_forecast:
        if isinstance(day['date'], str):
            day['date'] = datetime.strptime(day['date'], '%Y-%m-%d').date()

        if isinstance(day['date'], datetime):
            compare_date = day['date'].date()
        else:
            compare_date = day['date']
            
        if compare_date == target_date:
            forecast = day
            break
    
    if not forecast:
        return "मैं केवल अगले 5-7 दिनों का मौसम बता सकता हूं। (I can only provide forecasts for the next 5-7 days.)"
    
    # Format response in natural language
    response = format_weather_response(forecast, query_info)
    return response

def main():
    print("पुणे का मौसम जानने के लिए स्वागत है! (Welcome to Pune Weather Assistant!)")
    print("आप किसी भी भाषा में पूछ सकते हैं। (You can ask in any language.)")
    print("Testing with the following queries:\n")
    
    # Test cases - add or modify queries here
    test_queries = [
        "what is the weather in Mumbai tomorrow"
    ]

    query = test_queries[0]
    print(f"\n--- Testing query: '{query}' ---")
    try:
        response = get_weather_forecast_for_query(query)
        print("\n" + response + "\n")
        print("-" * 50)
    except Exception as e:
        print(f"\nSorry, there was an error: {str(e)}")
        print("Please try again with a different question.\n")

if __name__ == "__main__":
    main()
