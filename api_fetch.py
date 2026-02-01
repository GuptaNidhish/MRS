import requests

def get_ott_providers_by_id(movie_id, api_key, region="IN"):
    """
    Fetches the OTT (streaming) providers for a specific movie ID in a region.
    
    Args:
        movie_id (int/str): The TMDB Movie ID (e.g., 550).
        api_key (str): Your TMDB API Key.
        region (str): The country code (e.g., 'IN', 'US') to filter providers.
    """
    base_url = "https://api.themoviedb.org/3"
    
    # URL to fetch providers directly using the ID
    provider_url = f"{base_url}/movie/{movie_id}/watch/providers"
    params = {"api_key": api_key}
    
    try:
        response = requests.get(provider_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Access the results for the specific region
        results = data.get('results', {})
        
        if region in results:
            country_data = results[region]
            
            print(f"--- Providers for Movie ID {movie_id} in {region} ---")
            
            # 1. Check for 'flatrate' (Subscription/Streaming like Netflix, Hotstar)
            if 'flatrate' in country_data:
                print("Streaming (Subscription):")
                for provider in country_data['flatrate']:
                    print(f"- {provider['provider_name']}")
            else:
                print("Not currently streaming on subscription platforms.")
                
            # 2. Check for 'rent' (VOD like Google Play, Apple TV)
            if 'rent' in country_data:
                rent_list = [p['provider_name'] for p in country_data['rent']]
                print(f"Available for Rent: {', '.join(rent_list)}")

            # 3. Check for 'buy' (Digital Purchase)
            if 'buy' in country_data:
                buy_list = [p['provider_name'] for p in country_data['buy']]
                print(f"Available to Buy: {', '.join(buy_list)}")

        else:
            print(f"No provider data available for Movie ID {movie_id} in region '{region}'.")

    except requests.exceptions.RequestException as e:
        print(f"API Error: {e}")

# --- USAGE ---
MY_API_KEY = "446ae1109e2833e6c21b03e857d6a1dc"
RRR_MOVIE_ID = 579974  # The ID for RRR

get_ott_providers_by_id(RRR_MOVIE_ID, MY_API_KEY, region="IN")