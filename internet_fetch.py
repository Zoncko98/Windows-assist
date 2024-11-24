import requests

### FETCH LIVE DATA FROM THE INTERNET ###
def fetch_live_data(url):
    """
    Perform a direct HTTP GET request to fetch raw data from a given URL.
    
    :param url: The URL to fetch data from.
    :return: Truncated response text (first 2000 characters) or an error message.
    """
    try:
        # Perform the HTTP GET request
        response = requests.get(url, timeout=5)

        # Check the HTTP status code
        if response.status_code == 200:
            # Return the first 2000 characters to avoid oversized processing
            return response.text[:2000]
        else:
            # Handle failed HTTP responses
            return f"Failed to fetch data. HTTP Status Code: {response.status_code}"
    except Exception as e:
        # Handle unexpected connection issues
        return f"Error during web request: {str(e)}"


### TRUNCATE RAW TEXT ###
def truncate_text(data, max_length=500):
    """
    Truncate raw text data to a manageable length for model input.

    :param data: The full-length data string.
    :param max_length: Maximum number of characters to return.
    :return: Truncated string.
    """
    return data[:max_length]


### HANDLE SEARCH-LIKE REQUESTS ###
def wikipedia_query(topic):
    """
    Construct a Wikipedia-style URL and fetch the top section of the page.

    :param topic: The topic to search for (e.g., "Artificial Intelligence").
    :return: First 2000 characters of the Wikipedia page or an error message.
    """
    # Construct the Wikipedia URL
    url = f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"
    return fetch_live_data(url)