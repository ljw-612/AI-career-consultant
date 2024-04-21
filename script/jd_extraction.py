from bs4 import BeautifulSoup
import requests

def get_webpage_text(url):
    # Send a GET request to the URL
    response = requests.get(url)
    # If the GET request is successful, the status code will be 200
    if response.status_code == 200:
        # Get the content of the response
        webpage_content = response.content
        # Create a BeautifulSoup object and specify the parser
        soup = BeautifulSoup(webpage_content, 'html.parser')
        # Extract all the text from the webpage
        text = soup.get_text()
        
        webpage_text = text.replace("\n", " ")
        return webpage_text
    else:
        return "Unable to retrieve the webpage."

if __name__ == '__main__':
    # Ask the user to input a webpage URL
    url = "https://ocbridge.ai/job/data-science-intern/"
    # url = st.text_input("Enter a webpage URL")
    if url:
        # Get the text from the webpage
        webpage_text = get_webpage_text(url)
        # Display the text
        # st.write(webpage_text)