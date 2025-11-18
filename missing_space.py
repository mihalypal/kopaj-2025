import requests
import re
from bs4 import BeautifulSoup


def find_missing_spaces(url):
    try:
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text()

        pattern = r'([a-z]+)\.([A-Z][a-z]+)'
        matches = re.finditer(pattern, text)

        missing_spaces = [match.group(0) for match in matches]

        return missing_spaces
    except Exception as e:
        return f"Error: {str(e)}"


# ezt returnoli: ['solutions.Let', 'all.Since', 'development.Multinational', 'needs.Stability', 'solutions.Efficiency', 'value.Short', 'experience.Recent', 'sources.Medical', 'team.Learn', 'ndroid.Learn', 'life.Learn', 'reliable.Learn', 'details.If', 'together.Do']
url = "https://bishop-co.com/"
url = "https://bishop-co.com/case-studies/szin"
result = find_missing_spaces(url)[7]
print(f"A megoldas: '{result}'")
