import requests

def getName(str):
    l = str.split("/")
    return l[-1]

url = 'https://dbpedia.org/sparql'
query = """
SELECT ?lake ?country
WHERE {
    ?lake a dbo:Lake.
}
LIMIT 10
"""
r = requests.get(url, params = {'format': 'json', 'query': query})

data = r.json()



for item in data['results']['bindings']:
    print("Result:")
    print("    -> lake -", getName(item['lake']['value']))