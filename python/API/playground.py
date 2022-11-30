
import requests

output = requests.get('http://127.0.0.1:8000/files/image.jpeg')
print(output)
print(output.json())



