import requests
r = requests.post("http://localhost:5006/grammar",
                  json={"text": "저는 어제 도서관 가요"})
print(r.json())