import requests

resp = requests.post("http://127.0.0.1:5000/predict",
                     files={"file": open('/Users/arnoldasjanuska/PycharmProjects/PetBreed/static/img/uploads/image1.jpeg','rb')})
print(resp.json())
