import ollama

response = ollama.chat(model='deepseek-coder-v2:lite', messages=[
    {'role': 'user', 'content': 'Ecris un script Python simple pour nettoyer un CSV avec Pandas.'},
])

print("Réponse de l'IA locale :")
print(response['message']['content'])
