import pandas as pd
import random

crops = [
    "rice", "wheat", "maize", "cotton", "sugarcane",
    "barley", "millet", "soybean", "groundnut",
    "tea", "coffee", "banana", "mango", "apple",
    "orange", "grapes", "tomato", "potato",
    "onion", "carrot", "pea", "cabbage",
    "broccoli", "lettuce", "spinach"
]

data = []

for _ in range(1200):
    crop = random.choice(crops)
    N = random.randint(10, 120)
    P = random.randint(5, 60)
    K = random.randint(5, 60)
    temperature = round(random.uniform(15, 35), 2)
    humidity = round(random.uniform(40, 90), 2)
    ph = round(random.uniform(5.0, 7.5), 2)
    rainfall = round(random.uniform(50, 300), 2)
    data.append([N, P, K, temperature, humidity, ph, rainfall, crop])

columns = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "crop"]
df = pd.DataFrame(data, columns=columns)
df.to_csv('dataset.csv', index=False)
print('dataset.csv created with', len(df), 'rows')
