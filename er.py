import Algorithmia

input = {
  "image": "data://deeplearning/example_data/elon_musk.jpg",
  "numResults": 7
}
client = Algorithmia.client('YOUR_API_KEY')
algo = client.algo('deeplearning/EmotionRecognitionCNNMBP/1.0.1')
print(algo.pipe(input).result)