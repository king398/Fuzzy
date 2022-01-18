from thefuzz import fuzz
from thefuzz import process
import torch
from transformers import AutoTokenizer, AutoModel
from torch import Tensor, device


def cos_sim(a: Tensor, b: Tensor):
	"""
	borrowed from sentence transformers repo
	Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
	:return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
	"""
	if not isinstance(a, torch.Tensor):
		a = torch.tensor(a)

	if not isinstance(b, torch.Tensor):
		b = torch.tensor(b)

	if len(a.shape) == 1:
		a = a.unsqueeze(0)

	if len(b.shape) == 1:
		b = b.unsqueeze(0)

	a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
	b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
	return torch.mm(a_norm, b_norm.transpose(0, 1))


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
	token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
	input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
	return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


word1 = "law of multiple proportions states that if two elements form more than one compound,then the ratios of the masses of the second element which combine with a fixed mass of the first element will always be ratios of small whole numbers",
word2 = "idk"
tokenizer = AutoTokenizer.from_pretrained('shahrukhx01/paraphrase-mpnet-base-v2-fuzzy-matcher')
model = AutoModel.from_pretrained('shahrukhx01/paraphrase-mpnet-base-v2-fuzzy-matcher')
# what the student answers
len_of_answer = len(word1)
len_of_question = len(word2)
len(word1)
if (len(word2) / len(word1)) * 100 <= 30:
	score = 0


else:
	if len(word1) > 50:
		word1 = " ".join([char for char in word1])  ## divide the word to char level to fuzzy match
		word2 = " ".join([char for char in word2])  ## divide the word to char level to fuzzy match
		words = [word1, word2]
		encoded_input = tokenizer(words, padding=True, truncation=True, return_tensors='pt')

		# Compute token embeddings
		with torch.no_grad():
			model_output = model(**encoded_input)

		# Perform pooling. In this case, max pooling.
		fuzzy_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

		print("Fuzzy Match score:")
		score = cos_sim(fuzzy_embeddings[0], fuzzy_embeddings[1]) * 100
	else:
		score = fuzz.token_set_ratio(word1, word2)
