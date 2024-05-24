from typing import Dict
from transformers import AutoTokenizer, BertForQuestionAnswering
import torch

class NLPManager:
    def __init__(self):
        self.model_name = './fine-tuned-bert-qa'
        self.model = BertForQuestionAnswering.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.number_map = {
            "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
            "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"
            }

    def qa(self, context: str) -> Dict[str, str]:
        # perform NLP question-answering
        to_answer = context
        questions = ["What is the target to neutralize?",
                     "What is the heading in numbers?",
                     "What is the tool to neutralize the target?"]
        answers = []
        for question in questions:
            
            inputs = self.tokenizer.encode_plus(question, to_answer, add_special_tokens=True, return_tensors="pt")
            input_ids = inputs["input_ids"].tolist()[0]
            
            outputs = self.model(**inputs)
            answer_start_scores=outputs.start_logits
            answer_end_scores=outputs.end_logits

            answer_start = torch.argmax(answer_start_scores)
            answer_end = torch.argmax(answer_end_scores) + 1
            answer = self.tokenizer.convert_tokens_to_string(
                self.tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end])
            )

            # Combine the tokens in the answer and print it out.""
            answer = answer.replace("#","")
            answers.append(answer)
        
        answers[1] = "".join([self.number_map[word] for word in answers[1].split() if word in self.number_map])
        answers[2] = answers[2].replace(" - ", "-")
        toReturn = {"heading": answers[1], "tool": answers[2], "target": answers[0]}
        print(toReturn)
        
        return toReturn