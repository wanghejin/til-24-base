from transformers import BertTokenizer, BertTokenizerFast, BertForQuestionAnswering, TrainingArguments, Trainer
import torch
import pandas as pd

"""
}
"key": 0,
"transcript": "Turret, prepare to deploy electromagnetic pulse.
                Heading zero six five, target is grey and white fighter jet.
                Engage when ready.",
"tool": "electromagnetic pulse",
"heading": "065",
"target": "grey and white fighter jet"
}
"""
model_name = "bert-base-uncased"
tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

def data_to_df(instances):
        # questions: [target, heading, tool]
        # obtain ALL context, questions and answers
        question_keys = ["What is the target to neutralize?",
                        "What is the heading in numbers?",
                        "What is the tool to neutralize the target?"]
        answer_keys = ["target", "heading", "tool"]
        context = []
        questions = []
        answers = []
        answer_start = []
        num_to_word = {"0":"zero", "1":"one", "2":"two", "3":"three", "4":"four", "5":"five", "6":"six",
                       "7":"seven", "8":"eight", "9":"nine"}
        for _instance in instances: # each _instance has format above
            for i in range(len(question_keys)):
                _context = _instance["transcript"]
                context.append(_context)
                _question = question_keys[i]
                questions.append(_question)
                _answer = _instance[answer_keys[i]]
                if answer_keys[i]=="heading":
                    _answer = " ".join([num_to_word[num] for num in [*_answer]])
                answers.append(_answer)
                answer_start.append(_context.find(_answer))
        return pd.DataFrame({'context':context,
                            'questions':questions,
                            'answers':answers,
                            'answer_start':answer_start})

def preprocess_df(entry):
        
        context = entry['context']
        questions = entry['questions']
        answers = entry['answers']
        answer_start = entry['answer_start']

        # obtain tokenized input, offset mappings (used to calc start/end positions of answers in context)
        inputs = tokenizer(questions, context,
                                max_length=384, truncation="only_second",
                                return_offsets_mapping=True,
                                padding="max_length")
        offset_mapping = inputs.pop("offset_mapping")
        start_positions = []
        end_positions = []
        for i, offset in enumerate(offset_mapping):
            answer = answers[i]
            start_char = answer_start[i]
            end_char = start_char + len(answer)
            sequence_ids = inputs.sequence_ids(i)

            context_start = sequence_ids.index(1)
            context_end = len(sequence_ids) - 1 - sequence_ids[::1].index(1)
            if not (offset[context_start][0] <= start_char and offset[context_end - 1][1] >= end_char):
                start_positions.append(0)
                end_positions.append(0)
            else:
                start_positions.append(sequence_ids.index(1, offset.index((start_char, end_char))))
                end_positions.append(sequence_ids.index(1, offset.index((end_char, start_char + len(answer["text"][0])))))
        inputs['start_pos'] = start_positions
        inputs['end_pos'] = end_positions
        return inputs

##################################################
import json
import math
from datasets import Dataset # pip install

with open("til-24-base/nlp/src/nlp.jsonl", "r") as f:
    instances = [json.loads(line.strip()) for line in f if line.strip() != ""]
df = data_to_df(instances)
dataset = Dataset.from_pandas(df)
train_dataset, test_dataset = dataset.train_test_split(test_size=0.2).values()
tokenized_train_dataset = train_dataset.map(preprocess_df, batched=True)
tokenized_test_dataset = test_dataset.map(preprocess_df, batched=True)
##################################################

training_args = TrainingArguments(
     output_dir='./results',
     evaluation_strategy='epoch',
     learning_rate= 2*(math.e**(-5)),
     per_device_train_batch_size=16,
     per_device_eval_batch_size=16,
     num_train_epochs=3,
     weight_decay=0.01,
     logging_dir='./logs',
     logging_steps=10
)

trainer = Trainer(
     model=model,
     args=training_args,
     train_dataset=tokenized_train_dataset,
     test_dataset=tokenized_test_dataset,
     tokenizer=tokenizer
)

trainer.train()
model.save_pretrained('./fine-tuned-bert-qa')
tokenizer.save_pretrained('./fine-tuned-bert-qa')

