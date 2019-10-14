from allennlp.models.archival import load_archive
from allennlp.common.file_utils import cached_path
from allennlp.predictors.predictor import Predictor
import reading_comprehension
import json
import argparse

@Predictor.register('nabert_predictor')
class NABERTPredictor(Predictor):
    def predict_json(self, json_dict):
        for instance in self._dataset_reader.get_instance(json_dict):
            if instance is not None:
                predicted_dict = self.predict_instance(instance)
        return predicted_dict

if __name__ == "__main__":
    parse = argparse.ArgumentParser("")
    parse.add_argument("model")
    parse.add_argument("dataset")
    parse.add_argument("output_file")
    parse.add_argument("--cuda_device", type=int, default=-1)
    args = parse.parse_args()
    file_path = cached_path(args.model)
    archive = load_archive(file_path, cuda_device=0)
    predictor = Predictor.from_archive(archive, 'nabert_predictor')
    predictions = {}
    with open(args.dataset) as fr:
        for line in fr:
            predicted_dict = predictor.predict_json(json.loads(line))
            predictions[predicted_dict["question_id"]] = [predicted_dict["answer"]["value"]]
    json.dump(predictions, open(args.output_file, 'w'))
