import json
import shutil
import sys

from allennlp.commands import main

config_file = "/mnt/750GB/workspace/nabert/training_configs/nabert.json"

serialization_dir = "/mnt/750GB/data/augment_qa/demo"

# Training will fail if the serialization directory already
# has stuff in it. If you are running the same training loop
# over and over again for debugging purposes, it will.
# Hence we wipe it out in advance.
# BE VERY CAREFUL NOT TO DO THIS FOR ACTUAL TRAINING!
shutil.rmtree(serialization_dir, ignore_errors=True)

# Assemble the command into sys.argv-f
sys.argv = [
    "allennlp",  # command name, not used by main
    "train",
    config_file,
    "-s", serialization_dir,
    "--include-package", "reading_comprehension",
    "-o", json.dumps({"trainer": {"cuda_device": -1}, "iterator":{"batch_size": 2}})]
                      # "train_data_path": "/home/ddua/data/qa_eval/annotation_format/demo.json",
                      # "validation_data_path": "/home/ddua/data/qa_eval/annotation_format/demo.json"
                     # })]

# Assemble the command into sys.argv
# main()
