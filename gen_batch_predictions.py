# from training import CorpusTraining
from training_v2 import CorpusTraining

XML_FILES = ["data/stim_v1_original.xml", "data/stim_v2.xml"]

def predict_batch():
    trainer = CorpusTraining()

    all_predictions = []
    trainer.reg.load_models()

    with open('batch_predictions.txt', 'w') as file:
        for xml in XML_FILES:
            count = 0
            print("From file ", xml, file=file)
            trainer.workspaces = {}
            trainer.parse_workspace_data_from_xml(xml)
            for ws in trainer.workspaces:
                obj, context = trainer.workspaces[ws]
                predicted = trainer.reg.generate_output(obj, context)
                count += 1
                print(count, predicted, file=file)


if __name__ == "__main__":
    predict_batch()
