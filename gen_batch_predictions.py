from training import CorpusTraining


def predict_batch(xml_workspace_filename):
    trainer = CorpusTraining()
    trainer.parse_workspace_data_from_xml(xml_workspace_filename)
    count = 0

    trainer.reg.load_models()
    for ws in trainer.workspaces:
        obj, context = trainer.workspaces[ws]
        predicted = trainer.reg.generate_output(obj, context)
        count += 1
        print(count, predicted)

if __name__ == "__main__":
    predict_batch("data/stim_v2.xml")
