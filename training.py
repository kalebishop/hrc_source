import xml.etree.ElementTree as et
import csv
from colorsys import hsv_to_rgb
import statistics
import copy

from sklearn.utils.validation import check_is_fitted

from base_classes import Object, Context
from speech_module import SpeechModule
from re_generator import REG

class CorpusTraining:
    def __init__(self):
        # self.Theta = p_threshold
        w2c = "data/w2c_4096.txt"
        self.sm = SpeechModule(w2c)
        self.reg = REG()
        self.workspaces = {}

    def save_models(self):
        self.reg.save_models()

    def get_train_x_y(self, xml_workspace_filename, csv_responses_filename):
        self.parse_workspace_data_from_xml(xml_workspace_filename)

        responses = self.parse_responses_from_csv(csv_responses_filename)
        tokenized_responses = self.process_all_outputs(responses)

        feature_inputs = self.assemble_x(tokenized_responses)
        # TODO update to work with all responses (rather than one per qid)
        feature_outputs = self.assemble_Y(tokenized_responses)

        # raise NotImplementedError

        return feature_inputs, feature_outputs

    def train(self, feature_inputs, feature_outputs, save=True):
        clr_x, sz_x, dim_x = feature_inputs
        clr_y, sz_y, dim_y = feature_outputs

        # truncate to appropriate sizes
        clr_y = clr_y[:len(clr_x)]
        sz_y = sz_y[:len(sz_x)]
        dim_y = dim_y[:len(dim_x)]

        print(len(clr_x), len(clr_y))
        print(len(dim_x), len(dim_y))
        self.reg.train_model("color", clr_x, clr_y)
        self.reg.train_model("size", sz_x, sz_y)
        self.reg.train_model("dim", dim_x, dim_y)

        # self.reg.save_models()

    def parse_workspace_data_from_xml(self, filename):
        self.tree = et.parse(filename)
        self.root = self.tree.getroot() # data (elements are workspaces)

        for ws in self.root:
            id = ws.attrib["id"]
            obj_lst = []
            key_item = None

            for item in ws:
                feature_dict = {}
                item_id = item.attrib["id"]

                for datum in item:
                    if datum.tag == "type":
                         feature_dict["type"] = datum.text
                    elif datum.tag == "hsv":
                        # parse as tuple
                        # print(datum.text.split(', '))
                        h, s, v = [float(x) for x in datum.text.split(', ')]
                        # normalize from gimp conventions to [0, 1] range
                        h /= 360.0
                        s /= 100.0
                        v /= 100.0
                        rgb = hsv_to_rgb(h, s, v)
                        feature_dict["rgb"] = [255 * d for d in rgb]
                    elif datum.tag == "location" or datum.tag == "dimensions":
                        x = int(datum[0].text)
                        y = int(datum[1].text)
                        feature_dict[datum.tag] = (x, y)

                o = Object()
                o.from_dict(feature_dict)
                if item_id == "KEY":
                    key_item = o

                obj_lst.append(o)

            print(obj_lst)
            self.workspaces[id] = (key_item, Context(obj_lst))

    def assemble_x_for_q(self, obj, context, tokenized_response):
        labels, tokens = tokenized_response
        type = obj.get_feature_val("type")

        features = ["color", "size", "dimensions"]

        color_x = []
        size_x = []
        dim_x = []

        for t in tokens:
            if "color" in features:
                _, clr_score, clr_data, clr_kept_objects = self.reg.get_model_input("color", obj, context)
                color_x.append([clr_score, clr_data])
            if "size" in features:
                _, sz_score, sz_data, sz_kept_objects = self.reg.get_model_input("size", obj, context)
                size_x.append([sz_score, sz_data])
            if "dimensions" in features:
                _, dim_score, dim_data, dim_kept_objects = self.reg.get_model_input("dimensions", obj, context)
                dim_x.append([dim_score, dim_data])

            if t == self.sm.COLOR_I:
                kept = clr_kept_objects
                features.remove("color")
            elif t == self.sm.SIZE_I:
                kept = sz_kept_objects
                features.remove("size")
            elif t == self.sm.DIM_I:
                kept = dim_kept_objects
                features.remove("dimensions")

            context = self.reg.update_context(kept)

        # you'll always have a last one pre-noun
        if "color" in features:
            _, clr_score, clr_data, clr_kept_objects = self.reg.get_model_input("color", obj, context)
            color_x.append([clr_score, clr_data])
        if "size" in features:
            _, sz_score, sz_data, sz_kept_objects = self.reg.get_model_input("size", obj, context)
            size_x.append([sz_score, sz_data])
        if "dimensions" in features:
            _, dim_score, dim_data, dim_kept_objects = self.reg.get_model_input("dimensions", obj, context)
            dim_x.append([dim_score, dim_data])

        return color_x, size_x, dim_x

    def assemble_x(self, tokenized_responses):
        color_full_x = []
        size_full_x = []
        dim_full_x = []

        index = 0
        for ws in self.workspaces:
            obj, context = self.workspaces[ws]
            for pid_response in tokenized_responses[index]:
                color_x, size_x, dim_x = self.assemble_x_for_q(obj, context, pid_response)
                color_full_x += color_x
                size_full_x += size_x
                dim_full_x += dim_x
            index+=1

        return color_full_x, size_full_x, dim_full_x

    def test_labeling(self, key, c):
        # show usage
        # key, env = self.workspaces['Q3.2']
        # c = Context(env)
        w2c = "data/w2c_4096.txt"
        self.sm = SpeechModule(w2c)
        clr = self.sm.label_feature(key, c, "color")
        sz = self.sm.label_feature(key, c, "size")
        dim = self.sm.label_feature(key, c, "dimensions")

        # print(clr, sz, dm)
        return clr, sz, dim

    def parse_responses_from_csv(self, filename):
        with open(filename) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=",")
            header = next(csvreader)

            qs_to_indicies = {}
            indicies_to_qs = {}

            # get field names from first row
            for i in range(len(header)):
                field = header[i]
                # print(field)
                if field[0] == "Q":
                    indicies_to_qs[i] = field
                    qs_to_indicies[field] = i

            row = next(csvreader)
            all_responses = [[] for x in qs_to_indicies.keys()]

            for row in csvreader:
                count = 0
                for index in indicies_to_qs.keys():
                    response = row[index]
                    if response: # check response is not empty
                    # add response to appropriate list
                        all_responses[count].append(response)
                        count += 1
        return all_responses

    def assemble_Y(self, tokenized_responses):
        color_Ys = []
        size_Ys = []
        dim_Ys = []

        for qid in tokenized_responses:
            for labels, tokens in qid:
                clr, sz, dim = self.assemble_Y_for_q(tokens)
                color_Ys += clr
                size_Ys += sz
                dim_Ys += dim

        return color_Ys, size_Ys, dim_Ys

    def assemble_Y_for_q(self, tokenized_response):
        if not tokenized_response:
            return [False], [False], [False]

        color_y = []
        size_y = []
        dim_y = []

        base = [self.sm.COLOR_I, self.sm.SIZE_I, self.sm.DIM_I]
        features = copy.copy(base)
        for token in tokenized_response:
            res = list(map(lambda x: token == x, base))
            if self.sm.COLOR_I in features:
                color_y.append(res[0])
            if self.sm.SIZE_I in features:
                size_y.append(res[1])
            if self.sm.DIM_I in features:
                dim_y.append(res[2])

            features.remove(token)

        # last pre-noun space
        if self.sm.COLOR_I in features:
            color_y.append(False)
        if self.sm.SIZE_I in features:
            size_y.append(False)
        if self.sm.DIM_I in features:
            dim_y.append(False)

        return color_y, size_y, dim_y

    def process_all_outputs(self, all_responses):
        # NEW VERSION: use all responses (requires data cleaning)
        parsed_responses = []
        for qid in all_responses:
            qid_y = []
            for r in qid:
                labels, tokens = self.sm.process_speech_string(r)
                qid_y.append((labels, tokens))
            parsed_responses.append(qid_y)
        return parsed_responses

if __name__ == "__main__":
    trainer = CorpusTraining()
    xml_file = "data/stim_v1.xml"
    csv_file = "data/latest.csv"

    inputs, outputs = trainer.get_train_x_y(xml_file, csv_file)
    # #
    # responses = [["screwdriver", "blue screwdriver"], ["bottle", "red bottle"]]
    #
    # tokenized = trainer.process_all_outputs(responses)
    # trainer.parse_workspace_data_from_xml(xml_file)
    # # #
    # inputs = trainer.assemble_x(tokenized)
    # outputs = trainer.assemble_Y(tokenized)

    clr_x, sz_x, dim_x = inputs
    clr_y, sz_y, dim_y = outputs

    trainer.train(inputs, outputs)
    trainer.save_models()
