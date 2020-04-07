import numpy as np
import copy
import pickle
import os
import copy

from sklearn.linear_model import LogisticRegression

from speech_module import SpeechModule
from base_classes import Object, Context

class SpeechLearner:
    def __init__(self):
        self.clf = LogisticRegression(max_iter=1000)

    def train(self, X, y):
        self.clf.fit(X, y)
        print(self.clf.classes_)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_probs(self, X):
        return self.clf.predict_proba(X)

    def plot_learned_function(self, data):
        # TODO implement
        raise NotImplementedError
        pass

    def save_model(self, filename):
        pickle.dump(self.clf, open(filename, 'wb'))

    def load_model(self, filename):
        self.clf = pickle.load(open(filename, 'rb'))

class REG:
    def __init__(self):
        w2c = "data/w2c_4096.txt"
        # w2c = "/ros/catkin_ws/src/hrc_discrim_learning/src/hrc_discrim_learning/data/w2c_4096.txt"
        self.sm = SpeechModule(w2c)
        self.theta = 0.5

        self.features = ["color", "size", "dim", "none"]

        self.model = SpeechLearner()

    def save_models(self, filename="model/lgreg.pkl"):
        self.model.save_model(filename)

    def load_models(self, filename="model/lgreg.pkl"):
        self.model.load_model(filename)
        self.mapping = self.model.clf.classes_

    def train_model(self, x, y):
        self.model.train(x, y)
        self.mapping = self.model.clf.classes_

    def _generate_single_output(self, object, context, type_env, feature_set):
        """ UPDATED """
        features= copy.copy(feature_set)
        model_input, labels, results = self.get_model_input(object, context)
        # prediction = self.mapping[self.model.predict([model_input])]
        pdist = self.model.predict_probs([model_input])[0]
        # map prob distributions as a dict
        pdist_as_dict = {}
        for i in range(4):
            cls = self.mapping[i]
            pdist_as_dict[cls] = pdist[i]

        if len(type_env) > 1 and "none" in features:
            features.remove("none")
        best = max(features, key=lambda x: pdist_as_dict[x])

        if best == "none":
            return None, None, None
        else:
            return best, labels[best], results[best]

    def generate_output(self, object, context):
        # context should include object
        """ UPDATED """
        import pdb; pdb.set_trace()
        output = ""
        type = object.get_feature_val('type')
        type_env = context.get_type_match(type)

        feature_set = copy.copy(self.features)
        while feature_set:
            feature, label, new_context = self._generate_single_output(object, context, type_env, feature_set)
            if not label:
                break
            output += (label + " ")
            context = Context(new_context)
            type_env = context.get_type_match(type)
            feature_set.remove(feature)

        output+=type
        return output


        # output = ""
        #
        # type = object.get_feature_val("type")
        # type_env = context.get_type_match(type)
        #
        # # next: iterate through possible features
        # feature_set = copy.copy(self.features)
        # while feature_set:
        #     feature, label, new_context = self._generate_single_output(object, context, type_env, feature_set)
        #     if not label:
        #         output += type
        #         return output
        #
        #     output += (label + " ")
        #     context = Context(new_context)
        #     type_env = context.get_type_match(type)
        #     feature_set.remove(feature)
        #
        # # return "ERR: check REGenerator"
        # return output

    def get_model_input(self, object, context):
        v = []
        labels = {}
        results = {}
        for feature in self.features[:-1]:
            label, score, data, kept_objects = self._get_model_feature_input(feature, object, context)
            labels[feature] = label
            results[feature] = kept_objects
            v += [score, data]

        return v, labels, results

    def _get_model_feature_input(self, feature, object, context):
        # context should include object
        label, data = self.sm.label_feature(object, context, feature)

        if feature == "color" or feature == "type":
            score, kept_objects = self.elim_objects_discrete(context, label, feature)
        else:
            score, kept_objects = self.elim_objects_gradable(context, feature, label, data)

        return label, score, data, kept_objects

    def elim_objects_discrete(self, context, label, feature):
        # we want to eliminate everything that the term label can NOT apply to
        score = 0
        kept_objects = []
        for o in context.env:
            this_label, data = self.sm.label_feature(o, context, feature)
            if label == this_label:
                kept_objects.append(o)
            else:
                score += 1

        return score, kept_objects

    def elim_objects_gradable(self, context, feature, label, label_score):
        # we want to eliminate everything that the term can NOT apply to
        # that is, everything that the term fits LESS well than the target object
        score = 0
        kept_objects = []
        for o in context.env:
            this_label, data = self.sm.label_feature(o, context, feature)
            if this_label == label and data >= label_score:
                kept_objects.append(o)
            else:
                score += 1

        return score, kept_objects

    def update_context(self, kept_objects):
        return Context(kept_objects)
