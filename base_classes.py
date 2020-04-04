import math
import statistics

def process_workspace_from_msg(msg):
    new_workspace = []
    for obj_msg in msg.ObjectArray:
        new_workspace.append(Object(obj_msg))
    return Context(new_workspace)

class Object:
    def __init__(self, msg=None):
        # TODO update commented section

        if msg:
            self.features = {
                "id" : msg.id,           # object id
                "type": msg.type,        # object type (block, screwdriver, etc)
                "rgb": (msg.color.r, msg.color.g, msg.color.b),      # object color as RGBA
                "dimensions": (msg.x_dim, msg.y_dim, msg.z_dim), # object dimentions (estimated)
                # "pose": msg.pose.position # object pose (estimated) as Position msg (xyz)
            }

    def from_dict(self, dict):
        self.features = dict

    def get_feature_val(self, feature):
        if feature == "color":
            try:
                return self.features["color"]
            except KeyError:
                return self.features["rgb"]
        else:
            return self.features[feature]

    def _set_feature_val(self, feature, val):
        self.features[feature] = val

class Context:
    def __init__(self, objs, name=""):
        self.env = objs
        self.env_size = len(objs)
        self._intialize_feature_distributions()

        # self.sp_clf = spatial_model

    def object_lookup(self, id):
        return self.env[id]

    def return_all_feature_vals(self, feature):
        res = []
        for obj in self.env:
            res.append(obj.get_feature_val(feature))
        return res

    def get_type_match(self, type):
        out_lst = []
        for o in self.env:
            if o.get_feature_val('type') == type:
                out_lst.append(o)
        return out_lst

    # def feature_match(self, feature, value):
    #     matches = {}
    #     count = 0
    #     for id in self.env:
    #         obj = self.env[id]
    #         if self.get_obj_label(obj, feature) == value:
    #             matches[id] = obj
    #             count += 1
    #     return matches, count

    def get_obj_label(self, obj, feature):
        # if feature == "location":
        #     return self.sp_clf.predict(obj.get_feature_val("location"), self)
        # else:
        return obj.get_feature_val(feature)

    def _intialize_feature_distributions(self):
        if self.env_size == 1:
            o = self.env[0]
            o._set_feature_val("z_size", 0)
            o._set_feature_val("z_dim", 0)
            return

        obj_sizes = {}
        obj_ratios = {}
        for o in self.env:
            type = o.get_feature_val("type")
            dims = [float(d) for d in o.get_feature_val("dimensions")]
            sz = 1.0
            for d in dims:
                sz *= d

            try:
                obj_sizes[type].append(sz)
            except KeyError:
                obj_sizes[type] = [sz]

            larger = max([dims[0], dims[1]])
            smaller = min([dims[0], dims[1]])
            try:
                obj_ratios[type].append(larger/smaller)
            except KeyError:
                obj_ratios[type] = [larger/smaller]

        for type in obj_sizes.keys():
            if len(obj_sizes[type]) == 1:
                sz_xbar = obj_sizes[type][0]
                sz_sd = 0

                dim_xbar = obj_ratios[type][0]
                dim_sd = 0
            else:
                sz_xbar = statistics.mean(obj_sizes[type])
                sz_sd   = statistics.stdev(obj_sizes[type], sz_xbar)

                dim_xbar = statistics.mean(obj_ratios[type])
                dim_sd   = statistics.stdev(obj_ratios[type], dim_xbar)

            obj_sizes[type] = (sz_xbar, sz_sd)
            obj_ratios[type] = (dim_xbar, dim_sd)

        for o in self.env:
            type = o.get_feature_val("type")
            dims = [float(d) for d in o.get_feature_val("dimensions")]

            sz = 1.0
            for d in dims:
                sz *= d
            larger = max([dims[0], dims[1]])
            smaller = min([dims[0], dims[1]])
            ratio = larger/smaller

            sz_xbar, sz_sd = obj_sizes[type]
            dim_xbar, dim_sd = obj_ratios[type]

            z_size = (sz - sz_xbar) / sz_sd if sz_sd != 0 else 0
            z_dim = (ratio - dim_xbar) / dim_sd if dim_sd != 0 else 0

            o._set_feature_val("z_size", z_size)
            o._set_feature_val("z_dim", z_dim)

        # grab size and dimensions
        # all_sizes = []
        # all_ratios = []
        # for o in self.env:
        #     dims = [float(d) for d in o.get_feature_val("dimensions")]
        #     sz = 1.0
        #     for d in dims:
        #         sz *= d
        #     all_sizes.append(sz)
        #     all_ratios.append(dims[0]/dims[1])
        #
        # self.size_xbar = statistics.mean(all_sizes)
        # self.size_sd = statistics.stdev(all_sizes, self.size_xbar)
        #
        # self.dim_xbar = statistics.mean(all_ratios)
        # self.dim_sd = statistics.stdev(all_ratios, self.dim_xbar)


    # def _initialize_workspace_location_info(self):
    #     # should all this be calculated dynamically?
    #     # calculate centroid (based on x, y)
    #     sum_x = 0
    #     sum_y = 0
    #
    #     # store info on max and min x, y, z(workspace bounding box)
    #     x_bounds = [math.inf, -math.inf]
    #     y_bounds = [math.inf, -math.inf]
    #     z_bounds = [math.inf, -math.inf]
    #
    #     for o in self.env:
    #         x, y, z = o.get_feature_class_value("location")
    #         sum_x += x
    #         sum_y += y
    #
    #         x_bounds[0] = min(x_bounds[0], x)
    #         x_bounds[1] = max(x_bounds[1], x)
    #
    #         y_bounds[0] = min(y_bounds[0], y)
    #         y_bounds[1] = max(y_bounds[1], y)
    #
    #         z_bounds[0] = min(z_bounds[0], z)
    #         z_bounds[1] = max(z_bounds[1], z)
    #
    #     self.workspace_centroid = (sum_x / self.env_size, sum_y / self.env_size, 0)
    #
    #     x_net_max = max(abs(x_bounds[0] - self.workspace_centroid[0]), abs(x_bounds[1] - self.workspace_centroid[0]))
    #     y_net_max = max(abs(y_bounds[0] - self.workspace_centroid[0]), abs(y_bounds[1] - self.workspace_centroid[0]))
    #     z_net_max = max(abs(z_bounds[0] - self.workspace_centroid[0]), abs(z_bounds[1] - self.workspace_centroid[0]))
    #
    #     self.bounds = {'x': x_net_max, 'y': y_net_max, 'z': z_net_max}
    #     self.max_distance_norm = math.hypot(x_net_max, y_net_max)
