import csv
import types
from statistics import mode

PID_COUNT = -1

COLOR_I = 0
SIZE_I = 1
DIM_I = 2

COLOR = ["red", "yellow", "blue", "green", "purple", "grey", "white", "violet"]
SIZE = ["big", "biggest", "small", "smallest"]
DIM = ["long", "longest", "loing", "short", "shortest", "length", "rectang", "retang", "square", "cub", "brick"]

def get_modal_response_per_stim():
    # read from csv titled latest.csv
    with open("data/study_v1_responses.csv", newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",")
        header = next(csvreader)
        stim_dict = {}

        qs_to_indicies = {}
        indicies_to_qs = {}

        # get field names from first row
        for i in range(len(header)):
            field = header[i]
            # print(field)
            if field[0] == "Q":
                indicies_to_qs[i] = field
                qs_to_indicies[field] = i

                stim_dict[field] = []

        row = next(csvreader)
        row = next(csvreader)


        for row in csvreader:
            for index in indicies_to_qs.keys():
                response = row[index].lower()
                # print(response)

                # analyze response feature usage
                # entry = stim_dict[indicies_to_qs[index]]
                if response:
                    stim_dict[indicies_to_qs[index]].append(response)

        for q in stim_dict:
            responses = stim_dict[q]
            r_modal = mode(responses)
            stim_dict[q] = [r_modal]

    return stim_dict


def get_usage_counts_per_stim():
    # read from csv titled latest.csv
    with open("data/study_v2_responses.csv", newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",")
        header = next(csvreader)
        stim_dict = {}

        qs_to_indicies = {}
        indicies_to_qs = {}

        # get field names from first row
        for i in range(len(header)):
            field = header[i]
            # print(field)
            if field[0] == "Q":
                indicies_to_qs[i] = field
                qs_to_indicies[field] = i

                stim_dict[field] = [0, 0, 0]

        row = next(csvreader)
        row = next(csvreader)


        for row in csvreader:
            for index in indicies_to_qs.keys():
                response = row[index].lower()
                # print(response)

                # analyze response feature usage
                entry = stim_dict[indicies_to_qs[index]]

                for c in COLOR:
                    if c in response:
                        entry[COLOR_I] += 1
                        break
                for s in SIZE:
                    if s in response:
                        entry[SIZE_I] += 1
                        break
                for d in DIM:
                    if d in response:
                        entry[DIM_I] += 1
                        break

    return stim_dict

def write_csv_from_dict(dict, output_filename, header_fields=[]):
    with open(output_filename, "w", newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        csvwriter.writerow(header_fields)
        for key in dict:
            csvwriter.writerow([key] + dict[key])

def get_usage_counts_per_pid():
    # read from csv titled latest.csv
    with open("data/study_v2_responses.csv", newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",")
        header = next(csvreader)
        response_dict = {}

        question_indicies = []

        # get field names from first row
        for i in range(len(header)):
            field = header[i]
            # print(field)
            if field[0] == "Q":
                question_indicies.append(i)

        row = next(csvreader)
        row = next(csvreader)
        # iterate over first 40 samples for testing
        # pid_count = 0
        # import pdb; pdb.set_trace()
        for row in csvreader:
            pid = row[-1]
            if not pid:
                continue

            response_dict[pid] = []
            total_color = total_size = total_dim = 0

            for index in question_indicies:
                response = row[index].lower()
                # print(response)

                # analyze response feature usage
                # entry = stim_dict[indicies_to_qs[index]]

                for c in COLOR:
                    if c in response:
                        total_color+=1
                        break
                for s in SIZE:
                    if s in response:
                        total_size+=1
                        break
                for d in DIM:
                    if d in response:
                        total_dim += 1
                        break

                # q_entry = str(q_entry[COLOR_I]) + str(q_entry[SIZE_I]) + str(q_entry[DIM_I])
                response_dict[pid] = [total_color, total_size, total_dim]

    return response_dict

def get_response_tables_by_feature():
    # read from csv titled latest.csv
    with open("data/study_v2_responses.csv", newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",")
        header = next(csvreader)
        response_dict = {}

        question_indicies = []

        # get field names from first row
        for i in range(len(header)):
            field = header[i]
            # print(field)
            if field[0] == "Q":
                question_indicies.append(i)

        row = next(csvreader)
        row = next(csvreader)
        # pid_count = 0
        # import pdb; pdb.set_trace()
        count = 0
        for row in csvreader:
            pid = row[-1]
            if not pid:
                continue


            for index in question_indicies:
                used_color = used_size = used_dim = 0
                response = row[index].lower()
                # print(response)

                response_dict[count] = [pid]
                # analyze response feature usage
                # entry = stim_dict[indicies_to_qs[index]]

                for c in COLOR:
                    if c in response:
                        used_color = 1
                        break
                for s in SIZE:
                    if s in response:
                        used_size = 1
                        break
                for d in DIM:
                    if d in response:
                        used_dim = 1
                        break

                # q_entry = str(q_entry[COLOR_I]) + str(q_entry[SIZE_I]) + str(q_entry[DIM_I])
                response_dict[count] += [used_color, used_size, used_dim]
                count+=1

    return response_dict

def main():
    # dict = get_usage_counts_per_stim()
    # header_fields = ["qid", "color", "size", "dim"]
    # output_filename = "data/v2_data_by_qid.csv"
    #
    # write_csv_from_dict(dict, output_filename, header_fields)
    #
    #
    # dict = get_usage_counts_per_pid()
    # header_fields = ["pid", "color", "size", "dim"]
    # output_filename = "data/v2_data_by_pid.csv"
    #
    # write_csv_from_dict(dict, output_filename, header_fields)
    #
    #
    # dict = get_response_tables_by_feature()
    # header_fields = ["pid", "used_color", "used_size", "used_dim"]
    # output_filename = "data/v2_data_by_qid_and_pid.csv"
    #
    # write_csv_from_dict(dict, output_filename, header_fields)
    dict = get_modal_response_per_stim()
    output_filename = "data/study_v2_modal_responses.csv"
    write_csv_from_dict(dict, output_filename)



if __name__ == "__main__":
    main()
