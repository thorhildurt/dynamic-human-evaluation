class DataframeCollections:
    def __init__(self):
        self.df_y_proportion_x_total_labels = []
        self.df_y_labels_x_requests = []
        self.df_y_proportion_x_samples = []
        self.df_y_proportion_x_workers = []

    def update_dataframes(self, p, rd_mean, var, evaluation_results):
        update_y_proportion_x_samples(self.df_y_proportion_x_samples, p, rd_mean, var, evaluation_results)

    def update_y_proportion_x_samples_row(self, project_id, rd_mean, method_id, method_name,
                                          method_name_short, labels_per_request, answer_label, proportion, req_id):
        df_row = {
            'project_id': project_id,
            'evaluation_id': method_id,
            'labels_per_request': labels_per_request,
            'answer_label': answer_label,
            'proportion_of_1': proportion,
            'req_id': req_id,
            'rd_mean': rd_mean,
        }
        self.df_y_proportion_x_samples.append(df_row)


def update_y_proportion_x_samples(df, project_id, rd_mean, var, evaluation_results):
    answers = evaluation_results.request_answers
    total_sum = 0
    for i in range(len(evaluation_results.request_index_tick)):
        total_sum += answers[i]
        proportion = total_sum / (i + 1)
        df_row = {
            'project_id': project_id,
            'evaluation_id': evaluation_results.method_id,
            'evaluation_method': evaluation_results.method_name,
            'short_method_name': evaluation_results.method_name_short,
            'labels_per_request': evaluation_results.labels_per_request[i],
            'answer_label': answers[i],
            'proportion_of_1': proportion,
            'req_id': (i + 1),
            'rd_mean': rd_mean
        }
        df.append(df_row)
