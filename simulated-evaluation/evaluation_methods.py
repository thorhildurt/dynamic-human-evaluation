import numpy as np


class EvaluationResults:
    def __init__(self, method_id, method_name, method_name_short, requests, procedure, workers=0):
        self.method_id = method_id
        self.method_name = method_name
        self.method_name_short = method_name_short
        self.procedure = procedure
        self.request_index_tick = np.arange(len(requests))
        self.request_answers = []
        self.accuracy_per_label = [0.5]
        self.label_work_count = [0]
        self.labels_per_request = []
        self.proportion_correct_choice = 0
        self.total_workers = 0
        self.req_answers_per_worker = np.full((workers, len(requests)), -1)
        self.random_answers_cnt = 0
        self.requests = requests


# the same random worker evaluates all requests
def one_worker_evaluation(dfc, p, mean, workers, requests, procedure="two_choice"):
    method_id = f"ONE"
    method_name = "One Worker Labels Each Request"
    method_name_short = "One Worker"
    e = EvaluationResults(method_id=method_id, method_name=method_name, method_name_short=method_name_short, requests=requests, procedure=procedure)
    e.total_workers = 1
    # one worker to label all requests
    worker = np.random.choice(workers)
    total_sum = 0
    for idx, req in enumerate(requests):
        req_id = idx + 1
        answer = _get_answer(req, worker, e, procedure)
        e.request_answers.append(answer)
        # calculate the current proportion of correct answers
        total_sum += answer
        proportion = total_sum / (idx + 1)
        e.accuracy_per_label.append(proportion)
        e.label_work_count.append(idx+1)
        label_per_req = 1
        e.labels_per_request.append(label_per_req)

        # update results dataframe
        dfc.update_y_proportion_x_samples_row(p, mean, method_id, method_name, method_name_short, label_per_req, answer, proportion, req_id)

    e.proportion_correct_choice = total_sum / len(requests)

    return e


def one_random_worker_evaluation(dfc, p, mean, workers, requests, procedure="two_choice"):
    method_id = f"ONE_R"
    method_name = "Random Worker Labels Each Request"
    method_name_short = 'Random Worker'
    e = EvaluationResults(method_id=method_id, method_name=method_name, method_name_short=method_name_short, requests=requests, procedure=procedure)
    e.total_workers = 1
    total_sum = 0
    # one worker to label all requests
    for idx, req in enumerate(requests):
        req_id = idx + 1
        worker = np.random.choice(workers)
        answer = _get_answer(req, worker, e, procedure)
        e.request_answers.append(answer)
        # calculate the current proportion of correct answers
        total_sum += answer
        proportion = total_sum / (idx + 1)
        e.accuracy_per_label.append(proportion)
        e.label_work_count.append(idx+1)
        label_per_req = 1
        e.labels_per_request.append(label_per_req)

        # update results dataframe
        dfc.update_y_proportion_x_samples_row(p, mean, method_id, method_name, method_name_short, label_per_req, answer,
                                              proportion, req_id)
    e.proportion_correct_choice = total_sum / len(requests)

    return e


# n random workers evaluate the same request and the simple
# majority vote selects the final answer for a request.
# we do this for all requests
def n_workers_evaluation_mv(dfc, p, mean, workers, requests, n, procedure="two_choice"):
    method_id = f"MV_{n}"
    method_name = f"{n} Workers Label Each Request (Majority Vote)"
    method_name_short = f'{n} Workers'
    e = EvaluationResults(method_id=method_id, method_name=method_name, method_name_short=method_name_short, requests=requests, procedure=procedure, workers=n)
    e.total_workers = n
    work_count = 0

    requests = np.asarray(requests)
    total_sum = 0
    for idx1, req in enumerate(requests):
        req_id = idx1 + 1
        temp_answers = []
        # for each request we sample n workers
        n_workers = np.random.choice(workers, n, replace=False)
        for idx2, worker in enumerate(n_workers):
            answer = _get_answer(req, worker, e, procedure)
            temp_answers.append(answer)
            work_count = work_count + 1
            e.label_work_count.append(work_count)
            e.req_answers_per_worker[idx2][idx1] = answer
        voting_rate = temp_answers.count(True) / n
        if voting_rate > 0.5:
            final_answer = 1
            e.request_answers.append(final_answer)
        elif voting_rate == 0.5:
            # 50/50 probability due to conflict
            final_answer = _get_random_answer(e)
            e.request_answers.append(final_answer)
        else:
            final_answer = 0
            e.request_answers.append(final_answer)
        total_sum += final_answer
        proportion = total_sum / (idx1 + 1)
        li = [proportion] * n
        e.accuracy_per_label.extend(li)
        e.labels_per_request.append(n)

        # update results dataframe
        dfc.update_y_proportion_x_samples_row(p, mean, method_id, method_name, method_name_short, n, final_answer,
                                              proportion, req_id)
    e.proportion_correct_choice = total_sum / len(requests)

    return e


# 2 random workers label each request, if there is a conflict we
# get 1 additional random worker to resolve the conflict
def conflict_resolution(dfc, p, mean, workers, requests, procedure="two_choice"):
    method_id = f"MAX3"
    method_name = f"Max Three Random Workers Label Each Request"
    method_name_short = 'Max 3 Workers'
    e = EvaluationResults(method_id=method_id, method_name=method_name, method_name_short=method_name_short, requests=requests, procedure=procedure, workers=len(workers))

    worker_cnt = 0
    total_sum = 0
    total_worker_set = set()
    for idx, req in enumerate(requests):
        req_id = idx + 1
        temp_answers = []
        for labels_cnt in range(1, 4):
            worker_index = np.random.choice(np.arange(len(workers)))
            total_worker_set.add(worker_index)
            answer = _get_answer(req, workers[worker_index], e, procedure)
            e.req_answers_per_worker[worker_index][idx] = answer
            temp_answers.append(answer)
            worker_cnt = worker_cnt + 1
            e.label_work_count.append(worker_cnt)
            if labels_cnt == 2 and (temp_answers.count(True) / labels_cnt) > 0.5:
                final_answer = 1
                e.request_answers.append(final_answer)
                break
            elif labels_cnt == 2 and (temp_answers.count(False) / labels_cnt) > 0.5:
                final_answer = 0
                e.request_answers.append(final_answer)
                break
            elif labels_cnt == 3:
                if (temp_answers.count(True) / labels_cnt) > 0.5:
                    final_answer = 1
                    e.request_answers.append(final_answer)
                else:
                    final_answer = 0
                    e.request_answers.append(final_answer)
                break
        total_sum += final_answer
        proportion = total_sum / (idx + 1)
        li = [proportion]*labels_cnt
        e.accuracy_per_label.extend(li)
        e.labels_per_request.append(labels_cnt)

        # update results dataframe
        dfc.update_y_proportion_x_samples_row(p, mean, method_id, method_name, method_name_short, labels_cnt, final_answer,
                                              proportion, req_id)

    e.proportion_correct_choice = total_sum / len(requests)
    e.total_workers = len(total_worker_set)

    return e


def _get_answer(req_difficulty, worker_capability, e, method):
    answer = _get_answer_via_two_choice(req_difficulty, worker_capability, e)
    return answer


def _get_random_answer(e):
    e.random_answers_cnt = e.random_answers_cnt + 1
    return np.random.binomial(1, 0.5)


def _get_answer_via_two_choice(req_difficulty, worker_capability, e):
    p_a = req_difficulty * worker_capability
    # transform the probability to perform bernoulli trial with p in [0, 1]
    p_a = (p_a + 1) / 2

    return np.random.binomial(1, p_a)