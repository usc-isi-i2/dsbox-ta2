import os
import pprint
import multiprocessing as mps

dataset_base = "/inputs"
output_base = "/output"

configs_base = "/configs"

ta2_test_address = "/user_opt/dsbox/dsbox-ta2/python/ta2-test"

if not os.path.exists(configs_base):
    os.mkdir(configs_base)


def test_config_template(obase, dbase, name):
    return pprint.pformat({
        'dataset_schema': os.path.join(dbase, f'{name}/TEST/dataset_TEST/datasetDoc.json'),
        'original_root': os.path.join(dbase, f'{name}/{name}_dataset'),
        'original_schema': os.path.join(dbase, f'{name}/{name}_dataset/datasetDoc.json'),
        'problem_root': os.path.join(dbase, f'{name}/TEST/problem_TEST'),
        'problem_schema': os.path.join(dbase, f'{name}/TEST/problem_TEST/problemDoc.json'),
        'test_data_root': os.path.join(dbase, f'{name}/TEST/dataset_TEST'),
        'test_data_schema': os.path.join(dbase, f'{name}/TEST/dataset_TEST/datasetDoc.json'),
        'training_data_root': os.path.join(dbase, f'{name}/TEST/dataset_TEST'),
        'temp_storage_root': os.path.join(obase, f'{name}/supporting_files'),
        'user_problems_root': os.path.join(obase, f'{name}/user_problems'),
        'executables_root': os.path.join(obase, f'{name}/executables'),
        'pipeline_logs_root': os.path.join(obase, f'{name}/pipelines'),
        'cpus': '16',
        'ram': '10Gi',
    }).replace('\'', '\"')


all_datasets = os.listdir(dataset_base)

def run_test(dataset_name: str):
    if "configs" in dataset_name:
        return
    dataset_adr = os.path.join(dataset_base, dataset_name)
    print(dataset_adr)

    data_base_config_address = os.path.join(configs_base, f"{dataset_name}")
    if not os.path.exists(data_base_config_address):
        os.mkdir(data_base_config_address)

    with open(os.path.join(data_base_config_address, "test_config.json"), 'w') as f:
        f.write(test_config_template(obase=output_base, dbase=dataset_base, name=dataset_name))
        test_conf = os.path.join(data_base_config_address, "test_config.json")

    print("-" * 100)
    os.system(f"python3 {ta2_test_address} --timeout 60 {test_conf} ")
    print("#" * 100)


with mps.Pool() as p:
    p.map(func=run_test, iterable=all_datasets)

# for data_name in all_datasets:
#     if "configs" in data_name:
#         continue
#     dataset_adr = os.path.join(dataset_base, data_name)
#     print(dataset_adr)
#
#     data_base_config_address = os.path.join(configs_base, f"{data_name}")
#     if not os.path.exists(data_base_config_address):
#         os.mkdir(data_base_config_address)
#
#     with open(os.path.join(data_base_config_address, "test_config.json"), 'w') as f:
#         f.write(test_config_template(obase=output_base, dbase=dataset_base, name=data_name))
#         test_conf = os.path.join(data_base_config_address, "test_config.json")
#
#     print("-" * 100)
#     os.system(f"python3 {ta2_test_address} --timeout 60 {test_conf} ")
#     print("#" * 100)
