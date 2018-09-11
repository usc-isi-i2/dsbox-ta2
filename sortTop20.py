import os
import json
import shutil
import pandas as pd
from pprint import pprint
from multiprocessing import Pool
import logging

_logger = logging.getLogger(__name__)

def worker(i: int):
    _logger.error("I am worker "+str(i))

if __name__ == '__main__':
    FILE_FORMATTER = "[%(levelname)s] - %(asctime)s - %(name)s - %(message)s"
    FILE_LOGGING_LEVEL = logging.INFO
    LOG_FILENAME = 'dsbox.log'

    logging.basicConfig(
        level=FILE_LOGGING_LEVEL,
        format=FILE_FORMATTER,
        datefmt='%m-%d %H:%M:%S',
        filename=LOG_FILENAME,
    )
    with Pool(4) as p:
        p.map(worker, range(4))


def _process_pipeline_submission() -> None:
    output_directory = "/dsbox_efs/runs/0807-run3/seed/38_sick"
    pipelines_root: str = os.path.join(output_directory, 'pipelines')
    executables_root: str = os.path.join(output_directory, 'executables')
    supporting_root: str = os.path.join(output_directory, 'supporting_files')

    # Read all the json files in the pipelines
    piplines_name_list = os.listdir(pipelines_root)
    if len(piplines_name_list) < 20:
        return

    pipelines_df = pd.DataFrame(0.0, index=piplines_name_list, columns=["rank"])
    for name in piplines_name_list:
        with open(os.path.join(pipelines_root, name)) as f:
            try:
                rank = json.load(f)['pipeline_rank']
            except (json.decoder.JSONDecodeError, KeyError) as e:
                rank = 0
        pipelines_df.at[name, 'rank'] = rank

    # sort them based on their rank field
    pipelines_df.sort_values(by='rank', ascending=True, inplace=True)

    # make sure that "pipeline_considered" directory exists
    considered_root = os.path.join(os.path.dirname(pipelines_root), 'pipelines_considered')
    try:
        os.mkdir(considered_root)
    except FileExistsError:
        pass

    # pick the top 20 and move the rest to "pipeline_considered" directory
    for name in pipelines_df.index[20:]:
        os.rename(src=os.path.join(pipelines_root, name),
                  dst=os.path.join(considered_root, name))

    # delete the exec and supporting files related the moved pipelines
    for name in pipelines_df.index[20:]:
        pipeName = name.split('.')[0]
        try:
            os.remove(os.path.join(executables_root, pipeName + '.json'))
        except FileNotFoundError:
            traceback.print_exc()
            pass

        try:
            shutil.rmtree(os.path.join(supporting_root, pipeName))
        except FileNotFoundError:
            traceback.print_exc()
            pass
