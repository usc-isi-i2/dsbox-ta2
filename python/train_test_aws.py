import os
import subprocess
import sys
from multiprocessing.pool import ThreadPool
from pathlib import Path

timeout = 60
cpus = 12
num_threads = 10

mega_better_than_eval_set = {
    "LL0_1548_autouniv_au4_2500",
    "LL0_1493_one_hundred_plants_texture",
    "LL0_1481_kr_vs_k",
    "LL0_180_covertype",
    "LL0_458_analcatdata_authorship",
    "LL0_1569_poker_hand",
    "LL0_1475_first_order_theorem_proving",
    "LL0_40648_GAMETES_Epistasis_3_Way_20atts_0.2H_EDM_1_1",
    "LL0_1044_eye_movements",
    "LL0_1036_sylva_agnostic",
    "LL0_1065_kc3",
    "LL0_840_autoHorse",
    "LL0_1479_hill_valley",
    "LL0_6332_cylinder_bands",
    "LL0_1487_ozone_level_8hr",
    "LL0_941_lowbwt",
    "LL0_1467_climate_model_simulation_crashes",
    "LL0_1220_click_prediction_small",
    "LL0_28_optdigits",
    "LL0_1520_robot_failures_lp5",
    "LL0_12_mfeat_factors",
    "LL0_16_mfeat_karhunen",
    "LL0_35_dermatology",
    "LL0_953_splice",
    "LL0_454_analcatdata_halloffame",
    "LL0_188_eucalyptus",
    "LL0_1054_mc2",
    "LL0_179_adult",
    "LL0_307_vowel",
    "LL0_14_mfeat_fourier",
    "LL0_1040_sylva_prior",
    "LL0_31_credit_g",
    "LL0_1053_jm1",
    "LL0_1496_ringnorm",
    "LL0_40705_tokyo1",
    "LL0_32_pendigits",
    "LL0_1531_volcanoes_b1",
    "LL0_60_waveform_5000",
    "LL0_1497_wall_robot_navigation",
    "LL0_375_japanesevowels",
    "LL0_40499_texture"
}


def call_ta2search(command):
    print(command)

    p = subprocess.Popen(command, shell=True)

    try:
        p.communicate(timeout=timeout * 60)
    except subprocess.TimeoutExpired:
        p.kill()
        print(command, "took too long and was terminated" + "\n\n")


tp = ThreadPool(num_threads)

home = str(Path.home())
config_dir = sys.argv[1]

for conf in os.listdir(config_dir):
    if conf in mega_better_than_eval_set:
        command = "python3 ta2-search " + os.path.join(config_dir, conf, "search_config.json ") + " --timeout " + str(
            timeout) + " --cpus " + str(cpus)

        tp.apply_async(call_ta2search, (command,))

tp.close()
tp.join()
