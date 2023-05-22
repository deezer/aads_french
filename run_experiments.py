"""
*** Combining all the steps in one .py file to run experiments from configuration files ***
"""

import argparse
import logging
import time

from preprocessing.tokenize_corpus import *

from experiments_configs.configs_helpers import *

from regex_implementations.run_regexModel import *
from flair_implementations.rdwdgb_quote import *
from transformer_implementations.run_transformerModel import *

from evaluation_schemes.evaluation import *


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%% MAIN %%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if __name__ == "__main__":

    # python run_experiments.py --configs_folder experiments_configs/best_configs/main

    # 0.1 Set logging level to "info"
    logging.getLogger().setLevel(logging.INFO)

    # 0.2 Parse given arguments --> get configs
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--configs_folder",
        help="Folder containing config(s) files.",
        default=None,
        required=True,
    )

    parser.add_argument(
        "--configs_list",
        nargs="+",
        help="List of config files names (within <configs_folder>).",
        required=False,
        default=None,
    )

    args = parser.parse_args()

    # 0.3 Check that at least a config has been given
    if (not args.configs_list) and not (args.configs_folder):
        raise ValueError(
            """
        No configuration files or folder were given to the program.
        At least one of '--configs_list' or '--configs_folder' must be given.
        Run " $ python run_experiments.py --help " for help.
        """
        )

    if not args.configs_list:
        args.configs_list = []
        logging.info(
            "Parsing all json config files from {path_foleder}.".format(
                path_foleder=args.configs_folder
            )
        )
    else:
        logging.info(
            "Parsing {files} from {path_foleder}.".format(
                path_foleder=args.configs_folder, files=", ".join(args.configs_list)
            )
        )

    # 1. Parsing the configuration files
    experiments_arguments = read_all_configs(
        path_config_folder=args.configs_folder, configs_list=args.configs_list
    )

    start_program = time.time()
    logging.info("🗣 RUNNING {} (W)DSR EXPRIMENTS".format(len(experiments_arguments)))

    # 2. Run the experiment for each of the config:
    for i_exp, args in enumerate(experiments_arguments):
        general_args, model_args, data_args, training_args, eval_args = args
        # 2.1 Display current config to be run
        logging.info(
            loaded_config_card(
                args,
                header_name="LODADED CONFIG {i} — {mtype}".format(
                    i=i_exp + 1, mtype=general_args.model_type
                ),
            )
        )
        start_curr_exp = time.time()

        # [2.2] RUN EXPERIMENT BASED ON MODEL TYPE
        if general_args.model_type == "regex":
            run_regexModel((model_args, data_args, training_args, eval_args))

        elif general_args.model_type == "flair":
            run_flairModel((model_args, data_args, training_args, eval_args))

        elif general_args.model_type == "transformer":
            run_transformerModel((model_args, data_args, training_args, eval_args))

        curr_exp_time = time.time() - start_curr_exp
        curr_exp_time_str = time.strftime("%H:%M:%S", time.gmtime(curr_exp_time))
        logging.info(
            "\n💫 Experiment {i} done.\nRun Time : {t}.\n".format(
                i=i_exp + 1, t=curr_exp_time_str
            )
        )

    program_time = time.time() - start_program
    program_time_str = time.strftime("%H:%M:%S", time.gmtime(program_time))
    logging.info(
        "\n⏰ Total Run Time : {t}. Experiments done 🎊".format(t=program_time_str)
    )
