import csv
import logging
import os
import sys

import torch


def create_folders_if_necessary(path):
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)


def get_storage_dir():
    if "RL_STORAGE" not in os.environ:
        parent_dir = os.getcwd()
        if "scripts" in parent_dir:
            parent_dir = os.path.dirname(parent_dir)
        os.environ["RL_STORAGE"] = os.path.join(parent_dir, "runs")
    return os.environ["RL_STORAGE"]


def get_log_dir(run_name):
    return os.path.join(get_storage_dir(), run_name)


def get_status_path(model_dir):
    return os.path.join(model_dir, "status.pt")


def get_status(model_dir, device):
    path = get_status_path(model_dir)
    return torch.load(path, map_location=device)


def save_status(status, model_dir):
    path = get_status_path(model_dir)
    create_folders_if_necessary(path)
    torch.save(status, path)
    return path


def get_txt_logger(model_dir):
    path = os.path.join(model_dir, "log.txt")
    create_folders_if_necessary(path)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.FileHandler(filename=path), logging.StreamHandler(sys.stdout)],
    )

    return logging.getLogger()


def get_csv_logger(model_dir):
    csv_path = os.path.join(model_dir, "log.csv")
    create_folders_if_necessary(csv_path)
    csv_file = open(csv_path, "a")
    return csv_file, csv.writer(csv_file)
