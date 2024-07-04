import argparse
import datetime
import sys
import time

import gymnasium as gym
import torch
from torch.utils.tensorboard import SummaryWriter

from algo.a2c import A2CAlgo
from algo.models import (
    ActorModel,
    ObsCriticModel,
    ObsStateCriticModel,
    StateCriticModel,
)
from utils.common_func import *
from utils.env_utils import make_env
from utils.storage_utils import *


def parse_args():
    # fmt: off

    # Parse arguments
    parser = argparse.ArgumentParser()
    # Training params
    parser.add_argument("--algo", type=str, default="acc", choices=['acc', 'aec', 'aoc', 'uaac'],
                        help="algorithm to use: acc | aec | aoc | uaac")
    parser.add_argument("--env-name", type=str, default="MiniGrid-Empty-16x16-v0",
                        choices=["MiniGrid-Empty-16x16-v0", "MiniGrid-LavaGapS7-v0", "MiniGrid-MultiRoom-N2-S4-v0",
                                 "MiniGrid-SimpleCrossingS9N1-v0", "MiniGrid-SimpleCrossingS9N2-v0", ],
                        help="name of the environment to train on")
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed (default: 0)")
    parser.add_argument("--num-envs", type=int, default=16,
                        help="number of parallel envs (default: 16)")
    parser.add_argument("--num-frames", type=int, default=8,
                        help="number of frames to run for each process per update (default: 8)")
    parser.add_argument("--total-frames", type=int, default=10 ** 7,
                        help="number of frames of training (default: 1e7)")
    # Log params
    parser.add_argument("--run_name", type=str, default=None,
                        help="name of the experiment (default: {ENV}_{ALGO}_{TIME})")
    parser.add_argument("--log-interval", type=int, default=10,
                        help="number of updates between two logs (default: 1)")
    parser.add_argument("--save-interval", type=int, default=1000,
                        help="number of updates between two saves (default: 10, 0 means no saving)")
    parser.add_argument("--capture-video", action="store_true", default=False,
                        help="whether to capture videos of the agent performances")
    parser.add_argument("--track", action="store_true", default=False,
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default=None,
                        help="the wandb's project name")
    parser.add_argument("--wandb-group-name", type=str, default=None,
                        help="the wandb's group name")
    # Algo params
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="discount factor (default: 0.99)")
    parser.add_argument("--gae-lambda", type=float, default=1.0,
                        help="lambda coefficient in GAE formula, 1 means no gae (default: 1)")
    parser.add_argument("--entropy-coef", type=float, default=0.01,
                        help="entropy term coefficient (default: 0.01)")
    parser.add_argument("--value-loss-coef", type=float, default=0.5,
                        help="value loss term coefficient (default: 0.5)")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="maximum norm of gradient (default: 0.5)")
    parser.add_argument("--clip-eps", type=float, default=0.2,
                        help="clipping epsilon for PPO (default: 0.2)")
    # Optimizer params
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate (default: 0.001)")
    parser.add_argument("--optim-eps", type=float, default=1e-8,
                        help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
    parser.add_argument("--optim-alpha", type=float, default=0.99,
                        help="RMSprop optimizer alpha (default: 0.99)")
    # Network params
    parser.add_argument("--recurrent", action="store_true", default=True,
                        help="add a LSTM to the model")

    args = parser.parse_args()

    # fmt: on
    return args


if __name__ == "__main__":
    args = parse_args()

    # Set run dir
    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    default_run_name = f"{args.env_name}_{args.algo}_seed{args.seed}_{date}"
    run_name = args.run_name or default_run_name
    args.full_log_dir = get_log_dir(run_name)

    # Load loggers and Tensorboard writer
    txt_logger = get_txt_logger(args.full_log_dir)
    csv_file, csv_logger = get_csv_logger(args.full_log_dir)
    if args.track:
        import wandb

        wandb.tensorboard.patch(root_logdir=args.full_log_dir)
        wandb.init(
            name=run_name,
            entity="your id",
            project=args.wandb_project_name,
            group=args.wandb_group_name,
            monitor_gym=True,
            save_code=True,
            config=vars(args),
        )
    tb_writer = SummaryWriter(args.full_log_dir)

    # Log command and all script arguments
    txt_logger.info("{}\n".format(" ".join(sys.argv)))
    txt_logger.info(f"Save logs at {args.full_log_dir} \n")
    txt_logger.info("{}\n".format(args))
    tb_writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Set device
    txt_logger.info(f"Device: {device}\n")

    # Set seed for all randomness sources
    set_seed(args.seed)

    # Load environments
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_name,
                args.seed + 10000 * i,
                i,
                capture_video=args.capture_video,
                log_dir=args.full_log_dir,
            )
            for i in range(args.num_envs)
        ]
    )
    txt_logger.info("Environments loaded\n")

    # Load training status
    try:
        status = get_status(args.full_log_dir, device)
    except OSError:
        status = {"num_frames": 0, "update": 0}
    txt_logger.info(f"Training status loaded: {status} \n")

    # Load model
    actor_model = ActorModel(envs.single_observation_space, envs.single_action_space, args.recurrent)
    critic_model_list = []
    reshape_reward_fn = None
    reshape_adv_fn = lambda adv: adv[..., 0]
    if args.algo == "aec":
        critic_model_list.append(ObsCriticModel(envs.single_observation_space, args.recurrent))
    elif args.algo == "aoc":
        critic_model_list.append(StateCriticModel(envs.single_observation_space, args.recurrent))
    elif args.algo == "uaac":
        critic_model_list.append(ObsStateCriticModel(envs.single_observation_space, args.recurrent))
    elif args.algo == "acc":
        critic_model_list.append(ObsCriticModel(envs.single_observation_space, args.recurrent))
        critic_model_list.append(StateCriticModel(envs.single_observation_space, args.recurrent))
        reshape_adv_fn = lambda adv: torch.max(adv, dim=-1)[0]

    if "model_state" in status:
        actor_model.load_state_dict(status["model_state"]["actor_model"])
        for i, critic_model in enumerate(critic_model_list):
            critic_model.load_state_dict(status["model_state"]["critic_model"][i])
        txt_logger.info("Model loaded\n")
    actor_model.to(device)
    txt_logger.info("{}\n".format(actor_model))
    for critic_model in critic_model_list:
        critic_model.to(device)
        txt_logger.info("{}\n".format(critic_model))

    # Load algo
    algo = A2CAlgo(
        envs,
        actor_model,
        critic_model_list,
        device,
        args.num_frames,
        args.gamma,
        args.lr,
        args.gae_lambda,
        args.entropy_coef,
        args.value_loss_coef,
        args.max_grad_norm,
        reshape_reward_fn,
        reshape_adv_fn,
        args.optim_alpha,
        args.optim_eps,
    )

    if "optimizer_state" in status:
        algo.optimizer.load_state_dict(status["optimizer_state"])
    txt_logger.info("Optimizer loaded\n")

    # Train model
    num_frames = status["num_frames"]
    update = status["update"]
    start_time = time.time()

    last_save_update = update
    txt_logger.info("Begin training ...\n")
    while num_frames < args.total_frames:
        # Update model parameters
        update_start_time = time.time()
        exps, logs1 = algo.collect_experiences()
        logs2 = algo.update_parameters(exps)
        logs = {**logs1, **logs2}
        update_end_time = time.time()

        num_frames += logs["num_frames"]
        update += 1

        # Print logs
        if update % args.log_interval == 0:
            fps = logs["num_frames"] / (update_end_time - update_start_time)
            duration = int(time.time() - start_time)
            avg_return = synthesize(logs["avg_return"])
            return_per_episode = synthesize(logs["return_per_episode"])
            reshaped_return_per_episode = synthesize(logs["reshaped_return_per_episode"])
            num_frames_per_episode = synthesize(logs["num_frames_per_episode"])

            header = ["update", "frames", "FPS", "duration"]
            data = [update, num_frames, fps, duration]
            header += ["avg_return_" + key for key in avg_return.keys()]
            data += avg_return.values()

            header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
            data += num_frames_per_episode.values()
            header += [
                "entropy",
                "value",
                "policy_loss",
                "value_loss",
                "grad_norm",
            ]
            data += [
                logs["entropy"],
                logs["value"],
                logs["policy_loss"],
                logs["value_loss"],
                logs["grad_norm"],
            ]

            log_str = (
                "Update {:6} | Frames {:7} | FPS {:4.0f} | Duration {} "
                "| Rew:μσmM {:1.2f} {:1.2f} {:1.2f} {:1.2f} | Num:μσmM {:2.1f} {:2.1f} {:2.1f} {:2.1f} "
                "| H {:1.3f} | V {:1.3f} | pL {:1.3f} | vL {:1.3f} | ∇ {:1.3f} "
            )
            txt_logger.info(log_str.format(*data))

            header += ["reshaped_return_" + key for key in reshaped_return_per_episode.keys()]
            data += reshaped_return_per_episode.values()
            header += ["return_" + key for key in return_per_episode.keys()]
            data += return_per_episode.values()

            if status["num_frames"] == 0:
                csv_logger.writerow(header)
            csv_logger.writerow(data)
            csv_file.flush()

            for field, value in zip(header, data):
                tb_writer.add_scalar(field, value, num_frames)

        # Save status
        if args.save_interval > 0 and update % args.save_interval == 0:
            last_save_update = update
            status = {
                "num_frames": num_frames,
                "update": update,
                "model_state": {
                    "actor_model": actor_model.state_dict(),
                    "critic_model": [critic_model.state_dict() for critic_model in critic_model_list],
                },
                "optimizer_state": algo.optimizer.state_dict(),
            }
            model_path = save_status(status, args.full_log_dir)
            if args.track:
                wandb.save(model_path)
            txt_logger.info(f"Status saved to {model_path}")

    if args.save_interval > 0 and update != last_save_update:
        status = {
            "num_frames": num_frames,
            "update": update,
            "model_state": {
                "actor_model": actor_model.state_dict(),
                "critic_model": [critic_model.state_dict() for critic_model in critic_model_list],
            },
            "optimizer_state": algo.optimizer.state_dict(),
        }
        model_path = save_status(status, args.full_log_dir)
        if args.track:
            wandb.save(model_path)
        txt_logger.info(f"Status saved to {model_path}")
