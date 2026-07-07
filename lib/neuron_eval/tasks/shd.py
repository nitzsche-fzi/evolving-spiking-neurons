import torch
import tonic
from torch.utils.data import DataLoader
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import time

from lib.neuron_eval.combined_networks.dense_net import DenseNet
from lib.neuron_eval.combined_networks.loss import PopulationCELoss, accumulate_spike_loss
from lib.neuron_eval.combined_networks.clip_grad_norm import clip_population_grad_norm
from .sampler import MultiEpochBatchSampler

from esn.augmentations import AudioTransform, AudioPad

class SHD:
    """Training/evaluation task for the Spiking Heidelberg Digits dataset.

    Wraps the tonic SHD dataset (spoken-digit audio encoded as spike trains) and
    provides everything the GA needs to score a population of neurons on it:
    dataloaders, a training loop with a combined classification + spike-rate
    loss, and an evaluation pass. :meth:`evaluate` is the entry point run in a
    worker process; it trains a :class:`DenseNet` over the whole population and
    streams progress/results back over a multiprocessing queue.
    """

    def __init__(
        self,
        batch_size,
        lr,
        layer_sizes,
        n_epochs,
        surrogate_method,
        surrogate_alpha,
        spike_rate_reg,
        dataset_path,
        desired_sensor_size,
        dt,
        inp_mean, 
        inp_var,  
        noise,
        random_time_scale,
        grad_clip_norm,
        weight_decay=0.0,
    ):
        self.batch_size = batch_size
        self.lr = lr
        self.layer_sizes = layer_sizes
        self.surrogate_method = surrogate_method
        self.surrogate_alpha = surrogate_alpha
        self.dataset_path = dataset_path
        self.desired_sensor_size = desired_sensor_size
        self.dt = dt
        self.inp_mean = inp_mean
        self.inp_var = inp_var
        self.noise = noise
        self.random_time_scale = random_time_scale
        self.spike_rate_reg = spike_rate_reg
        self.n_epochs = n_epochs
        self.grad_clip_norm = grad_clip_norm
        self.weight_decay = weight_decay

        self.training_dataset = tonic.datasets.SHD(
            save_to=self.dataset_path,
            train=True,
            transform=AudioTransform(
                original_sensor_size=[700, 1, 1],
                desired_sensor_size=self.desired_sensor_size,
                dt=self.dt,
                random_time_scale=self.random_time_scale,
                squeeze_thresh=1
            )
        )

        self.testing_dataset = tonic.datasets.SHD(
            save_to=self.dataset_path,
            train=False,
            transform=AudioTransform(
                original_sensor_size=[700, 1, 1],
                desired_sensor_size=self.desired_sensor_size,
                dt=self.dt,
                random_time_scale=[1.0, 1.0],
                squeeze_thresh=1
            )
        )

        self.n_training_data_points = len(self.training_dataset)

    def set_name(self, name):
        self.name = name

    def get_train_dataloader(self, num_workers=4):
        sampler = MultiEpochBatchSampler(self.training_dataset, self.batch_size, self.n_epochs)
        return DataLoader(
            self.training_dataset,
            batch_sampler=sampler,
            collate_fn=AudioPad(batch_first=False, noise=self.noise),
            pin_memory=False,
            num_workers=num_workers,
        )

    def get_test_dataloader(self, num_workers=4):
        return DataLoader(
            self.testing_dataset,
            batch_size=self.batch_size,
            collate_fn=AudioPad(batch_first=False, noise=0.0),
            pin_memory=False,
            num_workers=num_workers,
            drop_last=False,
            shuffle=False
        )

    def train_step(self, population_size, net, data, criterion):
        """Runs one training batch: rollout, combined loss, and backward pass.

        Returns the metrics needed for progress reporting, including the
        classification and spike losses, per-individual accuracies, the largest
        absolute gradient, and the per-individual spike count.
        """
        in_data, out_data = data
        in_data = in_data.view(in_data.shape[0], in_data.shape[1], 1, -1).to(net.device)
        in_data = in_data.expand(-1,-1, population_size, -1)
        # in_data dims: [n_steps, batch_size, population_size, features]
        out_data = out_data.to(net.device)
        state = net.init_state()
        n_time_steps = in_data.shape[0]
        for i in range(n_time_steps):
            outputs, state = net(in_data[i], state)
        classification_loss, accuracies = criterion(outputs, out_data)
        spike_count = state[-1] # shape [population_size]
        spike_loss = accumulate_spike_loss(spike_count)
        total_loss = classification_loss + self.spike_rate_reg * spike_loss
        total_loss.backward()
        # Track the largest absolute gradient for monitoring training stability.
        max_grad = 0.0
        for param in net.parameters():
            if param.grad is not None:
                max_grad = max(max_grad, torch.max(torch.abs(param.grad)).item())

        return n_time_steps, classification_loss, total_loss, spike_loss, accuracies, max_grad, spike_count

    def eval_step(self, net, data, population_size):
        """Runs one evaluation batch and returns per-individual accuracies and spike counts.

        The final (possibly short) batch is zero-padded up to ``batch_size`` and
        the padded entries are dropped from the returned accuracies.
        """
        in_data, out_data = data
        in_data = in_data.view(in_data.shape[0], in_data.shape[1], 1, -1).to(net.device)

        to_pad = 0
        if in_data.shape[1] != self.batch_size:
            to_pad = self.batch_size - in_data.shape[1]
            pad_shape = (in_data.shape[0], to_pad, in_data.shape[2], in_data.shape[3])
            padding = torch.zeros(pad_shape, device=in_data.device, dtype=in_data.dtype)
            in_data = torch.cat([in_data, padding], dim=1)
            pad_targets = torch.full((to_pad,), -100, dtype=out_data.dtype, device=out_data.device)
            out_data = torch.cat([out_data, pad_targets], dim=0)

        in_data = in_data.expand(-1, -1, population_size, -1)
        out_data = out_data.to(net.device)

        state = net.init_state()
        n_time_steps = in_data.shape[0]
        for i in range(n_time_steps):
            outputs, state = net(in_data[i], state)

        outputs = outputs.argmax(dim=-1)
        out_data = out_data.unsqueeze(1).expand(-1, population_size)
        accuracies = (outputs == out_data).float().cpu().tolist()
        accuracies = accuracies[to_pad:] if to_pad > 0 else accuracies

        # Extract spike count from state
        spike_count = state[-1].cpu().tolist()  # shape: [population_size]

        return accuracies, spike_count

    def evaluate(self, seed, neuron_params, neuron_type, device, message_queue, do_compile=True):
        """Trains and evaluates a whole population on SHD (worker entry point).

        Builds a :class:`DenseNet` for the given neuron parameters, trains it for
        the configured number of epochs, evaluates it on the test set, and sends
        ``training started/progress/finished`` and ``evaluation
        progress/finished`` messages (including final accuracies and spike
        counts) over ``message_queue``.
        """
        population_size = len(neuron_params)
        for param in neuron_params:
            param.surrogate_method = self.surrogate_method
            param.surrogate_alpha = self.surrogate_alpha

        net = DenseNet(
            seed=seed,
            neuron_params=neuron_params,
            neuron_type=neuron_type,
            network_shape=self.layer_sizes,
            batch_size=self.batch_size,
            device=device,
            in_mean=self.inp_mean,
            in_var=self.inp_var,
        )

        optimizer = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = PopulationCELoss(self.batch_size, population_size, 20) #20 classes in SHD
        if do_compile:
            net.compile(mode="max-autotune-no-cudagraphs", fullgraph=True, dynamic=False)
            criterion.compile(mode="max-autotune-no-cudagraphs", fullgraph=True, dynamic=False)

            
        train_step = self.train_step

        dataloader = self.get_train_dataloader()
        net.train()
        train_step_time = 0.0
        n_spiking_neurons = net.get_spiking_neuron_count()
        n_training_steps = len(dataloader)
        message_time = 0.0

        message_queue.put({
            "type": "training started",
            "device": device,  # identifier for who is sending the message
            "population_size": population_size,
            "n_training_steps": n_training_steps,
            "n_spiking_neurons": n_spiking_neurons,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
        })

        for step, data in enumerate(dataloader):
            train_step_start = time.time()
            optimizer.zero_grad()
            n_time_steps, summed_loss, classification_loss, spike_loss, \
                accuracies, max_grad, spike_count = train_step(
                    population_size, net, data, criterion
                )
            # Clip per-value rather than per-population norm (see
            # clip_population_grad_norm for the alternative).
            torch.nn.utils.clip_grad_value_(net.parameters(), self.grad_clip_norm)
            optimizer.step()
            
            spike_rate = (spike_count / (n_spiking_neurons * n_time_steps))
            train_step_time = time.time() - train_step_start

            message_time_start = time.time()
            message_queue.put({
                "type" : "training progress",
                "step": step + 1,  # step is zero-indexed, so we add 1 for progress
                "device": device, #identifier for who is sending the message
                "summed_loss": summed_loss.item(),
                "classification_loss": classification_loss.item(),
                "spike_loss": spike_loss.item(),
                "spike_rate" : spike_rate.cpu().tolist(),  # convert to list for JSON serialization
                "accuracies": accuracies.cpu().tolist(), 
                "train_step_time": train_step_time, 
                "max_grad": max_grad, 
                "message_time": message_time
            })
            message_time_end = time.time()
            message_time = message_time_end - message_time_start

        message_queue.put({
            "type": "training finished",
            "device": device,
        })

        eval_dataloader = self.get_test_dataloader()
        eval_accuracies = []
        eval_spike_counts = []
        n_eval_steps = len(eval_dataloader)

        for step, data in enumerate(eval_dataloader):
            step_accuracies, step_spike_counts = self.eval_step(net, data, population_size)
            eval_accuracies += step_accuracies
            eval_spike_counts.append(step_spike_counts)
            message_queue.put({
                "type": "evaluation progress",
                "progress": (step + 1) / n_eval_steps,
                "device": device
            })

        eval_accuracies = torch.tensor(eval_accuracies, device=device, dtype=torch.float32)
        eval_accuracies = eval_accuracies.mean(dim=0)
        
        # Sum spike counts across all evaluation steps
        eval_spike_counts = torch.tensor(eval_spike_counts, device=device, dtype=torch.float32)
        total_spike_counts = eval_spike_counts.sum(dim=0)  # sum over steps, shape: [population_size]
        # For SHD, n_unroll_steps varies per sample, so we need to calculate it from the data
        total_timesteps = sum(len(data[0]) for data in eval_dataloader)  # sum of all timesteps across all samples

        message_queue.put({
            "type": "evaluation finished",
            "device": device,
            "results": eval_accuracies.cpu().tolist(),
            "total_spike_counts": total_spike_counts.cpu().tolist(),
            "total_timesteps": total_timesteps,
        })
