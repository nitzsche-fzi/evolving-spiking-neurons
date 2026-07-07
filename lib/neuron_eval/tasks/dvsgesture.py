import torch
import tonic
import torch.optim as optim
from torch.utils.data import DataLoader
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import time

from lib.neuron_eval.combined_networks.dense_net import DenseNet
from lib.neuron_eval.combined_networks.loss import PopulationCELoss, accumulate_spike_loss
from lib.neuron_eval.combined_networks.clip_grad_norm import clip_population_grad_norm
from .sampler import MultiEpochBatchSampler

from esn.augmentations import FrameTransform

class DVSGesture:
    """Training/evaluation task for the DVS128 Gesture dataset.

    Wraps the tonic DVSGesture dataset (event-camera recordings of hand
    gestures, rendered into frame tensors) and provides the dataloaders,
    training loop and evaluation pass used to score a population of neurons.
    :meth:`evaluate` is the worker entry point: it trains a :class:`DenseNet`
    over the whole population and streams progress/results back over a
    multiprocessing queue.
    """

    def __init__(
        self,
        batch_size=32,
        lr=0.0005346750832445695,
        layer_sizes=[2048, 256, 128, 11],
        n_epochs=50,
        surrogate_method="atan_derivative",
        surrogate_alpha=2.7066,
        spike_rate_reg=0.00001,
        dataset_path="/shared/datasets",
        desired_sensor_size=[32, 32, 2],
        dt=8000,
        n_steps=100,
        inp_mean=0.2166, 
        inp_var=1.8677,  
        random_start_offset=False,
        noise=1.1523e-05,
        random_time_scale=[0.8083, 1.1917],
        random_image_scale=[0.978, 1.022],
        random_image_offset=[-0.0807, 0.0807],
        grad_clip_norm=1.0,
        weight_decay=0.0,
        dropout_p=0.0,
    ):
        self.batch_size = batch_size
        self.lr = lr
        self.layer_sizes = layer_sizes
        self.surrogate_method = surrogate_method
        self.surrogate_alpha = surrogate_alpha 
        self.dataset_path = dataset_path
        self.desired_sensor_size = desired_sensor_size
        self.dt = dt
        self.n_unroll_steps = n_steps
        self.inp_mean = inp_mean
        self.inp_var = inp_var
        self.random_start_offset = random_start_offset
        self.noise = noise
        self.random_time_scale = random_time_scale
        self.random_image_scale = random_image_scale
        self.random_image_offset = random_image_offset
        self.spike_rate_reg = spike_rate_reg
        self.n_epochs = n_epochs
        self.grad_clip_norm = grad_clip_norm
        self.weight_decay = weight_decay

        self.training_dataset = tonic.datasets.DVSGesture(
            save_to=self.dataset_path, train=True, 
            transform=FrameTransform(
                original_sensor_size=tonic.datasets.DVSGesture.sensor_size,
                desired_sensor_size=self.desired_sensor_size,
                dt=self.dt,
                n_steps=self.n_unroll_steps,
                random_start_offset=self.random_start_offset,
                noise=self.noise,
                random_time_scale=self.random_time_scale,
                random_image_scale=self.random_image_scale,
                random_image_offset=self.random_image_offset
            )
        )

        self.testing_dataset = tonic.datasets.DVSGesture(save_to=self.dataset_path, train=False,
            transform=FrameTransform(
                original_sensor_size=tonic.datasets.DVSGesture.sensor_size,
                desired_sensor_size=self.desired_sensor_size,
                dt=self.dt,
                n_steps=self.n_unroll_steps,
                random_start_offset=False,
                noise=0.0,
                random_time_scale=[1.0, 1.0],
                random_image_scale=[1.0, 1.0],
                random_image_offset=[0.0, 0.0]
            )
        )

        self.n_training_data_points = len(self.training_dataset)

    def set_name(self, name):
        """Sets the name of the task."""
        self.name = name

    def get_train_dataloader(self):
        """Returns the dataloader for training or evaluation."""
        sampler = MultiEpochBatchSampler(self.training_dataset, self.batch_size, self.n_epochs)
        dataloader = DataLoader(
            self.training_dataset,
            batch_sampler=sampler,
            collate_fn=tonic.collation.PadTensors(batch_first=False),
            pin_memory=False,
            num_workers=4
        )
        return dataloader
    
    def get_test_dataloader(self):
        """Returns the dataloader for testing."""
        return DataLoader(
            self.testing_dataset,
            batch_size=self.batch_size,
            collate_fn=tonic.collation.PadTensors(batch_first=False),
            pin_memory=False,
            num_workers=4,
            drop_last=False,
            shuffle=False
        )
    
    def train_step(self, population_size, net, data, criterion):
        """Runs one training batch: rollout, combined loss, and backward pass.

        Returns the classification and spike losses, per-individual accuracies,
        the largest absolute gradient, and the per-individual spike count.
        """
        in_data, out_data = data
        in_data = in_data.view(in_data.shape[0], in_data.shape[1], 1, -1).to(net.device)
        in_data = in_data.expand(-1,-1, population_size, -1)
        # in_data dims: [n_steps, batch_size, population_size, features]
        out_data = out_data.to(net.device)
        state = net.init_state()
        for i in range(self.n_unroll_steps):
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
        return classification_loss, total_loss, spike_loss, accuracies, max_grad, spike_count

    def eval_step(self,net,data,population_size):
        """
        Evaluation step for the network.
        Returns accuracies and spike counts for each network in the population.
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
        in_data = in_data.expand(-1,-1, population_size, -1)
        #in_data dims: [n_steps, batch_size, population_size, features]
        assert in_data.shape[1] == self.batch_size, \
            f"Expected batch size {self.batch_size}, but got {in_data.shape[1]}"
        assert in_data.shape[2] == population_size, \
            f"Expected population size {population_size}, but got {in_data.shape[2]}"
        assert in_data.shape[0] == self.n_unroll_steps, \
            f"Expected n_steps {self.n_unroll_steps}, but got {in_data.shape[0]}"
        assert out_data.shape[0] == self.batch_size, \
            f"Expected batch size {self.batch_size}, but got {out_data.shape[0]}"
        out_data = out_data.to(net.device)
        state = net.init_state()
        for i in range(self.n_unroll_steps):
            outputs, state = net(in_data[i], state)
        #outputs has shape [batch_size, population_size, n_classes]
        #we need to argmax over the n_classes
        outputs = outputs.argmax(dim=-1)  # [batch_size, population_size]
        #out_data has shape [batch_size]
        #expand it to [batch_size, population_size] for comparison
        out_data = out_data.unsqueeze(1).expand(-1, population_size)
        accuracies = (outputs == out_data).float().cpu().tolist() #shape: [batch_size, population_size]
        accuracies = accuracies[to_pad:] if to_pad > 0 else accuracies  # remove padded entries if any
        
        # Extract spike count from state
        spike_count = state[-1].cpu().tolist()  # shape: [population_size]
        
        return accuracies, spike_count

    def evaluate(self, seed, neuron_params, neuron_type, device, message_queue, do_compile=True):
        """Trains and evaluates a whole population on DVSGesture (worker entry point).

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

        # Note: train_step is not itself torch.compiled because the CUDA-graph
        # backend does not support the backward pass; only net and criterion are
        # compiled below.
        optimizer = optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = PopulationCELoss(self.batch_size, population_size, 11) #11 classes for DVSGesture
        if do_compile:
            net.compile(mode="max-autotune", fullgraph=True, dynamic=False)
            criterion.compile(mode="max-autotune", fullgraph=True, dynamic=False)

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
            torch.compiler.cudagraph_mark_step_begin()
            summed_loss, classification_loss, spike_loss, \
                accuracies, max_grad, spike_count = train_step(
                    population_size, net, data, criterion
                )
            # Clip per-value rather than per-population norm (see
            # clip_population_grad_norm for the alternative).
            torch.nn.utils.clip_grad_value_(net.parameters(), self.grad_clip_norm)
            optimizer.step()
            spike_rate = spike_count / (n_spiking_neurons * self.n_unroll_steps) # shape [population_size]
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
            torch.compiler.cudagraph_mark_step_begin()
            step_accuracies, step_spike_counts = self.eval_step(net,data,population_size)
            eval_accuracies += step_accuracies
            eval_spike_counts.append(step_spike_counts)
            message_queue.put({
                "type": "evaluation progress",
                "progress": (step + 1) / n_eval_steps,
                "device": device, #identifier for who is sending the message
            })

        #eval_accuracies is of shape (len(eval_dataloader),population_size), a matrix of 1s and 0s
        #reduce it to population size
        eval_accuracies = torch.tensor(eval_accuracies, device=device, dtype=torch.float32)
        eval_accuracies = eval_accuracies.mean(dim=0)  # average over batches
        
        # Sum spike counts across all evaluation steps
        eval_spike_counts = torch.tensor(eval_spike_counts, device=device, dtype=torch.float32)
        total_spike_counts = eval_spike_counts.sum(dim=0)  # sum over steps, shape: [population_size]
        total_timesteps = n_eval_steps * self.n_unroll_steps
        
        message_queue.put({
            "type": "evaluation finished",
            "device": device,
            "results": eval_accuracies.cpu().tolist(),
            "total_spike_counts": total_spike_counts.cpu().tolist(),
            "total_timesteps": total_timesteps,
        })
