import torch
import copy
import pandas as pd
import pdb


class MetricsLogger:
    def __init__(self, num_epochs: int, log_frequency: int):
        self.num_epochs = num_epochs
        self.log_frequency = log_frequency
        self.num_early_training_epochs = 0
        self.early_training_log_frequency = 10

        if self.num_early_training_epochs == 0:
            self.num_loged_epochs = (num_epochs) // log_frequency
        else:
            self.num_loged_epochs = (num_epochs) // log_frequency 

        # Metrics to log
        self.metric_fns = {
            "loss": self.compute_loss,
            "accuracy": self.compute_accuracy,
            "weights_l2": self.compute_weights_l2,
            "grad_l2_norm": self.compute_grad_l2_norm,
            "grad_cosine_similarity": self.compute_grad_cosine_similarity,
            "abs_grad_l1_norm": self.compute_abs_grad_l1_norm,
            "zero_grad_percentage": self.compute_zero_grad_percentage,
            "zero_terms": self.compute_zero_terms,
            "softmax_collapse": self.compute_softmax_collapse,
        }

        self.metrics_df = pd.DataFrame(columns=["epoch", "input_type", "metric_name", "layer", "value"])

        self._train_output = None
        self._train_targets = None
        self._test_output = None
        self._test_targets = None

    def _get_epoch_position(self, epoch: int) -> int:
        epoch_position = epoch // self.log_frequency
        if epoch > self.num_early_training_epochs:
            epoch_position += self.num_early_training_epochs // self.early_training_log_frequency
        return epoch_position

    def _run_full_batch_forward(self, model, data, targets, args):
        model.eval()
        device = next(model.parameters()).device
        if isinstance(data, torch.Tensor):
            data = data.to(device)
            targets = targets.to(device)
        else: # Handle dataloaders
            data = data.dataset.dataset.data[data.dataset.indices].to(device)
            targets = data.dataset.dataset.targets[data.dataset.indices].to(device)
            
        with torch.no_grad():
            output = model(data)
            if args.use_transformer:
                output = output[:,-1]
        return output, targets

    def log_metrics(self, model, epoch, save_model_checkpoints, saved_models,
                    all_data, all_targets, all_test_data, all_test_targets,
                    args, loss_function, full_batch=True, train_loader=None, test_loader=None):
        """
        Compute and log all metrics.
        """

        epoch_position = self._get_epoch_position(epoch)

        if epoch in save_model_checkpoints:
            saved_models[epoch] = copy.deepcopy(model.state_dict())

        if full_batch:
            self._train_output, self._train_targets = self._run_full_batch_forward(model, all_data, all_targets, args)
            self._test_output, self._test_targets = self._run_full_batch_forward(model, all_test_data, all_test_targets, args)
        else:
            # This is slow, but necessary for mini-batch logging.
            train_outputs, train_targets = [], []
            for data, target, *_ in train_loader:
                output, target = self._run_full_batch_forward(model, data, target, args)
                train_outputs.append(output)
                train_targets.append(target)
            self._train_output = torch.cat(train_outputs)
            self._train_targets = torch.cat(train_targets)

            test_outputs, test_targets = [], []
            for data, target, *_ in test_loader:
                output, target = self._run_full_batch_forward(model, data, target, args)
                test_outputs.append(output)
                test_targets.append(target)
            self._test_output = torch.cat(test_outputs)
            self._test_targets = torch.cat(test_targets)


        for metric_name, metric_fn in self.metric_fns.items():
            rows = metric_fn(model=model,
                             epoch=epoch,
                             epoch_position=epoch_position,
                             args=args,
                             loss_function=loss_function)
            if rows is not None and len(rows) > 0:
                self.metrics_df = pd.concat([self.metrics_df, pd.DataFrame(rows)], ignore_index=True)


        self._train_output = None
        self._train_targets = None
        self._test_output = None
        self._test_targets = None


    def compute_loss(self, model, epoch, epoch_position, args, loss_function):
        ce_dtype = getattr(torch, args.cross_entropy_dtype)
        train_loss_val = loss_function(self._train_output, self._train_targets, dtype=ce_dtype).item()
        test_loss_val = loss_function(self._test_output, self._test_targets, dtype=ce_dtype).item()

        return [
            {
                "epoch": epoch_position,
                "input_type": "train",
                "metric_name": "loss",
                "layer": None,
                "value": train_loss_val
            },
            {
                "epoch": epoch_position,
                "input_type": "test",
                "metric_name": "loss",
                "layer": None,
                "value": test_loss_val
            }
        ]

    def compute_accuracy(self, model, epoch, epoch_position, args, loss_function):
        train_preds = self._train_output.argmax(dim=1)
        test_preds = self._test_output.argmax(dim=1)

        accuracy = lambda correct: correct.sum().item() / correct.size(0)

        train_acc = accuracy(train_preds == self._train_targets)
        test_acc = accuracy(test_preds == self._test_targets)

        return [
            {
                "epoch": epoch_position,
                "input_type": "train",
                "metric_name": "accuracy",
                "layer": None,
                "value": train_acc
            },
            {
                "epoch": epoch_position,
                "input_type": "test",
                "metric_name": "accuracy",
                "layer": None,
                "value": test_acc
            }
        ]

    def compute_weights_l2(self, model, epoch, epoch_position, args, loss_function):
        results = []
        for name, param in model.named_parameters():
            val = param.square().sum().sqrt().item()
            results.append({
                "epoch": epoch_position,
                "input_type": "general",
                "metric_name": "weights_l2",
                "layer": name,
                "value": val
            })
        return results

    def compute_zero_terms(self, model, epoch, epoch_position, args, loss_function):
        ce_dtype = getattr(torch, args.cross_entropy_dtype)
        full_loss = loss_function(self._train_output, self._train_targets, reduction="none", dtype=ce_dtype)
        zero_val = ((full_loss == 0).sum().item() / (full_loss.shape[0]))
        return [{
            "epoch": epoch_position,
            "input_type": "train",
            "metric_name": "zero_terms",
            "layer": None,
            "value": zero_val
        }]

    def compute_softmax_collapse(self, model, epoch, epoch_position, args, loss_function):
        ce_dtype = getattr(torch, args.cross_entropy_dtype)
        output = self._train_output.to(ce_dtype)
        output_off = output - output.amax(dim=1, keepdim=True)
        exp_output = torch.exp(output_off)
        sum_exp = torch.sum(exp_output, dim=-1, keepdim=True)
        log_softmax = output_off.amax(dim=1, keepdim=True)- torch.log(sum_exp)
        softmax_collapse = (sum_exp==1).float().mean().item()

        return [{
            "epoch": epoch_position,
            "input_type": "train",
            "metric_name": "softmax_collapse",
            "layer": None,
            "value": softmax_collapse
        }]

    def compute_grad_l2_norm(self, model, epoch, epoch_position, args, loss_function):
        results = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                val = param.grad.square().sum().sqrt().item()
                results.append({
                    "epoch": epoch_position,
                    "input_type": "general",
                    "metric_name": "grad_l2_norm",
                    "layer": name,
                    "value": val
                })
        return results

    def compute_grad_cosine_similarity(self, model, epoch, epoch_position, args, loss_function):
        results = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.square().sum().sqrt()
                weight_norm = param.square().sum().sqrt()
                if grad_norm > 0 and weight_norm > 0:
                    val = (param * param.grad).sum() / (grad_norm * weight_norm)
                    val = val.item()
                else:
                    val = 0
                results.append({
                    "epoch": epoch_position,
                    "input_type": "general",
                    "metric_name": "grad_cosine_similarity",
                    "layer": name,
                    "value": val
                })
        return results

    def compute_abs_grad_l1_norm(self, model, epoch, epoch_position, args, loss_function):
        results = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                val = param.grad.abs().sum().item()
                results.append({
                    "epoch": epoch_position,
                    "input_type": "general",
                    "metric_name": "abs_grad_l1_norm",
                    "layer": name,
                    "value": val
                })
        return results
    
    def compute_zero_grad_percentage(self, model, epoch, epoch_position, args, loss_function):
        results = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                val = (param.grad == 0).float().mean().item()
                results.append({
                    "epoch": epoch_position,
                    "input_type": "general",
                    "metric_name": "zero_grad_percentage",
                    "layer": name,
                    "value": val
                })
        return results