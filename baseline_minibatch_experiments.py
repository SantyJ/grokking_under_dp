import random
import time
import torch
import json
import os
from logger import MetricsLogger
from torch.utils.data import DataLoader
from utils import (
    evaluate,
    softmax_cross_entropy,
    get_specified_args,
    get_dataset,
    get_model,
    parse_args,
    get_optimizer,
    stablemax_cross_entropy,
)


torch.set_num_threads(5)

parser, args = parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)

# Force mini-batch by default for this script
args.full_batch = False

train_dtype = getattr(torch, args.train_dtype)
device = args.device
print("Using device:", device)

train_dataset, test_dataset = get_dataset(args)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

torch.save(train_dataset, "last_train_loader.pt")
torch.save(test_dataset, "last_test_loader.pt")

if args.lr is None:
    args.lr = 0.01

args.lr = args.lr / (args.alpha ** 2)

model = get_model(args)
logger = MetricsLogger(args.num_epochs, args.log_frequency)
optimizer = get_optimizer(model, args)

loss_functions = {
    "cross_entropy": softmax_cross_entropy,
    "stablemax": stablemax_cross_entropy,
}
loss_function = loss_functions[args.loss_function]
ce_dtype = getattr(torch, args.cross_entropy_dtype)
save_model_checkpoints = range(0, args.num_epochs, args.log_frequency)
saved_models = {epoch: None for epoch in save_model_checkpoints}

loss = torch.inf
start_time = time.time()
model = model.to(device=device, dtype=train_dtype)

for epoch in range(args.num_epochs):
    model.train()
    for data, target, *_ in train_loader:
        data = data.to(device=device)
        target = target.to(device=device).long()
        if not (args.use_transformer or args.use_embedding):
            data = data.to(train_dtype)

        optimizer.zero_grad()
        output = model(data)
        if args.use_transformer:
            output = output[:, -1]
        output = output * args.alpha
        loss = loss_function(output, target, dtype=ce_dtype)
        loss.backward()
        optimizer.step()

    if epoch % logger.log_frequency == 0:
        model.train()
        optimizer.zero_grad()
        # Use first batch for grad metrics
        data, target, *_ = next(iter(train_loader))
        data = data.to(device=device)
        target = target.to(device=device).long()
        if not (args.use_transformer or args.use_embedding):
            data = data.to(train_dtype)

        output = model(data)
        if args.use_transformer:
            output = output[:, -1]
        output = output * args.alpha
        loss_for_logging = loss_function(output, target, dtype=ce_dtype)
        loss_for_logging.backward()

        logger.log_metrics(
            model=model,
            epoch=epoch,
            save_model_checkpoints=save_model_checkpoints,
            saved_models=saved_models,
            all_data=train_loader.dataset,
            all_targets=None,
            all_test_data=test_loader.dataset,
            all_test_targets=None,
            args=args,
            loss_function=loss_function,
            full_batch=False,
            train_loader=train_loader,
            test_loader=test_loader,
        )
        optimizer.zero_grad()

        print(f"Epoch {epoch}: Training loss: {loss.item():.6f}")
        if epoch > 0:
            print(f"Time taken for the last {args.log_frequency} epochs: {(time.time() - start_time)/60:.2f} min")
        start_time = time.time()

model.eval().to("cpu")
test_loss, test_accuracy = evaluate(
    model,
    test_loader,
    loss_function=loss_function,
    use_embedding=args.use_transformer or args.use_embedding,
    dtype=ce_dtype,
)
print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}")
args.lr = args.lr

specified_args = get_specified_args(parser, args)
if len(specified_args.keys()) == 0:
    experiment_key = f"{args.dataset}_default"
else:
    experiment_key = f"{args.dataset}|" + "|".join([f"{key}-{str(specified_args[key])}" for key in specified_args.keys()])

experiment_key = experiment_key[:255]

torch.save(saved_models, "last_run_saved_model_checkpoints.pt")
torch.save(optimizer, "last_optimizer.pt")

os.makedirs(f"loggs/{experiment_key}", exist_ok=True)
logger.metrics_df.to_csv(f"loggs/{experiment_key}/metrics.csv", index=False)

with open(f"loggs/{experiment_key}/args.json", "w") as f:
    json.dump(vars(args), f, indent=4)

print(f"Saving run: {experiment_key}")
