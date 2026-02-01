from torch.nn.parallel import DistributedDataParallel as DDP
from functools import partial
from typing import Any, Dict, Union
import torch
import random
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import log_loss
import pickle as pk
import numpy as np
from torch import nn
import time
import os
import json
from alignn.models.alignn_atomwise import ALIGNNAtomWise
from data import get_train_val_loaders
from alignn.config import TrainingConfig

def dumpjson(data=[], filename=""):
    """Provide helper function to write a json file."""
    f = open(filename, "w")
    f.write(json.dumps(data))
    f.close()

def group_decay(model):
    """Omit weight decay from bias and batchnorm params."""
    decay, no_decay = [], []

    for name, p in model.named_parameters():
        if "bias" in name or "bn" in name or "norm" in name:
            no_decay.append(p)
        else:
            decay.append(p)

    return [
        {"params": decay},
        {"params": no_decay, "weight_decay": 0},
    ]


def setup_optimizer(params, config: TrainingConfig):
    """Set up optimizer for param groups."""
    if config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay,
        )
    return optimizer

def train_dgl(
    config: Union[TrainingConfig, Dict[str, Any]],
    model: nn.Module = None,
    # checkpoint_dir: Path = Path("./"),
    train_val_test_loaders=[],
    rank=0,
    world_size=0,
    # log_tensorboard: bool = False,
):
    """Training entry point for DGL networks.

    `config` should conform to alignn.conf.TrainingConfig, and
    if passed as a dict with matching keys, pydantic validation is used
    """
    # print("rank", rank)
    # setup(rank, world_size)
    if rank == 0:
        print("config:")
        # print(config)
        if type(config) is dict:
            try:
                print(config)
                config = TrainingConfig(**config)
            except Exception as exp:
                print("Check", exp)

    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    # checkpoint_dir = os.path.join(config.output_dir)
    # deterministic = False
    classification = False
    tmp = config.dict()
    f = open(os.path.join(config.output_dir, "config.json"), "w")
    f.write(json.dumps(tmp, indent=4))
    f.close()
    global tmp_output_dir
    tmp_output_dir = config.output_dir
    if config.classification_threshold is not None:
        classification = True

    line_graph = False
    if config.model.alignn_layers > 0:
        line_graph = True
    if world_size > 1:
        use_ddp = True
    else:
        use_ddp = False
        device = "cpu"
        if torch.cuda.is_available():
            device = torch.device("cuda")
    if not train_val_test_loaders:
        # use input standardization for all real-valued feature sets
        # print("config.neighbor_strategy",config.neighbor_strategy)
        # import sys
        # sys.exit()
        (
            train_loader,
            val_loader,
            test_loader,
            prepare_batch,
        ) = get_train_val_loaders(
            dataset=config.dataset,
            target=config.target,
            n_train=config.n_train,
            n_val=config.n_val,
            n_test=config.n_test,
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
            batch_size=config.batch_size,
            atom_features=config.atom_features,
            neighbor_strategy=config.neighbor_strategy,
            standardize=config.atom_features != "cgcnn",
            line_graph=line_graph,
            id_tag=config.id_tag,
            pin_memory=config.pin_memory,
            workers=config.num_workers,
            save_dataloader=config.save_dataloader,
            use_canonize=config.use_canonize,
            filename=config.filename,
            cutoff=config.cutoff,
            max_neighbors=config.max_neighbors,
            output_features=config.model.output_features,
            classification_threshold=config.classification_threshold,
            target_multiplication_factor=config.target_multiplication_factor,
            standard_scalar_and_pca=config.standard_scalar_and_pca,
            keep_data_order=config.keep_data_order,
            output_dir=config.output_dir,
            use_lmdb=config.use_lmdb,
        )
    else:
        train_loader = train_val_test_loaders[0]
        val_loader = train_val_test_loaders[1]
        test_loader = train_val_test_loaders[2]
        prepare_batch = train_val_test_loaders[3]
    # rank=0
    if use_ddp:
        device = torch.device(f"cuda:{rank}")
    prepare_batch = partial(prepare_batch, device=device)
    if classification:
        config.model.classification = True
    _model = {
        "alignn_atomwise": ALIGNNAtomWise,
    }
    if config.random_seed is not None:
        random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)
        torch.cuda.manual_seed_all(config.random_seed)
      
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(config.random_seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = str(":4096:8")
        torch.use_deterministic_algorithms(True)
    if model is None:
        net = _model.get(config.model.name)(config.model)
    else:
        net = model

    best_model_name = "best_model.pt"
    if os.path.exists(best_model_name):
        print("loading model from: ", best_model_name)
        net.load_state_dict(torch.load(best_model_name, map_location=device))

    print("net parameters", sum(p.numel() for p in net.parameters()))
    # print("device", device)
    net.to(device)
    print(next(net.parameters()).is_cuda)
    params = group_decay(net)
    optimizer = setup_optimizer(params, config)

    if config.scheduler == "none":
        # always return multiplier of 1 (i.e. do nothing)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: 1.0
        )

    elif config.scheduler == "onecycle":
        steps_per_epoch = len(train_loader)
        # pct_start = config.warmup_steps / (config.epochs * steps_per_epoch)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,
            epochs=config.epochs,
            steps_per_epoch=steps_per_epoch,
            # pct_start=pct_start,
            pct_start=0.3,
        )
    elif config.scheduler == "step":
        # pct_start = config.warmup_steps / (config.epochs * steps_per_epoch)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
        )

    print(config.model.name)
    if config.model.name == "alignn_atomwise":

        def get_batch_errors(dat=[]):
            """Get errors for samples."""
            target_out = []
            pred_out = []
            grad = []
            atomw = []
            stress = []
            mean_out = 0
            mean_atom = 0
            mean_grad = 0
            mean_stress = 0
            # natoms_batch=False
            # print ('lendat',len(dat))
            for i in dat:
                if i["target_grad"]:
                    # if config.normalize_graph_level_loss:
                    #      natoms_batch = 0
                    for m, n in zip(i["target_grad"], i["pred_grad"]):
                        x = np.abs(np.array(m) - np.array(n))
                        grad.append(np.mean(x))
                        # if config.normalize_graph_level_loss:
                        #     natoms_batch += np.array(i["pred_grad"]).shape[0]
                if i["target_out"]:
                    for j, k in zip(i["target_out"], i["pred_out"]):
                        # if config.normalize_graph_level_loss and
                        # natoms_batch:
                        #   j=j/natoms_batch
                        #   k=k/natoms_batch
                        # if config.normalize_graph_level_loss and
                        # not natoms_batch:
                        # tmp = 'Add above in atomwise if not train grad.'
                        #   raise ValueError(tmp)

                        target_out.append(j)
                        pred_out.append(k)
                if i["target_stress"]:
                    for p, q in zip(i["target_stress"], i["pred_stress"]):
                        x = np.abs(np.array(p) - np.array(q))
                        stress.append(np.mean(x))
                if i["target_atomwise_pred"]:
                    for m, n in zip(
                        i["target_atomwise_pred"], i["pred_atomwise_pred"]
                    ):
                        x = np.abs(np.array(m) - np.array(n))
                        atomw.append(np.mean(x))
            if "target_out" in i:
                # if i["target_out"]:
                target_out = np.array(target_out)
                pred_out = np.array(pred_out)
                # print('target_out',target_out,target_out.shape)
                # print('pred_out',pred_out,pred_out.shape)
                if classification:
                    mean_out = log_loss(target_out, pred_out)
                else:
                    mean_out = mean_absolute_error(target_out, pred_out)
            if "target_stress" in i:
                # if i["target_stress"]:
                mean_stress = np.array(stress).mean()
            if "target_grad" in i:
                # if i["target_grad"]:
                mean_grad = np.array(grad).mean()
            if "target_atomwise_pred" in i:
                # if i["target_atomwise_pred"]:
                mean_atom = np.array(atomw).mean()
            # print ('natoms_batch',natoms_batch)
            # if natoms_batch!=0:
            #   mean_out = mean_out/natoms_batch
            # print ('dat',dat)
            return mean_out, mean_atom, mean_grad, mean_stress
        
        best_loss = np.inf
        criterion = nn.L1Loss()
        if classification:
            criterion = nn.NLLLoss()
        params = group_decay(net)
        optimizer = setup_optimizer(params, config)
        # optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        history_train = []
        history_val = []
        print("config.epochs: ", range(config.epochs))
        for e in range(config.epochs):
            # optimizer.zero_grad()
            train_init_time = time.time()
            running_loss = 0
            train_result = []
            # for dats in train_loader:
            print("training...")
            i = 0
            for dats, jid in zip(train_loader, train_loader.dataset.ids):
                i = i+1
                if(i % 100 == 0):
                    print("training iteratiion: ", i)
                info = {}
                #print("=============")
                #print(dats[0])
                #print("*************")
                #print(dats[1])

                # info["id"] = jid
                optimizer.zero_grad()
                if (config.model.alignn_layers) > 0:
                    result = net([dats[0].to(device), dats[1].to(device)])
                else:
                    result = net(dats[0].to(device))

                #print(result["out"].size)    
                # info = {}
                info["target_out"] = []
                info["pred_out"] = []
                info["target_atomwise_pred"] = []
                info["pred_atomwise_pred"] = []
                info["target_grad"] = []
                info["pred_grad"] = []
                info["target_stress"] = []
                info["pred_stress"] = []
                
                loss1 = 0  # Such as energy
                loss2 = 0  # Such as bader charges
                loss3 = 0  # Such as forces
                loss4 = 0  # Such as stresses
                if config.model.output_features is not None:
                    # print('result["out"]',result["out"])
                    # print('dats[2]',dats[2])
                    loss1 = config.model.graphwise_weight * criterion(
                        result["out"],
                        dats[-1].to(device),
                        # result["out"], dats[2].to(device)
                    )
                    info["target_out"] = dats[-1].cpu().numpy().tolist()
                    # info["target_out"] = dats[2].cpu().numpy().tolist()
                    info["pred_out"] = [
                        result["out"].cpu().detach().numpy().tolist()
                    ]
                    # graphlevel_loss += np.mean(
                    #    np.abs(
                    #        dats[2].cpu().numpy()
                    #        - result["out"].cpu().detach().numpy()
                    #    )
                    # )
                    # print("target_out", info["target_out"][0])
                    # print("pred_out", info["pred_out"][0])
                # print(
                #    "config.model.atomwise_output_features",
                #    config.model.atomwise_output_features,
                # )
                if (
                    config.model.atomwise_output_features > 0
                    # config.model.atomwise_output_features is not None
                    and config.model.atomwise_weight != 0
                ):
                    loss2 = config.model.atomwise_weight * criterion(
                        result["atomwise_pred"].to(device),
                        dats[0].ndata["atomwise_target"].to(device),
                    )
                    info["target_atomwise_pred"] = (
                        dats[0].ndata["atomwise_target"].cpu().numpy().tolist()
                    )
                    info["pred_atomwise_pred"] = (
                        result["atomwise_pred"].cpu().detach().numpy().tolist()
                    )
                    # atomlevel_loss += np.mean(
                    #    np.abs(
                    #        dats[0].ndata["atomwise_target"].cpu().numpy()
                    #        - result["atomwise_pred"].cpu().detach().numpy()
                    #    )
                    # )

                if config.model.calculate_gradient:
                    loss3 = 1 * criterion(
                        result["grad"].to(device),
                        dats[0].ndata["atomwise_grad"].to(device),
                    )
                    info["target_grad"] = (
                        dats[0].ndata["atomwise_grad"].cpu().numpy().tolist()
                    )
                    info["pred_grad"] = (
                        result["grad"].cpu().detach().numpy().tolist()
                    )
                    # gradlevel_loss += np.mean(
                    #    np.abs(
                    #        dats[0].ndata["atomwise_grad"].cpu().numpy()
                    #        - result["grad"].cpu().detach().numpy()
                    #    )
                    # )
                    # print("target_grad", info["target_grad"][0])
                    # print("pred_grad", info["pred_grad"][0])
                if config.model.stresswise_weight != 0:
                    # print(
                    #    'result["stress"]',
                    #    result["stresses"],
                    #    result["stresses"].shape,
                    # )
                    # print(
                    #    'dats[0].ndata["stresses"]',
                    #    torch.cat(tuple(dats[0].ndata["stresses"])),
                    #    dats[0].ndata["stresses"].shape,
                    # )  # ,torch.cat(dats[0].ndata["stresses"]),
                    # torch.cat(dats[0].ndata["stresses"]).shape)
                    # print('result["stresses"]',result["stresses"],result["stresses"].shape)
                    # print(dats[0].ndata["stresses"],dats[0].ndata["stresses"].shape)
                    loss4 = config.model.stresswise_weight * criterion(
                        (result["stresses"]).to(device),
                        torch.cat(tuple(dats[0].ndata["stresses"])).to(device),
                    )
                    info["target_stress"] = (
                        torch.cat(tuple(dats[0].ndata["stresses"]))
                        .cpu()
                        .numpy()
                        .tolist()
                        # dats[0].ndata["stresses"][0].cpu().numpy().tolist()
                    )
                    info["pred_stress"] = (
                        result["stresses"].cpu().detach().numpy().tolist()
                    )
                    # print("target_stress", info["target_stress"][0])
                    # print("pred_stress", info["pred_stress"][0])
                train_result.append(info)
                loss = loss1 + loss2 + loss3 + loss4
                loss.backward()
                optimizer.step()
                # optimizer.zero_grad() #never
                running_loss += loss.item()

            
            mean_out, mean_atom, mean_grad, mean_stress = get_batch_errors(train_result)

            scheduler.step()
            train_final_time = time.time()
            train_ep_time = train_final_time - train_init_time
            # if rank == 0: # or world_size == 1:
            history_train.append([mean_out, mean_atom, mean_grad, mean_stress])
            dumpjson(
                filename=os.path.join(config.output_dir, "history_train.json"),
                data=history_train,
            )
            print("validating...")
            val_loss = 0
            val_result = []
            # for dats in val_loader:
            for dats, jid in zip(val_loader, val_loader.dataset.ids):
                info = {}
                info["id"] = jid
                optimizer.zero_grad()
                # result = net([dats[0].to(device), dats[1].to(device)])
                if (config.model.alignn_layers) > 0:
                    result = net([dats[0].to(device), dats[1].to(device)])
                else:
                    result = net(dats[0].to(device))
                # info = {}
                info["target_out"] = []
                info["pred_out"] = []
                info["target_atomwise_pred"] = []
                info["pred_atomwise_pred"] = []
                info["target_grad"] = []
                info["pred_grad"] = []
                info["target_stress"] = []
                info["pred_stress"] = []
                loss1 = 0  # Such as energy
                loss2 = 0  # Such as bader charges
                loss3 = 0  # Such as forces
                loss4 = 0  # Such as stresses
                if config.model.output_features is not None:
                    loss1 = config.model.graphwise_weight * criterion(
                        result["out"], dats[-1].to(device)
                    )
                    info["target_out"] = dats[-1].cpu().numpy().tolist()
                    info["pred_out"] = [
                        result["out"].cpu().detach().numpy().tolist()
                    ]

                if (
                    config.model.atomwise_output_features > 0
                    and config.model.atomwise_weight != 0
                ):
                    loss2 = config.model.atomwise_weight * criterion(
                        result["atomwise_pred"].to(device),
                        dats[0].ndata["atomwise_target"].to(device),
                    )
                    info["target_atomwise_pred"] = (
                        dats[0].ndata["atomwise_target"].cpu().numpy().tolist()
                    )
                    info["pred_atomwise_pred"] = (
                        result["atomwise_pred"].cpu().detach().numpy().tolist()
                    )
                if config.model.calculate_gradient:
                    loss3 = 1 * criterion(
                        result["grad"].to(device),
                        dats[0].ndata["atomwise_grad"].to(device),
                    )
                    info["target_grad"] = (
                        dats[0].ndata["atomwise_grad"].cpu().numpy().tolist()
                    )
                    info["pred_grad"] = (
                        result["grad"].cpu().detach().numpy().tolist()
                    )
                if config.model.stresswise_weight != 0:
                    # loss4 = config.model.stresswise_weight * criterion(
                    #    result["stress"].to(device),
                    #    dats[0].ndata["stresses"][0].to(device),
                    # )
                    loss4 = config.model.stresswise_weight * criterion(
                        # torch.flatten(result["stress"].to(device)),
                        # (dats[0].ndata["stresses"]).to(device),
                        # torch.flatten(dats[0].ndata["stresses"]).to(device),
                        # torch.flatten(torch.cat(dats[0].ndata["stresses"])).to(device),
                        # dats[0].ndata["stresses"][0].to(device),
                        (result["stresses"]).to(device),
                        torch.cat(tuple(dats[0].ndata["stresses"])).to(device),
                    )
                    info["target_stress"] = (
                        torch.cat(tuple(dats[0].ndata["stresses"]))
                        .cpu()
                        .numpy()
                        .tolist()
                        # dats[0].ndata["stresses"][0].cpu().numpy().tolist()
                    )
                    info["pred_stress"] = (
                        result["stresses"].cpu().detach().numpy().tolist()
                    )
                loss = loss1 + loss2 + loss3 + loss4
                val_result.append(info)
                val_loss += loss.item()
            mean_out, mean_atom, mean_grad, mean_stress = get_batch_errors(
                val_result
            )
            current_model_name = "current_model.pt"
            torch.save(
                net.state_dict(),
                os.path.join(config.output_dir, current_model_name),
            )
            saving_msg = ""
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_name = "best_model.pt"
                torch.save(
                    net.state_dict(),
                    os.path.join(config.output_dir, best_model_name),
                )
                # print("Saving data for epoch:", e)
                saving_msg = "Saving model"
                dumpjson(
                    filename=os.path.join(
                        config.output_dir, "Train_results.json"
                    ),
                    data=train_result,
                )
                dumpjson(
                    filename=os.path.join(
                        config.output_dir, "Val_results.json"
                    ),
                    data=val_result,
                )
                best_model = net
            history_val.append([mean_out, mean_atom, mean_grad, mean_stress])
            dumpjson(
                filename=os.path.join(config.output_dir, "history_val.json"),
                data=history_val,
            )
        if rank == 0 or world_size == 1:
            test_loss = 0
            test_result = []
            for dats, jid in zip(test_loader, test_loader.dataset.ids):
                # for dats in test_loader:
                info = {}
                info["id"] = jid
                optimizer.zero_grad()
                # print('dats[0]',dats[0])
                # print('test_loader',test_loader)
                # print('test_loader.dataset.ids',test_loader.dataset.ids)
                # result = net([dats[0].to(device), dats[1].to(device)])
                if (config.model.alignn_layers) > 0:
                    result = net([dats[0].to(device), dats[1].to(device)])
                else:
                    result = net(dats[0].to(device))
                loss1 = 0  # Such as energy
                loss2 = 0  # Such as bader charges
                loss3 = 0  # Such as forces
                loss4 = 0  # Such as stresses
                if (
                    config.model.output_features is not None
                    and not classification
                ):
                    # print('result["out"]',result["out"])
                    # print('dats[2]',dats[2])
                    loss1 = config.model.graphwise_weight * criterion(
                        result["out"], dats[-1].to(device)
                    )
                    info["target_out"] = dats[-1].cpu().numpy().tolist()
                    info["pred_out"] = [
                        result["out"].cpu().detach().numpy().tolist()
                    ]

                if config.model.atomwise_output_features > 0:
                    loss2 = config.model.atomwise_weight * criterion(
                        result["atomwise_pred"].to(device),
                        dats[0].ndata["atomwise_target"].to(device),
                    )
                    info["target_atomwise_pred"] = (
                        dats[0].ndata["atomwise_target"].cpu().numpy().tolist()
                    )
                    info["pred_atomwise_pred"] = (
                        result["atomwise_pred"].cpu().detach().numpy().tolist()
                    )

                if config.model.calculate_gradient:
                    loss3 = 1 * criterion(
                        result["grad"].to(device),
                        dats[0].ndata["atomwise_grad"].to(device),
                    )
                    info["target_grad"] = (
                        dats[0].ndata["atomwise_grad"].cpu().numpy().tolist()
                    )
                    info["pred_grad"] = (
                        result["grad"].cpu().detach().numpy().tolist()
                    )
                if config.model.stresswise_weight != 0:
                    loss4 = config.model.stresswise_weight * criterion(
                        # torch.flatten(result["stress"].to(device)),
                        # (dats[0].ndata["stresses"]).to(device),
                        # torch.flatten(dats[0].ndata["stresses"]).to(device),
                        result["stresses"].to(device),
                        torch.cat(tuple(dats[0].ndata["stresses"])).to(device),
                        # torch.flatten(torch.cat(dats[0].ndata["stresses"])).to(device),
                        # dats[0].ndata["stresses"][0].to(device),
                    )
                    # loss4 = config.model.stresswise_weight * criterion(
                    #    result["stress"][0].to(device),
                    #    dats[0].ndata["stresses"].to(device),
                    # )
                    info["target_stress"] = (
                        torch.cat(tuple(dats[0].ndata["stresses"]))
                        .cpu()
                        .numpy()
                        .tolist()
                    )
                    info["pred_stress"] = (
                        result["stresses"].cpu().detach().numpy().tolist()
                    )
                test_result.append(info)
                loss = loss1 + loss2 + loss3 + loss4
                if not classification:
                    test_loss += loss.item()
            print("TestLoss", e, test_loss)
            dumpjson(
                filename=os.path.join(config.output_dir, "Test_results.json"),
                data=test_result,
            )
            last_model_name = "last_model.pt"
            torch.save(
                net.state_dict(),
                os.path.join(config.output_dir, last_model_name),
            )
            # return test_result
    if rank == 0 or world_size == 1:
        if (
            config.write_predictions
            and not classification
            and config.model.output_features == 1
            #and config.model.gradwise_weight == 0
        ):
            best_model.eval()
            # net.eval()
            f = open(
                os.path.join(
                    config.output_dir, "prediction_results_test_set.csv"
                ),
                "w",
            )
            f.write("id,target,prediction\n")
            targets = []
            predictions = []
            with torch.no_grad():
                ids = test_loader.dataset.ids  # [test_loader.dataset.indices]
                for dat, id in zip(test_loader, ids):
                    g, lg, target = dat
                    out_data = best_model([g.to(device), lg.to(device)])["out"]
                    # out_data = net([g.to(device), lg.to(device)])["out"]
                    out_data = out_data.cpu().numpy().tolist()
                    if config.standard_scalar_and_pca:
                        sc = pk.load(
                            open(os.path.join(tmp_output_dir, "sc.pkl"), "rb")
                        )
                        out_data = sc.transform(
                            np.array(out_data).reshape(-1, 1)
                        )[0][0]
                    target = target.cpu().numpy().flatten().tolist()
                    if len(target) == 1:
                        target = target[0]
                    f.write("%s, %6f, %6f\n" % (id, target, out_data))
                    targets.append(target)
                    predictions.append(out_data)
            f.close()

            print(
                "Test MAE:",
                mean_absolute_error(np.array(targets), np.array(predictions)),
            )
            best_model.eval()
            # net.eval()
            f = open(
                os.path.join(
                    config.output_dir, "prediction_results_train_set.csv"
                ),
                "w",
            )
            f.write("target,prediction\n")
            targets = []
            predictions = []
            with torch.no_grad():
                ids = train_loader.dataset.ids  # [test_loader.dataset.indices]
                for dat, id in zip(train_loader, ids):
                    g, lg, target = dat
                    out_data = best_model([g.to(device), lg.to(device)])["out"]
                    # out_data = net([g.to(device), lg.to(device)])["out"]
                    out_data = out_data.cpu().numpy().tolist()
                    if config.standard_scalar_and_pca:
                        sc = pk.load(
                            open(os.path.join(tmp_output_dir, "sc.pkl"), "rb")
                        )
                        out_data = sc.transform(
                            np.array(out_data).reshape(-1, 1)
                        )[0][0]
                    target = target.cpu().numpy().flatten().tolist()
                    # if len(target) == 1:
                    #    target = target[0]
                    # if len(out_data) == 1:
                    #    out_data = out_data[0]
                    for ii, jj in zip(target, out_data):
                        f.write("%6f, %6f\n" % (ii, jj))
                        targets.append(ii)
                        predictions.append(jj)
            f.close()


def train_prop_model(
    prop="e_form",
    dataset="megnet",
    write_predictions=True,
    name="alignn_atomwise",
    save_dataloader=True,
    train_ratio=None,
    classification_threshold=None,
    val_ratio=None,
    test_ratio=None,
    learning_rate=0.001,
    batch_size=None,
    scheduler=None,
    n_epochs=None,
    id_tag=None,
    num_workers=None,
    weight_decay=None,
    alignn_layers=None,
    gcn_layers=None,
    edge_input_features=None,
    triplet_input_features=None,
    embedding_features=None,
    hidden_features=None,
    output_features=None,
    random_seed=None,
    n_early_stopping=None,
    cutoff=None,
    max_neighbors=None,
):
    """Train models for a dataset and a property."""
    if scheduler is None:
        scheduler = "onecycle"
    if batch_size is None:
        batch_size = 32
    if n_epochs is None:
        n_epochs = 20
    if num_workers is None:
        num_workers = 0
    config = {
        "dataset": dataset,
        "target": prop,
        "epochs": n_epochs,  # 00,#00,
        "batch_size": batch_size,  # 0,
        "weight_decay": 1e-05,
        "learning_rate": learning_rate,
        "criterion": "mse",
        "optimizer": "adamw",
        "scheduler": scheduler,
        "save_dataloader": save_dataloader,
        "pin_memory": False,
        "write_predictions": write_predictions,
        "num_workers": num_workers,
        "classification_threshold": classification_threshold,
        "model": {"name": name, },
    }
    if n_early_stopping is not None:
        config["n_early_stopping"] = n_early_stopping
    if cutoff is not None:
        config["cutoff"] = cutoff
    if max_neighbors is not None:
        config["max_neighbors"] = max_neighbors
    if weight_decay is not None:
        config["weight_decay"] = weight_decay
    if alignn_layers is not None:
        config["model"]["alignn_layers"] = alignn_layers
    if gcn_layers is not None:
        config["model"]["gcn_layers"] = gcn_layers
    if edge_input_features is not None:
        config["model"]["edge_input_features"] = edge_input_features
    if hidden_features is not None:
        config["model"]["hidden_features"] = hidden_features
    if embedding_features is not None:
        config["model"]["embedding_features"] = embedding_features
    if output_features is not None:
        config["model"]["output_features"] = output_features
    if random_seed is not None:
        config["random_seed"] = random_seed
    # if model_name is not None:
    #    config['model']['name']=model_name

    if id_tag is not None:
        config["id_tag"] = id_tag
    if train_ratio is not None:
        config["train_ratio"] = train_ratio
        if val_ratio is None:
            raise ValueError("Enter val_ratio.")

        if test_ratio is None:
            raise ValueError("Enter test_ratio.")
        config["val_ratio"] = val_ratio
        config["test_ratio"] = test_ratio
   
    if dataset == "megnet":
        config["id_tag"] = "id"
        if prop == "e_form" or prop == "gap pbe":
            config["n_train"] = 60000
            config["n_val"] = 5000
            config["n_test"] = 4239
            config["num_workers"] = 4

    t1 = time.time()
    result = train_dgl(config)
    t2 = time.time()
    print("train=", result["train"])
    print("validation=", result["validation"])
    print("Toal time:", t2 - t1)


train_prop_model(learning_rate=0.001,name="alignn_atomwise",dataset="megnet",prop="e_form")