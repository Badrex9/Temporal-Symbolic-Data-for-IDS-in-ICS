"""
run_gta_train_eval.py
======================
Entraine et evalue GTA (code officiel, zackchen-lb/GTA) sur SWaT ou WADI,
en utilisant directement Exp_GTA_DAD (train + test), puis sauvegarde
pred.npy / true.npy / label.npy pour le calcul d'AUROC en aval.

PROTOCOLE D'ENTRAINEMENT -- fidele a la Section V.B.4 du papier GTA :
  "We train our models for up to 50 epochs and early stopping strategy
  is applied with patience of 10. [...] Adam optimizer with learning
  rate initialized as 1e-4 and beta1, beta2 as 0.9, 0.99 [...] We run
  each experiment for 5 trials and report the mean value."
  => epochs=50, patience=10 (defauts de ce script, modifiable en CLI).
  => 5 essais moyennes : NON automatise ici (cout de calcul), a faire
     manuellement en relancant ce script 5 fois avec des seeds differentes
     si on veut reporter une moyenne fidele au protocole.
  => Adam betas=(0.9, 0.99) au lieu du defaut PyTorch (0.9, 0.999) --
     PATCH REQUIS sur le repo (voir ci-dessous), sinon les betas par
     defaut de PyTorch sont utilises silencieusement.

PREREQUIS :
  1. git clone https://github.com/zackchen-lb/GTA.git
  2. pip install torch_geometric --break-system-packages
  3. Patch bug gc_module :
       sed -i "s/self\.gc_modules\[0\]/self.gc_module/" GTA/models/gta.py
  4. Patch bugs pandas (apply(...,1) incompatible pandas recent) :
       sed -i "s/\.apply(lambda row:row\.month,1)/.apply(lambda row:row.month)/g" GTA/data/data_loader_dad.py
       sed -i "s/\.apply(lambda row:row\.day,1)/.apply(lambda row:row.day)/g" GTA/data/data_loader_dad.py
       sed -i "s/\.apply(lambda row:row\.weekday(),1)/.apply(lambda row:row.weekday())/g" GTA/data/data_loader_dad.py
       sed -i "s/\.apply(lambda row:row\.hour,1)/.apply(lambda row:row.hour)/g" GTA/data/data_loader_dad.py
       sed -i "s/\.apply(lambda row:row\.minute,1)/.apply(lambda row:row.minute)/g" GTA/data/data_loader_dad.py
       sed -i "s/\.apply(lambda row:row\.second,1)/.apply(lambda row:row.second)/g" GTA/data/data_loader_dad.py
       sed -i "s/\.drop(\['date'\],1)/.drop(['date'],axis=1)/g" GTA/data/data_loader_dad.py
       sed -i "s/\.drop(\[' Timestamp'\],1)/.drop([' Timestamp'],axis=1)/g" GTA/data/data_loader_dad.py
  5. Patch np.Inf (numpy 2.0) :
       sed -i "s/np\.Inf/np.inf/g" GTA/utils/tools.py
  6. Patch Adam betas pour matcher le papier (0.9, 0.99) :
       sed -i "s/optim.Adam(self.model.parameters(), lr=self.args.learning_rate)/optim.Adam(self.model.parameters(), lr=self.args.learning_rate, betas=(0.9, 0.99))/" GTA/exp/exp_gta_dad.py
  7. Preparer les donnees avec prepare_gta_data.py

Usage :
  python run_gta_train_eval.py --dataset swat --data_dir ./gta_data/
  python run_gta_train_eval.py --dataset wadi --data_dir ./gta_data/ --epochs 50

Sorties (comme le repo officiel) :
  ./results/<setting>/pred.npy
  ./results/<setting>/true.npy
  ./results/<setting>/label.npy
  ./checkpoints/<setting>/checkpoint.pth
"""

import sys, os, argparse
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "GTA"))
from exp.exp_gta_dad import Exp_GTA_DAD  # noqa: E402


class Args:
    """Namespace minimal reproduisant les args CLI de main_gta_dad.py."""
    pass


DATASET_CFG = {
    "swat": dict(data="SWaT", target="FIT_101",
                normal_file="SWaT_normaldata_downsampled.csv"),
    "wadi": dict(data="WADI", target="1_LS_001_AL",
                normal_file="WADI_14days_downsampled.csv"),
}


def detect_num_nodes(data_dir, cfg):
    """Compte le nombre de colonnes capteurs dans le CSV train genere par
    prepare_gta_data.py (total colonnes - 1 pour la colonne timestamp/date).
    Necessaire car le nombre reel de capteurs conserves depend du nettoyage
    (suppression des colonnes 100% vides), qui peut differer du chiffre
    hardcode dans le repo officiel (112 pour leur propre WADI downsample)."""
    import pandas as pd
    path = os.path.join(data_dir, cfg["normal_file"])
    df = pd.read_csv(path, nrows=1)
    return len(df.columns) - 1


def build_args(dataset, data_dir, epochs, batch_size, patience, lr,
              seq_len, label_len, pred_len):
    cfg = DATASET_CFG[dataset]
    num_nodes = detect_num_nodes(data_dir, cfg)
    args = Args()
    args.model = "gta"
    args.data = cfg["data"]
    args.root_path = data_dir
    args.data_path = ""  # non utilise par SWaT/WADI dataset classes (noms hardcodes)
    args.features = "M"
    args.target = cfg["target"]

    args.seq_len = seq_len
    args.label_len = label_len
    args.pred_len = pred_len
    args.num_nodes = num_nodes
    args.num_levels = 3
    args.d_model = 128
    args.n_heads = 8
    args.e_layers = 3
    args.d_layers = 2
    args.d_ff = 128
    args.factor = 5

    args.dropout = 0.05
    args.attn = "prob"
    args.embed = "fixed"
    args.activation = "gelu"
    args.num_workers = 0

    args.train_epochs = epochs
    args.batch_size = batch_size
    args.patience = patience
    args.learning_rate = lr
    args.lradj = "type1"

    args.use_gpu = torch.cuda.is_available()
    args.gpu = 0

    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["swat", "wadi"])
    parser.add_argument("--data_dir", required=True,
                        help="Dossier contenant les CSV *_downsampled.csv "
                             "generes par prepare_gta_data.py")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Papier GTA section V.B.4 : 'up to 50 epochs' "
                             "avec early stopping (patience=10)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=10,
                        help="Papier GTA section V.B.4 : patience=10")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seq_len", type=int, default=60)
    parser.add_argument("--label_len", type=int, default=30)
    parser.add_argument("--pred_len", type=int, default=24)
    args_cli = parser.parse_args()

    args = build_args(
        args_cli.dataset, args_cli.data_dir, args_cli.epochs,
        args_cli.batch_size, args_cli.patience, args_cli.lr,
        args_cli.seq_len, args_cli.label_len, args_cli.pred_len,
    )

    if args.use_gpu:
        torch.cuda.set_device(args.gpu)

    setting = f"gta_{args.data}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}"

    print(f"\n{'='*65}")
    print(f"  GTA -- entrainement + evaluation officielle")
    print(f"  dataset={args.data}  num_nodes={args.num_nodes} (auto-detecte)")
    print(f"  epochs={args.train_epochs}  batch_size={args.batch_size}")
    print(f"  device={'cuda' if args.use_gpu else 'cpu'}")
    print(f"{'='*65}\n")

    exp = Exp_GTA_DAD(args)

    print(">>>>>>> training >>>>>>>")
    exp.train(setting)

    print(">>>>>>> testing >>>>>>>")
    exp.test(setting)

    print(f"\n  Resultats sauvegardes dans ./results/{setting}/")
    print(f"  Lancer ensuite : python compute_auroc_gta.py --setting {setting}\n")


if __name__ == "__main__":
    main()
