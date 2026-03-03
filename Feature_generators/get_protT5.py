import esm
from Bio import SeqIO
import time
import pandas as pd
from tqdm import tqdm
import os
from transformers import T5EncoderModel, T5Tokenizer
import torch
import h5py
import numpy as np
from pathlib import Path
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using device: {}".format(device))


readfile = "../data/sequence/MPD476.fasta"
data = pd.read_excel('../data/dna_MPD476.xlsx')
save_dir_residue = '../data/feature/MPD476_protT5_residue'
Path(save_dir_residue).mkdir(parents=True, exist_ok=True)


seq_path = readfile
per_residue = True
per_residue_path = "./protT5/output/per_residue_embeddings.h5"  # where to store the embeddings
per_protein = True
per_protein_path = "./protT5/output/per_protein_embeddings.h5"  # where to store the embeddings
sec_struct = False
sec_struct_path = "./protT5/output/ss3_preds.fasta" # file for storing predictions
assert per_protein is True or per_residue is True or sec_struct is True, print(
    "Minimally, you need to active per_residue, per_protein or sec_struct. (or any combination)")


class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # This is only called "elmo_feature_extractor" for historic reason
        # CNN weights are trained on ProtT5 embeddings
        self.elmo_feature_extractor = torch.nn.Sequential(
            torch.nn.Conv2d(1024, 32, kernel_size=(7, 1), padding=(3, 0)),  # 7x32
            torch.nn.ReLU(),
            torch.nn.Dropout(0.25),
        )
        n_final_in = 32
        self.dssp3_classifier = torch.nn.Sequential(
            torch.nn.Conv2d(n_final_in, 3, kernel_size=(7, 1), padding=(3, 0))  # 7
        )

        self.dssp8_classifier = torch.nn.Sequential(
            torch.nn.Conv2d(n_final_in, 8, kernel_size=(7, 1), padding=(3, 0))
        )
        self.diso_classifier = torch.nn.Sequential(
            torch.nn.Conv2d(n_final_in, 2, kernel_size=(7, 1), padding=(3, 0))
        )

    def forward(self, x):
        # IN: X = (B x L x F); OUT: (B x F x L, 1)
        x = x.permute(0, 2, 1).unsqueeze(dim=-1)
        x = self.elmo_feature_extractor(x)  # OUT: (B x 32 x L x 1)
        d3_Yhat = self.dssp3_classifier(x).squeeze(dim=-1).permute(0, 2, 1)  # OUT: (B x L x 3)
        d8_Yhat = self.dssp8_classifier(x).squeeze(dim=-1).permute(0, 2, 1)  # OUT: (B x L x 8)
        diso_Yhat = self.diso_classifier(x).squeeze(dim=-1).permute(0, 2, 1)  # OUT: (B x L x 2)
        return d3_Yhat, d8_Yhat, diso_Yhat


def load_sec_struct_model():
    checkpoint_dir = "./protT5/sec_struct_checkpoint/secstruct_checkpoint.pt"
    state = torch.load(checkpoint_dir)
    model = ConvNet()
    model.load_state_dict(state['state_dict'])
    model = model.eval()
    model = model.to(device)
    print('Loaded sec. struct. model from epoch: {:.1f}'.format(state['epoch']))

    return model


def get_T5_model():
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
    model = model.to(device)  # move model to GPU
    model = model.eval()  # set model to evaluation model
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)

    return model, tokenizer


def read_fasta(fasta_path, split_char="!", id_field=0):
    '''
        Reads in fasta file containing multiple sequences.
        Split_char and id_field allow to control identifier extraction from header.
        E.g.: set split_char="|" and id_field=1 for SwissProt/UniProt Headers.
        Returns dictionary holding multiple sequences or only single
        sequence, depending on input file.
    '''

    seqs = dict()
    with open(fasta_path, 'r') as fasta_f:
        for line in fasta_f:
            # get uniprot ID from header and create new entry
            if line.startswith('>'):
                uniprot_id = line.replace('>', '').strip().split(split_char)[id_field]
                # replace tokens that are mis-interpreted when loading h5
                uniprot_id = uniprot_id.replace("/", "_").replace(".", "_")
                seqs[uniprot_id] = ''
            else:
                # repl. all whie-space chars and join seqs spanning multiple lines, drop gaps and cast to upper-case
                seq = ''.join(line.split()).upper().replace("-", "")
                # repl. all non-standard AAs and map them to unknown/X
                seq = seq.replace('U', 'X').replace('Z', 'X').replace('O', 'X')
                seqs[uniprot_id] += seq
    example_id = next(iter(seqs))
    print("Read {} sequences.".format(len(seqs)))
    print("Example:\n{}\n{}".format(example_id, seqs[example_id]))

    return seqs


def get_embeddings(model, tokenizer, seqs, per_residue, per_protein, sec_struct,
                   max_residues=4000, max_seq_len=1000, max_batch=100):
    if sec_struct:
        sec_struct_model = load_sec_struct_model()

    results = {"residue_embs": dict(),
               "protein_embs": dict(),
               "sec_structs": dict()
               }

    # sort sequences according to length (reduces unnecessary padding --> speeds up embedding)
    seq_dict = sorted(seqs.items(), key=lambda kv: len(seqs[kv[0]]), reverse=True)
    start = time.time()
    batch = list()
    for seq_idx, (pdb_id, seq) in enumerate(seq_dict, 1):
        seq = seq
        seq_len = len(seq)
        seq = ' '.join(list(seq))
        batch.append((pdb_id, seq, seq_len))

        # count residues in current batch and add the last sequence length to
        # avoid that batches with (n_res_batch > max_residues) get processed
        n_res_batch = sum([s_len for _, _, s_len in batch]) + seq_len
        if len(batch) >= max_batch or n_res_batch >= max_residues or seq_idx == len(seq_dict) or seq_len > max_seq_len:
            pdb_ids, seqs, seq_lens = zip(*batch)
            batch = list()

            # add_special_tokens adds extra token at the end of each sequence
            token_encoding = tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding="longest")
            input_ids = torch.tensor(token_encoding['input_ids']).to(device)
            attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)

            try:
                with torch.no_grad():
                    # returns: ( batch-size x max_seq_len_in_minibatch x embedding_dim )
                    embedding_repr = model(input_ids, attention_mask=attention_mask)
            except RuntimeError:
                print("RuntimeError during embedding for {} (L={})".format(pdb_id, seq_len))
                continue

            if sec_struct:  # in case you want to predict secondary structure from embeddings
                d3_Yhat, d8_Yhat, diso_Yhat = sec_struct_model(embedding_repr.last_hidden_state)

            for batch_idx, identifier in enumerate(pdb_ids):  # for each protein in the current mini-batch
                s_len = seq_lens[batch_idx]
                # slice off padding --> batch-size x seq_len x embedding_dim
                emb = embedding_repr.last_hidden_state[batch_idx, :s_len]
                if sec_struct:  # get classification results
                    results["sec_structs"][identifier] = torch.max(d3_Yhat[batch_idx, :s_len], dim=1)[
                        1].detach().cpu().numpy().squeeze()
                if per_residue:  # store per-residue embeddings (Lx1024)
                    results["residue_embs"][identifier] = emb.detach().cpu().numpy().squeeze()
                if per_protein:  # apply average-pooling to derive per-protein embeddings (1024-d)
                    protein_emb = emb.mean(dim=0)
                    results["protein_embs"][identifier] = protein_emb.detach().cpu().numpy().squeeze()

    passed_time = time.time() - start
    avg_time = passed_time / len(results["residue_embs"]) if per_residue else passed_time / len(results["protein_embs"])
    print('\n############# EMBEDDING STATS #############')
    print('Total number of per-residue embeddings: {}'.format(len(results["residue_embs"])))
    print('Total number of per-protein embeddings: {}'.format(len(results["protein_embs"])))
    print("Time for generating embeddings: {:.1f}[m] ({:.3f}[s/protein])".format(
        passed_time / 60, avg_time))
    print('\n############# END #############')
    return results


def save_embeddings(emb_dict, out_path):
    with h5py.File(str(out_path), "w") as hf:
        for sequence_id, embedding in emb_dict.items():
            # noinspection PyUnboundLocalVariable
            hf.create_dataset(sequence_id, data=embedding)
    return None


def write_prediction_fasta(predictions, out_path):
    class_mapping = {0:"H",1:"E",2:"L"}
    with open(out_path, 'w+') as out_f:
        out_f.write( '\n'.join(
            [ ">{}\n{}".format(
                seq_id, ''.join( [class_mapping[j] for j in yhat] ))
            for seq_id, yhat in predictions.items()
            ]
              ) )
    return None


# Load the encoder part of ProtT5-XL-U50 in half-precision (recommended)
model, tokenizer = get_T5_model()

# Load example fasta.
seqs = read_fasta(seq_path)

# Compute embeddings and/or secondary structure predictions
results = get_embeddings(model, tokenizer, seqs,
                         per_residue, per_protein, sec_struct)

# Store per-residue embeddings
if per_residue:
    save_embeddings(results["residue_embs"], per_residue_path)
if per_protein:
    save_embeddings(results["protein_embs"], per_protein_path)
if sec_struct:
    write_prediction_fasta(results["sec_structs"], sec_struct_path)


np.set_printoptions(threshold=np.inf)
residue_embs = results['residue_embs']
protein_embs = results['protein_embs']
# sec_structs = results['sec_structs']



def get_201_protT5_array(pdb_id_mut, pos):
    esm_output = residue_embs[pdb_id_mut]
    esm_array = np.zeros((201, 1024))
    pos_0 = pos - 1
    start_idx = max(0, pos_0 - 100)
    end_idx = min(len(esm_output), pos_0 + 101)
    # 若切片长度不足201行，用0填充（或其他策略）
    actual_length = end_idx - start_idx  # 实际截取的行数
    # offset：pos在截取切片中的位置（切片内索引）
    offset_in_slice = pos_0 - start_idx
    pad_top = 100 - offset_in_slice
    pad_bottom = 201 - actual_length - pad_top

    output_slice = np.pad(
        esm_output[start_idx:end_idx, :],
        ((pad_top, pad_bottom), (0, 0)),
        mode='constant',
        constant_values=0
    )
    esm_array[:, :] = output_slice[:201, :]

    if not np.array_equal(esm_array[100], esm_output[pos-1]):
        print("==========pos=========="+pdb_id_mut)

    return esm_array




for idx in tqdm(range(len(data))):
    Nucleic_Acid = data.iloc[idx]['Nucleic_Acid']
    pdb_id = data.iloc[idx]['pdb_id']
    chain = data.iloc[idx]['chain']
    mutation = data.iloc[idx]['mutation_new']
    pos = int(mutation[1:-1])
    wild_aa = mutation[0]
    mutation_aa = mutation[-1]
    wild_seq = data.iloc[idx]['wild_sequence']
    mutation_old = data.iloc[idx]['mutation_old']
    pos_old = int(mutation_old[1:-1])

    mut_seq = wild_seq[:pos - 1] + mutation_aa + wild_seq[pos:]
    pdb_id_mut = pdb_id + '_' + mutation

    protT5_residue_array = get_201_protT5_array(pdb_id_mut, pos)

    protT5_residue_path = f'{save_dir_residue}/{Nucleic_Acid}_{pdb_id.lower()}_0_{pos_old}_{wild_aa}_{chain}_{mutation_aa}.pt'
    protT5_residue_feature = torch.tensor(protT5_residue_array, dtype=torch.float)
    torch.save(protT5_residue_feature, protT5_residue_path)







