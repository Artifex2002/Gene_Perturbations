import gzip
import pandas as pd
from pyfaidx import Fasta

# ========= CONFIG =========
GTF_PATH = "gencode.v44.annotation.gtf.gz"
GENOME_FASTA = "GRCh38.primary_assembly.genome.fa"

FLANK = 1000  # bp upstream/downstream of TSS. Change to 300 if you want ±300bp

# ==========================

# ---- 1. Load gene list from your pairs table ----
pairs = pd.read_csv("tf_gene_pairs.csv")  # or however you saved it
gene_list = sorted(pairs["gene"].unique())
gene_set = set(gene_list)
print(f"Number of unique genes in pairs: {len(gene_set)}")

# ---- 2. Build gene -> (chrom, strand, TSS) map from GENCODE ----
# We'll use 'transcript' entries to get TSS; then pick one transcript per gene.

rows = []
with gzip.open(GTF_PATH, "rt") as fh:
    for line in fh:
        if line.startswith("#"):
            continue
        chrom, src, feature, start, end, score, strand, frame, attrs = line.strip().split("\t")
        if feature != "transcript":
            continue

        # parse attributes into a dict
        attr_dict = {}
        for field in attrs.strip().split(";"):
            field = field.strip()
            if not field:
                continue
            key, val = field.split(" ", 1)
            attr_dict[key] = val.strip('"')

        gene_name = attr_dict.get("gene_name")
        if gene_name not in gene_set:
            # we only care about genes that are in our pairs table
            continue

        start = int(start)
        end = int(end)

        # TSS depends on strand:
        tss = start if strand == "+" else end

        rows.append((gene_name, chrom, strand, tss))

tss_df = pd.DataFrame(rows, columns=["gene", "chrom", "strand", "tss"])
print(f"Found TSS entries for {tss_df['gene'].nunique()} of our genes.")

# If multiple transcripts per gene, pick one.
# Simple rule: take the "most upstream" TSS for + strand, "most downstream" for - strand.
tss_df = (
    tss_df
    .sort_values(["gene", "chrom", "strand", "tss"])
    .drop_duplicates(subset=["gene"], keep="first")
    .set_index("gene")
)

print("Example TSS rows:")
print(tss_df.head())
print(f"Final TSS map size: {tss_df.shape[0]} genes.")

# ---- 3. Extract promoter sequences from the genome FASTA ----

genome = Fasta(GENOME_FASTA)
comp = str.maketrans("ACGTacgt", "TGCAtgca")

def fetch_promoter(chrom, tss, strand, flank=FLANK):
    # Note: pyfaidx uses 0-based, end-exclusive indexing.
    start = max(0, tss - flank)
    end = tss + flank
    seq = genome[chrom][start:end].seq  # this is genomic orientation

    if strand == "-":
        # reverse-complement for minus strand
        seq = seq.translate(comp)[::-1]
    return seq

gene_seq = {}
missing_chrom = set()

for gene, row in tss_df.iterrows():
    chrom = row["chrom"]
    strand = row["strand"]
    tss = row["tss"]

    # handle 'chr1' vs '1' differences if needed
    if chrom not in genome:
        alt_chrom = chrom.replace("chr", "")
        if alt_chrom in genome:
            chrom = alt_chrom
        else:
            missing_chrom.add(chrom)
            continue

    try:
        gene_seq[gene] = fetch_promoter(chrom, tss, strand, flank=FLANK)
    except KeyError:
        missing_chrom.add(chrom)

print(f"Got promoter sequences for {len(gene_seq)} genes.")
if missing_chrom:
    print("Missing chromosomes in FASTA:", missing_chrom)

# ---- 4. Attach promoter sequences to pairs and optionally write FASTA ----

pairs["promoter_seq"] = pairs["gene"].map(gene_seq)
before = pairs.shape[0]
pairs = pairs.dropna(subset=["promoter_seq"])
after = pairs.shape[0]
print(f"Dropped {before - after} TF-gene rows with missing promoter sequence.")
print("Final pairs shape:", pairs.shape)

# Save updated table for ML / embedding step
pairs.to_csv(f"tf_gene_pairs_with_promoters_flank{FLANK}.csv", index=False)

# Also write promoters to a FASTA file (one per gene) if you want
with open(f"promoters_flank{FLANK}.fasta", "w") as out:
    for g, seq in gene_seq.items():
        out.write(f">{g}\n{seq}\n")

print("Done writing promoters and updated pairs table.")
