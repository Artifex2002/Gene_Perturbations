import gzip
import pandas as pd
from pyfaidx import Fasta

# paths and flank
GTF_PATH = "gencode.v44.annotation.gtf.gz"
GENOME_FASTA = "GRCh38.primary_assembly.genome.fa"

FLANK = 1000  # bp upstream/downstream of TSS.


# this load gene list from pairs table 
pairs = pd.read_csv("tf_gene_pairs.csv")  # or however   saved it
gene_list = sorted(pairs["gene"].unique())
gene_set = set(gene_list)
print(f"Number of unique genes in pairs: {len(gene_set)}")

# gene -> (chrom, strand, TSS) map from GENCODE 

# we'll use 'transcript' entries to get TSS and then pick one transcript per gene.

rows = []
with gzip.open(GTF_PATH, "rt") as fh:
    for line in fh:
        if line.startswith("#"):
            continue
        chrom, src, feature, start, end, score, strand, frame, attrs = line.strip().split("\t")
        if feature != "transcript":
            continue

        # parse attributes into a dictionairy 
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

# If multiple transcripts per gene, pick one
# take the "most upstream" TSS for + strand, "most downstream" for - strand.
tss_df = (
    tss_df
    .sort_values(["gene", "chrom", "strand", "tss"])
    .drop_duplicates(subset=["gene"], keep="first")
    .set_index("gene")
)



print("ex. TSS rows look like:")
print(tss_df.head())
print(f"Final TSS map size: {tss_df.shape[0]} genes.")

# extract promoter sequences from genome FASTA 
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

# attach promoter sequences to pairs and write FASTA

pairs["promoter_seq"] = pairs["gene"].map(gene_seq)
before = pairs.shape[0]
pairs = pairs.dropna(subset=["promoter_seq"])
after = pairs.shape[0]
print(f"Dropped {before - after} TF-gene rows with missing promoter sequence.")
print("Final pairs shape:", pairs.shape)

# save updated table for embedding step
pairs.to_csv(f"tf_gene_pairs_with_promoters_flank{FLANK}.csv", index=False)

# Also write promoters to a FASTA file (one per gene) if wanted. this is a lightweight file that can be used by some embedders etc.
with open(f"promoters_flank{FLANK}.fasta", "w") as out:
    for g, seq in gene_seq.items():
        out.write(f">{g}\n{seq}\n")

print("Done writing promoters and updated pairs table.")
