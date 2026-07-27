"""Export haplotype matrices to the formats RFMix and FLARE expect.

Both tools consume phased diploid VCFs plus a genetic map, so haplotypes are
paired into pseudo-diploid individuals. Because both tools report per-haplotype
ancestry, the pairing is transparent: haplotype 2i and 2i+1 of the matrix
become the two phased alleles of individual i, and their calls are recovered
separately.

Positions from a coalescent simulation are continuous and may collide once
rounded to integers, which a VCF cannot represent. Colliding sites are dropped
rather than jittered, and the retained index is returned so that ground-truth
labels can be subset to match exactly.
"""

import subprocess
from pathlib import Path

import numpy as np

CM_PER_BP = 1e-8 * 100.0  # constant-rate map: 1 cM per Mb at rho = 1e-8


def _dedup_positions(positions):
    """Round to integer bp and keep the first occurrence of each position."""
    pos = np.round(np.asarray(positions, dtype=np.float64)).astype(np.int64)
    pos = np.maximum(pos, 1)
    keep = np.ones(pos.size, dtype=bool)
    keep[1:] = pos[1:] > pos[:-1]
    return pos[keep], np.flatnonzero(keep)


def write_vcf(path, haps, positions, chrom, sample_prefix, bgzip=True):
    """Write a phased diploid VCF. haps is (n_sites, n_haplotypes), even count."""
    n_sites, n_haps = haps.shape
    assert n_haps % 2 == 0, "haplotype count must be even to pair into diploids"
    n_ind = n_haps // 2
    names = [f"{sample_prefix}{i}" for i in range(n_ind)]

    lines = [
        "##fileformat=VCFv4.2",
        f"##contig=<ID={chrom},length={int(positions[-1]) + 1000}>",
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(names),
    ]
    # Interleave allele columns so that column i is "h_{2i}|h_{2i+1}".
    left = haps[:, 0::2].astype(np.uint8)
    right = haps[:, 1::2].astype(np.uint8)
    zero = ord("0")
    for k in range(n_sites):
        gt = np.empty(n_ind * 4 - 1, dtype=np.uint8)
        gt[:] = ord("\t")
        gt[0::4] = left[k] + zero
        gt[1::4] = ord("|")
        gt[2::4] = right[k] + zero
        lines.append(
            f"{chrom}\t{positions[k]}\t.\tA\tG\t.\tPASS\t.\tGT\t"
            + gt.tobytes().decode("ascii")
        )

    text = "\n".join(lines) + "\n"
    path = Path(path)
    if bgzip:
        import pysam

        raw = path.with_suffix("") if path.suffix == ".gz" else path
        raw.write_text(text)
        pysam.tabix_compress(str(raw), str(path), force=True)
        pysam.tabix_index(str(path), preset="vcf", force=True)
        raw.unlink()
    else:
        path.write_text(text)
    return names


def write_genetic_map(path, positions, chrom, plink=False):
    """Constant-rate genetic map.

    RFMix wants  chrom <tab> physical_pos <tab> genetic_pos_cM
    PLINK/FLARE wants  chrom <tab> marker_id <tab> genetic_pos_cM <tab> physical_pos
    """
    cm = np.asarray(positions, dtype=np.float64) * CM_PER_BP
    with open(path, "w") as fh:
        for p, c in zip(positions, cm):
            if plink:
                fh.write(f"{chrom}\t{chrom}:{p}\t{c:.8f}\t{p}\n")
            else:
                fh.write(f"{chrom}\t{p}\t{c:.8f}\n")


def write_sample_map(path, names_a, names_b, pop_a="POPA", pop_b="POPB"):
    with open(path, "w") as fh:
        for n in names_a:
            fh.write(f"{n}\t{pop_a}\n")
        for n in names_b:
            fh.write(f"{n}\t{pop_b}\n")


def run(cmd, log_path, timeout=3600):
    """Run an external tool, capturing output; returns (ok, tail_of_log)."""
    with open(log_path, "w") as fh:
        try:
            r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                               timeout=timeout, check=False)
            ok = r.returncode == 0
        except subprocess.TimeoutExpired:
            return False, "TIMEOUT"
    tail = Path(log_path).read_text()[-1500:]
    return ok, tail
