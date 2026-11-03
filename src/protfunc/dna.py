from dataclasses import dataclass
from Bio.Seq import Seq

@dataclass(frozen=True)
class ORFCandidate:
    frame: str
    start: int
    end: int
    protein: str
    complete: bool

def clean_dna(raw: str, min_length: int = 3, max_length: int = 1_000_000) -> str:
    if not isinstance(raw, str): raise ValueError("DNA input must be text")
    lines = [x.strip() for x in raw.splitlines() if x.strip()]
    if lines and lines[0].startswith(">"):
        lines = lines[1:]
    seq = "".join("".join(lines).split()).upper()
    if not seq: raise ValueError("DNA sequence is empty")
    bad = sorted(set(seq) - set("ACGTN"))
    if bad: raise ValueError(f"Invalid DNA characters: {''.join(bad)} (allowed: ACGTN)")
    if len(seq) < min_length: raise ValueError(f"DNA length must be at least {min_length} bases")
    if len(seq) > max_length: raise ValueError(f"DNA length exceeds maximum of {max_length} bases")
    return seq

def find_orfs(raw: str, min_aa: int = 1) -> list[ORFCandidate]:
    dna = clean_dna(raw); found = []
    for strand, seq in (("+", Seq(dna)), ("-", Seq(dna).reverse_complement())):
        for offset in range(3):
            bases = str(seq[offset:]); codons = [bases[i:i+3] for i in range(0, len(bases)-2, 3)]; starts = []
            for i, codon in enumerate(codons):
                if codon == "ATG": starts.append(i)
                if codon in {"TAA", "TAG", "TGA"}:
                    for start in starts:
                        peptide = str(Seq("".join(codons[start:i])).translate())
                        if len(peptide) >= min_aa:
                            a,b=offset+start*3,offset+(i+1)*3
                            if strand == "-": a,b=len(dna)-b,len(dna)-a
                            found.append(ORFCandidate(f"{strand}{offset+1}",a,b,peptide,True))
                    starts=[]
            for start in starts:
                peptide=str(Seq("".join(codons[start:])).translate())
                if len(peptide)>=min_aa:
                    a,b=offset+start*3,offset+len(codons)*3
                    if strand=="-": a,b=len(dna)-b,len(dna)-a
                    found.append(ORFCandidate(f"{strand}{offset+1}",a,b,peptide,False))
    return sorted(found,key=lambda x:(len(x.protein),x.complete),reverse=True)

def longest_orf(raw: str, min_aa: int = 1):
    candidates=find_orfs(raw,min_aa)
    if not candidates: raise ValueError("No ORF found (expected an ATG start codon and a translated peptide)")
    return candidates[0],candidates[1:]
