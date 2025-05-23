# ----------------------------------------------------------------------------
# Copyright (c) 2016-2025, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from math import ceil
import os
import subprocess
import tempfile

from joblib import Parallel, delayed, effective_n_jobs
import pandas as pd
import skbio

from qiime2.plugin import Int, Str, Float, Range, Choices
from q2_types.feature_data import (FeatureData, Sequence, DNAIterator,
                                   DNASequencesDirectoryFormat, DNAFASTAFormat,
                                   SequenceCharacteristics)
from q2_feature_classifier._skl import _chunks
from q2_feature_classifier.classifier import _autotune_reads_per_batch

from .plugin_setup import plugin


def _seq_to_regex(seq):
    """Build a regex out of a IUPAC consensus sequence"""
    result = []
    for base in str(seq):
        if base in skbio.DNA.degenerate_chars:
            result.append('[{0}]'.format(
                ''.join(sorted(skbio.DNA.degenerate_map[base]))))
        else:
            result.append(base)

    return ''.join(result)


def _primers_to_regex(f_primer, r_primer):
    return '({0}.*{1})'.format(_seq_to_regex(f_primer),
                               _seq_to_regex(r_primer.reverse_complement()))


def _local_aln(primer, sequence):
    best_score = None
    for one_primer in sorted([str(s) for s in primer.expand_degenerates()]):
        # `sequence` may contain degenerates. These will usually be N
        # characters, which SSW will score as zero. Although undocumented, SSW
        # will treat other degenerate characters as a mismatch. We acknowledge
        # that this approach is a heuristic to finding an optimal alignment and
        # may be revisited in the future if there's an aligner that explicitly
        # handles degenerates.
        this_aln = \
            skbio.alignment.local_pairwise_align_ssw(skbio.DNA(one_primer),
                                                     sequence)
        score = this_aln[1]
        if best_score is None or score > best_score:
            best_score = score
            best_aln = this_aln
    return best_aln


def _semisemiglobal(primer, sequence, reverse=False):
    if reverse:
        primer = primer.reverse_complement()

    # locally align the primer
    (aln_prim, aln_seq), score, (prim_pos, seq_pos) = \
        _local_aln(primer, sequence)
    amplicon_pos = seq_pos[1]+len(primer)-prim_pos[1]

    # naively extend the alignment to be semi-global
    bits = [primer[:prim_pos[0]], aln_prim, primer[prim_pos[1]+1:]]
    aln_prim = ''.join(map(str, bits))
    bits = ['-'*(prim_pos[0]-seq_pos[0]),
            sequence[max(seq_pos[0]-prim_pos[0], 0):seq_pos[0]],
            aln_seq,
            sequence[seq_pos[1]+1:amplicon_pos],
            '-'*(amplicon_pos-len(sequence))]
    aln_seq = ''.join(map(str, bits))

    # count the matches
    matches = sum(s in skbio.DNA.degenerate_map.get(p, {p})
                  for p, s in zip(aln_prim, aln_seq))

    if reverse:
        amplicon_pos = max(seq_pos[0]-prim_pos[0], 0)

    return amplicon_pos, matches, len(aln_prim)


def _exact_match(seq, f_primer, r_primer):
    try:
        regex = _primers_to_regex(f_primer, r_primer)
        match = next(seq.find_with_regex(regex))
        beg, end = match.start + len(f_primer), match.stop - len(r_primer)
        return seq[beg:end]
    except StopIteration:
        return None


def _approx_match(seq, f_primer, r_primer, identity):
    beg, b_matches, b_length = _semisemiglobal(f_primer, seq)
    end, e_matches, e_length = _semisemiglobal(r_primer, seq, reverse=True)
    if (b_matches + e_matches) / (b_length + e_length) >= identity:
        return seq[beg:end]
    return None


def _gen_reads(sequence, f_primer, r_primer, trim_right, trunc_len, trim_left,
               identity, min_length, max_length, read_orientation):
    f_primer = skbio.DNA(f_primer)
    r_primer = skbio.DNA(r_primer)
    amp = None
    if read_orientation in ['forward', 'both']:
        amp = _exact_match(sequence, f_primer, r_primer)
    if not amp and read_orientation in ['reverse', 'both']:
        amp = _exact_match(sequence.reverse_complement(), f_primer, r_primer)
    if not amp and read_orientation in ['forward', 'both']:
        amp = _approx_match(sequence, f_primer, r_primer, identity)
    if not amp and read_orientation in ['reverse', 'both']:
        amp = _approx_match(
            sequence.reverse_complement(), f_primer, r_primer, identity)
    if not amp:
        return
    # we want to filter by max length before trimming
    if max_length > 0 and len(amp) > max_length:
        return
    if trim_right > 0:
        amp = amp[:-trim_right]
    if trunc_len > 0:
        amp = amp[:trunc_len]
    if trim_left > 0:
        amp = amp[trim_left:]
    if min_length > 0 and len(amp) < min_length:
        return
    if not amp:
        return
    return amp


def extract_reads(sequences: DNASequencesDirectoryFormat, f_primer: str,
                  r_primer: str, trim_right: int = 0,
                  trunc_len: int = 0, trim_left: int = 0,
                  identity: float = 0.8, min_length: int = 50,
                  max_length: int = 0, n_jobs: int = 1,
                  batch_size: int = 'auto', read_orientation: str = 'both') \
                  -> DNAFASTAFormat:
    """Extract the read selected by a primer or primer pair. Only sequences
    which match the primers at greater than the specified identity are returned

    Parameters
    ----------
    sequences : DNASequencesDirectoryFormat
        An aligned list of skbio.sequence.DNA query sequences
    f_primer : skbio.sequence.DNA
        Forward primer sequence
    r_primer : skbio.sequence.DNA
        Reverse primer sequence
    trim_right : int, optional
        `trim_right` nucleotides are removed from the 3' end if trim_right is
        positive. Applied before trunc_len.
    trunc_len : int, optional
        Read is cut to trunc_len if trunc_len is positive. Applied after
        trim_right.
    trim_left : int, optional
        `trim_left` nucleotides are removed from the 5' end if trim_left is
        positive. Applied after trim_right and trunc_len.
    identity : float, optional
        Minimum combined primer match identity threshold. Default: 0.8
    min_length: int, optional
        Minimum amplicon length. Shorter amplicons are discarded. Default: 50
    max_length: int, optional
        Maximum amplicon length. Longer amplicons are discarded.
    n_jobs: int, optional
        Number of seperate processes to break the task into.
    batch_size: int, optional
        Number of samples to be processed in one batch.
    read_orientation: str, optional
        'Orientation of primers relative to the sequences: "forward" searches '
        'for primer hits in the forward direction, "reverse" searches the '
        'reverse-complement, and "both" searches both directions.'
    Returns
    -------
    q2_types.DNAFASTAFormat
        containing the reads
    """
    if min_length > trunc_len - (trim_left + trim_right) and trunc_len > 0:
        raise ValueError('The minimum length setting is greater than the '
                         'length of the truncated sequences. This will cause '
                         'all sequences to be removed from the dataset. To '
                         'proceed, set '
                         'min_length ≤ trunc_len - (trim_left  + '
                         'trim_right).')

    n_jobs = effective_n_jobs(n_jobs)
    if batch_size == 'auto':
        batch_size = _autotune_reads_per_batch(
            sequences.file.view(DNAFASTAFormat), n_jobs)
    sequences = sequences.file.view(DNAIterator)
    ff = DNAFASTAFormat()
    with open(str(ff), 'a') as fh:
        with Parallel(n_jobs) as parallel:
            for chunk in _chunks(sequences, batch_size):
                amplicons = parallel(delayed(_gen_reads)(sequence, f_primer,
                                                         r_primer,
                                                         trim_right,
                                                         trunc_len,
                                                         trim_left,
                                                         identity,
                                                         min_length,
                                                         max_length,
                                                         read_orientation)
                                     for sequence in chunk)
                for amplicon in amplicons:
                    if amplicon is not None:
                        skbio.write(amplicon, format='fasta', into=fh)
    if os.stat(str(ff)).st_size == 0:
        raise RuntimeError("No matches found")
    return ff


plugin.methods.register_function(
    function=extract_reads,
    inputs={'sequences': FeatureData[Sequence]},
    parameters={'trunc_len': Int,
                'trim_left': Int,
                'trim_right': Int,
                'f_primer': Str,
                'r_primer': Str,
                'identity': Float,
                'min_length': Int % Range(0, None),
                'max_length': Int % Range(0, None),
                'n_jobs': Int % Range(1, None),
                'batch_size': Int % Range(1, None) | Str % Choices(['auto']),
                'read_orientation': Str % Choices(['both', 'forward',
                                                   'reverse'])},
    outputs=[('reads', FeatureData[Sequence])],
    name='Extract reads from reference sequences.',
    description='Extract simulated amplicon reads from a reference database. '
                'Performs in-silico PCR to extract simulated amplicons from '
                'reference sequences that match the input primer sequences '
                '(within the mismatch threshold specified by `identity`). '
                'Both primer sequences must be in the 5\' -> 3\' orientation. '
                'Sequences that fail to match both primers will be excluded. '
                'Reads are extracted, trimmed, and filtered in the following '
                'order: 1. reads are extracted in specified orientation; 2. '
                'primers are removed; 3. reads longer than `max_length` are '
                'removed; 4. reads are trimmed with `trim_right`; 5. reads '
                'are truncated to `trunc_len`; 6. reads are trimmed with '
                '`trim_left`; 7. reads shorter than `min_length` are removed.',
    parameter_descriptions={
        'f_primer': 'forward primer sequence (5\' -> 3\').',
        'r_primer': 'reverse primer sequence (5\' -> 3\'). Do not use reverse-'
                    'complemented primer sequence.',
        'trim_right': 'trim_right nucleotides are removed from the 3\' end if '
                      'trim_right is positive. Applied before trunc_len and '
                      'trim_left.',
        'trunc_len': 'read is cut to trunc_len if trunc_len is positive. '
                     'Applied after trim_right but before trim_left.',
        'trim_left': 'trim_left nucleotides are removed from the 5\' end if '
                     'trim_left is positive. Applied after trim_right and '
                     'trunc_len.',
        'identity': 'minimum combined primer match identity threshold.',
        'min_length': 'Minimum amplicon length. Shorter amplicons are '
                      'discarded. Applied after trimming and truncation, so '
                      'be aware that trimming may impact sequence retention. '
                      'Set to zero to disable min length filtering.',
        'max_length': 'Maximum amplicon length. Longer amplicons are '
                      'discarded. Applied before trimming and truncation, '
                      'so plan accordingly. Set to zero (default) to disable '
                      'max length filtering.',
        'n_jobs': 'Number of seperate processes to run.',
        'batch_size': 'Number of sequences to process in a batch. The `auto` '
                      'option is calculated from the number of sequences and '
                      'number of jobs specified.',
        'read_orientation': 'Orientation of primers relative to the '
                            'sequences: "forward" searches for primer hits in '
                            'the forward direction, "reverse" searches '
                            'reverse-complement, and "both" searches both '
                            'directions.'}
)


def _write_barcode_fasta(
        temp_dir_path: str, f_primer: str, r_primer: str,
) -> None:
    with open(os.path.join(temp_dir_path, "barcodes.fasta"), "w") as fh:
        fh.write(f">forward\n{f_primer}\n")
        fh.write(f">reverse\n{r_primer}\n")
    with open(os.path.join(temp_dir_path, "forward.fasta"), "w") as fh:
        fh.write(f">forward\n{f_primer}\n")
    with open(os.path.join(temp_dir_path, "reverse.fasta"), "w") as fh:
        fh.write(f">reverse\n{r_primer}\n")


def _usearch_search_oligodb(
        sequences_path: str, temp_dir_path: str, direction: str,
        strand: str = "both", max_diffs: int = 2,
) -> pd.DataFrame:
    primers_path = os.path.join(temp_dir_path, f"{direction}.fasta")
    result_path = os.path.join(temp_dir_path, f"search_oligodb-{direction}.tsv")
    cmd = [
        "usearch",
        "-search_oligodb", str(sequences_path),
        "-db", str(primers_path),
        "-maxdiffs", str(max_diffs),
        "-strand", strand,
        "-userfields", "query+qstrand+qrow+trowdots",
        "-userout", result_path
    ]
    subprocess.run(cmd, check=True)

    fwd = direction == "forward"
    return pd.read_table(
        result_path, sep="\t",
        names=[
            "Feature ID", "strand", f"primer_{'F' if fwd else 'R'}_seq",
            f"primer_{'F' if fwd else 'R'}_dot"
        ]
    )


def _usearch_search_pcr(
        sequences_path: str, temp_dir_path: str, strand: str = "both",
        max_diffs: int = 2, min_length: int = 50, max_length: int = 0,
) -> pd.DataFrame:
    primers_path = os.path.join(temp_dir_path, f"barcodes.fasta")
    result_path = os.path.join(temp_dir_path, f"search_pcr.tsv")
    cmd = [
        "usearch",
        "-search_pcr" , str(sequences_path),
        "-db", str(primers_path),
        "-strand", strand,
        "-maxdiffs", str(max_diffs),
        "-pcrout", result_path,
        "-ampout", "/dev/null",
    ]
    if min_length is not None:
        cmd.extend(["-minamp", str(min_length)])
    if max_length >0:
        cmd.extend(["-maxamp", str(max_length)])
    subprocess.run(cmd, check=True)

    rv = pd.read_table(
        result_path, sep="\t",
        names=[
            "Feature ID",
            "start", "end", "length",
            "primer_F_label", "pcr_primer_F_strand", "pcr_primer_F_dot",
            "primer_R_label", "pcr_primer_R_strand", "pcr_primer_R_dot",
            "amplicon_length", "amplicon_sequence",
            "primer_F_mismatches", "primer_R_mismatches", "primers_mismatches",
        ]
    ).drop(
        # Amplicon sequences still contain the primers, amplicon length will be
        #  recomputed later
        ["start", "end", "length", "primer_F_label", "primer_R_label",
         "amplicon_length"],
        axis=1
    )
    # Some sequences may have multiple amplicons, we only want to keep the
    #  one with the least mismatches
    rv.sort_values(["Feature ID", "primers_mismatches"], inplace=True)
    rv.drop_duplicates("Feature ID", keep="first", inplace=True)
    return rv


def _combine_usearch_results(
        oligodb_forward_df: pd.DataFrame, oligodb_reverse_df: pd.DataFrame,
        search_pcr_df: pd.DataFrame,
) -> pd.DataFrame:
    """Combine the results from search_oligodb and search_pcr.
    The aim is to replace the "primer dot sequences" with the actual matching
    sequences."""
    # There are some discrepancies in the strand direction reported by
    #  search_oligodb and search_pcr, so we have to merge twice per direction
    # Positive (p) strand in search_pcr and positive (p) strand in search_oligodb
    merged_p_p = (
        search_pcr_df[
            (search_pcr_df["pcr_primer_F_strand"] == "+")
            & (search_pcr_df["pcr_primer_R_strand"] == "-")
        ]
        # Merge forward primers in the expected direction
        .merge(
            oligodb_forward_df.loc[oligodb_forward_df["strand"] == "+"].drop('strand', axis=1),
            how="inner",
            left_on=["Feature ID", "pcr_primer_F_dot"],
            right_on=["Feature ID", "primer_F_dot"],
        )
        # Merge reverse primers in the expected direction
        .merge(
            oligodb_reverse_df[oligodb_reverse_df["strand"] == "-"].drop('strand', axis=1),
            how="inner",
            left_on=["Feature ID", "pcr_primer_R_dot"],
            right_on=["Feature ID", "primer_R_dot"],
        )
    )  # noqa: E501 improves readability

    # Positive (p) strand in search_pcr and negative (n) strand in search_oligodb
    merged_p_n = (
        search_pcr_df[
            (search_pcr_df["pcr_primer_F_strand"] == "+")
            & (search_pcr_df["pcr_primer_R_strand"] == "-")
        ]
        # Merge forward primers wrongly (?) detected in the opposite direction
        #  by search_oligodb
        .merge(
            oligodb_forward_df[oligodb_forward_df["strand"] == "-"].drop('strand', axis=1),
            how="inner",
            left_on=["Feature ID", "pcr_primer_R_dot"],
            right_on=["Feature ID", "primer_F_dot"],
        )
        # Merge reverse primers wrongly (?) detected in the opposite direction
        #  by search_oligodb
        .merge(
            oligodb_reverse_df[oligodb_reverse_df["strand"] == "+"].drop('strand', axis=1),
            how="inner",
            left_on=["Feature ID", "pcr_primer_F_dot"],
            right_on=["Feature ID", "primer_R_dot"],
        )
    )  # noqa: E501 improves readability

    # Negative (n) strand in search_pcr and negative (n) strand in search_oligodb
    merged_n_n = (
        search_pcr_df[
            (search_pcr_df["pcr_primer_F_strand"] == "-")
            & (search_pcr_df["pcr_primer_R_strand"] == "+")
        ]
        .merge(
            oligodb_forward_df[oligodb_forward_df["strand"] == "-"].drop('strand', axis=1),
            how="inner",
            left_on=["Feature ID", "pcr_primer_F_dot"],
            right_on=["Feature ID", "primer_F_dot"],
        )
        .merge(
            oligodb_reverse_df[oligodb_reverse_df["strand"] == "+"].drop('strand', axis=1),
            how="inner",
            left_on=["Feature ID", "pcr_primer_R_dot"],
            right_on=["Feature ID", "primer_R_dot"],
        )
    )  # noqa: E501 improves readability

    # Negative (n) strand in search_pcr and positive (p) strand in search_oligodb
    merged_n_p = (
        search_pcr_df[
            (search_pcr_df["pcr_primer_F_strand"] == "-")
            & (search_pcr_df["pcr_primer_R_strand"] == "+")
        ]
        .merge(
            oligodb_forward_df[oligodb_forward_df["strand"] == "+"].drop('strand', axis=1),
            how="inner",
            left_on=["Feature ID", "pcr_primer_R_dot"],
            right_on=["Feature ID", "primer_F_dot"],
        )
        .merge(
            oligodb_reverse_df[oligodb_reverse_df["strand"] == "-"].drop('strand', axis=1),
            how="inner",
            left_on=["Feature ID", "pcr_primer_F_dot"],
            right_on=["Feature ID", "primer_R_dot"],
        )
    )  # noqa: E501 improves readability

    rv = pd.concat([merged_p_p, merged_p_n, merged_n_n, merged_n_p])
    rv = rv[[
        "Feature ID", "primer_F_seq", "primer_R_seq", "amplicon_sequence",
        "primer_F_mismatches", "primer_R_mismatches", "primers_mismatches"
    ]]
    return rv


def _clean_amplicons(pcr_summary: pd.DataFrame) -> pd.DataFrame:
    """search_pcr returns amplicons with the primers """
    pcr_summary["trimmed_sequence"] = ""
    pcr_summary["amplicon_length"] = 0
    pcr_summary["trimmed_length"] = 0
    for index, row in pcr_summary.iterrows():
        f_primer = row["primer_F_seq"]
        r_primer = row["primer_R_seq"]
        amplicon = row["amplicon_sequence"]
        if amplicon[:len(f_primer)] == f_primer:
            # The reported sequence is in the expected orientation, we just need
            #  to trim the primers
            trimmed_amplicon = amplicon[len(f_primer):-len(r_primer)]
        else:
            # We need to trim and reverse complement the amplicon so it is in
            #  the expected orientation
            amplicon = str(skbio.DNA(amplicon).reverse_complement())
            trimmed_amplicon = amplicon[len(r_primer):-len(f_primer)]
        pcr_summary.at[index, "amplicon_sequence"] = amplicon
        pcr_summary.at[index, "amplicon_length"] = len(amplicon)
        pcr_summary.at[index, "trimmed_sequence"] = trimmed_amplicon
        pcr_summary.at[index, "trimmed_length"] = len(trimmed_amplicon)

    pcr_summary = pcr_summary[[
        "Feature ID", "primer_F_seq", "primer_R_seq",
        "primer_F_mismatches", "primer_R_mismatches", "primers_mismatches",
        "amplicon_sequence", "amplicon_length", "trimmed_sequence",
        "trimmed_length",
    ]]
    return pcr_summary


def _generate_summary(
        oligodb_forward: pd.DataFrame, oligodb_reverse: pd.DataFrame,
        search_pcr: pd.DataFrame
) -> pd.DataFrame:
    pcr_summary = _combine_usearch_results(
        oligodb_forward, oligodb_reverse, search_pcr)
    pcr_summary = _clean_amplicons(pcr_summary)
    pcr_summary.set_index("Feature ID", inplace=True)
    return pcr_summary


def _write_reads(output_path: DNAFASTAFormat, summary_df: pd.DataFrame) -> None:
    with open(str(output_path), 'a') as fh:
        for index, row in summary_df.iterrows():
            fh.write(f">{row['Feature ID']}\n{row['trimmed_sequence']}\n")


def usearch_PCR(
        sequences: DNASequencesDirectoryFormat,
        f_primer: str, r_primer: str, read_orientation: str = "both",
        identity: float = 0.8, min_length: int = 50, max_length: int = 1000,
        # trim_right: int = 0, trim_left: int = 0, trunc_len: int = 0,
) -> (DNAFASTAFormat, pd.DataFrame):
    with tempfile.TemporaryDirectory() as temp_dir_path:
        _write_barcode_fasta(temp_dir_path, f_primer, r_primer)
        sequences_path = f"{sequences}/dna-sequences.fasta"
        # Search database for forward primers match
        oligodb_forward_df = _usearch_search_oligodb(
            sequences_path, temp_dir_path,"forward",
            strand=read_orientation, max_diffs=ceil(len(f_primer) * (1 - identity)))
        # Search database for reverse primers match
        oligodb_reverse_df = _usearch_search_oligodb(
            sequences_path, temp_dir_path,"reverse",
            strand=read_orientation, max_diffs=ceil(len(r_primer) * (1 - identity)))
        # Search database for forward and reverse primers match
        search_pcr_df = _usearch_search_pcr(
            sequences_path, temp_dir_path, strand=read_orientation,
            max_diffs=ceil(len(f_primer)*(1-identity)), min_length=min_length,
            max_length=max_length)
    # Merge the results
    pcr_summary_df = _combine_usearch_results(
        oligodb_forward_df, oligodb_reverse_df, search_pcr_df)
    # Clean the amplicons
    pcr_summary_df = _clean_amplicons(pcr_summary_df)
    # Extract the amplicon sequences
    sequences_output = DNAFASTAFormat()
    _write_reads(sequences_output, pcr_summary_df)
    return sequences_output, pcr_summary_df


plugin.methods.register_function(
    function=usearch_PCR,
    inputs={'sequences': FeatureData[Sequence]},
    parameters={
        'f_primer': Str,
        'r_primer': Str,
        'identity': Float,
        'min_length': Int % Range(0, None),
        'max_length': Int % Range(0, None),
        'read_orientation': Str % Choices(['both', 'forward', 'reverse'])},
    outputs=[
        ('reads', FeatureData[Sequence]),
        ('stats', FeatureData[SequenceCharacteristics]),
    ],
    name='Extract reads from reference sequences.',
    description='Extract simulated amplicon reads from a reference database. '
                'Performs in-silico PCR to extract simulated amplicons from '
                'reference sequences that match the input primer sequences '
                '(within the mismatch threshold specified by `identity`). '
                'Both primer sequences must be in the 5\' -> 3\' orientation. '
                'Sequences that fail to match both primers will be excluded. '
                'Reads are extracted, trimmed, and filtered in the following '
                'order: 1. reads are extracted in specified orientation; 2. '
                'reads longer than `max_length` are removed; 3. reads shorter '
                'than `min_length` are removed. 4. forward and reverse primers '
                'are removed.',
    parameter_descriptions={
        'f_primer': 'forward primer sequence (5\' -> 3\').',
        'r_primer': 'reverse primer sequence (5\' -> 3\'). Do not use reverse-'
                    'complemented primer sequence.',
        'identity': 'minimum combined primer match identity threshold.',
        'min_length': 'Minimum amplicon length. Shorter amplicons are '
                      'discarded. Applied after trimming and truncation, so '
                      'be aware that trimming may impact sequence retention. '
                      'Set to zero to disable min length filtering.',
        'max_length': 'Maximum amplicon length. Longer amplicons are '
                      'discarded. Applied before trimming and truncation, '
                      'so plan accordingly. Set to zero (default) to disable '
                      'max length filtering.',
        'read_orientation': 'Orientation of primers relative to the '
                            'sequences: "forward" searches for primer hits in '
                            'the forward direction, "reverse" searches '
                            'reverse-complement, and "both" searches both '
                            'directions.'}
)
