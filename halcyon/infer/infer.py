"""
Infer the fasta files from fast5 files.
It split the signals and mergs the result by
pairwise alignemnt.
"""

import json
import os
import sys
from glob import glob
from os.path import basename, dirname, join
from typing import List, Optional, TextIO

from halcyon.infer.download import download
from halcyon.infer.inferer import Inferer
from logzero import logger
from more_itertools import chunked


def export_one_chunk(
    inferer: Inferer,
    fast5_paths: List[str],
    fasta_path: str,
    chunk_size: int,
    force: bool,
    export_meta: bool,
    outputs_qual_file: bool,
) -> None:
    """
    Process all fast5 existing in the same directory.
    The result will be in a single fastx file.
    """
    if (len(fast5_paths) == 0):
        return
    logger.info(f'{len(fast5_paths)} fast5 paths will be processed')
    if os.path.exists(fasta_path) and not force:
        logger.info('{} already exists.'.format(fasta_path))
        return
    logger.info('Export to {}'.format(fasta_path))
    quality_path = fasta_path.replace('.fasta', '.qual')
    output_dir = dirname(fasta_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    sub_chunks = list(chunked(fast5_paths, chunk_size))
    fasta_f = open(fasta_path, 'w')
    quality_f: Optional[TextIO]
    if outputs_qual_file:
        quality_f = open(quality_path, 'w')
    else:
        quality_f = None
    meta_d = {}
    for sub_chunk in sub_chunks:
        inferer_output = inferer.infer_fast5s(
            sub_chunk,
        )
        meta_d.update(inferer_output.meta_d)
        for signals_ID, seq_logits_pair in inferer_output.res_d.items():
            # Write to fasta
            fasta_str = '>{}\n{}\n'.format(
                signals_ID,
                seq_logits_pair.seq,
            )
            fasta_f.write(fasta_str)
            fasta_f.flush()
            # Write to quality
            if quality_f:
                quality_str = '>{}\n{}\n'.format(
                    signals_ID,
                    ','.join([
                        '{:3.2f}'.format(logit) for logit in
                        seq_logits_pair.logits
                    ]),
                )
                quality_f.write(quality_str)
                quality_f.flush()
    if export_meta:
        meta_path = os.path.join(
            dirname(fasta_path),
            '.{}.meta.json'.format(basename(fasta_path)),
        )
        with open(meta_path, 'w') as meta_f:
            json.dump(meta_d, meta_f)
    fasta_f.close()
    if quality_f:
        quality_f.close()
    return


def run(
    input_dir_path: str,
    output_fasta_path: str,
    config: str,
    signals_len: int,
    overlap_len: int,
    name: str,
    minibatch_size: int,
    chunk_size: int,
    beam_width: int,
    threads: int,
    gpus: List[int],
    ignores_alignment_history: bool,
    keeps_full_alignment: bool,
    outputs_qual_file: bool,
    exports_meta: bool,
    force: bool,
    verbose: bool,
) -> None:
    fast5_paths = glob(
        join(
            input_dir_path,
            '*.fast5',
        ))
    if not config:
        _config = download()
        if not _config:
            print('Failed to down load model.', file=sys.stderr)
            return None
        config = _config
    inferer = Inferer(
        config_path=config,
        signals_len=signals_len,
        overlap_len=overlap_len,
        name=name,
        minibatch_size=minibatch_size,
        beam_width=beam_width,
        num_threads=threads,
        gpus=gpus,
        ignore_alignment_history=ignores_alignment_history,
        keep_full_alignment=keeps_full_alignment,
        verbose=verbose,
    )
    export_one_chunk(
        inferer=inferer,
        fast5_paths=fast5_paths,
        fasta_path=output_fasta_path,
        chunk_size=chunk_size,
        force=force,
        export_meta=exports_meta,
        outputs_qual_file=outputs_qual_file,
    )
    inferer.close()
    return
